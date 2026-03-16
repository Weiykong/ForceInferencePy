"""
Topology Extraction for Force Inference.

SKELETON-GRAPH APPROACH — fully robust.

Pipeline:
  1. Threshold label image → binary boundary mask
  2. Skeletonize → guaranteed 1px-wide medial axis
  3. Find branch pixels (degree ≥ 3) → cluster → vertices
  4. Add border vertices (skeleton pixels touching image edge)
  5. Cut skeleton at branch/border neighborhoods → isolated 1-D segments
  6. Match each segment to its two nearest vertices → edges
  7. Sample labels on each side of segment → E_cells
  8. Order pixels along each segment → E_pixels
  9. Build C_v (per-cell vertex loop)

Tested on binary membrane image (white membranes, black background).
Works equally well on label images from segmentation.
"""

import numpy as np
from skimage import morphology, measure
from scipy.ndimage import convolve, label as nd_label
from scipy.spatial import cKDTree
import logging
from itertools import combinations
from typing import Optional, List, Tuple

from .core import Tissue

logger = logging.getLogger("ForceInference.Topology")


# =============================================================================
# Public API
# =============================================================================

def extract_topology(
    labels: np.ndarray,
    cell_labels: Optional[np.ndarray] = None,
    min_edge_len: float = 3.0,
    trace_pixels: bool = True,
    clean: bool = False,
    remove_outer_layer: bool = False,
    vertex_cluster_r: float = 3.0,
    vertex_search_r: float = 12.0,
    branch_cut_r: float = 1.0,
) -> Optional[Tissue]:
    """
    Extract tissue topology using skeleton-graph method.

    Args:
        labels:            Segmented label image (int). Background = 0.
                           Also accepts binary membrane images (255=membrane).
        min_edge_len:      Minimum edge length to keep after optional cleaning.
        trace_pixels:      If True, store ordered pixel paths in tissue.E_pixels.
        clean:             If True, collapse edges shorter than min_edge_len.
        remove_outer_layer: Remove cells touching the image border.
        vertex_cluster_r:  Radius (px) for merging nearby branch pixels into
                           one vertex. Increase for blurry/wide junctions.
        vertex_search_r:   Max distance (px) from a segment endpoint to its
                           nearest vertex. Increase if edges are being missed.
        branch_cut_r:      Radius (px) used to dilate branch pixels before
                           cutting the skeleton into segments. Use 0.0 to keep
                           very short edges between nearby junctions.

    Returns:
        Tissue object or None if extraction fails.
    """
    logger.info("Extracting topology (Skeleton-Graph)...")

    labels_proc = labels.copy()
    cell_labels_proc = cell_labels.copy() if cell_labels is not None else labels_proc
    H, W = labels_proc.shape

    # ------------------------------------------------------------------ #
    # 0.  Pre-processing                                                   #
    # ------------------------------------------------------------------ #
    if remove_outer_layer:
        bm = np.zeros_like(labels_proc, dtype=bool)
        bm[0, :] = bm[-1, :] = bm[:, 0] = bm[:, -1] = True
        border_labels = np.unique(labels_proc[bm])
        labels_proc[np.isin(labels_proc, border_labels)] = 0
        if cell_labels is not None:
            border_cell_lbls = np.unique(cell_labels_proc[bm])
            cell_labels_proc[np.isin(cell_labels_proc, border_cell_lbls)] = 0

    use_cell_labels_topology = (
        cell_labels is not None and len(np.unique(cell_labels_proc)) > 2
    )
    boundary_source = cell_labels_proc if use_cell_labels_topology else labels_proc
    # For label-driven topology, avoid extra dilation so close interfaces are
    # not merged before skeletonization.
    boundary = _labels_to_boundary(
        boundary_source, dilate_label_boundary=not use_cell_labels_topology
    )
    if not np.any(boundary):
        logger.warning("Empty boundary mask.")
        return None

    # ------------------------------------------------------------------ #
    # 1.  Skeletonize                                                       #
    # ------------------------------------------------------------------ #
    skel = morphology.skeletonize(boundary)
    logger.info(f"Skeleton: {int(np.sum(skel))} pixels")

    # ------------------------------------------------------------------ #
    # 2.  Vertices = clustered branch points + border touch points         #
    # ------------------------------------------------------------------ #
    kernel = np.ones((3, 3), dtype=np.uint8)
    kernel[1, 1] = 0
    nc = convolve(skel.astype(np.uint8), kernel, mode='constant', cval=0)

    skel_branch_mask = skel & (nc >= 3)
    branch_mask_for_cut = skel_branch_mask.copy()
    branch_vertices = _cluster_mask_components(skel_branch_mask, vertex_cluster_r)
    label_vertices = np.zeros((0, 2), dtype=float)
    if cell_labels is not None:
        label_junction_mask = _label_junctions_on_skeleton(
            cell_labels_proc, skel, snap_radius=2.5
        )
        if np.any(label_junction_mask):
            branch_mask_for_cut |= label_junction_mask
            label_vertices = _cluster_mask_components(
                label_junction_mask, max(0.75, 0.5 * vertex_cluster_r)
            )
            logger.info(
                f"Added {int(np.sum(label_junction_mask))} "
                "junction pixels from cell-label topology."
            )
            logger.info(f"Label-junction vertices: {len(label_vertices)}")

    # Border touch points: skeleton pixels on the image edge
    border_mask = np.zeros((H, W), dtype=bool)
    border_mask[0, :]  = skel[0, :]
    border_mask[-1, :] = skel[-1, :]
    border_mask[:, 0]  = skel[:, 0]
    border_mask[:, -1] = skel[:, -1]
    border_mask &= ~branch_mask_for_cut
    border_vertices = _cluster_mask_components(
        border_mask, max(1.0, vertex_cluster_r)
    )

    vertex_sets = [arr for arr in (branch_vertices, label_vertices, border_vertices) if len(arr) > 0]
    if not vertex_sets:
        logger.warning("No branch points found.")
        return None

    vertices = np.vstack(vertex_sets)
    # Only de-duplicate nearly identical points; keep close twin junctions separate.
    vertices = _cluster_points(vertices, radius=0.35)
    logger.info(
        f"Vertices: {len(vertices)} "
        f"(branch={len(branch_vertices)}, label={len(label_vertices)}, "
        f"border={len(border_vertices)})"
    )

    # ------------------------------------------------------------------ #
    # 3.  Cut skeleton → isolated segments = candidate edges               #
    # ------------------------------------------------------------------ #
    cut_r = max(0, int(np.ceil(float(branch_cut_r))))
    cut_branch = (
        morphology.dilation(skel_branch_mask, morphology.disk(cut_r))
        if cut_r > 0 else skel_branch_mask.copy()
    )
    cut_mask = cut_branch | branch_mask_for_cut | border_mask
    seg_skel = skel & ~cut_mask

    seg_labeled, n_segs = nd_label(seg_skel, structure=np.ones((3, 3), dtype=int))
    logger.info(f"Raw segments: {n_segs}")

    # ------------------------------------------------------------------ #
    # 4.  Match segments to (vertex, vertex) pairs                         #
    # ------------------------------------------------------------------ #
    vtree = cKDTree(vertices)

    cut_y, cut_x = np.where(cut_mask)
    cut_pts = np.column_stack((cut_x.astype(float), cut_y.astype(float)))
    cut_tree = cKDTree(cut_pts)

    edges = {}  # key=(i_v1, i_v2) → {'pts': ndarray, 'len': int}

    for sid in range(1, n_segs + 1):
        sy, sx = np.where(seg_labeled == sid)
        if len(sy) < 2:
            continue
        pts = np.column_stack((sx.astype(float), sy.astype(float)))

        ep1, ep2 = _segment_endpoints(pts, cut_tree)
        if ep1 is None:
            continue

        v_pair = _match_segment_vertices(vtree, ep1, ep2, vertex_search_r)
        if v_pair is None:
            continue
        i1, i2 = v_pair

        key = (min(i1, i2), max(i1, i2))
        seg_len = len(pts)
        if key not in edges or seg_len > edges[key]['len']:
            edges[key] = {'pts': pts, 'len': seg_len}

    logger.info(f"Matched edges: {len(edges)}")

    if not edges:
        return None

    # ------------------------------------------------------------------ #
    # 5.  Build arrays E, E_cells, E_pixels                                #
    # ------------------------------------------------------------------ #
    E_list, E_cells_list, E_pixels_list = [], [], []

    for (v1, v2), data in sorted(edges.items()):
        pts = data['pts']
        c1, c2 = _sample_edge_cells(pts, cell_labels_proc, H, W)
        E_list.append([v1, v2])
        if c1 == 0 and c2 == 0:
            E_cells_list.append([0, 0])
        elif c1 == 0:
            E_cells_list.append([c2, 0])
        elif c2 == 0:
            E_cells_list.append([c1, 0])
        else:
            E_cells_list.append(sorted([c1, c2]))
        if trace_pixels:
            ordered = _order_pixels(pts, vertices[v1], vertices[v2])
            E_pixels_list.append(ordered)

    E = np.array(E_list, dtype=int)
    E_cells = np.array(E_cells_list, dtype=int)

    # Sort: inner vertices first, border last (solver requires this ordering)
    margin = 2.0
    is_border_v = (
        (vertices[:, 0] <= margin) | (vertices[:, 0] >= W - margin) |
        (vertices[:, 1] <= margin) | (vertices[:, 1] >= H - margin)
    )
    inner_idx  = np.where(~is_border_v)[0]
    border_idx = np.where( is_border_v)[0]
    sorted_idx = np.concatenate([inner_idx, border_idx])
    num_inner  = len(inner_idx)
    vertices   = vertices[sorted_idx]
    old_to_new = np.empty(len(sorted_idx), dtype=int)
    old_to_new[sorted_idx] = np.arange(len(sorted_idx))
    E = old_to_new[E]

    V = np.column_stack((vertices, np.zeros(len(vertices))))

    # ------------------------------------------------------------------ #
    # 6.  Centroids + C_v                                                  #
    # ------------------------------------------------------------------ #
    is_label_image = len(np.unique(cell_labels_proc)) > 2
    if is_label_image:
        props = measure.regionprops(cell_labels_proc)
        max_lbl = int(cell_labels_proc.max())
        C_centroids = np.zeros((max_lbl, 2), dtype=float)
        for p in props:
            if p.label > 0:
                C_centroids[p.label - 1] = p.centroid[::-1]
    else:
        C_centroids = np.zeros((1, 2), dtype=float)

    C_v = _build_C_v(cell_labels_proc, V, E, E_cells)

    tissue = Tissue(V, E, E_cells, C_centroids, C_v, cell_labels_proc)

    tissue.num_inner_vertices = num_inner

    if trace_pixels:
        tissue.E_pixels = E_pixels_list

    # ------------------------------------------------------------------ #
    # 7.  Optional: collapse short edges                                   #
    # ------------------------------------------------------------------ #
    if clean and min_edge_len > 0 and len(tissue.E) > 0:
        tissue = _clean_tissue_graph(tissue, min_edge_len)

    logger.info(
        f"Done: {len(tissue.V)} vertices, {len(tissue.E)} edges, "
        f"{tissue.num_inner_vertices} inner vertices."
    )
    return tissue


# =============================================================================
# Core helpers
# =============================================================================

def _labels_to_boundary(
    labels: np.ndarray,
    dilate_label_boundary: bool = True,
) -> np.ndarray:
    """
    Binary boundary mask with adaptive dilation.

    Coverage-based rule (determined empirically on test.tif vs example.tif):

    DENSE membranes (coverage > 5%, e.g. test.tif at 7.1%):
        Already thick — disk(1) dilation merges nearby membranes, creates
        spurious branches and short segments.  No dilation gives more edges:
        disk(0) → 2687 edges,  disk(1) → 2489 edges  (worse)

    SPARSE membranes (coverage ≤ 5%, e.g. example.tif at 3.2%):
        Thin skeleton misses degree-3 pixels at junctions. disk(1) recovers
        genuine junctions without merging membranes:
        disk(0) → 619 edges,  disk(1) → 652 edges  (better)

    Label images: boundary is often 1px wide; optional dilation can recover
    thin junctions but may merge very close interfaces.
    """
    if len(np.unique(labels)) <= 2:
        b = (labels > 0)
        # Only dilate when membrane is sparse (thin junctions need widening)
        if b.mean() <= 0.05:
            b = morphology.dilation(b, morphology.disk(1))
        return b

    # Label image boundary from label transitions
    H, W = labels.shape
    b = np.zeros((H, W), dtype=bool)
    b[:-1, :] |= labels[:-1, :] != labels[1:, :]
    b[1:,  :] |= labels[:-1, :] != labels[1:, :]
    b[:, :-1] |= labels[:, :-1] != labels[:, 1:]
    b[:, 1:]  |= labels[:, :-1] != labels[:, 1:]
    if dilate_label_boundary:
        b = morphology.dilation(b, morphology.disk(1))
    return b


def _label_junctions_on_skeleton(
    cell_labels: np.ndarray,
    skel: np.ndarray,
    snap_radius: float = 2.5,
) -> np.ndarray:
    """
    Detect junctions from label topology (>=3 neighboring cell labels) and
    snap them to nearby skeleton pixels.
    """
    if len(np.unique(cell_labels)) <= 2:
        return np.zeros_like(skel, dtype=bool)

    boundary = _labels_to_boundary(cell_labels, dilate_label_boundary=False)
    by, bx = np.where(boundary)
    if len(bx) == 0:
        return np.zeros_like(skel, dtype=bool)

    # Candidate junctions on the label grid:
    # 8-neighborhood contains membrane/background (0) + >=3 cell ids.
    jmask = np.zeros_like(boundary, dtype=bool)
    H, W = cell_labels.shape
    for y, x in zip(by, bx):
        y0 = max(0, y - 1)
        y1 = min(H, y + 2)
        x0 = max(0, x - 1)
        x1 = min(W, x + 2)
        patch = cell_labels[y0:y1, x0:x1]
        uniq_all = np.unique(patch)
        uniq_nonzero = uniq_all[uniq_all > 0]
        has_membrane = np.any(uniq_all == 0)
        if (has_membrane and len(uniq_nonzero) >= 3) or len(uniq_nonzero) >= 4:
            jmask[y, x] = True

    if not np.any(jmask):
        return np.zeros_like(skel, dtype=bool)

    sy, sx = np.where(skel)
    if len(sx) == 0:
        return np.zeros_like(skel, dtype=bool)

    jy, jx = np.where(jmask)
    cand = np.column_stack((jx.astype(float), jy.astype(float)))
    # Compress nearby label-junction pixels before snapping to skeleton.
    cand = _cluster_points(cand, radius=1.0)

    skel_pts = np.column_stack((sx.astype(float), sy.astype(float)))
    tree = cKDTree(skel_pts)
    d, idx = tree.query(cand, distance_upper_bound=float(snap_radius))
    d = np.atleast_1d(d)
    idx = np.atleast_1d(idx)
    valid = np.isfinite(d) & (d < float(snap_radius)) & (idx < len(skel_pts))
    if not np.any(valid):
        return np.zeros_like(skel, dtype=bool)

    snapped = np.zeros_like(skel, dtype=bool)
    matched = skel_pts[idx[valid].astype(int)].astype(int)
    snapped[matched[:, 1], matched[:, 0]] = True
    return snapped


def _cluster_points(pts: np.ndarray, radius: float) -> np.ndarray:
    """Greedy: merge all points within `radius` into one centroid."""
    if len(pts) == 0:
        return pts
    tree = cKDTree(pts)
    visited = np.zeros(len(pts), dtype=bool)
    merged = []
    for i in range(len(pts)):
        if visited[i]:
            continue
        nb = tree.query_ball_point(pts[i], radius)
        merged.append(pts[nb].mean(axis=0))
        visited[nb] = True
    return np.array(merged)


def _cluster_mask_components(mask: np.ndarray, radius: float) -> np.ndarray:
    """
    Cluster each connected component independently so two nearby junctions
    are not merged just because they are spatially close.
    """
    xs = np.where(mask)[1]
    if len(xs) == 0:
        return np.zeros((0, 2), dtype=float)

    comp_lbl, n_comp = nd_label(mask, structure=np.ones((3, 3), dtype=int))
    merged = []
    for cid in range(1, n_comp + 1):
        cy, cx = np.where(comp_lbl == cid)
        pts = np.column_stack((cx.astype(float), cy.astype(float)))
        if len(pts) == 0:
            continue
        merged_pts = _cluster_points(pts, radius)
        if len(merged_pts) > 0:
            merged.append(merged_pts)
    if not merged:
        return np.zeros((0, 2), dtype=float)
    return np.vstack(merged)


def _match_segment_vertices(
    vtree: cKDTree,
    ep1: np.ndarray,
    ep2: np.ndarray,
    search_r: float,
) -> Optional[Tuple[int, int]]:
    """
    Match segment endpoints to two distinct vertices. Tries a few nearest
    candidates so short edges between nearby junctions are less likely to
    collapse onto the same vertex.
    """
    n_v = int(vtree.n)
    if n_v < 2:
        return None

    k = min(3, n_v)
    d1, i1 = vtree.query(ep1, k=k, distance_upper_bound=search_r)
    d2, i2 = vtree.query(ep2, k=k, distance_upper_bound=search_r)

    d1 = np.atleast_1d(d1)
    i1 = np.atleast_1d(i1)
    d2 = np.atleast_1d(d2)
    i2 = np.atleast_1d(i2)

    cand1 = [
        (int(idx), float(dist))
        for dist, idx in zip(d1, i1)
        if np.isfinite(dist) and dist < search_r and idx < n_v
    ]
    cand2 = [
        (int(idx), float(dist))
        for dist, idx in zip(d2, i2)
        if np.isfinite(dist) and dist < search_r and idx < n_v
    ]
    if not cand1 or not cand2:
        return None

    best = None
    best_cost = np.inf
    for idx1, dist1 in cand1:
        for idx2, dist2 in cand2:
            if idx1 == idx2:
                continue
            cost = dist1 + dist2
            if cost < best_cost:
                best = (idx1, idx2)
                best_cost = cost
    return best


def _segment_endpoints(
    pts: np.ndarray,
    cut_tree: cKDTree,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Identify the two endpoints of a skeleton segment as the pixels
    sitting closest to the cut (branch/border) region on each end.
    """
    if len(pts) < 2:
        return None, None

    d_cut, _ = cut_tree.query(pts)
    idx1 = int(np.argmin(d_cut))
    ep1 = pts[idx1]

    dist_from_ep1 = np.linalg.norm(pts - ep1, axis=1)
    score = dist_from_ep1 / (d_cut + 0.5)
    idx2 = int(np.argmax(score))
    ep2 = pts[idx2]

    if idx1 == idx2:
        idx2 = int(np.argmax(dist_from_ep1))
        ep2 = pts[idx2]

    if idx1 == idx2:
        return None, None

    return ep1, ep2


def _sample_edge_cells(
    pts: np.ndarray,
    labels: np.ndarray,
    H: int,
    W: int,
) -> Tuple[int, int]:
    """
    Find the two cell labels flanking a skeleton segment.
    Dilates the segment and reads the most-common label values in the halo.
    """
    xs = np.clip(pts[:, 0].astype(int), 0, W - 1)
    ys = np.clip(pts[:, 1].astype(int), 0, H - 1)

    # Robust interface vote from local 3x3 neighborhoods along the segment.
    pair_counts = {}
    for x, y in zip(xs, ys):
        y0 = max(0, y - 1)
        y1 = min(H, y + 2)
        x0 = max(0, x - 1)
        x1 = min(W, x + 2)
        uniq = np.unique(labels[y0:y1, x0:x1]).astype(int)
        if len(uniq) < 2:
            continue
        for a, b in combinations(sorted(uniq.tolist()), 2):
            key = (int(a), int(b))
            pair_counts[key] = pair_counts.get(key, 0) + 1

    if pair_counts:
        nonzero_pairs = {k: v for k, v in pair_counts.items() if k[0] > 0 and k[1] > 0}
        if nonzero_pairs:
            c1, c2 = max(nonzero_pairs.items(), key=lambda kv: kv[1])[0]
            return int(c1), int(c2)
        c1, c2 = max(pair_counts.items(), key=lambda kv: kv[1])[0]
        return int(c1), int(c2)

    # Fallback to halo vote if local vote has no evidence.
    seg_mask = np.zeros((H, W), dtype=bool)
    seg_mask[ys, xs] = True
    dilated = morphology.dilation(seg_mask, morphology.disk(2))
    neighbor_labels = labels[dilated & ~seg_mask]
    nonzero = neighbor_labels[neighbor_labels > 0]
    if len(nonzero) == 0:
        return 0, 0
    unique, counts = np.unique(nonzero, return_counts=True)
    top = unique[np.argsort(-counts)[:2]]
    return (int(top[0]), int(top[1])) if len(top) >= 2 else (int(top[0]), 0)


def _order_pixels(
    pts: np.ndarray,
    p1: np.ndarray,
    p2: np.ndarray,
) -> np.ndarray:
    """
    Order edge pixels from p1 to p2.
    Fast projection for straight edges; greedy NN for curved.
    """
    if len(pts) < 2:
        return pts

    edge_vec = p2 - p1
    length = np.linalg.norm(edge_vec)

    if length > 1e-6:
        d = edge_vec / length
        perp = np.abs((pts - p1) @ np.array([-d[1], d[0]]))
        if float(np.percentile(perp, 90)) < 3.0:
            return pts[np.argsort((pts - p1) @ d)]

    # Greedy nearest-neighbor
    d1 = np.linalg.norm(pts - p1, axis=1)
    start = int(np.argmin(d1))
    ordered = [start]
    remaining = list(range(len(pts)))
    remaining.remove(start)
    while remaining:
        curr = pts[ordered[-1]]
        dists = np.linalg.norm(pts[remaining] - curr, axis=1)
        nearest = remaining[int(np.argmin(dists))]
        ordered.append(nearest)
        remaining.remove(nearest)
    return pts[ordered]


def _build_C_v(
    labels: np.ndarray,
    V: np.ndarray,
    E: np.ndarray,
    E_cells: np.ndarray,
) -> List[List[int]]:
    """
    Build per-cell vertex loop C_v[lbl-1] = [v0, v1, ...] ordered CCW.
    Returns empty list for binary membrane images (no integer cell labels).
    """
    if len(np.unique(labels)) <= 2:
        return []  # binary image has no cell labels

    max_lbl = int(labels.max())
    C_v = [[] for _ in range(max_lbl)]
    cell_verts: List[set] = [set() for _ in range(max_lbl)]

    for e_idx, (v1, v2) in enumerate(E):
        for c in E_cells[e_idx]:
            if 0 < c <= max_lbl:
                cell_verts[c - 1].add(v1)
                cell_verts[c - 1].add(v2)

    props = measure.regionprops(labels)
    centroids = {p.label: np.array([p.centroid[1], p.centroid[0]])
                 for p in props if p.label > 0}

    for lbl in range(1, max_lbl + 1):
        verts = sorted(cell_verts[lbl - 1])
        if len(verts) < 2:
            continue
        if lbl not in centroids:
            C_v[lbl - 1] = verts
            continue
        cx, cy = centroids[lbl]
        angles = [np.arctan2(V[v, 1] - cy, V[v, 0] - cx) for v in verts]
        C_v[lbl - 1] = [verts[i] for i in np.argsort(angles)]

    return C_v


# =============================================================================
# Graph cleaning
# =============================================================================

def _clean_tissue_graph(tissue: "Tissue", min_edge_len: float) -> "Tissue":
    """Collapse edges shorter than min_edge_len by merging their vertices."""
    V, E = tissue.V, tissue.E
    E_cells = getattr(tissue, 'E_cells', None)
    C_v     = getattr(tissue, 'C_v', None)
    E_pixels = getattr(tissue, 'E_pixels', None)

    if len(E) == 0:
        return tissue

    for _ in range(5):
        if len(E) == 0:
            break
        d = np.linalg.norm(V[E[:, 0]] - V[E[:, 1]], axis=1)
        short = np.where(d < min_edge_len)[0]
        if len(short) == 0:
            break

        parent = np.arange(len(V))
        for idx in short:
            s, t = E[idx]
            r = min(parent[s], parent[t])
            parent[s] = parent[t] = r
        for i in range(len(V)):
            while parent[i] != parent[parent[i]]:
                parent[i] = parent[parent[i]]

        keep = np.unique(parent)
        mapping = {old: new for new, old in enumerate(keep)}

        new_V = np.zeros((len(keep), V.shape[1]))
        counts = np.zeros(len(keep))
        for i in range(len(V)):
            ni = mapping[parent[i]]
            new_V[ni] += V[i]
            counts[ni] += 1
        new_V /= counts[:, None]

        edge_to_data = {}
        for e_idx, (s, t) in enumerate(E):
            ns = mapping[parent[s]]
            nt = mapping[parent[t]]
            if ns != nt:
                key = (min(ns, nt), max(ns, nt))
                if key not in edge_to_data:
                    edge_to_data[key] = {
                        'cells': set(),
                        'pixels': E_pixels[e_idx] if E_pixels else None,
                    }
                if E_cells is not None:
                    for c in np.atleast_1d(E_cells[e_idx]):
                        edge_to_data[key]['cells'].add(int(c))

        new_keys = sorted(edge_to_data.keys())
        new_E = (np.array(new_keys, dtype=int) if new_keys
                 else np.zeros((0, 2), dtype=int))

        new_E_cells, new_pixels = [], []
        for key in new_keys:
            cells = sorted(edge_to_data[key]['cells'])
            if   len(cells) == 0: new_E_cells.append([0, 0])
            elif len(cells) == 1: new_E_cells.append([cells[0], 0])
            else:                 new_E_cells.append(cells[:2])
            if E_pixels is not None:
                new_pixels.append(edge_to_data[key]['pixels'])

        E_cells = (np.array(new_E_cells, dtype=int) if new_E_cells
                   else np.zeros((0, 2), dtype=int))

        if C_v is not None:
            remapped = []
            for seq in C_v:
                new_seq, last_v = [], None
                for vi in seq:
                    if vi < 0 or vi >= len(parent): continue
                    nv = mapping[parent[vi]]
                    if last_v is None or nv != last_v:
                        new_seq.append(nv)
                        last_v = nv
                if len(new_seq) > 1 and new_seq[0] == new_seq[-1]:
                    new_seq.pop()
                remapped.append(new_seq)
            C_v = remapped

        V, E = new_V, new_E
        if E_pixels is not None:
            E_pixels = new_pixels
        if len(E) == 0:
            break

    tissue.V, tissue.E = V, E
    if E_cells  is not None: tissue.E_cells  = E_cells
    if C_v      is not None: tissue.C_v      = C_v
    if E_pixels is not None: tissue.E_pixels = E_pixels
    return tissue
