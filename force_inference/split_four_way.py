"""
Split high-degree vertices into 3-way junctions.

In biological tissue, 4-way junctions are thermodynamically unstable;
they should be two 3-way junctions separated by a very short edge that
is below the image resolution. This module reconstructs that short edge
so that the force-balance solver is well-posed everywhere.
"""

import logging
from collections import Counter
from typing import List, Optional, Set, Tuple

import numpy as np

try:
    from .core import Tissue
except ImportError:
    from force_inference.core import Tissue

logger = logging.getLogger("ForceInference.SplitFourWay")


def split_high_degree_vertices(
    tissue: Tissue,
    split_length: float = 2.0,
    max_iterations: int = 5,
) -> Tissue:
    """
    Split every vertex with degree >= 4 into 3-way junctions.

    Args:
        tissue: Tissue object (modified in place).
        split_length: Full distance in pixels between the two new vertices.
        max_iterations: Safety cap; each pass reduces max degree by 1.

    Returns:
        The same Tissue object with updated V, E, E_cells, C_v, E_pixels,
        and num_inner_vertices.
    """
    offset_px = split_length / 2.0
    prev_high_count = None

    for iteration in range(max_iterations):
        deg = _vertex_degrees(tissue.E, len(tissue.V))
        high = np.where(deg >= 4)[0]
        if len(high) == 0:
            break

        if prev_high_count is not None and len(high) >= prev_high_count:
            logger.info(
                f"Iteration {iteration + 1}: no progress "
                f"({len(high)} high-degree vertices remain), stopping."
            )
            break
        prev_high_count = len(high)

        logger.info(
            f"Iteration {iteration + 1}: splitting {len(high)} "
            f"high-degree vertices (max degree = {deg[high].max()})"
        )

        order = high[np.argsort(-deg[high])]
        for v_idx in order:
            if _vertex_degree(tissue.E, int(v_idx)) < 4:
                continue
            tissue = _split_one_vertex(tissue, int(v_idx), offset_px)

        tissue = _rebuild_meta(tissue)

    deg_final = _vertex_degrees(tissue.E, len(tissue.V))
    deg_dist = Counter(deg_final.tolist())
    logger.info(
        f"After splitting: {len(tissue.V)} vertices, {len(tissue.E)} edges, "
        f"degree distribution = {dict(sorted(deg_dist.items()))}"
    )
    return tissue


def _split_one_vertex(
    tissue: Tissue,
    v_idx: int,
    offset_px: float,
) -> Tissue:
    """
    Split one vertex (degree >= 4) into two vertices connected by
    a synthetic short edge. Modifies tissue in place.
    """
    V = tissue.V
    E = tissue.E
    E_cells = tissue.E_cells
    E_pixels = getattr(tissue, "E_pixels", None)

    incident = []
    for ei in range(len(E)):
        v1, v2 = E[ei]
        if v1 == v_idx:
            incident.append((ei, int(v2)))
        elif v2 == v_idx:
            incident.append((ei, int(v1)))

    if len(incident) < 4:
        return tissue

    pos = V[v_idx, :2]

    angles = []
    for _ei, other_v in incident:
        d = V[other_v, :2] - pos
        angles.append(np.arctan2(d[1], d[0]))

    order = np.argsort(angles)
    incident = [incident[i] for i in order]
    angles = [angles[i] for i in order]
    n = len(incident)

    gaps = []
    for i in range(n):
        a1 = angles[i]
        a2 = angles[(i + 1) % n]
        gaps.append((a2 - a1) % (2 * np.pi))

    best_score = -1.0
    best_cut1 = 0
    best_cut2 = n // 2
    min_group = 2

    for c1 in range(n):
        for c2 in range(c1 + 1, n):
            size_a = c2 - c1
            size_b = n - size_a
            if size_a < min_group or size_b < min_group:
                continue
            score = gaps[c1] + gaps[c2]
            if score > best_score:
                best_score = score
                best_cut1 = c1
                best_cut2 = c2

    cut1, cut2 = best_cut1, best_cut2
    group_a_indices = list(range(cut1 + 1, cut2 + 1))
    group_b_indices = list(range(cut2 + 1, n)) + list(range(0, cut1 + 1))

    if len(group_a_indices) > len(group_b_indices):
        group_a_indices, group_b_indices = group_b_indices, group_a_indices

    group_a = [incident[i] for i in group_a_indices]
    group_b = [incident[i] for i in group_b_indices]

    a_lo = angles[cut1]
    a_hi = angles[(cut1 + 1) % n]
    gap_angle = (a_hi - a_lo) % (2 * np.pi)
    mid_angle = a_lo + gap_angle / 2.0
    split_dir = np.array([np.cos(mid_angle), np.sin(mid_angle)])

    pos_a = pos + offset_px * split_dir
    pos_b = pos - offset_px * split_dir

    idx_a = len(V)
    idx_b = idx_a + 1

    new_v_a = np.zeros((1, V.shape[1]))
    new_v_a[0, :2] = pos_a
    if V.shape[1] > 2:
        new_v_a[0, 2:] = V[v_idx, 2:]

    new_v_b = np.zeros((1, V.shape[1]))
    new_v_b[0, :2] = pos_b
    if V.shape[1] > 2:
        new_v_b[0, 2:] = V[v_idx, 2:]

    tissue.V = np.vstack([V, new_v_a, new_v_b])

    # Update E_pixels endpoints BEFORE remapping tissue.E so we can still
    # query which end of each edge was v_idx.
    if E_pixels is not None:
        for ei, _other in group_a:
            pix = E_pixels[ei]
            if pix is not None and len(pix) >= 1:
                pix_arr = np.asarray(pix, dtype=float).copy()
                if E[ei, 0] == v_idx:
                    pix_arr[0] = pos_a
                elif E[ei, 1] == v_idx:
                    pix_arr[-1] = pos_a
                E_pixels[ei] = pix_arr
        for ei, _other in group_b:
            pix = E_pixels[ei]
            if pix is not None and len(pix) >= 1:
                pix_arr = np.asarray(pix, dtype=float).copy()
                if E[ei, 0] == v_idx:
                    pix_arr[0] = pos_b
                elif E[ei, 1] == v_idx:
                    pix_arr[-1] = pos_b
                E_pixels[ei] = pix_arr

    for ei, _other in group_a:
        if tissue.E[ei, 0] == v_idx:
            tissue.E[ei, 0] = idx_a
        elif tissue.E[ei, 1] == v_idx:
            tissue.E[ei, 1] = idx_a

    for ei, _other in group_b:
        if tissue.E[ei, 0] == v_idx:
            tissue.E[ei, 0] = idx_b
        elif tissue.E[ei, 1] == v_idx:
            tissue.E[ei, 1] = idx_b

    cells_a: Set[int] = set()
    cells_b: Set[int] = set()
    for ei, _other in group_a:
        cells_a.update(int(c) for c in E_cells[ei] if c > 0)
    for ei, _other in group_b:
        cells_b.update(int(c) for c in E_cells[ei] if c > 0)

    shared = cells_a & cells_b
    if len(shared) >= 2:
        shared_sorted = sorted(shared)
        new_c1, new_c2 = shared_sorted[0], shared_sorted[1]
    elif len(shared) == 1:
        new_c1 = shared.pop()
        remaining = (cells_a | cells_b) - {new_c1}
        new_c2 = min(remaining) if remaining else 0
    else:
        new_c1 = min(cells_a) if cells_a else 0
        new_c2 = min(cells_b) if cells_b else 0

    tissue.E = np.vstack([tissue.E, np.array([[idx_a, idx_b]], dtype=tissue.E.dtype)])
    tissue.E_cells = np.vstack([
        tissue.E_cells,
        np.array([[new_c1, new_c2]], dtype=tissue.E_cells.dtype),
    ])

    # Track the new edge as synthetic so solvers can treat it appropriately.
    if tissue.E_synthetic is None:
        tissue.E_synthetic = np.zeros(len(tissue.E) - 1, dtype=bool)
    tissue.E_synthetic = np.append(tissue.E_synthetic, True)

    if E_pixels is not None:
        tissue.E_pixels.append(np.array([pos_a, pos_b]))

    return tissue


def _rebuild_meta(tissue: Tissue) -> Tissue:
    """Compact orphaned vertices, reorder inner/border vertices, rebuild C_v."""
    V = tissue.V
    E = tissue.E
    E_cells = tissue.E_cells
    E_pixels = getattr(tissue, "E_pixels", None)
    E_synthetic = getattr(tissue, "E_synthetic", None)
    H, W = tissue.labels.shape

    used = set(E[:, 0].tolist()) | set(E[:, 1].tolist())
    used_sorted = sorted(used)
    old_to_new = np.full(len(V), -1, dtype=int)
    for new_idx, old_idx in enumerate(used_sorted):
        old_to_new[old_idx] = new_idx

    new_V = V[used_sorted]
    new_E = old_to_new[E]

    margin = 2.0
    is_border = (
        (new_V[:, 0] <= margin)
        | (new_V[:, 0] >= W - margin)
        | (new_V[:, 1] <= margin)
        | (new_V[:, 1] >= H - margin)
    )
    inner_idx = np.where(~is_border)[0]
    border_idx = np.where(is_border)[0]
    sorted_idx = np.concatenate([inner_idx, border_idx])
    num_inner = len(inner_idx)

    final_V = new_V[sorted_idx]
    sort_map = np.full(len(new_V), -1, dtype=int)
    sort_map[sorted_idx] = np.arange(len(sorted_idx))
    final_E = sort_map[new_E]

    tissue.V = final_V
    tissue.E = final_E
    tissue.E_cells = E_cells
    tissue.num_inner_vertices = num_inner

    if E_pixels is not None:
        tissue.E_pixels = E_pixels

    # Preserve the synthetic-edge mask (edge ordering does not change here,
    # only vertex indices are remapped, so the mask is valid as-is).
    if E_synthetic is not None:
        tissue.E_synthetic = E_synthetic

    tissue.C_v = _build_C_v(tissue.labels, final_V, final_E, E_cells)
    return tissue


def _build_C_v(
    labels: np.ndarray,
    V: np.ndarray,
    E: np.ndarray,
    E_cells: np.ndarray,
) -> List[List[int]]:
    """Build per-cell vertex loops ordered CCW."""
    from skimage import measure

    if len(np.unique(labels)) <= 2:
        return []

    max_lbl = int(labels.max())
    cell_verts: List[Set[int]] = [set() for _ in range(max_lbl)]

    for e_idx, (v1, v2) in enumerate(E):
        for c in E_cells[e_idx]:
            if 0 < c <= max_lbl:
                cell_verts[c - 1].add(int(v1))
                cell_verts[c - 1].add(int(v2))

    props = measure.regionprops(labels)
    centroids = {
        p.label: np.array([p.centroid[1], p.centroid[0]])
        for p in props
        if p.label > 0
    }

    C_v: List[List[int]] = [[] for _ in range(max_lbl)]
    for lbl in range(1, max_lbl + 1):
        verts = sorted(cell_verts[lbl - 1])
        if len(verts) < 2:
            C_v[lbl - 1] = verts
            continue
        if lbl not in centroids:
            C_v[lbl - 1] = verts
            continue
        cx, cy = centroids[lbl]
        angs = [np.arctan2(V[v, 1] - cy, V[v, 0] - cx) for v in verts]
        C_v[lbl - 1] = [verts[i] for i in np.argsort(angs)]

    return C_v


def _vertex_degrees(E: np.ndarray, n_verts: int) -> np.ndarray:
    """Return degree array for all vertices."""
    deg = np.zeros(n_verts, dtype=int)
    for v1, v2 in E:
        deg[v1] += 1
        deg[v2] += 1
    return deg


def _vertex_degree(E: np.ndarray, v_idx: int) -> int:
    """Degree of one vertex."""
    return int(np.sum((E[:, 0] == v_idx) | (E[:, 1] == v_idx)))


def extract_and_split(
    labels: np.ndarray,
    split_length: float = 2.0,
    **topology_kwargs,
) -> Optional[Tissue]:
    """
    Extract topology with the label-driven method, then split 4-way vertices.
    """
    from force_inference.topology_label import extract_topology_label

    tissue = extract_topology_label(labels, **topology_kwargs)
    if tissue is None:
        return None

    return split_high_degree_vertices(tissue, split_length=split_length)
