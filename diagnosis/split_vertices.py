"""
Split high-degree (4-way, 5-way) vertices into pairs of 3-way vertices
connected by short synthetic edges.

Biological motivation:
  In real epithelial tissue, 4-way junctions are thermodynamically unstable.
  Every apparent 4-way junction is actually two 3-way (triple) junctions
  separated by a very short interface that falls below the image resolution.
  The segmentation/labeling collapses them into one point.

Solver motivation:
  At a 3-way vertex, force balance gives 2 equations for 3 tension unknowns.
  At a 4-way vertex, it gives 2 equations for 4 unknowns — locally
  underdetermined. Splitting restores a well-posed problem everywhere.

Algorithm:
  For each 4-way vertex:
    1. Compute the outgoing direction of each incident edge.
    2. Sort edges by angle around the vertex.
    3. Find the two largest angular gaps between consecutive edges.
    4. These two gaps define a natural split: edges on each side of
       the gaps form two groups of two.
    5. Create two new vertices offset from the original along the
       bisector of the larger gap.
    6. Rewire each group to its new vertex.
    7. Add a short synthetic edge between the two new vertices.
    8. The cell pair for the new edge is determined by the cells
       meeting inside the two gap regions.

  For 5-way vertices: same principle produces a 3-way + 4-way, then
  the 4-way is split on the next iteration.

Usage:
    from force_inference.split_vertices import split_high_degree_vertices
    tissue = split_high_degree_vertices(tissue, split_distance=2.0)
"""

import numpy as np
from collections import Counter, defaultdict
from typing import Optional, List, Tuple, Set
from skimage import measure
import logging

from .core import Tissue

logger = logging.getLogger("ForceInference.SplitVertices")


def split_high_degree_vertices(
    tissue: Tissue,
    split_distance: float = 2.0,
    max_degree_to_split: int = 6,
    max_iterations: int = 5,
) -> Tissue:
    """
    Split all vertices with degree >= 4 into chains of 3-way vertices.

    Args:
        tissue:            Tissue object (modified in-place and returned).
        split_distance:    Distance (px) between the two new vertices
                           created from each split. Should be small (1-3 px).
        max_degree_to_split: Only split vertices up to this degree.
        max_iterations:    Maximum rounds of splitting (each round reduces
                           max degree by 1, so 5 rounds handles up to 8-way).

    Returns:
        Modified Tissue with all junctions reduced to degree 3 (or degree 1-2
        for border vertices).
    """
    # ------------------------------------------------------------------ #
    # Convert to mutable lists for efficient modification                 #
    # ------------------------------------------------------------------ #
    V = [tissue.V[i].copy() for i in range(len(tissue.V))]
    E = [list(tissue.E[i]) for i in range(len(tissue.E))]
    E_cells = [list(tissue.E_cells[i]) for i in range(len(tissue.E_cells))]

    has_pixels = tissue.E_pixels is not None and len(tissue.E_pixels) == len(E)
    E_pixels = list(tissue.E_pixels) if has_pixels else None

    n_splits_total = 0

    for iteration in range(max_iterations):
        # ------------------------------------------------------------ #
        # Rebuild adjacency from scratch each iteration                 #
        # ------------------------------------------------------------ #
        degree, vertex_edges = _build_adjacency(E)

        high_deg = [v for v, d in degree.items() if 4 <= d <= max_degree_to_split]
        if not high_deg:
            break

        logger.info(
            f"Iteration {iteration}: {len(high_deg)} vertices with degree >= 4 "
            f"(max degree = {max(degree[v] for v in high_deg)})"
        )

        n_splits_iter = 0

        for v_idx in high_deg:
            # Re-check: degree may have changed from earlier splits this pass
            current_edges = [i for i in vertex_edges.get(v_idx, [])
                             if i < len(E) and v_idx in E[i]]
            if len(current_edges) < 4:
                continue

            result = _split_one_vertex(
                v_idx, current_edges, V, E, E_cells, E_pixels,
                split_distance
            )
            if result is not None:
                va_idx, vb_idx, new_edge_idx = result
                # Update adjacency for the new vertices
                vertex_edges[v_idx] = []  # dead vertex
                vertex_edges[va_idx] = [i for i in range(len(E))
                                        if va_idx in E[i]]
                vertex_edges[vb_idx] = [i for i in range(len(E))
                                        if vb_idx in E[i]]
                n_splits_iter += 1

        n_splits_total += n_splits_iter
        logger.info(f"  Split {n_splits_iter} vertices this iteration")

        if n_splits_iter == 0:
            break

    # ------------------------------------------------------------------ #
    # Collapse degree-2 chain vertices (artifacts from cascading splits) #
    # ------------------------------------------------------------------ #
    V, E, E_cells, E_pixels = _collapse_degree2_vertices(
        V, E, E_cells, E_pixels
    )

    # ------------------------------------------------------------------ #
    # Remove dead vertices (degree 0) and compact indices                #
    # ------------------------------------------------------------------ #
    V_arr = np.array(V)
    E_arr = np.array(E, dtype=int)

    # Find which vertices are actually used
    used_v = set()
    for v1, v2 in E_arr:
        used_v.add(v1)
        used_v.add(v2)
    used_v = sorted(used_v)
    old_to_new = np.full(len(V_arr), -1, dtype=int)
    old_to_new[used_v] = np.arange(len(used_v))

    new_V = V_arr[used_v]
    new_E = old_to_new[E_arr]
    new_E_cells = np.array(E_cells, dtype=int)

    # Re-sort: inner vertices first, border last
    H, W = tissue.labels.shape
    margin = 2.0
    is_border = (
        (new_V[:, 0] <= margin) | (new_V[:, 0] >= W - margin) |
        (new_V[:, 1] <= margin) | (new_V[:, 1] >= H - margin)
    )
    inner_idx = np.where(~is_border)[0]
    border_idx = np.where(is_border)[0]
    sorted_idx = np.concatenate([inner_idx, border_idx])
    num_inner = len(inner_idx)

    final_V = new_V[sorted_idx]
    remap = np.full(len(sorted_idx), -1, dtype=int)
    remap[sorted_idx] = np.arange(len(sorted_idx))
    final_E = remap[new_E]

    # Rebuild E_pixels with no index changes needed (pixel coords, not vertex refs)
    final_E_pixels = None
    if E_pixels is not None:
        final_E_pixels = E_pixels

    # ------------------------------------------------------------------ #
    # Rebuild Tissue                                                      #
    # ------------------------------------------------------------------ #
    # Rebuild C_v from the new topology
    C_v = _build_C_v(tissue.labels, final_V, final_E, new_E_cells)

    # Keep original centroids (cells haven't changed)
    tissue.V = final_V
    tissue.E = final_E
    tissue.E_cells = new_E_cells
    tissue.C_v = C_v
    tissue.num_inner_vertices = num_inner
    if final_E_pixels is not None:
        tissue.E_pixels = final_E_pixels

    # Clear stale geometry caches
    if hasattr(tissue, 'E_curvature'):
        del tissue.E_curvature
    if hasattr(tissue, 'E_tangents'):
        del tissue.E_tangents
    if hasattr(tissue, 'E_circles'):
        del tissue.E_circles

    # Log final stats
    final_deg, _ = _build_adjacency(list(map(list, final_E)))
    final_dist = Counter(final_deg.values())
    logger.info(
        f"Done: split {n_splits_total} high-degree vertices. "
        f"Final: {len(final_V)} vertices, {len(final_E)} edges. "
        f"Degree distribution: {dict(sorted(final_dist.items()))}"
    )

    return tissue


# ===================================================================== #
# Core split logic                                                       #
# ===================================================================== #

def _split_one_vertex(
    v_idx: int,
    edge_indices: List[int],
    V: List[np.ndarray],
    E: List[List[int]],
    E_cells: List[List[int]],
    E_pixels: Optional[List[np.ndarray]],
    split_distance: float,
) -> Optional[Tuple[int, int, int]]:
    """
    Split a single high-degree vertex into two vertices connected by
    a short synthetic edge.

    Peels off 2 edges (the pair spanning the smallest angular sector)
    into a new vertex, keeping the rest at the original vertex position
    (now with degree reduced by 1).

    Returns:
        (va_idx, vb_idx, new_edge_idx) or None if split failed.
    """
    n_edges = len(edge_indices)
    if n_edges < 4:
        return None

    v_pos = V[v_idx][:2].copy()

    # ---------------------------------------------------------------- #
    # 1. Compute outgoing direction & angle for each incident edge      #
    # ---------------------------------------------------------------- #
    edge_angles = []
    for ei in edge_indices:
        v1, v2 = E[ei]
        other = v2 if v1 == v_idx else v1
        other_pos = V[other][:2]
        d = other_pos - v_pos
        angle = np.arctan2(d[1], d[0])
        edge_angles.append((angle, ei))

    # Sort by angle (CCW from east)
    edge_angles.sort(key=lambda x: x[0])
    sorted_angles = [a for a, _ in edge_angles]
    sorted_edges = [ei for _, ei in edge_angles]

    # ---------------------------------------------------------------- #
    # 2. Find the two largest angular gaps                              #
    # ---------------------------------------------------------------- #
    n = len(sorted_angles)
    gaps = []
    for i in range(n):
        a_cur = sorted_angles[i]
        a_next = sorted_angles[(i + 1) % n]
        gap = (a_next - a_cur) % (2 * np.pi)
        gaps.append((gap, i))

    # Sort gaps descending — the two largest define the split
    gaps.sort(key=lambda x: -x[0])
    gap1_size, gap1_after = gaps[0]  # largest gap: between edge[gap1_after] and edge[gap1_after+1]
    gap2_size, gap2_after = gaps[1]  # second largest gap

    # ---------------------------------------------------------------- #
    # 3. Partition edges into two groups                                 #
    #    Walk from gap1 to gap2 → group_a                               #
    #    Walk from gap2 to gap1 → group_b                               #
    # ---------------------------------------------------------------- #
    # Indices go: gap1_after+1, ..., gap2_after → group A
    #             gap2_after+1, ..., gap1_after → group B
    group_a = []
    group_b = []
    i = (gap1_after + 1) % n
    while i != (gap2_after + 1) % n:
        group_a.append(sorted_edges[i])
        i = (i + 1) % n
    i = (gap2_after + 1) % n
    while i != (gap1_after + 1) % n:
        group_b.append(sorted_edges[i])
        i = (i + 1) % n

    # Pick the smaller group to peel off (the other stays at original position)
    if len(group_a) > len(group_b):
        group_a, group_b = group_b, group_a
        gap1_size, gap2_size = gap2_size, gap1_size
        gap1_after, gap2_after = gap2_after, gap1_after

    # group_a is the smaller group (will move to a new vertex)
    # group_b stays at the original position (or close to it)
    # Both groups must have at least 2 edges for a valid split
    # (otherwise we create a degree-2 vertex, not a triple junction)
    if len(group_a) < 2 or len(group_b) < 2:
        # Fallback for 4-way: force a 2+2 split using alternating edges
        if n_edges == 4:
            group_a = [sorted_edges[0], sorted_edges[2]]
            group_b = [sorted_edges[1], sorted_edges[3]]
            # Recompute gap info for split direction: use angle bisector
            # between the two edges in group_a
            a0 = sorted_angles[0]
            a2 = sorted_angles[2]
            mid_angle = (a0 + a2) / 2.0
            # Make sure mid_angle points "between" the two edges
            if abs(a2 - a0) > np.pi:
                mid_angle += np.pi
            split_dir = np.array([np.cos(mid_angle), np.sin(mid_angle)])
            # Rotate 90 degrees so the split edge is perpendicular
            split_dir = np.array([-split_dir[1], split_dir[0]])
        else:
            return None

    # ---------------------------------------------------------------- #
    # 4. Compute split direction and new vertex positions               #
    # ---------------------------------------------------------------- #
    # Direction: bisector of the larger gap (gap1) — this is where the
    # new short edge should point, perpendicular to the "natural" split.
    mid_angle = sorted_angles[gap1_after] + gap1_size / 2.0
    split_dir = np.array([np.cos(mid_angle), np.sin(mid_angle)])

    # group_a vertex: offset toward the gap1 bisector
    # group_b vertex: offset away from it
    half_d = split_distance / 2.0
    va_pos = v_pos + half_d * split_dir      # group_a (peeled off)
    vb_pos = v_pos - half_d * split_dir      # group_b (main mass)

    z_val = V[v_idx][2] if len(V[v_idx]) > 2 else 0.0
    va_idx = len(V)
    vb_idx = len(V) + 1
    V.append(np.array([va_pos[0], va_pos[1], z_val]))
    V.append(np.array([vb_pos[0], vb_pos[1], z_val]))

    # ---------------------------------------------------------------- #
    # 5. Rewire edges                                                   #
    # ---------------------------------------------------------------- #
    for ei in group_a:
        if E[ei][0] == v_idx:
            E[ei][0] = va_idx
        elif E[ei][1] == v_idx:
            E[ei][1] = va_idx

    for ei in group_b:
        if E[ei][0] == v_idx:
            E[ei][0] = vb_idx
        elif E[ei][1] == v_idx:
            E[ei][1] = vb_idx

    # ---------------------------------------------------------------- #
    # 6. Determine cell pair for the new synthetic edge                 #
    # ---------------------------------------------------------------- #
    # The cells "inside" the two largest gaps are the ones the new
    # edge separates.  These are exactly the cells that appear in BOTH
    # groups (since each gap-bordering edge contributes cells from both
    # sides of the gap).
    cells_a: Set[int] = set()
    cells_b: Set[int] = set()
    for ei in group_a:
        cells_a.update(E_cells[ei])
    for ei in group_b:
        cells_b.update(E_cells[ei])
    shared = cells_a & cells_b
    shared.discard(0)

    if len(shared) >= 2:
        new_cells = sorted(shared)[:2]
    elif len(shared) == 1:
        # Fallback: pair with the most common non-shared cell
        other = (cells_a | cells_b) - shared - {0}
        new_cells = [sorted(shared)[0], sorted(other)[0] if other else 0]
    else:
        # Fallback: pick one from each group
        ca = sorted(cells_a - {0})
        cb = sorted(cells_b - {0})
        new_cells = [ca[0] if ca else 0, cb[0] if cb else 0]

    # ---------------------------------------------------------------- #
    # 7. Add the new synthetic edge                                     #
    # ---------------------------------------------------------------- #
    new_edge_idx = len(E)
    E.append([va_idx, vb_idx])
    E_cells.append(new_cells)
    if E_pixels is not None:
        # Synthetic pixel path: just the two endpoints
        E_pixels.append(np.array([va_pos, vb_pos]))

    return va_idx, vb_idx, new_edge_idx


def _collapse_degree2_vertices(V, E, E_cells, E_pixels):
    """
    Remove degree-2 vertices by merging their two incident edges into one.

    A degree-2 vertex v with edges (a—v) and (v—b) can be replaced by a
    single edge (a—b) whose pixel path is the concatenation of the two
    original pixel paths.  This is safe because the two edges at a degree-2
    vertex always border the same cell pair.

    Iterates until no degree-2 vertices remain.
    """
    for _round in range(10):
        degree, vertex_edges = _build_adjacency(E)
        deg2 = [v for v, d in degree.items() if d == 2]
        if not deg2:
            break

        # Mark edges and vertices for removal
        dead_edges = set()
        dead_verts = set()

        for v_idx in deg2:
            if v_idx in dead_verts:
                continue
            edges_at_v = [i for i in vertex_edges.get(v_idx, [])
                          if i not in dead_edges and i < len(E)]
            if len(edges_at_v) != 2:
                continue

            ei_a, ei_b = edges_at_v
            # Find the other endpoints
            va = E[ei_a][0] if E[ei_a][1] == v_idx else E[ei_a][1]
            vb = E[ei_b][0] if E[ei_b][1] == v_idx else E[ei_b][1]

            if va == vb:
                # Self-loop — just remove both edges and the vertex
                dead_edges.add(ei_a)
                dead_edges.add(ei_b)
                dead_verts.add(v_idx)
                continue

            # Merge: keep edge ei_a, rewire to (va, vb), kill edge ei_b
            E[ei_a] = [va, vb]
            # Cell pair: prefer the one from the longer edge
            if E_pixels is not None:
                len_a = len(E_pixels[ei_a]) if E_pixels[ei_a] is not None else 0
                len_b = len(E_pixels[ei_b]) if E_pixels[ei_b] is not None else 0
                if len_b > len_a:
                    E_cells[ei_a] = E_cells[ei_b]
                # Concatenate pixel paths
                if len_a > 0 and len_b > 0:
                    E_pixels[ei_a] = np.vstack([E_pixels[ei_a], E_pixels[ei_b]])
                elif len_b > 0:
                    E_pixels[ei_a] = E_pixels[ei_b]

            dead_edges.add(ei_b)
            dead_verts.add(v_idx)

        # Compact: remove dead edges
        if dead_edges:
            keep = [i for i in range(len(E)) if i not in dead_edges]
            E = [E[i] for i in keep]
            E_cells = [E_cells[i] for i in keep]
            if E_pixels is not None:
                E_pixels = [E_pixels[i] for i in keep]

    return V, E, E_cells, E_pixels


# ===================================================================== #
# Helpers                                                                #
# ===================================================================== #

def _build_adjacency(E):
    """Build degree counter and vertex→edge-list mapping."""
    degree = Counter()
    vertex_edges = defaultdict(list)
    for i, (v1, v2) in enumerate(E):
        degree[v1] += 1
        degree[v2] += 1
        vertex_edges[v1].append(i)
        vertex_edges[v2].append(i)
    return degree, vertex_edges


def _build_C_v(
    labels: np.ndarray,
    V: np.ndarray,
    E: np.ndarray,
    E_cells: np.ndarray,
) -> List[List[int]]:
    """
    Rebuild per-cell vertex loop: C_v[label-1] = [v0, v1, ...] CCW.
    """
    max_lbl = int(labels.max())
    if max_lbl == 0:
        return []
    C_v = [[] for _ in range(max_lbl)]
    cell_verts: List[set] = [set() for _ in range(max_lbl)]

    for e_idx, (v1, v2) in enumerate(E):
        for c in E_cells[e_idx]:
            if 0 < c <= max_lbl:
                cell_verts[c - 1].add(int(v1))
                cell_verts[c - 1].add(int(v2))

    props = measure.regionprops(labels)
    centroids = {p.label: np.array([p.centroid[1], p.centroid[0]])
                 for p in props if p.label > 0}

    Vxy = V[:, :2] if V.ndim == 2 else V

    for lbl in range(1, max_lbl + 1):
        verts = sorted(cell_verts[lbl - 1])
        if len(verts) < 2:
            C_v[lbl - 1] = verts
            continue
        if lbl not in centroids:
            C_v[lbl - 1] = verts
            continue
        cx, cy = centroids[lbl]
        angles = [np.arctan2(Vxy[v, 1] - cy, Vxy[v, 0] - cx) for v in verts]
        C_v[lbl - 1] = [verts[i] for i in np.argsort(angles)]

    return C_v
