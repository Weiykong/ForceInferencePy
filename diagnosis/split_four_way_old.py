"""
Split high-degree vertices (≥4) into pairs of triple junctions.

In biological tissue, true 4-way junctions are thermodynamically unstable.
At the image level they appear when two triple junctions are too close to
resolve.  This module replaces each degree-≥4 vertex with a pair (or chain)
of degree-3 vertices connected by short synthetic edges.

Algorithm for a degree-4 vertex with edges e0..e3 ordered CCW by angle:

    1. Identify the 4 angular sectors → 4 cells (A, B, C, D) going CCW.
    2. Evaluate the two possible splits:
         Split α: groups {e0, e1} vs {e2, e3}  —  synthetic edge separates B, D
         Split β: groups {e1, e2} vs {e3, e0}  —  synthetic edge separates A, C
    3. Choose the split whose two gap angles (at the split points) are largest.
    4. Create two new vertices offset from the original, connected by a
       short edge whose cell labels are the "opposite" pair.
    5. Rewire original edges to the appropriate new vertex.

Degree-5+ vertices are handled by iterating: each pass reduces the maximum
degree by 1 (5→4+3, then 4→3+3).
"""

import numpy as np
import logging
from collections import Counter, defaultdict
from typing import List, Optional, Set, Tuple

# Allow standalone use or import from force_inference
try:
    from .core import Tissue
except ImportError:
    from force_inference.core import Tissue

logger = logging.getLogger("ForceInference.SplitFourWay")


def split_high_degree_vertices(
    tissue: Tissue,
    split_length: float = 2.0,
    max_iterations: int = 3,
) -> Tissue:
    """
    Replace every vertex with degree ≥ 4 by triple junctions + short edges.

    Args:
        tissue:         Tissue object (modified in place and returned).
        split_length:   Length (px) of the synthetic edge inserted between
                        the two new triple-junction vertices.  2 px is a
                        good default — large enough for the circle fitter
                        to assign a curvature of ~0, small enough to not
                        distort the geometry.
        max_iterations: Safety cap on the split loop.  Each pass reduces
                        max degree by 1, so 3 iterations handles up to
                        degree 6.

    Returns:
        The same Tissue object with updated V, E, E_cells, E_pixels, C_v.
    """
    for iteration in range(max_iterations):
        high_deg = _find_high_degree_vertices(tissue)
        if not high_deg:
            break
        logger.info(
            f"Iteration {iteration}: splitting {len(high_deg)} vertices "
            f"(degrees: {Counter(d for _, d in high_deg)})"
        )
        tissue = _split_one_pass(tissue, high_deg, split_length)

    # Rebuild C_v after all splits
    tissue.C_v = _rebuild_C_v(tissue)

    # Re-sort vertices: inner first, border last
    tissue = _resort_vertices(tissue)

    return tissue


# =========================================================================
# Internal
# =========================================================================

def _find_high_degree_vertices(tissue: Tissue) -> List[Tuple[int, int]]:
    """Return [(vertex_index, degree), ...] for all vertices with degree ≥ 4."""
    degree = Counter()
    for v1, v2 in tissue.E:
        degree[v1] += 1
        degree[v2] += 1
    return [(v, d) for v, d in degree.items() if d >= 4]


def _split_one_pass(
    tissue: Tissue,
    targets: List[Tuple[int, int]],
    split_length: float,
) -> Tissue:
    """
    Split each target vertex once.  A degree-4 becomes two degree-3 vertices.
    A degree-5 becomes one degree-3 and one degree-4 (caught next iteration).
    """
    V = tissue.V.copy()            # (N, 3)
    E = tissue.E.tolist()          # list of [v1, v2]
    E_cells = tissue.E_cells.tolist()  # list of [c1, c2]
    E_pixels = list(tissue.E_pixels) if tissue.E_pixels is not None else None

    # Build vertex → incident-edge index mapping
    vertex_edges = defaultdict(list)
    for i, (v1, v2) in enumerate(E):
        vertex_edges[v1].append(i)
        vertex_edges[v2].append(i)

    # Process each target — we append to V / E / E_cells, so indices grow.
    # We process in ONE pass; each split only touches edges incident to one
    # vertex, so splits are independent as long as no two targets share an
    # edge.  (Two 4-way vertices sharing an edge is extremely rare in
    # practice; if it happens, the second iteration cleans it up.)
    target_set = set(v for v, _ in targets)

    for v_idx, deg in targets:
        incident = vertex_edges[v_idx]
        if len(incident) < 4:
            # Already been modified by a neighbor's split this pass
            continue

        # ---- 1. Order edges by angle around the vertex ----
        Vxy = V[:, :2]
        edge_info = []  # (edge_index, angle, cell_set)
        for ei in incident:
            v1, v2 = E[ei]
            other = v2 if v1 == v_idx else v1
            d = Vxy[other] - Vxy[v_idx]
            angle = np.arctan2(d[1], d[0])
            cells = set(E_cells[ei])
            cells.discard(0)
            edge_info.append((ei, angle, cells))

        edge_info.sort(key=lambda x: x[1])
        n = len(edge_info)

        # ---- 2. Identify cells in angular sectors ----
        sector_cells = []  # cell label between edge[i] and edge[i+1]
        for i in range(n):
            j = (i + 1) % n
            shared = edge_info[i][2] & edge_info[j][2]
            if len(shared) == 1:
                sector_cells.append(shared.pop())
            elif len(shared) > 1:
                # Ambiguous — pick the one that's not in any other sector yet
                sector_cells.append(min(shared))
            else:
                # No shared cell — shouldn't happen with correct topology
                sector_cells.append(0)

        # ---- 3. Compute angular gaps ----
        angles = [info[1] for info in edge_info]
        gaps = []
        for i in range(n):
            j = (i + 1) % n
            gap = angles[j] - angles[i]
            if gap <= 0:
                gap += 2 * np.pi
            gaps.append(gap)

        # ---- 4. Choose best split ----
        #
        # We split the edges into two consecutive groups.  For degree d,
        # group A gets 2 edges and group B gets (d−2) edges.  (Always peel
        # off a pair to create one clean triple junction; the other vertex
        # may still be degree ≥ 4 and will be split in the next iteration.)
        #
        # The split happens at two gap positions.  For a group of 2 starting
        # at index i, the split gaps are at positions (i−1) and (i+1):
        #   group A = {edge[i], edge[i+1]}
        #   group B = {edge[i+2], ..., edge[i−1]}   (wrapping)
        #
        # We pick the starting index whose two split-gap sum is largest.

        best_i = 0
        best_gap_sum = -1.0
        for i in range(n):
            # Group A = {edge[i], edge[(i+1)%n]}
            # Split gaps are at position (i-1)%n and (i+1)%n
            gap_before = gaps[(i - 1) % n]  # gap between edge[i-1] and edge[i]
            gap_after = gaps[(i + 1) % n]   # gap between edge[i+1] and edge[i+2]
            total = gap_before + gap_after
            if total > best_gap_sum:
                best_gap_sum = total
                best_i = i

        # Group A: edges at indices best_i and (best_i+1)%n
        # Group B: all remaining edges
        ia = best_i
        ib = (best_i + 1) % n
        group_a_indices = [ia, ib]
        group_b_indices = [k for k in range(n) if k not in group_a_indices]

        group_a_edges = [edge_info[k][0] for k in group_a_indices]
        group_b_edges = [edge_info[k][0] for k in group_b_indices]

        # ---- 5. Determine synthetic edge cell labels ----
        #
        # The cell in the sector just before group A (gap_before) and the
        # cell in the sector between the last of A and the first of B
        # (gap_after) are on OPPOSITE sides of the synthetic edge.
        #
        # Sector before group A = sector at index (ia - 1) % n
        # Sector after group A  = sector at index ib
        #
        # These are the two cells separated by the synthetic edge.

        synth_c1 = sector_cells[(ia - 1) % n]
        synth_c2 = sector_cells[ib]

        # ---- 6. Create two new vertices ----
        #
        # The synthetic edge direction is the bisector of the gap_after
        # angle (between the last edge of group A and the first of group B).
        # Vertex A sits on the group-A side, vertex B on the group-B side.

        first_b_idx = group_b_indices[0]
        angle_a_last = angles[ib]
        angle_b_first = angles[first_b_idx]
        gap_after_angle = angle_b_first - angle_a_last
        if gap_after_angle <= 0:
            gap_after_angle += 2 * np.pi
        bisect_after = angle_a_last + gap_after_angle / 2

        # Offset direction: from vertex toward the gap bisector
        offset_dir = np.array([np.cos(bisect_after), np.sin(bisect_after)])
        half_len = split_length / 2.0

        pos_orig = Vxy[v_idx].copy()
        pos_a = pos_orig - half_len * offset_dir  # away from the after-gap
        pos_b = pos_orig + half_len * offset_dir   # toward the after-gap

        # Append new vertices
        new_va = len(V)
        new_vb = new_va + 1
        z_val = V[v_idx, 2] if V.shape[1] > 2 else 0.0
        V = np.vstack([V, [[pos_a[0], pos_a[1], z_val]],
                            [[pos_b[0], pos_b[1], z_val]]])

        # ---- 7. Rewire edges ----
        for ei in group_a_edges:
            if E[ei][0] == v_idx:
                E[ei][0] = new_va
            elif E[ei][1] == v_idx:
                E[ei][1] = new_va

        for ei in group_b_edges:
            if E[ei][0] == v_idx:
                E[ei][0] = new_vb
            elif E[ei][1] == v_idx:
                E[ei][1] = new_vb

        # ---- 8. Add synthetic edge ----
        E.append([new_va, new_vb])
        E_cells.append(sorted([synth_c1, synth_c2]))
        if E_pixels is not None:
            # Synthetic edge: just the two endpoint positions
            synth_pts = np.array([pos_a, pos_b])
            E_pixels.append(synth_pts)

        # Update vertex_edges for any subsequent splits this pass
        vertex_edges[new_va] = group_a_edges + [len(E) - 1]
        vertex_edges[new_vb] = group_b_edges + [len(E) - 1]
        # Mark original vertex as dead
        vertex_edges[v_idx] = []

    # ---- 9. Write back ----
    tissue.V = V
    tissue.E = np.array(E, dtype=int)
    tissue.E_cells = np.array(E_cells, dtype=int)
    if E_pixels is not None:
        tissue.E_pixels = E_pixels

    return tissue


def _rebuild_C_v(tissue: Tissue) -> List[List[int]]:
    """Rebuild per-cell vertex loops from E and E_cells."""
    from skimage import measure

    labels = tissue.labels
    V = tissue.V
    E = tissue.E
    E_cells = tissue.E_cells

    max_lbl = int(labels.max())
    cell_verts: List[set] = [set() for _ in range(max_lbl)]

    for e_idx, (v1, v2) in enumerate(E):
        for c in E_cells[e_idx]:
            if 0 < c <= max_lbl:
                cell_verts[c - 1].add(int(v1))
                cell_verts[c - 1].add(int(v2))

    props = measure.regionprops(labels)
    centroids = {
        p.label: np.array([p.centroid[1], p.centroid[0]])
        for p in props if p.label > 0
    }

    C_v: List[List[int]] = []
    for lbl in range(1, max_lbl + 1):
        verts = sorted(cell_verts[lbl - 1])
        if len(verts) < 2:
            C_v.append(verts)
            continue
        if lbl not in centroids:
            C_v.append(verts)
            continue
        cx, cy = centroids[lbl]
        verts_arr = np.array(verts)
        angles = np.arctan2(V[verts_arr, 1] - cy, V[verts_arr, 0] - cx)
        C_v.append([verts[i] for i in np.argsort(angles)])

    return C_v


def _resort_vertices(tissue: Tissue) -> Tissue:
    """Re-sort vertices: inner first, border last.  Update all references."""
    H, W = tissue.labels.shape
    margin = 2.0
    V = tissue.V
    is_border = (
        (V[:, 0] <= margin) | (V[:, 0] >= W - margin) |
        (V[:, 1] <= margin) | (V[:, 1] >= H - margin)
    )
    inner_idx = np.where(~is_border)[0]
    border_idx = np.where(is_border)[0]
    sorted_idx = np.concatenate([inner_idx, border_idx])
    num_inner = len(inner_idx)

    # Reindex
    old_to_new = np.full(len(V), -1, dtype=int)
    old_to_new[sorted_idx] = np.arange(len(sorted_idx))

    tissue.V = V[sorted_idx]
    tissue.E = old_to_new[tissue.E]
    tissue.num_inner_vertices = num_inner

    # Remap C_v
    if tissue.C_v:
        new_cv = []
        for seq in tissue.C_v:
            new_seq = [int(old_to_new[v]) for v in seq if v < len(old_to_new) and old_to_new[v] >= 0]
            new_cv.append(new_seq)
        tissue.C_v = new_cv

    return tissue


# =========================================================================
# Convenience: run the full pipeline with splitting
# =========================================================================

def extract_and_split(
    labels: np.ndarray,
    split_length: float = 2.0,
    **topology_kwargs,
) -> Optional[Tissue]:
    """
    Extract topology with label-driven method, then split 4-way vertices.

    This is a convenience wrapper that calls extract_topology_label followed
    by split_high_degree_vertices.

    Args:
        labels:          Segmented label image.
        split_length:    Length of synthetic edges (px).
        **topology_kwargs: Passed to extract_topology_label.

    Returns:
        Tissue with all vertices at degree 3 (or 1 for border stubs).
    """
    from force_inference.topology_label import extract_topology_label

    tissue = extract_topology_label(labels, **topology_kwargs)
    if tissue is None:
        return None

    tissue = split_high_degree_vertices(tissue, split_length=split_length)
    return tissue
