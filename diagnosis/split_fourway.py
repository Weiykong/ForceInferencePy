"""
split_fourway.py — Split high-degree vertices into 3-way junctions.

In biological tissue, 4-way junctions are thermodynamically unstable;
they should be two 3-way junctions separated by a very short edge that
is below the image resolution.  This module reconstructs that short edge
so that the force-balance solver is well-posed everywhere.

Algorithm for a degree-4 vertex V with edges e1…e4:
  1. Sort the four edges by outward angle from V.
  2. There are two ways to cut 4 angularly-ordered edges into two pairs
     of consecutive edges: cut at gaps (1,2)&(3,4) or (2,3)&(4,1).
     (The third combinatorial pairing is non-planar and discarded.)
  3. Pick the cut whose two angular gaps are most balanced (largest
     minimum gap) — this keeps the new 3-way junctions well-conditioned.
  4. Create two new vertices V_a and V_b offset ±δ from V along the
     bisector of the chosen gap direction.
  5. Re-wire: edges in group A → V_a, edges in group B → V_b.
  6. Insert a synthetic edge V_a ↔ V_b with cell labels inherited from
     the flanking edges.

For degree-5 (and higher) vertices the same idea generalises:
  - sort edges angularly
  - find the two largest angular gaps
  - split there into groups of 2 and 3 (for deg-5), giving one 3-way
    and one 4-way vertex; then recurse on the 4-way.

Public API
----------
    tissue = split_high_degree_vertices(tissue, offset_px=2.0)

The function modifies the Tissue *in-place* and returns it.

Compatible with both the Laplace and Bayesian solvers — run this
BEFORE compute_curvature() and the solver.  The curvature of the
new synthetic edge is set to 0 (straight line) and its pixel path
is a 2-point segment.
"""

import numpy as np
from collections import Counter, defaultdict
from typing import List, Tuple, Set, Optional, Dict
import logging

from .core import Tissue

logger = logging.getLogger("ForceInference.SplitFourway")


# =====================================================================
# Public API
# =====================================================================

def split_high_degree_vertices(
    tissue: Tissue,
    offset_px: float = 2.0,
    max_iterations: int = 5,
) -> Tissue:
    """
    Split every vertex with degree ≥ 4 into 3-way junctions.

    Args:
        tissue:        Tissue object (modified in place).
        offset_px:     Half-distance (in pixels) between the two new
                       vertices that replace each 4-way vertex.
                       Typical values: 1.0–3.0.  Smaller = more faithful
                       to original vertex position; larger = better
                       numerical conditioning for the solver.
        max_iterations: Safety cap — each pass reduces the maximum degree
                       by 1, so 5 iterations handles up to degree-8.

    Returns:
        The same Tissue object with updated V, E, E_cells, C_v,
        E_pixels, and num_inner_vertices.
    """
    prev_high_count = None
    for iteration in range(max_iterations):
        deg = _vertex_degrees(tissue.E, len(tissue.V))
        high = np.where(deg >= 4)[0]
        if len(high) == 0:
            break

        # Detect stall — if no progress, stop
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

        # Process in decreasing-degree order so the worst cases are
        # handled first (before index remapping makes them harder to find).
        order = high[np.argsort(-deg[high])]

        for v_idx in order:
            # Re-check degree — previous splits in this iteration may
            # have already reduced it.
            d = _vertex_degree(tissue.E, v_idx)
            if d < 4:
                continue
            tissue = _split_one_vertex(tissue, int(v_idx), offset_px)

        # After all splits in this pass, rebuild C_v and inner-vertex count
        tissue = _rebuild_meta(tissue)

    # Final stats
    deg_final = _vertex_degrees(tissue.E, len(tissue.V))
    deg_dist = Counter(deg_final.tolist())
    logger.info(
        f"After splitting: {len(tissue.V)} vertices, {len(tissue.E)} edges, "
        f"degree distribution = {dict(sorted(deg_dist.items()))}"
    )
    return tissue


# =====================================================================
# Core: split a single vertex
# =====================================================================

def _split_one_vertex(
    tissue: Tissue,
    v_idx: int,
    offset_px: float,
) -> Tissue:
    """
    Split vertex *v_idx* (degree ≥ 4) into two vertices connected by
    a new short edge.  Modifies tissue in place.
    """
    V = tissue.V                 # (N, 3)
    E = tissue.E                 # (M, 2)
    E_cells = tissue.E_cells     # (M, 2)
    E_pixels = getattr(tissue, "E_pixels", None)

    # ── 1. Collect incident edges and their outward directions ──────────
    incident = []   # list of (edge_idx, other_vertex_idx)
    for ei in range(len(E)):
        v1, v2 = E[ei]
        if v1 == v_idx:
            incident.append((ei, int(v2)))
        elif v2 == v_idx:
            incident.append((ei, int(v1)))

    if len(incident) < 4:
        return tissue   # nothing to split

    pos = V[v_idx, :2]   # (x, y) of the junction

    # Outward unit vector and angle for each incident edge
    angles = []
    for ei, other_v in incident:
        d = V[other_v, :2] - pos
        ang = np.arctan2(d[1], d[0])
        angles.append(ang)

    # ── 2. Sort edges by angle ──────────────────────────────────────────
    order = np.argsort(angles)
    incident = [incident[i] for i in order]
    angles = [angles[i] for i in order]
    n = len(incident)

    # ── 3. Find the best balanced split ─────────────────────────────────
    #    We pick two "cut points" in the circular edge ordering.  Each cut
    #    falls in one of the n angular gaps.  The two cuts divide the n
    #    edges into two contiguous groups.  We require each group to have
    #    at least 2 edges (so that with the synthetic edge both new
    #    vertices reach degree ≥ 3).
    #
    #    For degree 4: the only valid split is 2+2.
    #    For degree 5: valid splits are 2+3.
    #    For degree 6: valid splits are 2+4 and 3+3.
    #
    #    Among all valid splits, we pick the one that maximises the
    #    sum of the two gap sizes at the cut points (= best angular
    #    separation between groups).

    gaps = []
    for i in range(n):
        a1 = angles[i]
        a2 = angles[(i + 1) % n]
        gap = (a2 - a1) % (2 * np.pi)
        gaps.append(gap)

    best_score = -1.0
    best_cut1 = 0
    best_cut2 = n // 2
    min_group = 2  # minimum edges per group

    for c1 in range(n):
        for c2 in range(c1 + 1, n):
            # Group A: edges from (c1+1) to c2 inclusive
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

    # Build groups
    group_a_indices = list(range(cut1 + 1, cut2 + 1))
    group_b_indices = list(range(cut2 + 1, n)) + list(range(0, cut1 + 1))

    # Ensure the smaller group is A
    if len(group_a_indices) > len(group_b_indices):
        group_a_indices, group_b_indices = group_b_indices, group_a_indices

    group_a = [incident[i] for i in group_a_indices]
    group_b = [incident[i] for i in group_b_indices]

    # ── 4. Compute split direction and new vertex positions ─────────────
    #    Direction = bisector of the angular gap at cut1.
    a_lo = angles[cut1]
    a_hi = angles[(cut1 + 1) % n]
    gap_angle = (a_hi - a_lo) % (2 * np.pi)
    mid_angle_1 = a_lo + gap_angle / 2.0

    split_dir = np.array([np.cos(mid_angle_1), np.sin(mid_angle_1)])

    pos_a = pos + offset_px * split_dir
    pos_b = pos - offset_px * split_dir

    # ── 5. Create new vertices ──────────────────────────────────────────
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

    # ── 6. Re-wire existing edges ───────────────────────────────────────
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

    # ── 7. Determine cell labels for the synthetic edge ─────────────────
    # The new edge separates the cells that are "between" the two gaps.
    # Collect all cell labels from group A and group B edges, then find
    # the two labels that appear on both sides of the split.
    cells_a: Set[int] = set()
    cells_b: Set[int] = set()
    for ei, _ in group_a:
        cells_a.update(int(c) for c in E_cells[ei] if c > 0)
    for ei, _ in group_b:
        cells_b.update(int(c) for c in E_cells[ei] if c > 0)

    shared = cells_a & cells_b
    if len(shared) >= 2:
        # The synthetic edge separates the two shared cells
        shared_sorted = sorted(shared)
        new_c1, new_c2 = shared_sorted[0], shared_sorted[1]
    elif len(shared) == 1:
        # One shared cell + pick the most common from the other side
        new_c1 = shared.pop()
        remaining = (cells_a | cells_b) - {new_c1}
        new_c2 = min(remaining) if remaining else 0
    else:
        # Fallback: pick one cell from each group
        new_c1 = min(cells_a) if cells_a else 0
        new_c2 = min(cells_b) if cells_b else 0

    # ── 8. Insert synthetic edge ────────────────────────────────────────
    new_edge = np.array([[idx_a, idx_b]], dtype=tissue.E.dtype)
    tissue.E = np.vstack([tissue.E, new_edge])

    new_cells = np.array([[new_c1, new_c2]], dtype=tissue.E_cells.dtype)
    tissue.E_cells = np.vstack([tissue.E_cells, new_cells])

    if E_pixels is not None:
        synthetic_pixels = np.array([pos_a, pos_b])
        tissue.E_pixels.append(synthetic_pixels)

    # ── 9. Disconnect the old vertex ────────────────────────────────────
    # The original v_idx is now orphaned (degree 0).  We leave it in
    # the array — it will be cleaned up later or simply ignored by the
    # solver (no edges reference it).  Removing it would require global
    # index renumbering which we defer to _rebuild_meta.

    return tissue


# =====================================================================
# Post-processing: clean up orphaned vertices, rebuild C_v, etc.
# =====================================================================

def _rebuild_meta(tissue: Tissue) -> Tissue:
    """
    After splitting, compact the vertex array (remove orphans),
    rebuild inner/border ordering, and rebuild C_v.
    """
    V = tissue.V
    E = tissue.E
    E_cells = tissue.E_cells
    E_pixels = getattr(tissue, "E_pixels", None)
    H, W = tissue.labels.shape

    # ── 1. Find referenced vertices ─────────────────────────────────────
    used = set(E[:, 0].tolist()) | set(E[:, 1].tolist())
    used_sorted = sorted(used)
    old_to_new = np.full(len(V), -1, dtype=int)
    for new_idx, old_idx in enumerate(used_sorted):
        old_to_new[old_idx] = new_idx

    new_V = V[used_sorted]

    # Remap edges
    new_E = old_to_new[E]
    assert np.all(new_E >= 0), "Edge references orphaned vertex"

    # ── 2. Sort: inner first, border last ───────────────────────────────
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
    tissue.E_cells = E_cells   # unchanged
    tissue.num_inner_vertices = num_inner

    if E_pixels is not None:
        tissue.E_pixels = E_pixels  # unchanged (pixel arrays don't use vertex indices)

    # ── 3. Rebuild C_v ──────────────────────────────────────────────────
    tissue.C_v = _build_C_v(tissue.labels, final_V, final_E, E_cells)

    return tissue


def _build_C_v(
    labels: np.ndarray,
    V: np.ndarray,
    E: np.ndarray,
    E_cells: np.ndarray,
) -> List[List[int]]:
    """
    Build per-cell vertex loop C_v[lbl-1] = [v0, v1, ...] ordered CCW.
    """
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


# =====================================================================
# Utilities
# =====================================================================

def _vertex_degrees(E: np.ndarray, n_verts: int) -> np.ndarray:
    """Return degree array for all vertices."""
    deg = np.zeros(n_verts, dtype=int)
    for v1, v2 in E:
        deg[v1] += 1
        deg[v2] += 1
    return deg


def _vertex_degree(E: np.ndarray, v_idx: int) -> int:
    """Degree of a single vertex."""
    return int(np.sum((E[:, 0] == v_idx) | (E[:, 1] == v_idx)))
