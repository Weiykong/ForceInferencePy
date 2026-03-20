"""
Visualization module for Force Inference results.

INCLUDES FIX for zigzag/horizontal-lines in edge plotting.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection, PatchCollection
from matplotlib.patches import Polygon
from matplotlib.colors import Normalize
from scipy.interpolate import splprep, splev
import logging
from typing import Optional, Tuple

from .core import Tissue, ForceResult

logger = logging.getLogger("ForceInference.Vis")


# =============================================================================
# THE FIX: Smooth edge pixels before plotting
# =============================================================================

def _fix_edge_pixels(
    pixels: np.ndarray,
    n_output_points: int = 0,
    v1_pos: Optional[np.ndarray] = None,
    v2_pos: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Smooth already-ordered edge pixels using a spline fit.

    After the topology rework, E_pixels already has vertex positions
    prepended/appended (pixels[0] == v1, pixels[-1] == v2).  The spline
    smooths the interior zigzag caused by half-integer corner coordinates
    while keeping both endpoints exactly at the vertex positions.

    Args:
        pixels:         (N, 2) ordered edge points including vertex endpoints.
        n_output_points: Number of output points.  0 = auto (scales with
                         edge length so line density stays roughly constant).
        v1_pos / v2_pos: Optional explicit vertex positions.  When given,
                         these override pixels[0] / pixels[-1] as anchors,
                         which is useful when the stored endpoints may have
                         been rounded or snapped slightly.

    Returns:
        Smoothed (M, 2) path anchored exactly at the vertex positions.
    """
    pixels = np.asarray(pixels, dtype=float)
    n = len(pixels)

    # Determine anchor positions
    anchor_start = v1_pos if v1_pos is not None else pixels[0]
    anchor_end   = v2_pos if v2_pos is not None else pixels[-1]

    if n < 2:
        return np.array([anchor_start, anchor_end])

    if n == 2:
        # Just two points — straight line, no smoothing needed
        return np.array([anchor_start, anchor_end])

    # ------------------------------------------------------------------
    # Sort ALL pixels by projection onto the edge direction.
    #
    # This handles two failure modes from the upstream pixel orderer:
    #   (a) Interior-only sort left a low-projection pixel at the tail,
    #       creating a backward-curving spline tail.
    #   (b) Fallback edges (one junction vertex found by distance
    #       heuristic) have all their pixels near anchor_start; the last
    #       pixel also ends up near anchor_start.
    #
    # After sorting ALL pixels by projection, pixels[-1] is guaranteed to
    # be the pixel with the highest projection onto anchor_start→anchor_end.
    # If that projection is still less than half the edge length, the
    # stored pixel path simply doesn't reach anchor_end — fall back to a
    # straight line so the edge is at least drawn correctly.
    # ------------------------------------------------------------------
    edge_vec = anchor_end - anchor_start
    edge_len = float(np.linalg.norm(edge_vec))
    if edge_len > 1e-6 and n > 2:
        d = edge_vec / edge_len
        proj_all = (pixels - anchor_start) @ d
        pixels = pixels[np.argsort(proj_all)]
        proj_all = proj_all[np.argsort(proj_all)]   # keep in sync
        n = len(pixels)

        # If the farthest pixel barely reaches halfway toward anchor_end,
        # the stored path is clustered at anchor_start (bad fallback edge).
        # Use a straight line instead of a misleading tangled spline.
        if proj_all[-1] < 0.5 * edge_len:
            t = np.linspace(0, 1, max(n_output_points, 10))
            return np.outer(1 - t, anchor_start) + np.outer(t, anchor_end)

    # Auto n_output_points: ~1 point per pixel of arc length, min 10
    if n_output_points <= 0:
        arc = float(np.sum(np.linalg.norm(np.diff(pixels, axis=0), axis=1)))
        n_output_points = max(10, min(int(arc) + 1, 200))

    if n < 4:
        # Linear interpolation for very short edges
        t = np.linspace(0, 1, n_output_points)
        result = np.outer(1 - t, anchor_start) + np.outer(t, anchor_end)
        return result

    try:
        # Smoothing factor s — kept deliberately small so the spline stays
        # close to the label-boundary pixel path.
        #
        # Diagnosis (test.tif, 1 903 edges):
        #   s = n * 2.0  →  average ~1.4 px deviation per stored pixel
        #                    3.5 % of cell-cell interface pixels uncovered
        #                    with 1 px line-half-width (visible gaps at membrane)
        #   s = n * 0.25 →  average ~0.5 px deviation per stored pixel
        #                    1.0 % uncovered  (pixel staircase smoothed away,
        #                    path stays within ½ px of the membrane centre)
        #
        # Rule of thumb: s = n * amplitude²  where amplitude is the
        # per-point noise level we want to eliminate.
        # Pixel staircase amplitude ≈ 0.5 px  →  s = n * 0.25
        s = float(n) * 0.25

        tck, _ = splprep(
            [pixels[:, 0], pixels[:, 1]],
            s=s,
            k=min(3, n - 1),
        )
        u_new = np.linspace(0, 1, n_output_points)
        smooth_x, smooth_y = splev(u_new, tck)
        result = np.column_stack([smooth_x, smooth_y])

        # Anchor exactly at vertex positions
        result[0]  = anchor_start
        result[-1] = anchor_end

        return result

    except Exception:
        # Fall back: linear interpolation between the two endpoints
        t = np.linspace(0, 1, n_output_points)
        result = np.outer(1 - t, anchor_start) + np.outer(t, anchor_end)
        return result


# =============================================================================
# Plotting functions
# =============================================================================

def plot_tensions(ax: plt.Axes,
                  tissue: Tissue,
                  result: ForceResult,
                  cmap: str = 'turbo',
                  width: float = 2.0,
                  alpha: float = 1.0,
                  fix_zigzag: bool = True,
                  show_nan_edges: bool = True) -> None:
    """
    Plots the tissue edges colored by their inferred Tension.

    Edges whose tension is NaN (e.g. excluded Bayesian border edges) are drawn
    as thin light-gray lines so the full network is always visible.  Set
    ``show_nan_edges=False`` to suppress them entirely.

    Args:
        ax: Matplotlib axes
        tissue: Tissue object
        result: ForceResult object
        cmap: Colormap name
        width: Line width in points
        alpha: Transparency
        fix_zigzag: If True, apply spline smoothing to edge pixels
        show_nan_edges: If True, draw NaN-tension edges as a neutral gray line
    """
    if result is None or len(tissue.E) == 0:
        return

    T = result.tensions

    # Handle NaN tensions (excluded border edges)
    valid_mask = ~np.isnan(T)
    T_valid = T[valid_mask]
    if len(T_valid) == 0:
        return

    vmin, vmax = np.percentile(T_valid, 5), np.percentile(T_valid, 95)
    norm = Normalize(vmin=vmin, vmax=vmax)

    use_pixels = (tissue.E_pixels is not None) and (len(tissue.E_pixels) == len(tissue.E))

    lines = []
    valid_tensions = []
    nan_lines = []

    for i, (v1, v2) in enumerate(tissue.E):
        p1 = tissue.V[v1, :2]
        p2 = tissue.V[v2, :2]

        if use_pixels and len(tissue.E_pixels[i]) > 1:
            pixels = tissue.E_pixels[i]
            if fix_zigzag:
                pixels = _fix_edge_pixels(pixels, v1_pos=p1, v2_pos=p2)
            path = pixels if len(pixels) > 1 else np.array([p1, p2])
        else:
            path = np.array([p1, p2])

        if np.isnan(T[i]):
            nan_lines.append(path)
        else:
            lines.append(path)
            valid_tensions.append(T[i])

    # Draw NaN edges first (behind colored edges)
    if show_nan_edges and nan_lines:
        lc_nan = LineCollection(
            nan_lines,
            colors=(0.55, 0.55, 0.55, 0.45 * alpha),
            linewidths=max(0.8, width * 0.5),
            capstyle='round',
            joinstyle='round',
        )
        ax.add_collection(lc_nan)

    if len(lines) == 0:
        return

    valid_tensions = np.array(valid_tensions)

    lc = LineCollection(
        lines,
        array=valid_tensions,
        cmap=cmap,
        linewidths=width,
        norm=norm,
        alpha=alpha,
        capstyle='round',       # round caps fill the gap at junction points
        joinstyle='round',
    )
    ax.add_collection(lc)

    # ------------------------------------------------------------------
    # Coloured dot at every inner junction vertex.
    #
    # Three round-capped line ends meeting at a vertex leave a small
    # triangular gap.  A scatter dot coloured by the mean tension of the
    # incident edges seals this gap.
    #
    # Correct matplotlib scatter sizing (s is area in pt², not diameter²):
    #   dot radius r (pt) → s = π · r²
    #
    # We want r = linewidth so the dot diameter equals 2 × linewidth.
    # This fills junction gaps for realistic tissue angles (≥ ~60°).
    # ------------------------------------------------------------------
    num_inner = getattr(tissue, 'num_inner_vertices', len(tissue.V))
    v_tension_sum = np.zeros(num_inner)
    v_tension_cnt = np.zeros(num_inner, dtype=int)
    for i, (v1, v2) in enumerate(tissue.E):
        t_val = T[i]
        if np.isnan(t_val):
            continue
        if v1 < num_inner:
            v_tension_sum[v1] += t_val
            v_tension_cnt[v1] += 1
        if v2 < num_inner:
            v_tension_sum[v2] += t_val
            v_tension_cnt[v2] += 1
    has_tension = v_tension_cnt > 0
    v_mean = np.where(has_tension, v_tension_sum / np.maximum(v_tension_cnt, 1), np.nan)

    valid_v = np.where(has_tension)[0]
    if len(valid_v) > 0:
        vx = tissue.V[valid_v, 0]
        vy = tissue.V[valid_v, 1]
        vt = v_mean[valid_v]
        # s = π * r² where r = linewidth → dot diameter = 2 × linewidth
        dot_size = np.pi * width ** 2
        ax.scatter(
            vx, vy,
            s=dot_size,
            c=vt,
            cmap=cmap,
            norm=norm,
            edgecolors='none',
            zorder=3,
            alpha=alpha,
            linewidths=0,
        )

    if not hasattr(ax, '_tension_colorbar_added'):
        plt.colorbar(lc, ax=ax, label="Tension (normalized)")
        ax._tension_colorbar_added = True

    ax.set_aspect('equal')
    ax.axis('off')


def plot_pressures(ax: plt.Axes, 
                   tissue: Tissue, 
                   result: ForceResult, 
                   cmap: str = 'bwr', 
                   size: float = 30) -> None:
    """Plots cell centroids colored by inferred Pressure."""
    if result is None:
        return

    P = result.pressures
    
    valid_indices = []
    for c_idx, verts in enumerate(tissue.C_v):
        if len(verts) > 2 and c_idx < len(P):
            valid_indices.append(c_idx)
            
    if not valid_indices:
        return
        
    pts = tissue.C_centroids[valid_indices]
    vals = P[valid_indices]
    
    max_abs = np.max(np.abs(vals)) if len(vals) > 0 else 1.0
    norm = Normalize(vmin=-max_abs, vmax=max_abs)
    
    ax.scatter(pts[:, 0], pts[:, 1], c=vals, cmap=cmap, 
               s=size, edgecolors='k', norm=norm)
    
    ax.set_aspect('equal')
    ax.axis('off')


def plot_curvature(ax: plt.Axes, 
                   tissue: Tissue, 
                   cmap: str = 'plasma',
                   fix_zigzag: bool = True) -> None:
    """Plots edges colored by their calculated Curvature."""
    if tissue.E_curvature is None:
        logger.warning("No curvature data found. Cannot plot.")
        return
        
    vals = np.abs(tissue.E_curvature)
    norm = Normalize(vmin=0, vmax=np.percentile(vals, 95))
    
    lines = []
    valid_vals = []
    
    use_pixels = (tissue.E_pixels is not None) and (len(tissue.E_pixels) == len(tissue.E))

    for i, (v1, v2) in enumerate(tissue.E):
        p1 = tissue.V[v1, :2]
        p2 = tissue.V[v2, :2]
        if use_pixels and len(tissue.E_pixels[i]) > 1:
            pixels = tissue.E_pixels[i]
            if fix_zigzag:
                pixels = _fix_edge_pixels(pixels, v1_pos=p1, v2_pos=p2)
            lines.append(pixels)
            valid_vals.append(vals[i])
        else:
            lines.append(np.array([p1, p2]))
            valid_vals.append(vals[i])

    lc = LineCollection(lines, array=np.array(valid_vals), cmap=cmap,
                        linewidths=2, norm=norm)
    ax.add_collection(lc)
    ax.set_aspect('equal')
    ax.axis('off')


def plot_topology_check(ax: plt.Axes, 
                        tissue: Tissue,
                        fix_zigzag: bool = True) -> None:
    """
    Debug plot showing cells, edges, and junction numbers.
    """
    if len(tissue.C_v) == 0: 
        return

    # 1. Fill Cells
    patches = []
    for verts in tissue.C_v:
        if len(verts) < 3: 
            continue
        coords = tissue.V[verts, :2]
        poly = Polygon(coords, closed=True)
        patches.append(poly)
    
    if patches:
        colors = np.random.rand(len(patches)) 
        p = PatchCollection(patches, cmap='nipy_spectral', alpha=0.6, edgecolors='none')
        p.set_array(colors)
        ax.add_collection(p)

    # 2. Draw Edges
    if tissue.E_pixels is not None and len(tissue.E_pixels) == len(tissue.E):
        for i, px in enumerate(tissue.E_pixels):
            v1, v2 = tissue.E[i]
            p1 = tissue.V[v1, :2]
            p2 = tissue.V[v2, :2]
            if len(px) > 1:
                if fix_zigzag:
                    px = _fix_edge_pixels(px, v1_pos=p1, v2_pos=p2)
                ax.plot(px[:, 0], px[:, 1], color='black', linewidth=1.5)
            else:
                ax.plot([p1[0], p2[0]], [p1[1], p2[1]],
                        color='black', linewidth=1.5, linestyle='--')

    # 3. Draw Junctions
    if len(tissue.V) > 0:
        # Plot inner vertices
        inner_v = tissue.num_inner_vertices if hasattr(tissue, 'num_inner_vertices') else len(tissue.V)
        ax.scatter(tissue.V[:inner_v, 0], tissue.V[:inner_v, 1], 
                   c='white', s=15, zorder=10, edgecolors='black', linewidth=0.5)

    ax.set_aspect('equal')
    ax.set_facecolor('#303030')
    ax.axis('off')


def plot_stress_crosses(ax: plt.Axes, 
                        grid_coords: Tuple[np.ndarray, np.ndarray], 
                        grid_tensors: np.ndarray, 
                        scale: float = 1.0, 
                        min_mag: float = 0.01) -> None:
    """Plots principal stress crosses on a grid."""
    if grid_coords is None or grid_tensors is None:
        return

    grid_x, grid_y = grid_coords
    H, W = grid_tensors.shape[:2]
    
    for i in range(H):
        for j in range(W):
            sigma = grid_tensors[i, j]
            if np.all(sigma == 0): 
                continue
            
            evals, evecs = np.linalg.eigh(sigma)
            cx, cy = grid_x[i, j], grid_y[i, j]
            
            for k in range(2):
                val = evals[k]
                vec = evecs[:, k]
                if abs(val) < min_mag: 
                    continue
                color = 'r' if val > 0 else 'b'
                length = scale * abs(val)
                dx = vec[0] * length / 2
                dy = vec[1] * length / 2
                ax.plot([cx - dx, cx + dx], [cy - dy, cy + dy], 
                        color=color, lw=1.5, alpha=0.8)


def plot_cell_stress_crosses(
    ax: plt.Axes,
    tissue: Tissue,
    result: ForceResult,
    scale: float = 60.0,
    min_mag: float = 0.01,
) -> None:
    """Plot principal stress crosses at cell centroids (better positional alignment)."""
    if result is None or result.stress_tensors is None:
        return

    centroids = np.asarray(tissue.C_centroids, dtype=float)
    tensors = np.asarray(result.stress_tensors, dtype=float)
    n = min(len(centroids), len(tensors))
    if n == 0:
        return

    H, W = tissue.labels.shape if getattr(tissue, "labels", None) is not None else (None, None)

    for ci in range(n):
        cx, cy = centroids[ci]
        if not np.isfinite([cx, cy]).all():
            continue
        if cx == 0 and cy == 0:
            continue

        if H is not None and W is not None:
            x_idx = int(round(cx))
            y_idx = int(round(cy))
            if x_idx < 0 or x_idx >= W or y_idx < 0 or y_idx >= H:
                continue
            if tissue.labels[y_idx, x_idx] == 0:
                continue

        sigma = tensors[ci]
        if not np.isfinite(sigma).all() or np.allclose(sigma, 0):
            continue

        evals, evecs = np.linalg.eigh(sigma)
        for k in range(2):
            val = float(evals[k])
            if abs(val) < min_mag:
                continue
            vec = evecs[:, k]
            color = 'r' if val > 0 else 'b'
            length = scale * abs(val)
            dx = vec[0] * length / 2.0
            dy = vec[1] * length / 2.0
            ax.plot([cx - dx, cx + dx], [cy - dy, cy + dy], color=color, lw=1.2, alpha=0.8)

    # Keep overlays aligned to image pixel coordinates after line plotting.
    if H is not None and W is not None:
        ax.set_xlim(-0.5, W - 0.5)
        ax.set_ylim(H - 0.5, -0.5)
