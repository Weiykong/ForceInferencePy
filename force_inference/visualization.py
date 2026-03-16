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
from typing import Optional, Tuple, List

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
    # Pre-sort interior pixels by projection onto the edge direction.
    # This is a cheap robustness guard: if _order_pixels produced a
    # slightly wrong ordering (e.g. two disconnected corner sets appended
    # in the wrong sequence), the projection sort will un-jumble them.
    # For genuinely curved edges the projection order is approximate but
    # still much better than random, and the high-smoothing spline below
    # will absorb any residual misplacement.
    # ------------------------------------------------------------------
    edge_vec = anchor_end - anchor_start
    edge_len = float(np.linalg.norm(edge_vec))
    if edge_len > 1e-6 and n > 2:
        d = edge_vec / edge_len
        # Sort only the interior points (skip first and last which are anchors)
        interior = pixels[1:-1]
        if len(interior) > 1:
            proj = (interior - anchor_start) @ d
            interior = interior[np.argsort(proj)]
            pixels = np.vstack([pixels[0], interior, pixels[-1]])
            n = len(pixels)

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
        # Smoothing factor s: large enough that the spline can freely
        # eliminate the half-integer staircase (amplitude ~0.5 px) and
        # any minor ordering imperfections without over-fitting.
        # s = n * 2.0 allows average ~1.4 px deviation per point — well
        # above the staircase amplitude but gentle enough to keep genuine
        # edge curvature.
        s = float(n) * 2.0

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
                  fix_zigzag: bool = True) -> None:
    """
    Plots the tissue edges colored by their inferred Tension.
    
    Args:
        ax: Matplotlib axes
        tissue: Tissue object
        result: ForceResult object
        cmap: Colormap name
        width: Line width
        alpha: Transparency
        fix_zigzag: If True, apply zigzag fix to edge pixels
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

    for i, (v1, v2) in enumerate(tissue.E):
        # Skip edges with NaN tension (excluded border edges)
        if np.isnan(T[i]):
            continue

        p1 = tissue.V[v1, :2]
        p2 = tissue.V[v2, :2]

        if use_pixels and len(tissue.E_pixels[i]) > 1:
            pixels = tissue.E_pixels[i]

            # Smooth the edge path, anchoring exactly at vertex positions
            if fix_zigzag:
                pixels = _fix_edge_pixels(pixels, v1_pos=p1, v2_pos=p2)

            if len(pixels) > 1:
                lines.append(pixels)
                valid_tensions.append(T[i])

        else:
            # Straight line fallback (also used when E_pixels is absent)
            lines.append(np.array([p1, p2]))
            valid_tensions.append(T[i])

    if len(lines) == 0:
        return

    valid_tensions = np.array(valid_tensions)
    
    lc = LineCollection(lines, array=valid_tensions, cmap=cmap, 
                        linewidths=width, norm=norm, alpha=alpha)
    ax.add_collection(lc)
    
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
    
    sc = ax.scatter(pts[:, 0], pts[:, 1], c=vals, cmap=cmap, 
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
