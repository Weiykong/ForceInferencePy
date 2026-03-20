import matplotlib.pyplot as plt
import os
import logging
import numpy as np
import argparse
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
import tifffile

from force_inference import segmentation, topology

logging.basicConfig(level=logging.INFO)


def _fix_edge_pixels(px: np.ndarray) -> np.ndarray:
    """Remove zigzag by greedy nearest-neighbor reordering."""
    if len(px) < 3:
        return px
    ordered = [0]
    remaining = list(range(1, len(px)))
    while remaining:
        curr = px[ordered[-1]]
        dists = np.linalg.norm(px[remaining] - curr, axis=1)
        nearest = remaining[int(np.argmin(dists))]
        ordered.append(nearest)
        remaining.remove(nearest)
    return px[ordered]


def plot_topology_check(ax: plt.Axes, tissue, img: np.ndarray = None, fix_zigzag: bool = True) -> None:
    """
    Plot topology check overlaid on the original image.
    
    Args:
        ax:         Matplotlib axes to draw on.
        tissue:     Tissue object from extract_topology.
        img:        Original image array (grayscale or RGB). If None, plain background.
        fix_zigzag: Whether to reorder edge pixels to avoid zigzag artifacts.
    """

    # --- 1. Show original image as background ---
    if img is not None:
        if img.ndim == 2:
            ax.imshow(img, cmap='gray', origin='upper', interpolation='bilinear')
        else:
            ax.imshow(img, origin='upper', interpolation='bilinear')
    else:
        ax.set_facecolor('#303030')

    # --- 2. Fill Cells (semi-transparent) ---
    patches = []
    for verts in tissue.C_v:
        if len(verts) < 3:
            continue
        coords = tissue.V[verts, :2]
        poly = Polygon(coords, closed=True)
        patches.append(poly)

    if patches:
        colors = np.random.rand(len(patches))
        p = PatchCollection(patches, cmap='nipy_spectral', alpha=0.25, edgecolors='none')
        p.set_array(colors)
        ax.add_collection(p)

    # --- 3. Draw Edges ---
    if hasattr(tissue, 'E_pixels') and tissue.E_pixels is not None and len(tissue.E_pixels) == len(tissue.E):
        for i, px in enumerate(tissue.E_pixels):
            if len(px) > 1:
                if fix_zigzag:
                    px = _fix_edge_pixels(px)
                ax.plot(px[:, 0], px[:, 1], color='lime', linewidth=1.5, alpha=0.85)
            else:
                v1, v2 = tissue.E[i]
                ax.plot(
                    [tissue.V[v1, 0], tissue.V[v2, 0]],
                    [tissue.V[v1, 1], tissue.V[v2, 1]],
                    color='red', linewidth=1.5, linestyle='--', alpha=0.85
                )
    else:
        # Fallback: draw straight lines between vertices
        for v1, v2 in tissue.E:
            ax.plot(
                [tissue.V[v1, 0], tissue.V[v2, 0]],
                [tissue.V[v1, 1], tissue.V[v2, 1]],
                color='lime', linewidth=1.5, alpha=0.85
            )

    # --- 4. Draw Vertices ---
    if len(tissue.V) > 0:
        ax.scatter(
            tissue.V[:, 0], tissue.V[:, 1],
            c='yellow', s=20, zorder=10,
            edgecolors='black', linewidths=0.5
        )

    ax.autoscale()
    ax.set_aspect('equal')
    ax.axis('off')


def run_topocheck(filename, method="cellpose"):
    if not os.path.exists(filename):
        print(f"Image not found: {filename}")
        return

    # --- Load original image for background ---
    print("Loading original image...")
    img = tifffile.imread(filename)

    # Normalize to [0, 1] for display
    img_display = img.astype(float)
    img_display -= img_display.min()
    if img_display.max() > 0:
        img_display /= img_display.max()

    # --- Segmentation ---
    print(f"Segmenting image using {method}...")
    if method == "cellpose":
        try:
            labels, _ = segmentation.segment_cellpose(filename, model_type="cyto3")
        except ImportError:
            print("Cellpose not found, falling back to grayscale.")
            labels, _ = segmentation.segment_grayscale(filename, h_depth=8.0, min_cell_size=10)
    else:
        labels, _ = segmentation.segment_grayscale(filename, h_depth=8.0, min_cell_size=10)

    # --- Topology Extraction ---
    print("Extracting Topology...")
    tissue = topology.extract_topology(labels, min_edge_len=2.0, clean=True)

    if tissue is None:
        print("Failed to extract tissue.")
        return

    print(f"Stats: {len(tissue.V)} vertices, {len(tissue.E)} edges, {len(tissue.C_v)} cells.")

    # --- Plot: side by side (original | topology overlay) ---
    fig, axes = plt.subplots(1, 1, figsize=(18, 9))

    # Right: topology overlay on original
    plot_topology_check(axes, tissue, img=img_display, fix_zigzag=True)
    axes.set_title(
        f"Topology Check  |  V={len(tissue.V)}  E={len(tissue.E)}  Cells={len(tissue.C_v)}",
        fontsize=13
    )

    plt.tight_layout()
    plt.savefig('topology_check.png', dpi=150, bbox_inches='tight')
    print("Saved: topology_check.png")
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Topology Check Demo")
    parser.add_argument("--filename", type=str, default="../data/test.tif", help="Path to TIF image")
    parser.add_argument("--method", type=str, default="cellpose", choices=["cellpose", "grayscale"], 
                        help="Segmentation method (default: cellpose)")
    args = parser.parse_args()
    
    run_topocheck(args.filename, args.method)
