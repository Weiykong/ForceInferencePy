#!/usr/bin/env python3
"""
Visual debug: junction detection over original .tif

Steps:
  1. Segment the .tif → label map
  2. Apply full Voronoi fill (close all membrane gaps)
  3. Run extract_topology_label
  4. Draw detected vertices + edges over the original raw image

Usage:
    python test_junction_detection.py                         # uses test.tif
    python test_junction_detection.py data/example.tif
    python test_junction_detection.py data/test.tif data/example.tif
"""

import sys
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import tifffile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from force_inference.segmentation import segment_grayscale
from force_inference.topology_label import (
    extract_topology_label,
    _full_voronoi_labels,
    _classify_boundary_pixels,
)


# ── helpers ──────────────────────────────────────────────────────────────────

def load_raw(path):
    """Load raw .tif as a displayable grayscale float image."""
    raw = tifffile.imread(path).astype(float)
    if raw.ndim == 3:
        raw = raw.mean(axis=0) if raw.shape[0] <= 4 else raw[..., :3].mean(axis=-1)
    raw -= raw.min()
    if raw.max() > 0:
        raw /= raw.max()
    return raw


def run(tif_path, out_path):
    print(f"\n{'='*60}")
    print(f"  {tif_path}")
    print('='*60)

    # ── 1. segment ────────────────────────────────────────────────────────────
    labels, _ = segment_grayscale(tif_path, h_depth=2.0, min_cell_size=5)
    raw = load_raw(tif_path)
    H, W = labels.shape
    print(f"  Labels: {H}×{W},  cells: {int(labels.max())}")

    # ── 2. full Voronoi fill ──────────────────────────────────────────────────
    labels_voronoi = _full_voronoi_labels(labels)
    print(f"  Voronoi fill: background pixels remaining = "
          f"{int(np.sum(labels_voronoi == 0))}")

    # ── 3. classify boundary pixels on filled labels ──────────────────────────
    vertex_mask, edge_mask, _ = _classify_boundary_pixels(
        labels_voronoi, half_window=1
    )
    print(f"  Vertex pixels: {int(np.sum(vertex_mask))}  "
          f"Edge pixels: {int(np.sum(edge_mask))}")

    # ── 4. extract topology ───────────────────────────────────────────────────
    tissue = extract_topology_label(
        labels,
        use_skeleton_geometry=False,
        collapse_stubs=True,
        collapse_tiny_twins=False,
        trace_pixels=True,
    )
    if tissue is None:
        print("  FAILED: extract_topology_label returned None")
        return

    n_inner = getattr(tissue, 'num_inner_vertices', len(tissue.V))
    print(f"  Vertices: {len(tissue.V)} ({n_inner} inner)  "
          f"Edges: {len(tissue.E)}")

    # ── 5. draw ───────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(24, 8))
    fig.suptitle(os.path.basename(tif_path), fontsize=13)

    # Panel A: voronoi fill coloured
    axes[0].imshow(labels_voronoi, cmap='tab20', interpolation='nearest',
                   vmin=0, vmax=int(labels.max()))
    vy, vx = np.where(vertex_mask)
    axes[0].scatter(vx, vy, c='red', s=1, linewidths=0, alpha=0.7)
    axes[0].set_title(f'Voronoi fill + vertex pixels ({int(np.sum(vertex_mask))})',
                      fontsize=10)
    axes[0].axis('off')

    # Panel B: topology over raw image — full view
    _draw_topology(axes[1], raw, tissue,
                   title=f'Topology over raw image\n'
                         f'V={len(tissue.V)} ({n_inner} inner)  E={len(tissue.E)}')

    # Panel C: zoom into a 300×300 crop of the centre (or whole image if small)
    cy, cx = H // 2, W // 2
    r = min(150, H // 3, W // 3)
    axes[2].imshow(raw, cmap='gray')
    _draw_topology_on_ax(axes[2], tissue)
    axes[2].set_xlim(cx - r, cx + r)
    axes[2].set_ylim(cy + r, cy - r)   # y-axis is flipped in imshow
    axes[2].set_title('Centre crop', fontsize=10)
    axes[2].axis('off')

    plt.tight_layout()
    plt.savefig(out_path, dpi=130, bbox_inches='tight')
    plt.close()
    print(f"  Saved → {out_path}")


def _draw_topology_on_ax(ax, tissue, lw=1.2, vs=15):
    """Draw edges + vertices on an existing axes that already has a background."""
    V = tissue.V[:, :2]
    E_pix = getattr(tissue, 'E_pixels', None)

    cmap = plt.cm.hsv
    colors = cmap(np.linspace(0, 0.95, max(len(tissue.E), 1)))
    np.random.seed(42)
    np.random.shuffle(colors)

    for idx, (v1, v2) in enumerate(tissue.E):
        col = colors[idx % len(colors)]
        if E_pix is not None and idx < len(E_pix) and len(E_pix[idx]) > 1:
            pts = E_pix[idx]
            ax.plot(pts[:, 0], pts[:, 1],
                    color=col, lw=lw, alpha=0.9, solid_capstyle='round')
        else:
            ax.plot([V[v1, 0], V[v2, 0]], [V[v1, 1], V[v2, 1]],
                    color=col, lw=lw, alpha=0.9)

    n_inner = getattr(tissue, 'num_inner_vertices', len(V))
    ax.scatter(V[:n_inner, 0], V[:n_inner, 1],
               c='white', s=vs, zorder=10,
               linewidths=0.5, edgecolors='black', alpha=0.95)
    if n_inner < len(V):
        ax.scatter(V[n_inner:, 0], V[n_inner:, 1],
                   c='cyan', s=vs * 0.6, zorder=10,
                   linewidths=0.3, edgecolors='black', alpha=0.7)


def _draw_topology(ax, raw, tissue, title=''):
    ax.imshow(raw, cmap='gray', alpha=0.55)
    _draw_topology_on_ax(ax, tissue)
    ax.set_title(title, fontsize=10)
    ax.axis('off')


# ── main ─────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    import logging
    logging.basicConfig(level=logging.INFO, format='%(name)s: %(message)s')

    paths = sys.argv[1:] if len(sys.argv) > 1 else [
        os.path.join('data', 'test.tif'),
        os.path.join('data', 'example.tif'),
    ]

    for p in paths:
        if not os.path.exists(p):
            print(f"  Not found: {p}  — skipping")
            continue
        base = os.path.splitext(os.path.basename(p))[0]
        out = os.path.join(os.path.dirname(p), f'{base}_junction_debug.png')
        run(p, out)
