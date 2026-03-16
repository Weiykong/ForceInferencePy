#!/usr/bin/env python3
"""
End-to-end topology extraction test using the label-driven pipeline.

Usage:
    python test_topology.py --image path/to/labels.tif
    python test_topology.py --image test.tif --image2 example.tif

The script:
  1. Segments the image (or loads a pre-segmented label map)
  2. Runs extract_topology_label (full Voronoi + pixel detection pipeline)
  3. Saves a coloured edge overlay PNG next to the input

NOTE: This script now uses topology_label.py — not the old standalone
skeleton-based code that was here before.
"""

import argparse
import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from force_inference.topology_label import extract_topology_label


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_image(path):
    """Load a tif/png/jpg as a label map or grayscale image."""
    try:
        import tifffile
        arr = tifffile.imread(path)
    except Exception:
        from PIL import Image
        arr = np.array(Image.open(path))
    return arr


def segment_if_needed(arr):
    """
    If arr is already a label map (integer, many unique values) return it.
    Otherwise segment it as a grayscale membrane image.
    """
    unique = np.unique(arr)
    if arr.dtype in (np.int16, np.int32, np.int64, np.uint16, np.uint32) \
            or len(unique) > 10:
        return arr.astype(np.int32), None  # already labels
    # Treat as grayscale membrane image → segment
    try:
        from force_inference.segmentation import segment_grayscale
        labels, img = segment_grayscale(None, image_array=arr)
        return labels, arr
    except Exception as e:
        print(f"  Segmentation failed: {e}")
        print("  Treating as label map directly.")
        return arr.astype(np.int32), None


def visualize(image_bg, tissue, out_path, title=""):
    """
    Save a coloured-edge overlay.

    image_bg : background image (H,W) grayscale or (H,W,3) RGB
    tissue   : Tissue object from extract_topology_label
    """
    fig, ax = plt.subplots(1, 1, figsize=(14, 10))

    if image_bg is not None:
        ax.imshow(image_bg, cmap='gray', alpha=0.6)
    else:
        # Use label map as background
        ax.set_facecolor('#222222')

    if tissue is None:
        ax.set_title("FAILED — no topology extracted")
        plt.savefig(out_path, dpi=130, bbox_inches='tight')
        plt.close()
        return

    V = tissue.V[:, :2]
    E = tissue.E
    E_pix = getattr(tissue, 'E_pixels', None)

    cmap = plt.cm.hsv
    colors = cmap(np.linspace(0, 0.95, max(len(E), 1)))
    np.random.seed(42)
    np.random.shuffle(colors)

    for idx, (v1, v2) in enumerate(E):
        col = colors[idx % len(colors)]
        if E_pix is not None and idx < len(E_pix) and len(E_pix[idx]) > 1:
            pts = E_pix[idx]
            ax.plot(pts[:, 0], pts[:, 1],
                    color=col, lw=1.3, alpha=0.9, solid_capstyle='round')
        else:
            ax.plot([V[v1, 0], V[v2, 0]], [V[v1, 1], V[v2, 1]],
                    color=col, lw=1.3, alpha=0.9)

    ax.scatter(V[:, 0], V[:, 1],
               c='white', s=12, zorder=10,
               linewidths=0.4, edgecolors='black', alpha=0.9)

    n_inner = getattr(tissue, 'num_inner_vertices', len(V))
    ax.set_title(
        f"{title}\n"
        f"V={len(V)} ({n_inner} inner)  E={len(E)}  "
        f"cells={int(tissue.labels.max())}",
        fontsize=11
    )
    ax.axis('off')
    plt.tight_layout()
    plt.savefig(out_path, dpi=130, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_one(path, args):
    print(f"\n{'='*60}")
    print(f"Image: {path}")
    print('='*60)

    arr = load_image(path)
    print(f"  Loaded: shape={arr.shape} dtype={arr.dtype} "
          f"range=[{arr.min()}, {arr.max()}]")

    labels, img_bg = segment_if_needed(arr)
    n_cells = int(labels.max())
    print(f"  Cells: {n_cells}")

    tissue = extract_topology_label(
        labels,
        min_edge_len=args.min_edge_len,
        collapse_stubs=True,
        stub_edge_threshold=args.stub_threshold,
        collapse_tiny_twins=True,
        tiny_twin_threshold=args.twin_threshold,
        trace_pixels=True,
        use_skeleton_geometry=True,
    )

    if tissue:
        n_inner = getattr(tissue, 'num_inner_vertices', len(tissue.V))
        print(f"  Result: {len(tissue.V)} vertices ({n_inner} inner), "
              f"{len(tissue.E)} edges")
    else:
        print("  FAILED: extract_topology_label returned None")

    base = os.path.splitext(os.path.basename(path))[0]
    out = os.path.join(os.path.dirname(path), base + '_topology_label.png')
    visualize(img_bg if img_bg is not None else (labels > 0).astype(float),
              tissue, out, title=base)


def main():
    parser = argparse.ArgumentParser(
        description='Test label-driven topology extraction.')
    parser.add_argument('--image',  type=str, required=True,
                        help='Path to label tif or membrane image')
    parser.add_argument('--image2', type=str, default=None,
                        help='Optional second image for side-by-side comparison')
    parser.add_argument('--min_edge_len',    type=int,   default=3)
    parser.add_argument('--stub_threshold',  type=float, default=5.0)
    parser.add_argument('--twin_threshold',  type=float, default=3.0)
    args = parser.parse_args()

    run_one(args.image, args)
    if args.image2:
        run_one(args.image2, args)


if __name__ == '__main__':
    main()
