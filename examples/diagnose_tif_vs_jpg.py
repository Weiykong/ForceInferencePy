"""
Diagnostic: compare JPG membrane vs TIF label image topology extraction.
Run this to see exactly where junctions are lost.

Usage:
    python diagnose_tif_vs_jpg.py \
        --jpg  path/to/membrane.jpg \
        --tif  path/to/labels.tif \
        --out  /tmp/diag
"""

import argparse
import numpy as np
from PIL import Image
from skimage import morphology
from scipy.ndimage import convolve, label as nd_label
from scipy.spatial import cKDTree
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os


# ── helpers ──────────────────────────────────────────────────────────────────

def load_image(path):
    ext = os.path.splitext(path)[1].lower()
    arr = None
    if ext in ('.tif', '.tiff'):
        try:
            import tifffile
            arr = tifffile.imread(path)
        except ImportError:
            pass
    if arr is None:
        arr = np.array(Image.open(path).convert('L'))

    print(f"  raw shape={arr.shape}  dtype={arr.dtype}")

    # Collapse to 2D
    if arr.ndim == 3:
        if arr.shape[0] <= 4:          # (C, H, W) — take first channel
            arr = arr[0]
            print(f"  → took channel 0 → {arr.shape}")
        elif arr.shape[2] <= 4:        # (H, W, C) — convert to gray
            arr = np.array(Image.fromarray(arr).convert('L'))
            print(f"  → converted RGB→gray → {arr.shape}")
        else:                          # (Z, H, W) z-stack — max-project
            arr = arr.max(axis=0)
            print(f"  → max-projected z-stack → {arr.shape}")
    elif arr.ndim == 4:                # (Z, C, H, W) — max-project then first channel
        arr = arr.max(axis=0)[0]
        print(f"  → max-proj + channel 0 → {arr.shape}")

    if arr.ndim != 2:
        raise ValueError(f"Cannot reduce image to 2D: shape={arr.shape}")
    return arr


def labels_to_boundary_thin(labels):
    """Standard 4-connected boundary — 1 px wide (used for label images)."""
    H, W = labels.shape
    b = np.zeros((H, W), dtype=bool)
    b[:-1, :] |= labels[:-1, :] != labels[1:, :]
    b[1:,  :] |= labels[:-1, :] != labels[1:, :]
    b[:, :-1] |= labels[:, :-1] != labels[:, 1:]
    b[:, 1:]  |= labels[:, :-1] != labels[:, 1:]
    return b


def count_branch_points(skel):
    kernel = np.ones((3, 3), dtype=np.uint8); kernel[1, 1] = 0
    nc = convolve(skel.astype(np.uint8), kernel, mode='constant', cval=0)
    return skel & (nc >= 3), nc


def cluster_count(branch_mask, r=3.0):
    by, bx = np.where(branch_mask)
    if len(by) == 0:
        return 0
    coords = np.column_stack((bx.astype(float), by.astype(float)))
    tree = cKDTree(coords)
    visited = np.zeros(len(coords), dtype=bool)
    n = 0
    for i in range(len(coords)):
        if visited[i]: continue
        nb = tree.query_ball_point(coords[i], r)
        visited[nb] = True
        n += 1
    return n


def analyze(name, boundary, ax_row):
    # Ensure 2D
    if boundary.ndim != 2:
        raise ValueError(f"analyze() expects 2D array, got shape {boundary.shape}")
    boundary = boundary.astype(bool)
    skel = morphology.skeletonize(boundary)
    branch_mask, nc = count_branch_points(skel)
    n_clustered = cluster_count(branch_mask)
    branch_mask, nc = count_branch_points(skel)
    n_clustered = cluster_count(branch_mask)

    # color skeleton by degree
    disp = np.zeros((*skel.shape, 3), dtype=float)
    disp[skel & (nc == 1)] = [0.2, 0.2, 1.0]   # endpoint: blue
    disp[skel & (nc == 2)] = [0.0, 0.8, 0.0]   # edge: green
    disp[skel & (nc >= 3)] = [1.0, 0.0, 0.0]   # branch: red

    ax_row[0].imshow(boundary, cmap='gray')
    ax_row[0].set_title(f'{name}\nboundary ({np.sum(boundary)} px)')

    ax_row[1].imshow(disp)
    ax_row[1].set_title(f'Skeleton\ngreen=edge  red=branch  blue=endpoint')

    by, bx = np.where(branch_mask)
    ax_row[2].imshow(skel, cmap='gray', alpha=0.4)
    ax_row[2].scatter(bx, by, c='red', s=3, alpha=0.7)
    ax_row[2].set_title(f'Branch pixels: {len(by)}\nClustered vertices: {n_clustered}')

    for ax in ax_row:
        ax.axis('off')

    print(f"\n{'='*50}")
    print(f"{name}")
    print(f"  boundary px   : {np.sum(boundary)}")
    print(f"  skeleton px   : {np.sum(skel)}")
    print(f"  branch px     : {len(by)}")
    print(f"  vertices (clust r=3): {n_clustered}")

    # membrane thickness estimate
    col = boundary[:, boundary.shape[1]//2]
    runs = []
    in_run = False; run_len = 0
    for v in col:
        if v:
            in_run = True; run_len += 1
        elif in_run:
            runs.append(run_len); in_run = False; run_len = 0
    if runs:
        print(f"  membrane thickness (mid-col): "
              f"mean={np.mean(runs):.1f}px  max={max(runs)}px  min={min(runs)}px")
    else:
        print(f"  membrane thickness: (no runs found in mid column)")

    return skel, branch_mask


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--jpg', required=True)
    ap.add_argument('--tif', required=True)
    ap.add_argument('--out', default='/tmp/diag')
    ap.add_argument('--threshold', type=int, default=128)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    # Load
    jpg_raw = np.array(Image.open(args.jpg).convert('L'))
    jpg_bin = (jpg_raw > args.threshold).astype(np.uint8)

    tif_raw = load_image(args.tif)
    print(f"TIF: shape={tif_raw.shape} dtype={tif_raw.dtype} "
          f"min={tif_raw.min()} max={tif_raw.max()} "
          f"unique={len(np.unique(tif_raw))} values")

    # Determine TIF boundary type
    if len(np.unique(tif_raw)) <= 2:
        tif_boundary = (tif_raw > args.threshold).astype(np.uint8)
        tif_label = "TIF (binary, thresh)"
    else:
        # label image → thin boundary
        tif_boundary = labels_to_boundary_thin(tif_raw)
        tif_label = "TIF (label image → thin boundary)"
        # also test dilated boundary
        tif_boundary_dilated = morphology.dilation(tif_boundary, morphology.disk(1))

    # ── Figure 1: boundary & skeleton comparison ──────────────────────────
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    analyze("JPG binary membrane", jpg_bin, axes[0])
    _, _ = analyze(tif_label, tif_boundary, axes[1])

    plt.suptitle('Boundary → Skeleton comparison\n'
                 'If TIF has far fewer branch points than JPG → '
                 'boundary is too thin, need dilation', fontsize=11)
    plt.tight_layout()
    p1 = os.path.join(args.out, 'compare_skeleton.png')
    plt.savefig(p1, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"\nSaved: {p1}")

    # ── Figure 2: effect of dilation on TIF label image ───────────────────
    if len(np.unique(tif_raw)) > 2:
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        analyze("TIF — no dilation", tif_boundary, axes[0])
        analyze("TIF — disk(1) dilation", tif_boundary_dilated, axes[1])
        plt.suptitle('Effect of pre-dilation on TIF label image\n'
                     'disk(1) should recover missing junctions', fontsize=11)
        plt.tight_layout()
        p2 = os.path.join(args.out, 'compare_dilation.png')
        plt.savefig(p2, dpi=120, bbox_inches='tight')
        plt.close()
        print(f"Saved: {p2}")
        print("\n→ If disk(1) dilation recovers branches, apply the fix in topology.py")


if __name__ == '__main__':
    main()