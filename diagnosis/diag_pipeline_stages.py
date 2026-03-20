"""
Stage-by-stage diagnostic for the Laplace pipeline.

Usage:
    python diagnosis/diag_pipeline_stages.py
    python diagnosis/diag_pipeline_stages.py --file data/test.tif
    python diagnosis/diag_pipeline_stages.py --file data/test.tif --crop 820 620 180
"""

import argparse
import copy
from collections import Counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from force_inference import geometry, segmentation, solvers, visualization
from force_inference.split_four_way import split_high_degree_vertices
from force_inference.topology_label import extract_topology_label


def _degree_counts(tissue):
    deg = Counter()
    for v1, v2 in tissue.E:
        deg[int(v1)] += 1
        deg[int(v2)] += 1
    return Counter(deg.values())


def _apply_crop(ax, crop):
    if crop is None:
        return
    cx, cy, half = crop
    ax.set_xlim(cx - half, cx + half)
    ax.set_ylim(cy + half, cy - half)


def run(filename: str, out: str, crop=None, split_length: float = 4.0):
    labels, img_smooth = segmentation.segment_grayscale(
        filename, h_depth=2.0, min_cell_size=5
    )

    tissue_raw = extract_topology_label(
        labels,
        min_edge_len=1,
        use_skeleton_geometry=False,
        collapse_stubs=False,
        collapse_tiny_twins=False,
    )
    if tissue_raw is None:
        raise RuntimeError("Topology extraction failed")

    tissue_split = copy.deepcopy(tissue_raw)
    tissue_split = split_high_degree_vertices(tissue_split, split_length=split_length)

    tissue_curv = copy.deepcopy(tissue_split)
    tissue_curv = geometry.compute_curvature(tissue_curv)

    result = solvers.solve_laplace(tissue_curv, regularization=1.0, detrend=True)
    if result is None:
        raise RuntimeError("Laplace solver failed")

    fig, axes = plt.subplots(2, 2, figsize=(16, 16))

    ax = axes[0, 0]
    ax.imshow(img_smooth, cmap="gray", alpha=0.5)
    visualization.plot_topology_check(ax, tissue_raw)
    ax.set_title(
        f"1. Extracted topology\nV={len(tissue_raw.V)} E={len(tissue_raw.E)} "
        f"deg={dict(sorted(_degree_counts(tissue_raw).items()))}"
    )
    _apply_crop(ax, crop)

    ax = axes[0, 1]
    ax.imshow(img_smooth, cmap="gray", alpha=0.5)
    visualization.plot_topology_check(ax, tissue_split)
    ax.set_title(
        f"2. After split\nV={len(tissue_split.V)} E={len(tissue_split.E)} "
        f"deg={dict(sorted(_degree_counts(tissue_split).items()))}"
    )
    _apply_crop(ax, crop)

    ax = axes[1, 0]
    ax.imshow(img_smooth, cmap="gray", alpha=0.5)
    visualization.plot_topology_check(ax, tissue_curv)
    ax.set_title(
        f"3. After curvature\ncurved edges={int(np.sum(np.abs(tissue_curv.E_curvature) > 1e-6))}"
    )
    _apply_crop(ax, crop)

    ax = axes[1, 1]
    ax.imshow(img_smooth, cmap="gray", alpha=0.5)
    visualization.plot_tensions(ax, tissue_curv, result, cmap="turbo")
    ax.set_title(
        f"4. Laplace tensions\nresidual={result.residual:.3g}"
    )
    _apply_crop(ax, crop)

    plt.tight_layout()
    plt.savefig(out, dpi=180, bbox_inches="tight")
    print(f"Saved -> {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", default="data/test.tif")
    ap.add_argument("--out", default="pipeline_stages.png")
    ap.add_argument("--split-length", type=float, default=4.0)
    ap.add_argument(
        "--crop",
        nargs=3,
        type=int,
        metavar=("X", "Y", "HALF"),
        help="Crop around (X, Y) with half-size HALF",
    )
    args = ap.parse_args()
    run(
        args.file,
        args.out,
        tuple(args.crop) if args.crop else None,
        split_length=args.split_length,
    )


if __name__ == "__main__":
    main()
