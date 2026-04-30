#!/opt/homebrew/bin/python3.10
"""
Time-series force inference demo.

Simulates a 5-frame sequence by applying progressive noise to the label image,
solving each frame independently, then aligning the scales so tensions are
comparable across frames.

Run:
    python examples/demo_timeseries.py [--image data/test.tif] [--frames 5]
"""

from __future__ import annotations
import argparse, copy, sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tifffile

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from force_inference import segmentation, solvers
from force_inference.topology_label import extract_topology_label
from force_inference.split_four_way import split_high_degree_vertices
from force_inference.timeseries import TimeSeries, align_timeseries


# ─── helpers ────────────────────────────────────────────────────────────────

def _perturb_labels(labels: np.ndarray, rng: np.random.Generator,
                    strength: float = 0.5) -> np.ndarray:
    """Simulate a small amount of cell shape change by randomly eroding/dilating."""
    from scipy.ndimage import binary_erosion, binary_dilation, label as nd_label
    out = labels.copy()
    unique = [l for l in np.unique(labels) if l > 0]
    for lbl in rng.choice(unique, size=max(1, len(unique) // 5), replace=False):
        mask = labels == lbl
        if rng.random() < strength:
            mask = binary_erosion(mask, iterations=1)
        else:
            grow = binary_dilation(mask, iterations=1) & (labels == 0)
            mask = mask | grow
        out[mask] = lbl
    return out


def build_tissue(labels: np.ndarray):
    """Extract topology + split 4-way vertices."""
    tissue = extract_topology_label(
        labels, use_skeleton_geometry=False,
        collapse_stubs=True, collapse_tiny_twins=False,
    )
    if tissue is None:
        return None
    tissue = split_high_degree_vertices(copy.deepcopy(tissue), split_length=4.0)
    return tissue


def solve(tissue):
    """Run Bayesian solver; return best ForceResult (or None)."""
    if tissue is None:
        return None
    scan = solvers.solve_bayesian(tissue, mu=None)
    if scan is None:
        return None
    return scan.best_result


# ─── main ───────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, default=REPO / "data" / "test.tif")
    parser.add_argument("--frames", type=int, default=5)
    parser.add_argument("--out", type=Path, default=Path("/tmp/timeseries_demo.png"))
    args = parser.parse_args()

    # ── 1. Segment base image ─────────────────────────────────────────────────
    print(f"Segmenting {args.image} …")
    labels_base, gray = segmentation.segment_grayscale(
        str(args.image), h_depth=2.0, min_cell_size=5
    )

    rng = np.random.default_rng(42)
    times = list(range(args.frames))  # minutes

    ts = TimeSeries()

    for fi in range(args.frames):
        print(f"  Frame {fi}/{args.frames - 1} …", end=" ", flush=True)
        # Simulate slight shape change
        lbl = _perturb_labels(labels_base, rng, strength=0.3 * fi / max(args.frames - 1, 1))
        tissue = build_tissue(lbl)
        result = solve(tissue)
        if tissue is None or result is None:
            print("skipped (solver failed)")
            continue

        # Simulate a global tension increase (what we want alignment to recover)
        # In a real experiment you would NOT do this; this is just a test.
        true_scale = 1.0 + 0.15 * fi   # tension grows 15 % per frame
        result.tensions = result.tensions * true_scale

        ts.add_frame(tissue, result, time=float(fi))
        print(f"✓  {len(tissue.E)} edges, "
              f"median T (raw) = {np.nanmedian(result.tensions):.3f}")

    if len(ts) < 2:
        print("Not enough frames solved — exiting.")
        return

    # ── 2. Align scales ───────────────────────────────────────────────────────
    print("\nAligning scales (shared_edges strategy) …")
    ts.align(strategy="shared_edges", reference_frame=0)
    print("  Scale factors:", np.round(ts.scales, 4))

    # Also show what the median strategy gives
    ts_median = copy.deepcopy(ts)
    ts_median.align(strategy="median", reference_frame=0)

    # ── 3. Plot ───────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), facecolor="#111")
    fig.suptitle("Time-series force inference  ·  scale alignment demo",
                 color="white", fontsize=13, fontweight="bold")

    for ax in axes.flat:
        ax.set_facecolor("#1a1a1a")
        for sp in ax.spines.values():
            sp.set_edgecolor("#333")

    # Panel A — raw (unaligned) trajectories
    ax = axes[0, 0]
    trajs_raw = {}
    for fi in range(len(ts)):
        T = ts.results[fi].tensions   # NOT scaled
        for pair, traj in ts.edge_trajectories().items():
            break  # just to get the pair set; we redo below
    # Build raw trajs manually
    all_trajs = ts.edge_trajectories()
    t_arr = np.array(ts.times)
    for pair, traj in all_trajs.items():
        ax.plot(t_arr, traj / ts.scales, color="steelblue", lw=0.7, alpha=0.3)
    ax.set_title("Raw normalised tensions (each frame ~mean=1)", color="white", fontsize=10)
    ax.set_xlabel("Frame", color="#aaa"); ax.set_ylabel("T (normalised)", color="#aaa")
    ax.tick_params(colors="#aaa")

    # Panel B — shared-edge aligned trajectories
    ts.plot_trajectories(axes[0, 1], top_n=30, cmap="turbo", alpha=0.6)
    axes[0, 1].set_title("Aligned (shared_edges strategy)", color="white", fontsize=10)
    axes[0, 1].set_xlabel("Frame", color="#aaa"); axes[0, 1].set_ylabel("T (aligned)", color="#aaa")
    axes[0, 1].tick_params(colors="#aaa")

    # Panel C — scale factors
    ts.plot_scale_factors(axes[1, 0])
    axes[1, 0].set_title("Per-frame scale factors\n(should recover true_scale progression)",
                          color="white", fontsize=10)
    axes[1, 0].set_xlabel("Frame", color="#aaa"); axes[1, 0].set_ylabel("Scale", color="#aaa")
    axes[1, 0].tick_params(colors="#aaa")
    # Overlay true scale
    axes[1, 0].plot(t_arr, [1.0 + 0.15 * fi for fi in t_arr],
                    "r--", lw=2, label="true scale")
    axes[1, 0].legend(fontsize=8)

    # Panel D — distribution shift across frames (violin)
    ax = axes[1, 1]
    parts = ax.violinplot(
        [ts.aligned_tensions(fi)[~np.isnan(ts.aligned_tensions(fi))]
         for fi in range(len(ts))],
        positions=t_arr, widths=0.6, showmedians=True,
    )
    for pc in parts["bodies"]:
        pc.set_facecolor("steelblue"); pc.set_alpha(0.5)
    parts["cmedians"].set_color("cyan")
    ax.set_title("Aligned tension distribution per frame", color="white", fontsize=10)
    ax.set_xlabel("Frame", color="#aaa"); ax.set_ylabel("T (aligned)", color="#aaa")
    ax.tick_params(colors="#aaa")

    plt.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150, facecolor="#111", bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved → {args.out}")


if __name__ == "__main__":
    main()
