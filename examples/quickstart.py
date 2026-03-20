"""
ForceInferencePy — 5-minute quickstart
=======================================

Loads the bundled test image, extracts topology, runs Bayesian force
inference, and saves a tension-overlay figure to quickstart_output.png.

Usage:
    python examples/quickstart.py
"""

import os
import logging
import numpy as np
import argparse
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from force_inference import segmentation
from force_inference.topology_label import extract_topology_label
from force_inference.geometry import compute_curvature
from force_inference.solvers import solve_bayesian, BayesianScanResult

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")

def run_quickstart(img_path=None, method="cellpose"):
    # ------------------------------------------------------------------
    # 1. Locate the bundled example image
    # ------------------------------------------------------------------
    REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if img_path is None:
        img_path = os.path.join(REPO_ROOT, "data", "test.tif")

    if not os.path.exists(img_path):
        raise FileNotFoundError(
            f"Image not found at {img_path}.\n"
            "Run this script from the repository root or provide a valid path."
        )

    # ------------------------------------------------------------------
    # 2. Segment
    # ------------------------------------------------------------------
    print(f"Step 1 / 4 — Segmenting image using {method} …")
    if method == "cellpose":
        try:
            labels, img_processed = segmentation.segment_cellpose(img_path, model_type="cyto3")
        except ImportError:
            print("Cellpose not found, falling back to grayscale.")
            labels, img_processed = segmentation.segment_grayscale(img_path, h_depth=5, blur_sigma=1, min_cell_size=20)
    else:
        labels, img_processed = segmentation.segment_grayscale(img_path, h_depth=5, blur_sigma=1, min_cell_size=20)
    
    print(f"  Found {labels.max()} cells.")

    # ------------------------------------------------------------------
    # 3. Extract topology
    # ------------------------------------------------------------------
    print("Step 2 / 4 — Extracting topology …")
    tissue = extract_topology_label(labels)
    if tissue is None or len(tissue.E) == 0:
        raise RuntimeError("Topology extraction failed — no edges found.")
    print(f"  Vertices: {len(tissue.V)}, Edges: {len(tissue.E)}")

    # ------------------------------------------------------------------
    # 4. Compute curvature (needed by some solvers; optional for Bayesian)
    # ------------------------------------------------------------------
    print("Step 3 / 4 — Computing curvature …")
    tissue = compute_curvature(tissue)

    # ------------------------------------------------------------------
    # 5. Infer forces (Bayesian, automatic μ selection)
    # ------------------------------------------------------------------
    print("Step 4 / 4 — Inferring forces …")
    scan = solve_bayesian(tissue)  # returns BayesianScanResult when mu is None

    if scan is None:
        raise RuntimeError("Bayesian solver returned None — system may be too small.")

    if isinstance(scan, BayesianScanResult):
        result = scan.best_result
        print(f"  Best μ = {scan.best_mu:.3g}")
    else:
        result = scan

    tensions = result.tensions
    real_tensions = tensions[~np.isnan(tensions)]
    print(f"  Tension range: [{real_tensions.min():.3f}, {real_tensions.max():.3f}]")

    # ------------------------------------------------------------------
    # 6. Plot
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(img_processed, cmap="gray", origin="upper")
    ax.set_title("Bayesian tension overlay")
    ax.axis("off")

    Vxy = tissue.to_2d()
    t_min = np.nanmin(tensions)
    t_max = np.nanmax(tensions)
    t_range = max(t_max - t_min, 1e-9)

    cmap = plt.get_cmap("coolwarm")
    for i, (v1, v2) in enumerate(tissue.E):
        t = tensions[i]
        if np.isnan(t):
            continue
        color = cmap((t - t_min) / t_range)
        xs = [Vxy[v1, 0], Vxy[v2, 0]]
        ys = [Vxy[v1, 1], Vxy[v2, 1]]
        ax.plot(xs, ys, color=color, linewidth=1.5)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=t_min, vmax=t_max))
    plt.colorbar(sm, ax=ax, label="Tension (a.u.)", fraction=0.03, pad=0.02)

    OUT_PATH = os.path.join(REPO_ROOT, "quickstart_output.png")
    fig.savefig(OUT_PATH, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nDone! Output saved to: {OUT_PATH}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ForceInferencePy Quickstart")
    parser.add_argument("--filename", type=str, default=None, help="Path to TIF image")
    parser.add_argument("--method", type=str, default="cellpose", choices=["cellpose", "grayscale"], 
                        help="Segmentation method (default: cellpose)")
    args = parser.parse_args()
    
    run_quickstart(args.filename, args.method)
