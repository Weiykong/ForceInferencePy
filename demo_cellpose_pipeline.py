"""
Complete force inference pipeline using Cellpose segmentation.

This runs Cellpose segmentation → label-driven topology extraction →
4-way splitting → geometry → solver.

Usage:
    python demo_cellpose_pipeline.py data/test.tif
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import logging
import torch

# Explicitly check for MPS
if torch.backends.mps.is_available():
    print("DEBUG: MPS is available in demo script")

logging.basicConfig(level=logging.INFO, format="%(name)s: %(message)s")
logger = logging.getLogger("Pipeline")

from force_inference import geometry, segmentation, solvers
from force_inference.topology_label import extract_topology_label
from force_inference.split_four_way import split_high_degree_vertices


def run_pipeline(
    filename: str,
    # Cellpose parameters
    model_type: str = "cyto3",
    diameter: float = None,
    cellprob_threshold: float = 0.0,
    flow_threshold: float = 0.4,
    # Topology parameters
    collapse_stubs: bool = True,
    split_4way: bool = True,
    split_length: float = 2.0,
    # Solver parameters
    solver: str = "laplace",
    regularization: float = 1.0,
):
    """
    Full pipeline: Cellpose → topology → geometry → force inference.

    Args:
        filename:            Path to membrane image.
        model_type:          Cellpose model ("cyto3", "cpsam", "cyto2").
        diameter:            Cell diameter (None = auto).
        cellprob_threshold:  Lower → catches more faint cells (-2 to 0).
        flow_threshold:      Higher → more permissive (0.4 to 0.8).
        collapse_stubs:      Remove stub edges from topology.
        split_4way:          Split 4-way vertices into triple junctions.
        split_length:        Length of synthetic edges at split points.
        solver:              "laplace" or "bayesian".
        regularization:      Regularization strength for solver.
    """
    if not os.path.exists(filename):
        raise FileNotFoundError(filename)

    # ---- 1. Segmentation ----
    logger.info("Step 1: Cellpose segmentation...")
    labels, gray = segmentation.segment_cellpose(
        filename,
        model_type=model_type,
        diameter=diameter,
        cellprob_threshold=cellprob_threshold,
        flow_threshold=flow_threshold,
        gpu=True,
    )
    logger.info(f"  → {labels.max()} cells")

    # ---- 2. Topology extraction ----
    logger.info("Step 2: Topology extraction...")
    tissue = extract_topology_label(
        labels,
        use_skeleton_geometry=False,
        collapse_stubs=collapse_stubs,
        collapse_tiny_twins=False,
    )
    if tissue is None:
        logger.error("Topology extraction failed.")
        return None
    logger.info(f"  → {len(tissue.V)} vertices, {len(tissue.E)} edges")

    # ---- 3. Split 4-way vertices ----
    if split_4way:
        logger.info("Step 3: Splitting 4-way vertices...")
        tissue = split_high_degree_vertices(tissue, split_length=split_length)
        logger.info(f"  → {len(tissue.V)} vertices, {len(tissue.E)} edges")

    # ---- 4. Geometry ----
    logger.info("Step 4: Computing geometry (curvature + tangents)...")
    tissue = geometry.compute_curvature(tissue)

    # ---- 5. Solver ----
    logger.info(f"Step 5: Solving ({solver})...")
    if solver == "laplace":
        result = solvers.solve_laplace(
            tissue, regularization=regularization
        )
    elif solver == "bayesian":
        # Use patched solver if available, else original
        try:
            from solvers_patched import solve_bayesian_patched
            result = solve_bayesian_patched(tissue, mu=1e-2)
        except ImportError:
            result = solvers.solve_bayesian(tissue, mu=1e-2)
    else:
        raise ValueError(f"Unknown solver: {solver}")

    if result is None:
        logger.error("Solver failed.")
        return None

    T = result.tensions
    nan_count = np.sum(np.isnan(T))
    logger.info(
        f"  → T=[{np.nanmin(T):.3f}, {np.nanmax(T):.3f}], "
        f"NaN={nan_count}, residual={result.residual:.2f}"
    )

    return tissue, result, labels, gray


def plot_results(tissue, result, labels, gray, save_path=None):
    """Quick visualization of results."""
    from force_inference import visualization

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Tensions
    ax = axes[0]
    ax.imshow(gray, cmap="gray", alpha=0.5)
    visualization.plot_tensions(ax, tissue, result, cmap="turbo")
    ax.set_title(f"Tensions ({len(tissue.E)} edges)")

    # Pressures
    ax = axes[1]
    ax.imshow(gray, cmap="gray", alpha=0.5)
    visualization.plot_pressures(ax, tissue, result, cmap="coolwarm")
    ax.set_title(f"Pressures ({labels.max()} cells)")

    # Topology check
    ax = axes[2]
    ax.imshow(gray, cmap="gray", alpha=0.5)
    visualization.plot_topology_check(ax, tissue)
    ax.set_title("Topology")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"Saved: {save_path}")
    plt.show()


# =========================================================================
# Parameter tuning guide
# =========================================================================

TUNING_GUIDE = """
╔══════════════════════════════════════════════════════════════════╗
║                  CELLPOSE PARAMETER TUNING                      ║
╠══════════════════════════════════════════════════════════════════╣
║                                                                  ║
║  Problem: UNDER-segmented (cells merged, missing edges)          ║
║  Fix:                                                            ║
║    • Lower cellprob_threshold: -1, -2, -3 (catches faint cells) ║
║    • Smaller diameter: try 0.7× current value                    ║
║    • Higher flow_threshold: 0.6, 0.8 (more permissive)          ║
║                                                                  ║
║  Problem: OVER-segmented (too many cells, false boundaries)      ║
║  Fix:                                                            ║
║    • Higher cellprob_threshold: 1, 2 (stricter)                  ║
║    • Larger diameter: try 1.3× current value                     ║
║    • Lower flow_threshold: 0.2, 0.3 (less permissive)           ║
║                                                                  ║
║  Problem: Small cells missed                                     ║
║  Fix:                                                            ║
║    • Lower min_size: 5-10                                        ║
║    • Smaller diameter                                            ║
║                                                                  ║
║  Problem: Large cells split                                      ║
║  Fix:                                                            ║
║    • Larger diameter                                             ║
║    • Higher cellprob_threshold                                   ║
║                                                                  ║
║  Model choice:                                                   ║
║    • cyto3: best overall (recommended)                           ║
║    • cyto2: slightly older, sometimes better on specific images  ║
║    • cpsam: best general model in Cellpose 4                     ║
║                                                                  ║
╚══════════════════════════════════════════════════════════════════╝
"""


if __name__ == "__main__":
    fname = sys.argv[1] if len(sys.argv) > 1 else "./data/test.tif"

    if "--help" in sys.argv or "-h" in sys.argv:
        print(TUNING_GUIDE)
        sys.exit(0)

    output = run_pipeline(
        fname,
        model_type="cyto3",
        diameter=None,
        cellprob_threshold=0.0,
        split_4way=True,
        solver="laplace",
    )

    if output is not None:
        tissue, result, labels, gray = output
        save_path = fname.rsplit(".", 1)[0] + "_result.png"
        plot_results(tissue, result, labels, gray, save_path=save_path)
