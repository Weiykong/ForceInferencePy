import os

import matplotlib.pyplot as plt
import numpy as np
import tifffile
from force_inference.topology_label import extract_topology_label
from force_inference import geometry, segmentation, solvers, topology, visualization


def _load_membrane_binary(filename):
    raw = tifffile.imread(filename)
    if raw.ndim == 3:
        if raw.shape[0] <= 4:  # (C, H, W)
            raw = raw[0]
        elif raw.shape[2] <= 4:  # (H, W, C)
            raw = (
                0.299 * raw[:, :, 0]
                + 0.587 * raw[:, :, 1]
                + 0.114 * raw[:, :, 2]
            ).astype(raw.dtype)
        else:  # (Z, H, W)
            raw = raw.max(axis=0)
    membrane_binary = (raw > 128).astype(np.uint8)
    return membrane_binary


def run_laplace_demo(filename="/Users/weiyuankong/ForceInferencePy/data/example.tif"):
    if not os.path.exists(filename):
        print(f"File {filename} not found.")
        return

    # 1. Segment for cell labels and background image
    labels, img_smooth = segmentation.segment_grayscale(
        filename, h_depth=2.0, min_cell_size=5
    )

    # 2. Extract topology (label-driven, full-Voronoi pipeline)
    print("Extracting topology...")
    tissue = extract_topology_label(
        labels,
        use_skeleton_geometry=False,  # don't snap to original skeleton
        collapse_stubs=True,
        collapse_tiny_twins=False,
    )
    if tissue is None:
        print("Topology extraction failed.")
        return
    if tissue.num_inner_vertices == 0:
        print("No inner vertices found.")
        return

    # 3. Geometry (curvature + tangents)
    print("Computing geometry (Curvature & Tangents)...")
    tissue = geometry.compute_curvature(tissue)

    # 4. Laplace solver
    print("Solving Forces (Curved Edge Balance)...")
    result = solvers.solve_laplace(tissue, regularization=1.0, detrend=True)
    if result is None:
        print("Solver failed.")
        return

    # 5. Visualize
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(14, 6))

    ax1.imshow(img_smooth, cmap="gray", alpha=0.5)
    visualization.plot_tensions(ax1, tissue, result, cmap="turbo")
    ax1.set_title("Inferred Tensions (Curved Balance)")

    ax2.imshow(img_smooth, cmap="gray", alpha=0.5)
    visualization.plot_pressures(ax2, tissue, result, cmap="coolwarm")
    ax2.set_title("Inferred Pressures (Laplace Law)")

    ax3.imshow(img_smooth, cmap="gray", alpha=0.5)
    visualization.plot_topology_check(ax3, tissue)
    ax3.set_title("Topology Check (Junctions & Edges)")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    run_laplace_demo()
