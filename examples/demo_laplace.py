import os
import sys
import matplotlib.pyplot as plt
import numpy as np
import argparse
from force_inference.split_four_way import extract_and_split
from force_inference import geometry, segmentation, solvers, visualization

def run_laplace_demo(filename="./data/test.tif", split_length=4.0, method="cellpose"):
    if not os.path.exists(filename):
        print(f"File {filename} not found.")
        return

    # 1. Segment for cell labels and background image
    print(f"Segmenting with {method}...")
    if method == "cellpose":
        try:
            labels, img_smooth = segmentation.segment_cellpose(
                filename, model_type="cyto3", cellprob_threshold=0.0
            )
        except ImportError:
            print("Cellpose not installed. Falling back to grayscale watershed.")
            labels, img_smooth = segmentation.segment_grayscale(
                filename, h_depth=2.0, min_cell_size=5
            )
    else:
        labels, img_smooth = segmentation.segment_grayscale(
            filename, h_depth=2.0, min_cell_size=5
        )

    # 2. Extract topology and split 4-way junctions before curvature fitting
    print("Extracting topology and splitting high-degree junctions...")
    tissue = extract_and_split(
        labels,
        split_length=split_length,
        min_edge_len=1,
        use_skeleton_geometry=False, 
        collapse_stubs=True,
        collapse_tiny_twins=False,
    )
    if tissue is None:
        print("Topology extraction failed.")
        return

    # 3. Geometry (curvature + tangents)
    print("Computing geometry (Curvature & Tangents)...")
    tissue = geometry.compute_curvature(tissue)

    # 4. Laplace solver
    print("Solving Forces (Curved Edge Balance)...")
    result = solvers.solve_laplace(tissue, regularization=1.0)
    if result is None:
        print("Solver failed.")
        return

    # 5. Visualize
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))

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
    out_name = os.path.basename(filename).split('.')[0] + "_laplace_result.png"
    plt.savefig(out_name, dpi=150)
    print(f"Saved result to {out_name}")
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("filename", nargs="?", default="./data/test.tif")
    parser.add_argument("--method", choices=["cellpose", "grayscale"], default="cellpose")
    args = parser.parse_args()
    
    run_laplace_demo(filename=args.filename, method=args.method)
