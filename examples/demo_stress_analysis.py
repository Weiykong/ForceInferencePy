import matplotlib.pyplot as plt
import os
import argparse

from force_inference import segmentation, topology, solvers, geometry, visualization

def run_stress_demo(filename, method="cellpose"):
    # 1. Setup
    
    if not os.path.exists(filename):
        print(f"File {filename} not found.")
        return

    # 2. Pipeline: Segment -> Topology -> Solver
    print(f"Segmenting image using {method}...")
    if method == "cellpose":
        try:
            labels, img = segmentation.segment_cellpose(filename, model_type="cyto3")
        except ImportError:
            print("Cellpose not found, falling back to grayscale.")
            labels, img = segmentation.segment_grayscale(filename, h_depth=5.0)
    else:
        labels, img = segmentation.segment_grayscale(filename, h_depth=5.0)

    print("Extracting topology...")
    tissue = topology.extract_topology(labels, min_edge_len=3.0)
    
    if tissue is None:
        print("Topology extraction failed.")
        return

    # Solve for Tensions and Pressures (Required for Stress Calc)
    result = solvers.solve_bayesian(tissue, mu=1.0)
    
    if result is None:
        print("Solver failed.")
        return

    # 3. Calculate Cell-Level Stress Tensors
    # Uses Batchelor's formula: sigma = -P*I + (1/A) * Sum(T * L * u * u)
    print("Calculating Batchelor stress...")
    result = geometry.calculate_batchelor_stress(tissue, result)

    # 4. (Optional) Interpolate to Regular Grid for coarse-grained stress.
    # Kept available, but for display we plot at cell centroids for alignment.
    print("Interpolating to grid (Coarse Graining)...")
    grid_coords, grid_tensors = geometry.interpolate_stress_to_grid(
        tissue, result, grid_size=60
    )

    # 5. Visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

    # View 1: Standard Tensions (for reference)
    ax1.imshow(img, cmap='gray', alpha=0.5)
    visualization.plot_tensions(ax1, tissue, result, cmap='turbo', width=1.5)
    ax1.set_title("Cell-Level Tensions")
    
    # View 2: Cell-centered Stress Field (aligned to cells)
    # Plotting "Stress Crosses":
    # Red Arm  = Principal Tension (Pulling)
    # Blue Arm = Principal Compression (Pushing)
    ax2.imshow(img, cmap='gray', alpha=0.4)
    
    # Plot at cell centroids to avoid the visual "misalignment" of coarse grid points.
    visualization.plot_cell_stress_crosses(
        ax2, tissue, result, scale=60.0, min_mag=0.01
    )
    # Keep the overlay aligned to image pixel coordinates after line plotting.
    H, W = img.shape[:2]
    ax2.set_xlim(-0.5, W - 0.5)
    ax2.set_ylim(H - 0.5, -0.5)
    
    ax2.set_title("Cell-Centered Stress Field")
    ax2.axis('off')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stress Analysis Demo")
    parser.add_argument("--filename", type=str, default="./data/example.tif", help="Path to TIF image")
    parser.add_argument("--method", type=str, default="cellpose", choices=["cellpose", "grayscale"], 
                        help="Segmentation method (default: cellpose)")
    args = parser.parse_args()
    
    run_stress_demo(args.filename, args.method)
