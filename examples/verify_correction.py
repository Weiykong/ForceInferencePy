import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
from force_inference import segmentation, topology, geometry, solvers

def run_correction_test(filename, method="cellpose"):
    
    print(f"--- Loading {filename} using {method} ---")
    if method == "cellpose":
        try:
            labels, _ = segmentation.segment_cellpose(filename, model_type="cyto3")
        except ImportError:
            print("Cellpose not installed. Falling back to grayscale watershed.")
            labels, _ = segmentation.segment_grayscale(filename, h_depth=2.0, min_cell_size=50)
    else:
        labels, _ = segmentation.segment_grayscale(filename, h_depth=2.0, min_cell_size=50)
    
    # 1. Setup Topology
    tissue = topology.extract_topology(labels, min_edge_len=0.0, clean=False, trace_pixels=True)
    if tissue is None:
        print("Topology extraction failed.")
        return
    tissue = geometry.compute_curvature(tissue)
    
    # 2. Run BAD Solver (Low Reg, No Detrend) -> Expect Ramp
    print("Running Standard Solver (Expect Ramp)...")
    res_bad = solvers.solve_laplace(tissue, regularization=0.1)
    
    # 3. Run GOOD Solver (High Reg, Detrend) -> Expect Hills
    print("Running Corrected Solver (Expect Hills)...")
    res_good = solvers.solve_laplace(tissue, regularization=5.0)
    
    # 4. Visualization
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    
    # Plot A: The Ramp
    plot_map(ax[0], tissue, res_bad.pressures, "Low Regularization")
    
    # Plot B: The Correction
    plot_map(ax[1], tissue, res_good.pressures, "High Regularization")
    
    plt.tight_layout()
    plt.show()

def plot_map(ax, tissue, values, title):
    ax.imshow(tissue.labels, cmap='gray', alpha=0.3)
    cents = tissue.C_centroids
    
    # Normalize for view
    val_disp = values[values!=0]
    if len(val_disp)==0:
        return
    v_min, v_max = np.percentile(val_disp, [2, 98])
    
    sc = ax.scatter(cents[:,0], cents[:,1], c=values[:len(cents)], 
                    cmap='coolwarm', s=30, vmin=v_min, vmax=v_max, edgecolors='k')
    ax.set_title(title)
    plt.colorbar(sc, ax=ax)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("filename", nargs="?", default='./data/example.tif')
    parser.add_argument("--method", choices=["cellpose", "grayscale"], default="cellpose")
    args = parser.parse_args()
    
    if os.path.exists(args.filename):
        run_correction_test(args.filename, method=args.method)
    else:
        print(f"File {args.filename} not found.")
