import matplotlib.pyplot as plt
import numpy as np
import os
import tifffile

from force_inference import segmentation, topology, solvers, visualization
from force_inference.topology_label import extract_topology_label

def run_demo():
    filename = '/Users/weiyuankong/ForceInferencePy/data/example.tif'
    if not os.path.exists(filename):
        print(f"File {filename} not found.")
        return

    # 1. Load raw TIF and binarize membrane
    raw = tifffile.imread(filename)
    if raw.ndim == 3:
        if raw.shape[0] <= 4:           # (C, H, W)
            raw = raw[0]
        elif raw.shape[2] <= 4:         # (H, W, C)
            raw = (0.299*raw[:,:,0] + 0.587*raw[:,:,1]
                   + 0.114*raw[:,:,2]).astype(raw.dtype)
        else:                           # (Z, H, W)
            raw = raw.max(axis=0)

    membrane_binary = (raw > 128).astype(np.uint8)
    print(f"Image: {raw.shape}, membrane px: {np.sum(membrane_binary)}")

    # 2. Segment (for visualization background + pressures)
    print("Segmenting image...")
    labels, img_smooth = segmentation.segment_grayscale(
        filename, h_depth=2.0, min_cell_size=5)

    # 3. Extract topology from label map
    print("Extracting topology...")
    tissue = extract_topology_label(
        labels,
        use_skeleton_geometry=False,
        collapse_stubs=True,
        collapse_tiny_twins=False,
    )
    if tissue is None:
        print("Topology extraction failed.")
        return
    if tissue.num_inner_vertices == 0:
        print("No inner vertices found.")
        return
 
    if tissue is None:
        print("Topology extraction failed.")
        return

    print(f"  V={len(tissue.V)} ({tissue.num_inner_vertices} inner + "
          f"{len(tissue.V) - tissue.num_inner_vertices} border), "
          f"E={len(tissue.E)}")

    if tissue.num_inner_vertices == 0:
        print("No inner vertices — cannot solve.")
        return

    # 4. Solve
    print("Scanning for optimal mu (Bayesian)...")
    scan_result = solvers.solve_bayesian(tissue)

    if scan_result is None:
        print("Solver failed.")
        return

    best_mu = scan_result.best_mu
    result  = scan_result.best_result
    print(f"Solver complete. Optimal mu = {best_mu:.4f}")

    # 5. Visualize
    fig = plt.figure(figsize=(18, 6))
    gs  = fig.add_gridspec(1, 3)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[0, 2])

    ax1.imshow(img_smooth, cmap='gray', alpha=0.5)
    visualization.plot_tensions(ax1, tissue, result, cmap='turbo')
    ax1.set_title(f"Inferred Tensions (N={len(tissue.E)})")

    ax2.imshow(img_smooth, cmap='gray', alpha=0.5)
    visualization.plot_pressures(ax2, tissue, result, cmap='bwr')
    ax2.set_title("Inferred Pressures")

    ax3.semilogx(scan_result.mu_values, scan_result.log_evidences, 'b-o', lw=1.5)
    ax3.axvline(best_mu, color='r', linestyle='--', label=f'Optimal: {best_mu:.2g}')
    ax3.set_xlabel("Regularization Parameter (mu)")
    ax3.set_ylabel("Log Evidence")
    ax3.set_title("Bayesian Parameter Scan")
    ax3.legend()
    ax3.grid(True, which="both", ls="-", alpha=0.4)

    plt.tight_layout()
    plt.show()

    print(f"Edges: {len(tissue.E)}")
    print(f"Tension range: {np.nanmin(result.tensions):.3f} "
          f"to {np.nanmax(result.tensions):.3f}")
    print(f"NaN tensions: {np.sum(np.isnan(result.tensions))}")


if __name__ == "__main__":
    run_demo()