import numpy as np
import matplotlib.pyplot as plt
import logging
import os

from force_inference import segmentation, topology, geometry, solvers

logging.basicConfig(level=logging.INFO)

def run_comparison_demo(img_path='data/test_image.tif'):
    print(f"--- Loading {img_path} ---")
    labels, _ = segmentation.segment_grayscale(img_path, h_depth=2.0, min_cell_size=50)
    
    # ---------------------------------------------------------
    # A. Run BAYESIAN Inference
    # ---------------------------------------------------------
    print("\n--- Running Bayesian Inference ---")
    tissue_bayes = topology.extract_topology(labels, min_edge_len=4.0, clean=True, trace_pixels=False)
    if tissue_bayes is None:
        return
    res_bayes = solvers.solve_bayesian(tissue_bayes, mu=1.0)
    
    # ---------------------------------------------------------
    # B. Run LAPLACE Inference
    # ---------------------------------------------------------
    print("\n--- Running Laplace Inference ---")
    tissue_lap = topology.extract_topology(labels, min_edge_len=0.0, clean=False, trace_pixels=True)
    tissue_lap = geometry.compute_curvature(tissue_lap)
    res_lap = solvers.solve_laplace(tissue_lap, tension_val=1.0)

    if res_bayes is None or res_lap is None:
        print("Solver failed.")
        return

    # ---------------------------------------------------------
    # C. Data Alignment
    # ---------------------------------------------------------
    # Get Cell Tensions (Average of perimeter edges)
    T_cell_bayes = get_cell_mean_tensions(tissue_bayes, res_bayes.tensions)
    T_cell_lap = get_cell_mean_tensions(tissue_lap, res_lap.tensions)
    
    # Check if Laplace is constant (Standard Deviation ~ 0)
    is_constant = np.std(T_cell_lap) < 1e-6
    
    if is_constant:
        print("\n[INFO] Laplace Tension is constant (1.0). Correlation with Bayes is mathematically 0.")
        corr_t = 0.0
    else:
        # Only calc correlation if both vary
        corr_t = np.corrcoef(normalize(T_cell_bayes), normalize(T_cell_lap))[0, 1]

    # Normalize Pressures for fair comparison
    P_bayes_z = normalize(res_bayes.pressures)
    P_lap_z = normalize(res_lap.pressures)
    corr_p = np.corrcoef(P_bayes_z, P_lap_z)[0, 1]
    
    print(f"Pressure Correlation: {corr_p:.3f}")

    # ---------------------------------------------------------
    # D. Visualization
    # ---------------------------------------------------------
    fig = plt.figure(figsize=(14, 10))
    gs = fig.add_gridspec(2, 4)

    # --- Row 1: PRESSURES (Z-Scores) ---
    ax_p1 = fig.add_subplot(gs[0, 0])
    plot_map(ax_p1, tissue_bayes, P_bayes_z, "Bayes Pressure (Z)")
    
    ax_p2 = fig.add_subplot(gs[0, 1])
    plot_map(ax_p2, tissue_lap, P_lap_z, "Laplace Pressure (Z)")
    
    ax_p3 = fig.add_subplot(gs[0, 2])
    plot_map(ax_p3, tissue_bayes, P_bayes_z - P_lap_z, "Diff (Bayes - Lap)", cmap='coolwarm')
    
    ax_scat_p = fig.add_subplot(gs[0, 3])
    scatter_plot(ax_scat_p, P_bayes_z, P_lap_z, f"Pressure Corr={corr_p:.2f}")

    # --- Row 2: TENSIONS (RAW VALUES) ---
    # We plot RAW values so Laplace doesn't disappear
    ax_t1 = fig.add_subplot(gs[1, 0])
    plot_map(ax_t1, tissue_bayes, T_cell_bayes, "Bayes Tension (Raw)")
    
    ax_t2 = fig.add_subplot(gs[1, 1])
    # This will now show a flat color map (all 1.0)
    plot_map(ax_t2, tissue_lap, T_cell_lap, "Laplace Tension (Raw=1.0)")
    
    ax_t3 = fig.add_subplot(gs[1, 2])
    # For difference, we compare normalized patterns
    T_diff = normalize(T_cell_bayes) - normalize(T_cell_lap)
    plot_map(ax_t3, tissue_bayes, T_diff, "Pattern Diff (Norm)", cmap='coolwarm')
    
    ax_scat_t = fig.add_subplot(gs[1, 3])
    scatter_plot(ax_scat_t, T_cell_bayes, T_cell_lap, f"Tension Corr={corr_t:.2f}")

    plt.tight_layout()
    plt.show()

# --- Helpers ---

def normalize(data):
    """Returns Z-score. Handles constant data by returning zeros."""
    valid = data[data != 0]
    if len(valid) == 0:
        return data
    
    std = np.std(valid)
    if std < 1e-6:
        return np.zeros_like(data)
    
    return (data - np.mean(valid)) / std

def get_cell_mean_tensions(tissue, edge_tensions):
    """Averages edge tensions for each cell."""
    n_cells = tissue.labels.max()
    cell_tensions = np.zeros(n_cells)
    cell_counts = np.zeros(n_cells)

    n_edges = min(len(tissue.E_cells), len(edge_tensions))
    if len(tissue.E_cells) != len(edge_tensions):
        print(f"[WARN] Edge mismatch: E_cells={len(tissue.E_cells)} vs tensions={len(edge_tensions)}. Truncating.")

    for i in range(n_edges):
        c1, c2 = tissue.E_cells[i]
        val = edge_tensions[i]
        if not np.isfinite(val):
            continue
        if c1 > 0: 
            cell_tensions[c1-1] += val
            cell_counts[c1-1] += 1
        if c2 > 0: 
            cell_tensions[c2-1] += val
            cell_counts[c2-1] += 1
            
    mask = cell_counts > 0
    cell_tensions[mask] /= cell_counts[mask]
    return cell_tensions

def plot_map(ax, tissue, values, title, cmap='viridis'):
    """Plots cell values. Handles constant values gracefully."""
    ax.imshow(tissue.labels, cmap='gray', alpha=0.3)
    centroids = tissue.C_centroids
    if len(values) > len(centroids):
        values = values[:len(centroids)]

    # Robust scaling
    if np.ptp(values) < 1e-6: 
        # Constant data (e.g. Laplace Tension)
        v_min, v_max = values.min() - 0.1, values.max() + 0.1
    else:
        v_min, v_max = np.percentile(values, [2, 98])
        
    norm = plt.Normalize(v_min, v_max)
    
    # Plot even if value is 0 (unless it's background)
    valid_idx = np.where(values != -999)[0] # Just take all
    
    sc = ax.scatter(centroids[valid_idx, 0], centroids[valid_idx, 1], c=values[valid_idx], 
                    cmap=cmap, s=20, edgecolors='k', linewidth=0.5, norm=norm)
    
    # Draw faint edges
    lc = np.array([tissue.V[e] for e in tissue.E])
    from matplotlib.collections import LineCollection
    lines = LineCollection(lc[:,:,:2], colors='white', alpha=0.15, linewidths=0.5)
    ax.add_collection(lines)
    
    ax.set_title(title, fontsize=10)
    ax.axis('off')
    plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)

def scatter_plot(ax, x, y, title):
    if np.std(x) < 1e-6 or np.std(y) < 1e-6:
        ax.text(0.5, 0.5, "Constant Data\n(No Correlation)", 
                ha='center', va='center', transform=ax.transAxes)
    else:
        ax.scatter(x, y, alpha=0.5, s=10, c='k')
        
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("Bayesian")
    ax.set_ylabel("Laplace")
    ax.grid(True, alpha=0.3)


if __name__ == "__main__":
    if not os.path.exists('data/example.tif'):
        print("Please provide 'data/example.tif'")
    else:
        run_comparison_demo('data/example.tif')
