import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import os
import logging
import argparse

from force_inference import segmentation, topology, geometry, solvers

logging.basicConfig(level=logging.INFO)

def run_final_comparison(img_path='data/example.tif', method="cellpose"):
    print(f"--- Loading {img_path} using {method} ---")
    
    # 1. Segmentation (Shared)
    if method == "cellpose":
        try:
            labels, img = segmentation.segment_cellpose(img_path, model_type="cyto3")
        except ImportError:
            print("Cellpose not installed. Falling back to grayscale watershed.")
            labels, img = segmentation.segment_grayscale(img_path, h_depth=2.0, min_cell_size=50)
    else:
        labels, img = segmentation.segment_grayscale(img_path, h_depth=2.0, min_cell_size=50)

    # ==========================================
    # A. BAYESIAN Inference (Clean Graph)
    # ==========================================
    print("\n[Bayesian] Extracting Topology (Clean=True for 4-way junctions)...")
    # CRITICAL: clean=True collapses short edges to form 4-way vertices
    tissue_bayes = topology.extract_topology(labels, min_edge_len=4.0, clean=True, trace_pixels=False)
    
    if tissue_bayes is None:
        print("Bayesian topology extraction failed.")
        return

    print("[Bayesian] Solving Forces...")
    res_bayes = solvers.solve_bayesian(tissue_bayes, mu=1.0)
    
    # ==========================================
    # B. LAPLACE Inference (Raw Graph)
    # ==========================================
    print("\n[Laplace] Extracting Topology (Clean=False for curvature)...")
    # CRITICAL: clean=False keeps raw pixels for accurate curvature
    tissue_lap = topology.extract_topology(labels, min_edge_len=0.0, clean=False, trace_pixels=True)
    
    if tissue_lap is None:
        print("Laplace topology extraction failed.")
        return

    print("[Laplace] Computing Curvature & Solving...")
    tissue_lap = geometry.compute_curvature(tissue_lap)
    res_lap = solvers.solve_laplace(tissue_lap, regularization=5.0)

    if res_bayes is None or res_lap is None:
        print("One of the solvers failed.")
        return

    # ==========================================
    # C. ALIGNMENT (The "Apples to Apples" Step)
    # ==========================================
    
    # 1. Compare Cell Pressures (Direct Array Match)
    P_bayes_z = normalize_z(res_bayes.pressures)
    P_lap_z   = normalize_z(res_lap.pressures)
    
    valid_p = (P_bayes_z != 0) & (P_lap_z != 0)
    if np.any(valid_p):
        corr_p = np.corrcoef(P_bayes_z[valid_p], P_lap_z[valid_p])[0, 1]
    else:
        corr_p = 0.0
    
    # 2. Compare Edge Tensions (Interface Match)
    T_map_bayes = get_interface_tensions(tissue_bayes, res_bayes.tensions)
    T_map_lap   = get_interface_tensions(tissue_lap, res_lap.tensions)
    
    common_edges = sorted(list(set(T_map_bayes.keys()) & set(T_map_lap.keys())))
    
    if len(common_edges) > 0:
        T_bayes_aligned = np.array([T_map_bayes[k] for k in common_edges])
        T_lap_aligned   = np.array([T_map_lap[k] for k in common_edges])
        
        is_const_tension = np.std(T_lap_aligned) < 1e-6
        if is_const_tension:
            corr_t = 0.0
            t_title = "Tension (Bayes vs Constant)"
        else:
            corr_t = np.corrcoef(T_bayes_aligned, T_lap_aligned)[0, 1]
            t_title = f"Tension Corr={corr_t:.2f}"
    else:
        corr_t = 0.0
        t_title = "No common edges"

    print(f"\nStats over {len(common_edges)} shared edges:")
    print(f"  Pressure Correlation: {corr_p:.3f}")
    print(f"  Tension Correlation:  {corr_t:.3f}")

    # ==========================================
    # D. VISUALIZATION
    # ==========================================
    fig = plt.figure(figsize=(16, 8))
    gs = fig.add_gridspec(2, 4)

    # --- ROW 1: PRESSURES ---
    ax1 = fig.add_subplot(gs[0, 0])
    plot_cells(ax1, tissue_bayes, P_bayes_z, "Bayesian P (Z-Score)")
    
    ax2 = fig.add_subplot(gs[0, 1])
    plot_cells(ax2, tissue_lap, P_lap_z, "Laplace P (Z-Score)")
    
    ax3 = fig.add_subplot(gs[0, 2])
    if np.any(valid_p):
        ax3.scatter(P_bayes_z[valid_p], P_lap_z[valid_p], alpha=0.5, c='k', s=10)
        ax3.set_xlabel("Bayesian P")
        ax3.set_ylabel("Laplace P")
        ax3.set_title(f"Pressure Correlation: {corr_p:.2f}")
        ax3.grid(True, alpha=0.3)
    
    # --- ROW 2: TENSIONS ---
    ax4 = fig.add_subplot(gs[1, 0])
    plot_edges(ax4, tissue_bayes, res_bayes.tensions, "Bayesian Tension")
    
    ax5 = fig.add_subplot(gs[1, 1])
    plot_edges(ax5, tissue_lap, res_lap.tensions, "Laplace Tension")
    
    ax6 = fig.add_subplot(gs[1, 2])
    if len(common_edges) > 0 and not is_const_tension:
        ax6.scatter(T_bayes_aligned, T_lap_aligned, alpha=0.5, c='k', s=10)
        ax6.set_xlabel("Bayesian T")
        ax6.set_ylabel("Laplace T")
        ax6.set_title(t_title)
        ax6.grid(True, alpha=0.3)
    else:
        ax6.text(0.5, 0.5, t_title, ha='center', va='center')

    plt.tight_layout()
    plt.show()

# ==========================
# Helpers
# ==========================

def get_interface_tensions(tissue, tensions):
    mapping = {}
    for i, (c1, c2) in enumerate(tissue.E_cells):
        if c1 == 0 or c2 == 0:
            continue
        key = tuple(sorted((c1, c2)))
        mapping[key] = tensions[i]
    return mapping

def normalize_z(data):
    valid = data[data != 0]
    if len(valid) == 0:
        return data
    std = np.std(valid)
    if std < 1e-9:
        return np.zeros_like(data)
    return (data - np.mean(valid)) / std

def plot_cells(ax, tissue, values, title):
    ax.imshow(tissue.labels, cmap='gray', alpha=0.3)
    centroids = tissue.C_centroids
    if len(values) > len(centroids):
        values = values[:len(centroids)]
    valid_idx = np.where(values != 0)[0]
    if len(valid_idx) > 0:
        v_min, v_max = np.percentile(values[valid_idx], [2, 98])
    else:
        v_min, v_max = -1, 1
    sc = ax.scatter(centroids[valid_idx, 0], centroids[valid_idx, 1], 
                    c=values[valid_idx], cmap='coolwarm', s=30, 
                    edgecolors='k', linewidth=0.5, vmin=v_min, vmax=v_max)
    ax.set_title(title, fontsize=10)
    ax.axis('off')
    plt.colorbar(sc, ax=ax, fraction=0.046)

def plot_edges(ax, tissue, values, title):
    ax.imshow(tissue.labels, cmap='gray', alpha=0.3)
    if np.std(values) < 1e-6:
        v_min, v_max = values.min() - 0.1, values.max() + 0.1
    else:
        v_min, v_max = np.percentile(values, [5, 95])
    lines = []
    colors = []
    for i, (v1, v2) in enumerate(tissue.E):
        p1 = tissue.V[v1]
        p2 = tissue.V[v2]
        lines.append([p1[:2], p2[:2]])
        colors.append(values[i])
    lc = LineCollection(lines, array=np.array(colors), cmap='viridis', 
                        norm=plt.Normalize(v_min, v_max), linewidths=1.5)
    ax.add_collection(lc)
    ax.set_title(title, fontsize=10)
    ax.axis('off')
    plt.colorbar(lc, ax=ax, fraction=0.046)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("filename", nargs="?", default="data/example.tif")
    parser.add_argument("--method", choices=["cellpose", "grayscale"], default="cellpose")
    args = parser.parse_args()
    
    if os.path.exists(args.filename):
        run_final_comparison(args.filename, method=args.method)
    else:
        print(f"File {args.filename} not found.")
