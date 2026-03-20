import matplotlib.pyplot as plt
import os
import logging
import argparse

# Import your package
from force_inference import segmentation, topology, visualization

# Configure logging to see extraction details
logging.basicConfig(level=logging.INFO)

def run_check(filename, method="cellpose"):
    # 1. Setup
    if not os.path.exists(filename):
        print(f"Image not found: {filename}")
        return

    # 2. Segmentation
    print(f"Segmenting image using {method}...")
    if method == "cellpose":
        try:
            labels, _ = segmentation.segment_cellpose(filename, model_type="cyto3")
        except ImportError:
            print("Cellpose not found, falling back to grayscale.")
            labels, _ = segmentation.segment_grayscale(filename, h_depth=8.0, min_cell_size=10)
    else:
        labels, _ = segmentation.segment_grayscale(filename, h_depth=8.0, min_cell_size=10)

    # 3. Topology Extraction
    # Clean=True helps usually, but set to False if you want raw pixel logic
    print("Extracting Topology...")
    tissue = topology.extract_topology(labels, min_edge_len=2.0, clean=True)
    
    if tissue is None:
        print("Failed to extract tissue.")
        return

    print(f"Stats: {len(tissue.V)} vertices, {len(tissue.E)} edges, {len(tissue.C_v)} cells.")

    # 4. Visualization Check
    fig, ax = plt.subplots(figsize=(10, 10))
    
    visualization.plot_topology_check(ax, tissue)
    
    plt.title(f"Topology Check\nN_Cells={len(tissue.C_v)} | N_Edges={len(tissue.E)}")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Check Topology Demo")
    parser.add_argument("--filename", type=str, default="../data/test.tif", help="Path to TIF image")
    parser.add_argument("--method", type=str, default="cellpose", choices=["cellpose", "grayscale"], 
                        help="Segmentation method (default: cellpose)")
    args = parser.parse_args()
    
    run_check(args.filename, args.method)
