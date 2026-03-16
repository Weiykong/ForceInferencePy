import matplotlib.pyplot as plt
import os
import logging

# Import your package
from force_inference import segmentation, topology, visualization

# Configure logging to see extraction details
logging.basicConfig(level=logging.INFO)

def main():
    # 1. Setup
    filename = '../data/test.tif'
    
    if not os.path.exists(filename):
        print("Image not found.")
        return

    # 2. Segmentation
    print("Segmenting...")
    # Adjust h_depth/min_cell_size as per your image
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
    main()