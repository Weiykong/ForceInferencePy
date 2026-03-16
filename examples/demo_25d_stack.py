import matplotlib.pyplot as plt
import numpy as np
from force_inference import segmentation, topology, geometry, solvers, visualization

def create_mock_z_stack(shape):
    """Generates a synthetic 'dome' shape for demonstration."""
    H, W = shape
    Y, X = np.indices((H, W))
    # A Gaussian dome in the center, max height 50 pixels
    z_map = 50 * np.exp(-((X - W/2)**2 + (Y - H/2)**2) / (W/3)**2)
    
    # Convert 2D map to 3D stack (Z, Y, X)
    z_dim = 60
    stack = np.zeros((z_dim, H, W), dtype=np.uint8)
    for i in range(H):
        for j in range(W):
            z_idx = int(z_map[i, j])
            if 0 <= z_idx < z_dim:
                stack[z_idx, i, j] = 255 # Bright membrane pixel
    return stack

def run_25d_demo(filename):
    # 1. Segment (Standard 2D projection)
    
    labels, img = segmentation.segment_grayscale(filename)
    
    # 2. Topology
    tissue = topology.extract_topology(labels)
    
    # 3. Map Z-Depth (The 2.5D Step)
    print("Mapping Z-depth...")
    # In reality, load your stack: stack = io.imread('stack.tif')
    mock_stack = create_mock_z_stack(labels.shape) 
    
    # This updates tissue.V to have real Z coordinates
    tissue = geometry.map_z_to_vertices(tissue, mock_stack)
    
    # 4. Solve (Universal solver handles 3D V automatically)
    print("Solving forces (2.5D)...")
    result = solvers.solve_bayesian(tissue, mu=1.0)
    
    # 5. Visualize
    fig = plt.figure(figsize=(15, 5))
    
    # Top View (XY)
    ax1 = fig.add_subplot(131)
    ax1.imshow(img, cmap='gray', alpha=0.5)
    visualization.plot_tensions(ax1, tissue, result)
    ax1.set_title("2.5D Tensions (XY Projection)")
    
    # Side View (XZ) - To prove it's 3D
    ax2 = fig.add_subplot(132)
    V = tissue.V
    ax2.scatter(V[:, 0], V[:, 2], c=V[:, 2], cmap='magma', s=10)
    ax2.set_xlabel("X (Width)")
    ax2.set_ylabel("Z (Height)")
    ax2.set_title("Side View (XZ Profile)")
    ax2.invert_yaxis() # Microscope Z often points down
    
    # Pressures
    ax3 = fig.add_subplot(133)
    ax3.imshow(img, cmap='gray', alpha=0.5)
    visualization.plot_pressures(ax3, tissue, result)
    ax3.set_title("Pressures")
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    filename = './data/example.tif'
    run_25d_demo(filename)