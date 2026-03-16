#!/usr/bin/env python3
"""
Diagnostic test: Label-driven topology vs Skeleton-based topology.

Creates a synthetic label image with a known twin-junction configuration
and verifies that the label-driven approach preserves both junctions
while the skeleton-based approach merges them.

Also runs both methods on the real test data (test.tif) for comparison.
"""

import sys
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from force_inference.segmentation import segment_grayscale
from force_inference.topology import extract_topology as extract_skeleton
from force_inference.topology_label import extract_topology_label


# =============================================================================
# Test 1: Synthetic twin-junction
# =============================================================================

def make_twin_junction_image(size=60):
    """
    Create a label image with 4 cells meeting at a central junction,
    plus two additional cells to create a twin-junction configuration.

    Layout (schematic):
        +-----+-----+
        |  1  |  2  |
        +-----+-----+
        |  3  |  4  |
        +-----+-----+

    Each cell occupies approximately (size/2-1)×(size/2-1) pixels.
    The background (0) forms 1-pixel-wide membranes between cells.
    At the center, cells 1,2,3,4 all meet at a single junction point.

    For a twin-junction, we create a thin extension of cell 5 that
    splits the 1-2 interface and the 3-4 interface, creating two
    closely-spaced junctions.
    """
    img = np.zeros((size, size), dtype=np.int32)
    mid = size // 2

    # Four main cells (2×2 grid of tightly-packed blocks)
    # Each cell is sized to just touch its neighbors
    img[1:mid-1, 1:mid-1] = 1          # top-left
    img[1:mid-1, mid+1:size-1] = 2     # top-right
    img[mid+1:size-1, 1:mid-1] = 3     # bottom-left
    img[mid+1:size-1, mid+1:size-1] = 4  # bottom-right

    # Create a thin "wedge" cell (5) that intersects both the top and
    # bottom interfaces, creating two separate junctions where
    # (1,2,5) and (3,4,5) meet.
    # This wedge is only 1-2 pixels wide vertically but spans part
    # of both interfaces.
    wedge_col_start = mid - 1
    wedge_col_end = mid + 2
    img[1:mid-1, wedge_col_start:wedge_col_end] = 5  # splits 1-2
    img[mid+1:size-1, wedge_col_start:wedge_col_end] = 5  # splits 3-4

    # This creates:
    # - Junction A at approximately (mid-1, mid) where cells 1, 2, 5, (background) meet
    # - Junction B at approximately (mid+1, mid) where cells 3, 4, 5, (background) meet
    # The two junctions are 2 pixels apart, the classic twin-junction case.

    return img


def test_twin_junction():
    """Test that twin junctions are preserved."""
    print("=" * 60)
    print("TEST 1: Synthetic twin-junction image")
    print("=" * 60)

    labels = make_twin_junction_image(60)
    n_cells = len(np.unique(labels)) - 1  # exclude background
    print(f"Created label image with {n_cells} cells")

    # Expected: 2 junctions very close together at the center
    # The interface between cells 1&2 (top) and cells 3&4 (bottom)
    # should be connected by a short edge between the two junctions.

    # --- Skeleton-based ---
    print("\n--- Skeleton-based topology ---")
    tissue_skel = extract_skeleton(labels, vertex_cluster_r=3.0)
    if tissue_skel:
        print(f"  Vertices: {len(tissue_skel.V)}")
        print(f"  Edges:    {len(tissue_skel.E)}")
        # Check if there's an edge between cells 1&3 and cells 2&4
        # that passes through the center
        center_edges = []
        for i, (c1, c2) in enumerate(tissue_skel.E_cells):
            pair = tuple(sorted([c1, c2]))
            center_edges.append(pair)
        print(f"  Cell pairs: {sorted(set(center_edges))}")
        has_12 = (1, 2) in set(center_edges) or (2, 1) in set(center_edges)
        has_34 = (3, 4) in set(center_edges) or (4, 3) in set(center_edges)
        print(f"  Has edge between cells 1-2: {has_12}")
        print(f"  Has edge between cells 3-4: {has_34}")
    else:
        print("  FAILED: No topology extracted!")

    # --- Label-driven ---
    print("\n--- Label-driven topology ---")
    tissue_label = extract_topology_label(labels, vertex_cluster_r=2.0)
    if tissue_label:
        print(f"  Vertices: {len(tissue_label.V)}")
        print(f"  Edges:    {len(tissue_label.E)}")
        center_edges = []
        for i, (c1, c2) in enumerate(tissue_label.E_cells):
            pair = tuple(sorted([c1, c2]))
            center_edges.append(pair)
        print(f"  Cell pairs: {sorted(set(center_edges))}")
        has_12 = (1, 2) in set(center_edges)
        has_34 = (3, 4) in set(center_edges)
        print(f"  Has edge between cells 1-2: {has_12}")
        print(f"  Has edge between cells 3-4: {has_34}")

        # Verify twin junctions: there should be at least 2 vertices
        # near the center (mid_x, mid_y)
        mid = 30.0
        center_verts = []
        for i, v in enumerate(tissue_label.V):
            if abs(v[0] - mid) < 5 and abs(v[1] - mid) < 5:
                center_verts.append(i)
        print(f"  Vertices near center: {len(center_verts)}")
        if len(center_verts) >= 2:
            print("  ✓ Twin junctions PRESERVED!")
        else:
            print("  ✗ Twin junctions MERGED (only 1 vertex near center)")
    else:
        print("  FAILED: No topology extracted!")

    # Visualize
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))

    # Row 1: Label image and classification
    axes[0, 0].imshow(labels, cmap='tab20', interpolation='nearest')
    axes[0, 0].set_title('Label Image')

    # Show where vertex pixels are — must use full Voronoi first
    from force_inference.topology_label import _classify_boundary_pixels, _full_voronoi_labels
    labels_v = _full_voronoi_labels(labels)
    vertex_mask, edge_mask, _ = _classify_boundary_pixels(labels_v, half_window=1)
    axes[0, 1].imshow(vertex_mask.astype(int), cmap='RdYlGn', interpolation='nearest')
    axes[0, 1].set_title(f'Vertex Pixels (full Voronoi): {int(np.sum(vertex_mask))}')

    # Row 2: Topology overlays
    for ax_idx, (tissue, name) in enumerate([
        (tissue_skel, 'Skeleton-based'),
        (tissue_label, 'Label-driven'),
    ]):
        ax = axes[1, ax_idx]
        ax.imshow(labels, cmap='tab20', alpha=0.3, interpolation='nearest')
        if tissue:
            V2 = tissue.V[:, :2]
            for v1i, v2i in tissue.E:
                ax.plot([V2[v1i, 0], V2[v2i, 0]],
                        [V2[v1i, 1], V2[v2i, 1]], 'b-', lw=1.5, alpha=0.6)
            ax.scatter(V2[:, 0], V2[:, 1], c='red', s=40, zorder=5)
            ax.set_title(f'{name}\n{len(V2)} verts, {len(tissue.E)} edges')
        else:
            ax.set_title(f'{name}\nFAILED')

    plt.tight_layout()
    plt.savefig('test_twin_junction.png', dpi=150)
    print("\nSaved: test_twin_junction.png")


# =============================================================================
# Test 2: Real data
# =============================================================================

def test_real_data():
    """Compare both methods on real test data."""
    print("\n" + "=" * 60)
    print("TEST 2: Real test data (test.tif)")
    print("=" * 60)

    data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              'data', 'test.tif')
    if not os.path.exists(data_path):
        print(f"  Skipping: {data_path} not found")
        return

    labels, img = segment_grayscale(data_path)
    n_cells = len(np.unique(labels)) - 1
    print(f"Segmented: {n_cells} cells, image shape {labels.shape}")

    # Skeleton-based
    print("\n--- Skeleton-based ---")
    tissue_skel = extract_skeleton(labels, cell_labels=labels,
                                    vertex_cluster_r=3.0)
    if tissue_skel:
        print(f"  Vertices: {len(tissue_skel.V)}")
        print(f"  Edges:    {len(tissue_skel.E)}")
        unique_pairs = set()
        for c1, c2 in tissue_skel.E_cells:
            unique_pairs.add(tuple(sorted([c1, c2])))
        print(f"  Unique cell pairs: {len(unique_pairs)}")
    else:
        print("  FAILED")

    # Label-driven
    print("\n--- Label-driven ---")
    tissue_label = extract_topology_label(labels, vertex_cluster_r=2.0)
    if tissue_label:
        print(f"  Vertices: {len(tissue_label.V)}")
        print(f"  Edges:    {len(tissue_label.E)}")
        unique_pairs = set()
        for c1, c2 in tissue_label.E_cells:
            unique_pairs.add(tuple(sorted([c1, c2])))
        print(f"  Unique cell pairs: {len(unique_pairs)}")
    else:
        print("  FAILED")

    # Visualize side by side
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    axes[0].imshow(labels, cmap='tab20', interpolation='nearest')
    axes[0].set_title(f'Labels ({n_cells} cells)')

    for ax_idx, (tissue, name) in enumerate([
        (tissue_skel, 'Skeleton-based'),
        (tissue_label, 'Label-driven'),
    ], start=1):
        ax = axes[ax_idx]
        ax.imshow(labels, cmap='tab20', alpha=0.3, interpolation='nearest')
        if tissue:
            V2 = tissue.V[:, :2]
            for v1i, v2i in tissue.E:
                ax.plot([V2[v1i, 0], V2[v2i, 0]],
                        [V2[v1i, 1], V2[v2i, 1]], 'b-', lw=0.8)
            ax.scatter(V2[:, 0], V2[:, 1], c='red', s=15, zorder=5)
            ax.set_title(f'{name}\n{len(V2)} verts, {len(tissue.E)} edges')
        else:
            ax.set_title(f'{name}\nFAILED')

    plt.tight_layout()
    plt.savefig('test_real_comparison.png', dpi=150)
    print("\nSaved: test_real_comparison.png")


# =============================================================================
# Test 3: Micro twin-junction (the hardest case)
# =============================================================================

def test_micro_twin():
    """
    Test a clean 4-cell junction where cells just barely touch.

    This creates the geometry:

        1 1 0 2 2
        1 1 0 2 2
        0 0 0 0 0
        3 3 0 4 4
        3 3 0 4 4

    At the center pixel (2, 2), all four cells touch via their corners.
    This creates a single 4-way junction, plus 4 edges (1-2, 1-3, 2-4, 3-4).

    The label-driven approach should detect this junction correctly because
    the 3×3 window around (2,2) contains cells 1,2,3,4, triggering the
    ≥3 nonzero labels rule for a vertex pixel.
    """
    print("\n" + "=" * 60)
    print("TEST 3: Clean 4-cell junction")
    print("=" * 60)

    # Create a 5×5 label image with 4 cells
    img = np.zeros((5, 5), dtype=np.int32)
    img[0:2, 0:2] = 1          # top-left
    img[0:2, 3:5] = 2          # top-right
    img[3:5, 0:2] = 3          # bottom-left
    img[3:5, 3:5] = 4          # bottom-right
    # Rows/cols 2 are the membrane (background=0)

    print(f"Label image shape: {img.shape}")
    print(f"Unique labels: {np.unique(img)}")
    print(f"Image:\n{img}")

    tissue = extract_topology_label(img, vertex_cluster_r=1.0, min_edge_len=1)
    if tissue:
        print(f"  Vertices: {len(tissue.V)}")
        for i, v in enumerate(tissue.V):
            print(f"    v{i}: ({v[0]:.1f}, {v[1]:.1f})")
        print(f"  Edges: {len(tissue.E)}")
        for i, (v1, v2) in enumerate(tissue.E):
            c1, c2 = tissue.E_cells[i]
            print(f"    e{i}: v{v1}--v{v2}  cells={c1},{c2}")

        # We expect at least 4 edges (one per cell-pair interface)
        pairs = set()
        for c1, c2 in tissue.E_cells:
            pairs.add(tuple(sorted([c1, c2])))
        print(f"  Unique cell pairs: {sorted(pairs)}")
        expected = {(1, 2), (1, 3), (2, 4), (3, 4)}
        missing = expected - pairs
        if missing:
            print(f"  ✗ MISSING cell pairs: {missing}")
        else:
            print("  ✓ All 4 cell-pair interfaces found!")
    else:
        print("  FAILED: No topology extracted")


if __name__ == '__main__':
    import logging
    logging.basicConfig(level=logging.INFO,
                        format='%(name)s: %(message)s')

    test_twin_junction()
    test_micro_twin()
    test_real_data()
