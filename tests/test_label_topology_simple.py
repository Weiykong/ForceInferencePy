#!/usr/bin/env python3
"""
Simplified diagnostic test for label-driven topology.
Does not require segmentation, just tests the topology extraction.
"""

import sys
import os
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from force_inference.topology_label import extract_topology_label
from force_inference.topology import extract_topology as extract_skeleton


def test_4cell_junction():
    """
    Test a clean 4-cell junction where cells just barely touch.

    This creates the geometry:

        1 1 0 2 2
        1 1 0 2 2
        0 0 0 0 0
        3 3 0 4 4
        3 3 0 4 4

    At the center pixel (2, 2), all four cells touch via their corners.
    """
    print("\n" + "=" * 70)
    print("TEST 1: Clean 4-cell junction")
    print("=" * 70)

    # Create a 5×5 label image with 4 cells
    img = np.zeros((5, 5), dtype=np.int32)
    img[0:2, 0:2] = 1          # top-left
    img[0:2, 3:5] = 2          # top-right
    img[3:5, 0:2] = 3          # bottom-left
    img[3:5, 3:5] = 4          # bottom-right
    # Rows/cols 2 are the membrane (background=0)

    print(f"Label image:\n{img}\n")

    # Test skeleton-based
    print("--- Skeleton-based ---")
    tissue_skel = extract_skeleton(img, vertex_cluster_r=1.0)
    if tissue_skel:
        pairs_skel = set(tuple(sorted([c1, c2])) for c1, c2 in tissue_skel.E_cells)
        print(f"  Vertices: {len(tissue_skel.V)}")
        print(f"  Edges: {len(tissue_skel.E)}")
        print(f"  Cell pairs: {sorted(pairs_skel)}")
    else:
        print("  FAILED")

    # Test label-driven
    print("\n--- Label-driven ---")
    tissue_label = extract_topology_label(
        img, 
        vertex_cluster_r=1.0, 
        min_edge_len=1,
        collapse_stubs=False,
        collapse_tiny_twins=False
    )
    if tissue_label:
        pairs_label = set(tuple(sorted([c1, c2])) for c1, c2 in tissue_label.E_cells)
        print(f"  Vertices: {len(tissue_label.V)}")
        print(f"  Edges: {len(tissue_label.E)}")
        print(f"  Cell pairs: {sorted(pairs_label)}")

        expected = {(1, 2), (1, 3), (2, 4), (3, 4)}
        missing = expected - pairs_label
        extra = pairs_label - expected

        if not missing and not extra:
            print(f"  ✓ PASS: All expected cell pairs found!")
        else:
            if missing:
                print(f"  ✗ MISSING: {missing}")
            if extra:
                print(f"  ! EXTRA: {extra}")
    else:
        print("  FAILED")


def test_6cell_grid():
    """
    Test a 3×2 cell grid with complex junction topology.

    Geometry (2×2 blocks per cell, 1 pixel membranes):
        +---+---+---+
        | 1 | 2 | 3 |
        +---+---+---+
        | 4 | 5 | 6 |
        +---+---+---+
    """
    print("\n" + "=" * 70)
    print("TEST 2: 6-cell grid (3×2)")
    print("=" * 70)

    img = np.zeros((7, 10), dtype=np.int32)
    # Top row (cells 1, 2, 3)
    img[0:2, 0:2] = 1
    img[0:2, 3:5] = 2
    img[0:2, 6:8] = 3
    # Row separator at 2 (membrane)
    # Bottom row (cells 4, 5, 6)
    img[4:6, 0:2] = 4
    img[4:6, 3:5] = 5
    img[4:6, 6:8] = 6

    print(f"Label image shape: {img.shape}")
    print(f"Unique labels: {sorted(np.unique(img))}")

    # Test label-driven
    print("\n--- Label-driven ---")
    tissue = extract_topology_label(
        img, 
        vertex_cluster_r=1.0, 
        min_edge_len=1,
        collapse_stubs=False,
        collapse_tiny_twins=False
    )
    if tissue:
        pairs = set(tuple(sorted([c1, c2])) for c1, c2 in tissue.E_cells if c1 > 0 and c2 > 0)
        print(f"  Vertices: {len(tissue.V)}")
        print(f"  Edges: {len(tissue.E)}")
        print(f"  Cell pairs: {sorted(pairs)}")

        # Expected adjacencies
        # Row 1: 1-2, 2-3
        # Row 2: 4-5, 5-6
        # Between rows: 1-4, 2-5, 3-6
        expected = {(1, 2), (2, 3), (4, 5), (5, 6), (1, 4), (2, 5), (3, 6)}
        missing = expected - pairs
        extra = pairs - expected

        print(f"  Expected: {len(expected)}, Got: {len(pairs)}")
        if not missing:
            print(f"  ✓ No missing pairs")
        else:
            print(f"  ✗ MISSING: {missing}")
        if not extra:
            print(f"  ✓ No extra pairs")
        else:
            print(f"  ! EXTRA: {extra}")
    else:
        print("  FAILED")


def test_twin_junction_simple():
    """
    Create a 5-cell geometry that isolates the twin-junction problem.

    Two closely-spaced junctions where cell 5 (wedge) splits the interface
    between cells 1-2 (top) and cells 3-4 (bottom).

    Geometry:
        1 1 5 2 2
        1 1 5 2 2
        0 0 0 0 0
        3 3 5 4 4
        3 3 5 4 4

    This creates:
    - Junction A at (2, 1): cells 1, 2, 5 meet
    - Junction B at (2, 3): cells 3, 4, 5 meet
    - Short edge connecting A-B along cell 5's interface
    """
    print("\n" + "=" * 70)
    print("TEST 3: Twin-junction (wedge cell splits interfaces)")
    print("=" * 70)

    img = np.zeros((5, 5), dtype=np.int32)
    # Top-left: cell 1
    img[0:2, 0:2] = 1
    # Top-right: cell 2
    img[0:2, 3:5] = 2
    # Wedge: cell 5 (vertical spine at x=2)
    img[0:2, 2] = 5
    # Bottom-left: cell 3
    img[3:5, 0:2] = 3
    # Bottom-right: cell 4
    img[3:5, 3:5] = 4
    # Wedge continued
    img[3:5, 2] = 5

    print(f"Label image:\n{img}\n")

    # Test label-driven
    print("--- Label-driven ---")
    tissue = extract_topology_label(img, vertex_cluster_r=1.0, min_edge_len=1)
    if tissue:
        pairs = set(tuple(sorted([c1, c2])) for c1, c2 in tissue.E_cells if c1 > 0 and c2 > 0)
        print(f"  Vertices: {len(tissue.V)}")
        for i, v in enumerate(tissue.V):
            print(f"    v{i}: ({v[0]:.1f}, {v[1]:.1f})")
        print(f"  Edges: {len(tissue.E)}")
        for i, (v1, v2) in enumerate(tissue.E):
            c1, c2 = tissue.E_cells[i]
            if c1 > 0 and c2 > 0:
                print(f"    e{i}: v{v1}--v{v2}  cells=({c1},{c2})")
        print(f"  Cell pairs: {sorted(pairs)}")

        # We expect edges for: 1-2, 1-5, 2-5, 3-4, 3-5, 4-5
        expected = {(1, 2), (1, 5), (2, 5), (3, 4), (3, 5), (4, 5)}
        missing = expected - pairs

        if not missing:
            print(f"  ✓ PASS: All 6 cell-pair edges found!")
        else:
            print(f"  ✗ MISSING: {missing}")
    else:
        print("  FAILED")


if __name__ == '__main__':
    import logging
    logging.basicConfig(level=logging.INFO,
                        format='%(name)s: %(message)s')

    test_4cell_junction()
    test_6cell_grid()
    test_twin_junction_simple()

    print("\n" + "=" * 70)
    print("All tests completed")
    print("=" * 70)
