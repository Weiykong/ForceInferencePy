"""Shared pytest fixtures and configuration for the ForceInferencePy test suite."""
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend — must be set before importing pyplot

import numpy as np
import pytest

from force_inference.core import Tissue, ForceResult


# ---------------------------------------------------------------------------
# Minimal synthetic tissue helpers
# ---------------------------------------------------------------------------

def _make_4cell_labels(size: int = 30) -> np.ndarray:
    """Return a (size, size) label image with 4 equal-quadrant cells."""
    labels = np.zeros((size, size), dtype=np.int32)
    mid = size // 2
    labels[1:mid, 1:mid] = 1
    labels[1:mid, mid + 1 : size - 1] = 2
    labels[mid + 1 : size - 1, 1:mid] = 3
    labels[mid + 1 : size - 1, mid + 1 : size - 1] = 4
    return labels


@pytest.fixture
def four_cell_labels() -> np.ndarray:
    """Label image with 4 cells meeting at a central junction."""
    return _make_4cell_labels()


@pytest.fixture
def minimal_tissue() -> Tissue:
    """
    A hand-crafted Tissue with 3 interior vertices, 3 edges, and 2 cells.

    Layout (pixel coords, origin top-left):
        V0 (5, 5)  --- E0 --- V1 (15, 5)
              \\                  /
               E2              E1
                \\              /
                 V2 (10, 15)

    Cell 1: V0, V1, V2  (triangle, label 1)
    Cell 2: shares edges with cell 1 (label 2)
    """
    size = 20
    labels = np.zeros((size, size), dtype=np.int32)
    # Rough triangular cells
    from skimage.draw import polygon
    r1, c1 = polygon([1, 1, 9], [1, 9, 5], shape=labels.shape)
    labels[r1, c1] = 1
    r2, c2 = polygon([1, 9, 9], [9, 5, 14], shape=labels.shape)
    labels[r2, c2] = 2

    V = np.array([[5.0, 5.0, 0.0], [15.0, 5.0, 0.0], [10.0, 15.0, 0.0]])
    E = np.array([[0, 1], [1, 2], [0, 2]])
    E_cells = np.array([[1, 2], [1, 2], [1, 2]])
    C_centroids = np.array([[8.0, 5.0], [12.0, 8.0]])
    C_v = [[0, 1, 2], [0, 2, 1]]

    return Tissue(
        V=V,
        E=E,
        E_cells=E_cells,
        C_centroids=C_centroids,
        C_v=C_v,
        labels=labels,
    )


@pytest.fixture
def minimal_force_result(minimal_tissue: Tissue) -> ForceResult:
    """A ForceResult consistent with minimal_tissue (uniform tensions)."""
    n_edges = len(minimal_tissue.E)
    n_cells = len(minimal_tissue.C_v)
    return ForceResult(
        tensions=np.ones(n_edges, dtype=float),
        pressures=np.zeros(n_cells, dtype=float),
        residual=0.0,
    )
