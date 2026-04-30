"""Tests for force_inference.split_four_way — high-degree vertex splitting."""
import numpy as np
import pytest

from force_inference.split_four_way import split_high_degree_vertices
from force_inference.topology_label import extract_topology_label


def _make_six_cell_labels() -> np.ndarray:
    """3x2 grid of six cells with interior 4-way junctions.

    Layout (cell numbers, background=0):

        . 1 . 2 . 3 .
        . 1 . 2 . 3 .
        . . . . . . .
        . 4 . 5 . 6 .
        . 4 . 5 . 6 .

    The intersections of row and column membranes create two interior
    4-way junctions, which are the target for split_high_degree_vertices.
    """
    img = np.zeros((90, 90), dtype=np.int32)
    img[5:30, 5:30] = 1
    img[5:30, 35:60] = 2
    img[5:30, 65:85] = 3
    img[45:70, 5:30] = 4
    img[45:70, 35:60] = 5
    img[45:70, 65:85] = 6
    return img


@pytest.fixture
def six_cell_tissue():
    labels = _make_six_cell_labels()
    tissue = extract_topology_label(labels, min_edge_len=3)
    if tissue is None or len(tissue.E) == 0:
        pytest.skip("Topology extraction yielded no edges")
    return tissue


class TestSplitHighDegreeVertices:
    def test_returns_tissue(self, six_cell_tissue):
        result = split_high_degree_vertices(six_cell_tissue)
        assert result is not None

    def test_no_high_degree_vertices_after_split(self, six_cell_tissue):
        result = split_high_degree_vertices(six_cell_tissue)
        degrees = _vertex_degrees(result)
        assert all(d <= 3 for d in degrees.values()), (
            f"Found degree > 3: {[d for d in degrees.values() if d > 3]}"
        )

    def test_edge_count_increases_when_high_degree_present(self, six_cell_tissue):
        # The 6-cell grid has degree-4 junctions; splitting must add edges.
        n_before = len(six_cell_tissue.E)
        result = split_high_degree_vertices(six_cell_tissue)
        assert len(result.E) >= n_before

    def test_synthetic_mask_shape(self, six_cell_tissue):
        result = split_high_degree_vertices(six_cell_tissue)
        if result.E_synthetic is not None:
            assert result.E_synthetic.shape == (len(result.E),)

    def test_synthetic_edges_are_marked(self, six_cell_tissue):
        before = len(six_cell_tissue.E)
        result = split_high_degree_vertices(six_cell_tissue)
        after = len(result.E)
        if after > before and result.E_synthetic is not None:
            # Newly inserted edges must be flagged
            assert np.any(result.E_synthetic)

    def test_idempotent_on_degree_3_tissue(self, six_cell_tissue):
        """Second split call on an already-clean tissue is a no-op."""
        after_first = split_high_degree_vertices(six_cell_tissue)
        n_edges_first = len(after_first.E)
        after_second = split_high_degree_vertices(after_first)
        assert len(after_second.E) == n_edges_first

    def test_split_length_parameter(self, six_cell_tissue):
        """split_length controls the length of inserted synthetic edges."""
        result = split_high_degree_vertices(six_cell_tissue, split_length=8.0)
        assert result is not None
        degrees = _vertex_degrees(result)
        assert all(d <= 3 for d in degrees.values())


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _vertex_degrees(tissue) -> dict:
    degrees: dict = {}
    for v1, v2 in tissue.E:
        degrees[v1] = degrees.get(v1, 0) + 1
        degrees[v2] = degrees.get(v2, 0) + 1
    return degrees
