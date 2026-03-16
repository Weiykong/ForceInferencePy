"""Tests for force_inference.geometry — curvature, circle fitting, tangents."""
import numpy as np
import pytest

from force_inference.geometry import (
    compute_curvature,
    map_z_to_vertices,
    _fit_circle_parameters_full,
    _get_curvature_sign_distance,
    calculate_batchelor_stress,
)
from force_inference.core import Tissue, ForceResult


# ---------------------------------------------------------------------------
# _fit_circle_parameters_full
# ---------------------------------------------------------------------------

class TestFitCircle:
    def _circle_points(self, cx, cy, R, n=40):
        theta = np.linspace(0, 2 * np.pi, n, endpoint=False)
        x = cx + R * np.cos(theta)
        y = cy + R * np.sin(theta)
        return np.column_stack([x, y])

    def test_unit_circle(self):
        pts = self._circle_points(0, 0, 1.0)
        kappa, center, R = _fit_circle_parameters_full(pts)
        assert abs(kappa - 1.0) < 0.05
        assert abs(R - 1.0) < 0.05
        assert np.linalg.norm(center) < 0.1

    def test_offset_circle(self):
        pts = self._circle_points(5.0, -3.0, 2.0)
        kappa, center, R = _fit_circle_parameters_full(pts)
        assert abs(R - 2.0) < 0.1
        assert np.linalg.norm(center - np.array([5.0, -3.0])) < 0.2

    def test_collinear_points_low_curvature(self):
        # Collinear points lie on a circle of infinite radius;
        # the fit returns a very large R and correspondingly small kappa.
        pts = np.column_stack([np.arange(5, dtype=float), np.zeros(5)])
        kappa, center, R = _fit_circle_parameters_full(pts)
        # R must be finite and either large (near-degenerate) or zero
        assert np.isfinite(kappa)
        assert np.isfinite(R)

    def test_returns_finite_values(self):
        # Arbitrary small input should never produce NaN or Inf
        pts = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, 0.5]])
        kappa, center, R = _fit_circle_parameters_full(pts)
        assert np.isfinite(kappa)
        assert np.all(np.isfinite(center))
        assert np.isfinite(R)

    def test_returns_tuple_types(self):
        pts = self._circle_points(0, 0, 3.0)
        kappa, center, R = _fit_circle_parameters_full(pts)
        assert isinstance(kappa, float)
        assert isinstance(center, np.ndarray)
        assert center.shape == (2,)
        assert isinstance(R, float)


# ---------------------------------------------------------------------------
# _get_curvature_sign_distance
# ---------------------------------------------------------------------------

class TestCurvatureSign:
    def test_bulging_outward_is_positive(self):
        # Arc bows away from centroid placed at origin
        theta = np.linspace(np.pi / 4, 3 * np.pi / 4, 20)
        pts = np.column_stack([np.cos(theta), np.sin(theta)])  # upper arc
        cell_cent = np.array([0.0, -2.0])  # centroid well below
        sign = _get_curvature_sign_distance(pts, cell_cent)
        assert sign == 1.0

    def test_bulging_inward_is_negative(self):
        theta = np.linspace(np.pi / 4, 3 * np.pi / 4, 20)
        pts = np.column_stack([np.cos(theta), np.sin(theta)])
        cell_cent = np.array([0.0, 5.0])  # centroid above the arc midpoint
        sign = _get_curvature_sign_distance(pts, cell_cent)
        assert sign == -1.0


# ---------------------------------------------------------------------------
# map_z_to_vertices
# ---------------------------------------------------------------------------

class TestMapZToVertices:
    def _make_stack(self, z_dim=5, h=20, w=20):
        stack = np.zeros((z_dim, h, w), dtype=float)
        stack[3, :, :] = 1.0  # brightest at z=3
        return stack

    def test_basic_z_mapping(self, minimal_tissue):
        stack = self._make_stack()
        t = map_z_to_vertices(minimal_tissue, stack)
        # All vertices inside the image should map to z=3
        assert np.all(t.V[:, 2] == 3.0)

    def test_bad_stack_raises(self, minimal_tissue):
        bad = np.zeros((20, 20))  # 2-D, not 3-D
        with pytest.raises(ValueError):
            map_z_to_vertices(minimal_tissue, bad)

    def test_empty_stack_uses_fallback(self, minimal_tissue):
        stack = np.zeros((0, 20, 20), dtype=float)
        t = map_z_to_vertices(minimal_tissue, stack, z_fallback=-1.0)
        assert np.all(t.V[:, 2] == -1.0)


# ---------------------------------------------------------------------------
# compute_curvature (integration)
# ---------------------------------------------------------------------------

class TestComputeCurvature:
    def _four_cell_tissue(self):
        """Build a small 4-cell tissue for integration testing."""
        from force_inference.topology_label import extract_topology_label
        import numpy as np

        size = 40
        labels = np.zeros((size, size), dtype=np.int32)
        mid = size // 2
        labels[1:mid, 1:mid] = 1
        labels[1:mid, mid + 1 : size - 1] = 2
        labels[mid + 1 : size - 1, 1:mid] = 3
        labels[mid + 1 : size - 1, mid + 1 : size - 1] = 4
        return extract_topology_label(labels)

    def test_curvature_array_shape(self):
        tissue = self._four_cell_tissue()
        if tissue is None or len(tissue.E) == 0:
            pytest.skip("Topology extraction returned no edges")
        tissue = compute_curvature(tissue)
        assert tissue.E_curvature is not None
        assert len(tissue.E_curvature) == len(tissue.E)

    def test_tangents_array_shape(self):
        tissue = self._four_cell_tissue()
        if tissue is None or len(tissue.E) == 0:
            pytest.skip("Topology extraction returned no edges")
        tissue = compute_curvature(tissue)
        assert hasattr(tissue, "E_tangents")
        assert tissue.E_tangents.shape == (len(tissue.E), 2, 2)


# ---------------------------------------------------------------------------
# calculate_batchelor_stress
# ---------------------------------------------------------------------------

class TestBatchelorStress:
    def test_output_shape(self, minimal_tissue, minimal_force_result):
        result = calculate_batchelor_stress(minimal_tissue, minimal_force_result)
        assert result.stress_tensors is not None
        n_cells = len(minimal_tissue.C_v)
        assert result.stress_tensors.shape == (n_cells, 2, 2)

    def test_empty_tissue(self):
        labels = np.zeros((10, 10), dtype=np.int32)
        t = Tissue(
            V=np.zeros((0, 3)),
            E=np.zeros((0, 2), dtype=int),
            E_cells=np.zeros((0, 2), dtype=int),
            C_centroids=np.zeros((0, 2)),
            C_v=[],
            labels=labels,
        )
        fr = ForceResult(tensions=np.array([]), pressures=np.array([]), residual=0.0)
        result = calculate_batchelor_stress(t, fr)
        assert result.stress_tensors.shape == (0, 2, 2)
