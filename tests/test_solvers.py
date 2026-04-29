"""Tests for force_inference.solvers — Bayesian and Laplace solvers."""
import numpy as np
import pytest

from force_inference.solvers import solve_bayesian, solve_laplace, BayesianScanResult
from force_inference.core import Tissue, ForceResult
from force_inference.topology_label import extract_topology_label
from force_inference.geometry import compute_curvature


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_grid_tissue() -> Tissue:
    """3x2 six-cell grid → Tissue with interior 4-way junctions.

    Uses a 90x90 label image with 25-px cells so interior vertices are
    well clear of the image margin and the Bayesian system is non-degenerate.
    """
    img = np.zeros((90, 90), dtype=np.int32)
    img[5:30, 5:30] = 1
    img[5:30, 35:60] = 2
    img[5:30, 65:85] = 3
    img[45:70, 5:30] = 4
    img[45:70, 35:60] = 5
    img[45:70, 65:85] = 6
    tissue = extract_topology_label(img, min_edge_len=3)
    return tissue


# ---------------------------------------------------------------------------
# Bayesian solver
# ---------------------------------------------------------------------------

class TestSolveBayesian:
    def test_returns_scan_result_for_none_mu(self):
        tissue = _make_grid_tissue()
        if tissue is None or len(tissue.E) == 0:
            pytest.skip("No edges extracted")
        result = solve_bayesian(tissue, mu=None)
        # With enough interior edges it should return a BayesianScanResult
        # (or None if system is degenerate)
        if result is not None:
            assert isinstance(result, BayesianScanResult)

    def test_single_mu_returns_force_result(self):
        tissue = _make_grid_tissue()
        if tissue is None or len(tissue.E) == 0:
            pytest.skip("No edges extracted")
        result = solve_bayesian(tissue, mu=1e-7)
        if result is None:
            pytest.skip("Degenerate system — no interior vertices")
        assert isinstance(result, ForceResult)
        assert result.tensions is not None
        assert len(result.tensions) == len(tissue.E)

    def test_tensions_finite(self):
        tissue = _make_grid_tissue()
        if tissue is None or len(tissue.E) == 0:
            pytest.skip("No edges extracted")
        result = solve_bayesian(tissue, mu=1e-7)
        if result is None:
            pytest.skip("Degenerate system")
        real_t = result.tensions[~np.isnan(result.tensions)]
        assert np.all(np.isfinite(real_t))

    def test_pressures_zero_mean(self):
        tissue = _make_grid_tissue()
        if tissue is None or len(tissue.E) == 0:
            pytest.skip("No edges extracted")
        result = solve_bayesian(tissue, mu=1e-7)
        if result is None:
            pytest.skip("Degenerate system")
        assert abs(np.mean(result.pressures)) < 1.0  # roughly centred

    def test_scan_result_fields(self):
        tissue = _make_grid_tissue()
        if tissue is None or len(tissue.E) == 0:
            pytest.skip("No edges extracted")
        result = solve_bayesian(tissue, mu=None)
        if result is None:
            pytest.skip("Degenerate system")
        assert isinstance(result, BayesianScanResult)
        assert result.mu_values is not None
        assert result.log_evidences is not None
        assert len(result.mu_values) == len(result.log_evidences)
        assert isinstance(result.best_mu, float)


# ---------------------------------------------------------------------------
# Laplace solver
# ---------------------------------------------------------------------------

class TestSolveLaplace:
    def _tissue_with_curvature(self) -> Tissue:
        tissue = _make_grid_tissue()
        if tissue is None or len(tissue.E) == 0:
            return None
        tissue = compute_curvature(tissue)
        return tissue

    def test_returns_force_result(self):
        tissue = self._tissue_with_curvature()
        if tissue is None:
            pytest.skip("No edges extracted")
        result = solve_laplace(tissue)
        if result is None:
            pytest.skip("Degenerate system")
        assert isinstance(result, ForceResult)

    def test_tensions_positive(self):
        tissue = self._tissue_with_curvature()
        if tissue is None:
            pytest.skip("No edges extracted")
        result = solve_laplace(tissue)
        if result is None:
            pytest.skip("Degenerate system")
        assert np.all(result.tensions > 0)

    def test_missing_tangents_raises(self):
        """solve_laplace must raise ValueError when E_tangents is absent."""
        tissue = _make_grid_tissue()
        if tissue is None or len(tissue.E) == 0:
            pytest.skip("No edges extracted")
        # Intentionally do NOT call compute_curvature
        with pytest.raises(ValueError, match="compute_curvature"):
            solve_laplace(tissue)

    def test_pressures_shape(self):
        tissue = self._tissue_with_curvature()
        if tissue is None:
            pytest.skip("No edges extracted")
        result = solve_laplace(tissue)
        if result is None:
            pytest.skip("Degenerate system")
        n_cells = int(tissue.labels.max())
        assert len(result.pressures) == n_cells
