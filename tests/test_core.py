"""Tests for force_inference.core — Tissue and ForceResult data structures."""
import numpy as np
import pytest

import force_inference


class TestTissue:
    def test_construction(self, minimal_tissue):
        t = minimal_tissue
        assert t.V.shape == (3, 3)
        assert t.E.shape == (3, 2)
        assert t.E_cells.shape == (3, 2)
        assert len(t.C_v) == 2
        assert t.labels.ndim == 2

    def test_to_2d_shape(self, minimal_tissue):
        xy = minimal_tissue.to_2d()
        assert xy.shape == (len(minimal_tissue.V), 2)

    def test_to_2d_values(self, minimal_tissue):
        xy = minimal_tissue.to_2d()
        np.testing.assert_array_equal(xy, minimal_tissue.V[:, :2])

    def test_optional_fields_default_none(self, minimal_tissue):
        assert minimal_tissue.E_pixels is None
        assert minimal_tissue.E_curvature is None

    def test_optional_fields_assignable(self, minimal_tissue):
        minimal_tissue.E_pixels = [np.zeros((4, 2)) for _ in range(len(minimal_tissue.E))]
        assert minimal_tissue.E_pixels is not None
        assert len(minimal_tissue.E_pixels) == len(minimal_tissue.E)

    # --- new convenience properties ---

    def test_n_vertices(self, minimal_tissue):
        assert minimal_tissue.n_vertices == len(minimal_tissue.V)

    def test_n_edges(self, minimal_tissue):
        assert minimal_tissue.n_edges == len(minimal_tissue.E)

    def test_n_cells(self, minimal_tissue):
        assert minimal_tissue.n_cells == len(minimal_tissue.C_v)

    def test_repr_contains_counts(self, minimal_tissue):
        r = repr(minimal_tissue)
        assert "n_vertices=3" in r
        assert "n_edges=3" in r
        assert "n_cells=2" in r


class TestForceResult:
    def test_construction(self, minimal_force_result):
        fr = minimal_force_result
        assert fr.tensions.shape == (3,)
        assert fr.pressures.shape == (2,)
        assert fr.residual == 0.0
        assert fr.stress_tensors is None

    def test_stress_tensors_assignable(self, minimal_force_result):
        n = 2
        minimal_force_result.stress_tensors = np.zeros((n, 2, 2))
        assert minimal_force_result.stress_tensors.shape == (n, 2, 2)

    def test_tensions_dtype(self, minimal_force_result):
        assert minimal_force_result.tensions.dtype == float

    def test_pressures_dtype(self, minimal_force_result):
        assert minimal_force_result.pressures.dtype == float

    # --- ForceResult.summary() ---

    def test_summary_returns_string(self, minimal_force_result):
        s = minimal_force_result.summary()
        assert isinstance(s, str)

    def test_summary_contains_residual(self, minimal_force_result):
        s = minimal_force_result.summary()
        assert "residual" in s

    def test_summary_with_nan_tensions(self):
        from force_inference.core import ForceResult
        fr = ForceResult(
            tensions=np.array([1.0, np.nan, 2.0]),
            pressures=np.zeros(2),
            residual=0.5,
        )
        s = fr.summary()
        assert "edges solved : 2 / 3" in s

    def test_summary_with_stress_tensors(self, minimal_force_result):
        minimal_force_result.stress_tensors = np.zeros((2, 2, 2))
        s = minimal_force_result.summary()
        assert "stress tensors" in s


class TestPackageVersion:
    def test_version_string(self):
        assert hasattr(force_inference, "__version__")
        assert isinstance(force_inference.__version__, str)
        # Must be valid semver-like: X.Y.Z
        parts = force_inference.__version__.split(".")
        assert len(parts) == 3
        assert all(p.isdigit() for p in parts)
