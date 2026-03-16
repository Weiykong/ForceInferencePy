"""Tests for force_inference.core — Tissue and ForceResult data structures."""
import numpy as np



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
