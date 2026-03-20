"""Tests for force_inference.segmentation — segment_grayscale and segment_cellpose."""
import os
import tempfile
import numpy as np
import pytest
from unittest.mock import MagicMock, patch
from skimage import io as sk_io

from force_inference.segmentation import segment_grayscale, segment_cellpose


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_synthetic_tif(path: str, pattern: str = "grid") -> None:
    """Write a small synthetic grayscale TIFF to *path*."""
    size = 64
    img = np.zeros((size, size), dtype=np.uint8)

    if pattern == "grid":
        img[:] = 30
        img[size // 2, :] = 220
        img[:, size // 2] = 220
    elif pattern == "uniform":
        img[:] = 128

    sk_io.imsave(path, img)


# ---------------------------------------------------------------------------
# Tests for segment_grayscale
# ---------------------------------------------------------------------------

class TestSegmentGrayscale:
    def test_missing_file_raises(self):
        with pytest.raises(FileNotFoundError):
            segment_grayscale("/nonexistent/path/image.tif")

    def test_returns_two_arrays(self):
        with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as f:
            path = f.name
        try:
            _write_synthetic_tif(path)
            labels, processed = segment_grayscale(path, h_depth=5, blur_sigma=1, min_cell_size=5)
            assert isinstance(labels, np.ndarray)
            assert isinstance(processed, np.ndarray)
        finally:
            os.unlink(path)

    def test_labels_integer(self):
        with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as f:
            path = f.name
        try:
            _write_synthetic_tif(path)
            labels, _ = segment_grayscale(path, h_depth=5, blur_sigma=1, min_cell_size=5)
            assert np.issubdtype(labels.dtype, np.integer)
        finally:
            os.unlink(path)

# ---------------------------------------------------------------------------
# Tests for segment_cellpose
# ---------------------------------------------------------------------------

class TestSegmentCellpose:
    def test_missing_file_raises(self):
        # We mock cellpose.models to avoid import errors if not installed
        with patch('cellpose.models.CellposeModel') as mock_model:
            with pytest.raises(FileNotFoundError):
                segment_cellpose("/nonexistent/path/image.tif")

    @patch('cellpose.models.CellposeModel')
    def test_mock_cellpose_run(self, mock_cp_model):
        # Setup mock
        mock_instance = MagicMock()
        mock_cp_model.return_value = mock_instance
        
        # Mock eval result: (masks, flows, styles, diams)
        mock_masks = np.zeros((64, 64), dtype=np.int32)
        mock_masks[10:20, 10:20] = 1
        mock_instance.eval.return_value = (mock_masks, None, None, None)
        
        with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as f:
            path = f.name
        try:
            _write_synthetic_tif(path)
            labels, gray = segment_cellpose(path, model_type="cyto3")
            
            assert labels.shape == (64, 64)
            assert labels.max() == 1
            assert isinstance(gray, np.ndarray)
            mock_instance.eval.assert_called_once()
        finally:
            os.unlink(path)

    def test_cellpose_import_error_msg(self):
        with patch.dict('sys.modules', {'cellpose': None}):
            with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as f:
                path = f.name
            try:
                _write_synthetic_tif(path)
                # Reloading segment_cellpose logic might be tricky, but calling it should fail
                with pytest.raises(ImportError) as excinfo:
                    segment_cellpose(path)
                assert "Cellpose is required" in str(excinfo.value)
            finally:
                os.unlink(path)
