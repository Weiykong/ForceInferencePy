"""Tests for force_inference.segmentation — segment_grayscale."""
import os
import tempfile

import numpy as np
import pytest
from skimage import io as sk_io

from force_inference.segmentation import segment_grayscale


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_synthetic_tif(path: str, pattern: str = "grid") -> None:
    """Write a small synthetic grayscale TIFF to *path*."""
    size = 64
    img = np.zeros((size, size), dtype=np.uint8)

    if pattern == "grid":
        # Bright membranes on a dark background (cells are dark)
        img[:] = 30  # dark cell interior
        img[size // 2, :] = 220  # horizontal membrane
        img[:, size // 2] = 220  # vertical membrane
    elif pattern == "uniform":
        img[:] = 128

    sk_io.imsave(path, img)


# ---------------------------------------------------------------------------
# Tests
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

    def test_labels_same_spatial_shape(self):
        with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as f:
            path = f.name
        try:
            _write_synthetic_tif(path, pattern="grid")
            labels, processed = segment_grayscale(path, h_depth=5, blur_sigma=1, min_cell_size=5)
            assert labels.shape == processed.shape
        finally:
            os.unlink(path)

    def test_processed_is_float(self):
        with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as f:
            path = f.name
        try:
            _write_synthetic_tif(path)
            _, processed = segment_grayscale(path)
            assert np.issubdtype(processed.dtype, np.floating)
        finally:
            os.unlink(path)

    def test_at_least_one_cell_found(self):
        with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as f:
            path = f.name
        try:
            _write_synthetic_tif(path, pattern="grid")
            labels, _ = segment_grayscale(path, h_depth=3, blur_sigma=1, min_cell_size=3)
            assert labels.max() >= 1
        finally:
            os.unlink(path)

    def test_jpg_input(self):
        """segment_grayscale should handle JPEG input without error."""
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
            path = f.name
        try:
            _write_synthetic_tif(path)  # skimage saves as JPEG for .jpg
            labels, _ = segment_grayscale(path, h_depth=5, blur_sigma=1, min_cell_size=5)
            assert labels.ndim == 2
        finally:
            os.unlink(path)
