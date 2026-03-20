from __future__ import annotations

import numpy as np
import os
import logging
from scipy import ndimage
from skimage import io, morphology, measure, segmentation
from typing import Tuple

logger = logging.getLogger("ForceInference.Segmentation")

def segment_grayscale(img_path: str,
                      h_depth: float = 5.0,
                      blur_sigma: float = 1.0,
                      min_cell_size: int = 20) -> Tuple[np.ndarray, np.ndarray]:
    """
    Robust watershed segmentation for membrane-labeled tissue images.
    
    Pipeline:
    1. Invert image (assumes membranes are bright, cells are dark).
    2. Gaussian blur to smooth noise.
    3. H-Minima transform to find deep basins (cell centers).
    4. "Dust" removal: eliminates tiny markers (noise specs) before watershed.
    5. Watershed segmentation.

    Args:
        img_path: Path to the input image (TIFF, PNG, etc.).
        h_depth: Depth of the h-minima transform. Higher values merge shallow basins 
                 (reduces oversegmentation).
        blur_sigma: Standard deviation for Gaussian kernel.
        min_cell_size: Minimum size (in pixels) for a marker to be kept. 
                       Crucial for removing "dust" or noise.

    Returns:
        (labels, img_processed)
        labels: Integer mask where 0 is boundary/background, and 1..N are cells.
        img_processed: The pre-processed floating point image used for segmentation.
    """
    if not os.path.exists(img_path):
        raise FileNotFoundError(f"Image file not found: {img_path}")

    # 1. Load Image
    img = io.imread(img_path)
    
    # Handle dimensions (e.g., if RGBA or Z-stack, take first slice/channel)
    if img.ndim == 3:
        # Heuristic: if channels are last (H, W, C) and C < 5
        if img.shape[-1] < 5: 
            img = img[..., 0] 
        # Else assume (Z, H, W), take max projection or middle slice? 
        # For segmentation, usually max projection is safest if 3D.
        else:
            img = np.max(img, axis=0)

    # Normalize to float 0..255 or 0..1 range roughly
    img = img.astype(float)
    
    # 2. Inversion Heuristic
    # Watershed expects catchment basins (dark). Membranes are usually bright.
    # We invert so membranes become high ridges (bright) and cells become dark basins.
    # Check if background/cell interior is brighter than mean
    if np.mean(img) > 128: # Assuming 8-bit range approx
        logger.info("Inverting image intensity for segmentation...")
        img = img.max() - img
        
    # 3. Smoothing
    img_smooth = ndimage.gaussian_filter(img, sigma=blur_sigma)
    
    # 4. Marker Extraction (H-Minima)
    # h_minima finds local minima deeper than h_depth
    markers_mask = morphology.h_minima(img_smooth, h=h_depth)
    
    # 5. Dust Removal (The "Clean" step)
    # Remove connected components smaller than min_cell_size
    # Ensure boolean input to avoid skimage warning when only one marker label is present.
    markers_clean = morphology.remove_small_objects(markers_mask.astype(bool))# max_size=min_cell_size)
    
    # Label the markers (seeds for watershed)
    markers = measure.label(markers_clean)
    num_seeds = markers.max()
    logger.info(f"Segmentation initialized with {num_seeds} seeds after cleaning.")
    
    # 6. Watershed
    # We watershed on the smoothed image using the cleaned markers
    labels = segmentation.watershed(img_smooth, markers)
    
    return labels, img_smooth
