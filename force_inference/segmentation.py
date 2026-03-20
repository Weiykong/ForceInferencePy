from __future__ import annotations

import numpy as np
import os
import logging
import inspect
from scipy import ndimage
from skimage import io, morphology, measure, segmentation
from typing import Tuple, Optional

logger = logging.getLogger("ForceInference.Segmentation")

def segment_cellpose(
    img_path: str,
    model_type: str = "cyto3",
    diameter: Optional[float] = None,
    flow_threshold: float = 0.4,
    cellprob_threshold: float = 0.0,
    min_size: int = 15,
    gpu: bool = True,
    invert: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Segment a membrane-labeled tissue image using Cellpose.

    Compatible with both Cellpose 3 and Cellpose 4 (replaces models.Cellpose
    with models.CellposeModel).

    Args:
        img_path:        Path to image (TIFF, PNG, etc.).
        model_type:      Cellpose model:
                           "cyto3"  — Cellpose3 cytoplasm (recommended start)
                           "cpsam"  — Cellpose-SAM (best general, CP4 default)
                           "cyto2"  — older cytoplasm model
        diameter:        Expected cell diameter in pixels (None = auto).
        flow_threshold:  Max flow error (higher = more permissive, 0.4 default).
        cellprob_threshold: Cell probability threshold. Lower = more cells.
                         Try -2 to -4 for faint membranes.
                         (This maps to cellprob_threshold in CP3
                          or mask_threshold in CP4.)
        min_size:        Minimum cell area in pixels.
        gpu:             Use GPU if available.
        invert:          Invert image intensity before segmentation.

    Returns:
        (labels, img_gray)
    """
    try:
        from cellpose import models
    except ImportError:
        logger.error("Cellpose not installed. Run 'pip install cellpose'.")
        raise ImportError("Cellpose is required for segment_cellpose()")

    if not os.path.exists(img_path):
        raise FileNotFoundError(f"Image file not found: {img_path}")

    # ---- Load image ----
    img = io.imread(img_path)

    if img.ndim == 3:
        if img.shape[-1] <= 4:  # (H, W, C)
            gray = (
                0.299 * img[:, :, 0].astype(float)
                + 0.587 * img[:, :, 1].astype(float)
                + 0.114 * img[:, :, 2].astype(float)
            )
            img_for_cp = img[:, :, :3] if img.shape[-1] >= 3 else gray
        elif img.shape[0] <= 4:  # (C, H, W)
            gray = img[0].astype(float)
            img_for_cp = gray
        else:  # (Z, H, W)
            gray = np.max(img, axis=0).astype(float)
            img_for_cp = gray
    else:
        gray = img.astype(float)
        img_for_cp = gray

    if invert:
        img_for_cp = img_for_cp.max() - img_for_cp

    # ---- Build model (CP3 or CP4 compatible) ----
    logger.info(
        f"Running Cellpose ({model_type}), "
        f"diameter={'auto' if diameter is None else diameter}"
    )

    # Cellpose 4 removed models.Cellpose → use models.CellposeModel
    # Note: CellposeModel is available in CP3 as well.
    model = models.CellposeModel(model_type=model_type, gpu=gpu)

    # ---- Build eval kwargs ----
    eval_kwargs = dict(
        diameter=diameter,
        flow_threshold=flow_threshold,
        min_size=min_size,
    )

    # CP4 renamed cellprob_threshold → mask_threshold
    eval_sig = inspect.signature(model.eval)
    if "mask_threshold" in eval_sig.parameters:
        eval_kwargs["mask_threshold"] = cellprob_threshold
    elif "cellprob_threshold" in eval_sig.parameters:
        eval_kwargs["cellprob_threshold"] = cellprob_threshold

    # cpsam doesn't use channels; cyto models need [0,0] for grayscale
    if "channels" in eval_sig.parameters and model_type not in ("cpsam",):
        eval_kwargs["channels"] = [0, 0]

    # ---- Run ----
    result = model.eval(img_for_cp, **eval_kwargs)
    masks = result[0]

    n_cells = int(masks.max())
    logger.info(f"Cellpose found {n_cells} cells")

    return masks.astype(np.int32), gray

def segment_grayscale(img_path: str,
                      h_depth: float = 5.0,
                      blur_sigma: float = 1.0,
                      min_cell_size: int = 20) -> Tuple[np.ndarray, np.ndarray]:
    """
    DEPRECATED: Use segment_cellpose for better results.
    Robust watershed segmentation for membrane-labeled tissue images.
    
    Args:
        img_path: Path to the input image (TIFF, PNG, etc.).
        h_depth: Depth of the h-minima transform.
        blur_sigma: Standard deviation for Gaussian kernel.
        min_cell_size: Minimum size (in pixels) for a marker to be kept.

    Returns:
        (labels, img_processed)
    """
    logger.warning("segment_grayscale is deprecated; consider switching to segment_cellpose.")
    if not os.path.exists(img_path):
        raise FileNotFoundError(f"Image file not found: {img_path}")

    # 1. Load Image
    img = io.imread(img_path)
    
    if img.ndim == 3:
        if img.shape[-1] < 5: 
            img = img[..., 0] 
        else:
            img = np.max(img, axis=0)

    img = img.astype(float)
    
    # 2. Inversion Heuristic
    if np.mean(img) > 128: 
        logger.info("Inverting image intensity for segmentation...")
        img = img.max() - img
        
    # 3. Smoothing
    img_smooth = ndimage.gaussian_filter(img, sigma=blur_sigma)
    
    # 4. Marker Extraction (H-Minima)
    markers_mask = morphology.h_minima(img_smooth, h=h_depth)
    
    # 5. Dust Removal
    markers_clean = morphology.remove_small_objects(markers_mask.astype(bool))
    
    # Label the markers
    markers = measure.label(markers_clean)
    
    # 6. Watershed
    labels = segmentation.watershed(img_smooth, markers)
    
    return labels, img_smooth
