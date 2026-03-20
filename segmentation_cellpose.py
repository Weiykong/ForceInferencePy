"""
Cellpose-based segmentation for force inference.

Compatible with Cellpose 4.x (uses CellposeModel, not the removed Cellpose class).

Installation:
    pip install cellpose

Usage:
    from segmentation_cellpose import segment_cellpose
    labels, gray = segment_cellpose("test.tif")
"""

import numpy as np
import os
import logging
import inspect
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

    Compatible with both Cellpose 3 and Cellpose 4.

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
    from cellpose import models
    from skimage import io

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


if __name__ == "__main__":
    import sys
    import matplotlib.pyplot as plt

    fname = sys.argv[1] if len(sys.argv) > 1 else "test.tif"
    model = "cyto3"
    for arg in sys.argv:
        if arg.startswith("--model="):
            model = arg.split("=")[1]

    print(f"Segmenting {fname} with Cellpose ({model})...")
    labels, gray = segment_cellpose(fname, model_type=model)
    print(f"Found {labels.max()} cells")

    from skimage.segmentation import find_boundaries
    boundaries = find_boundaries(labels, mode="inner")

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    axes[0].imshow(gray, cmap="gray")
    axes[0].set_title("Raw image")
    axes[0].axis("off")

    axes[1].imshow(gray, cmap="gray", alpha=0.6)
    bnd_ov = np.zeros((*gray.shape, 4))
    bnd_ov[boundaries, :] = [0, 1, 1, 0.9]
    axes[1].imshow(bnd_ov)
    axes[1].set_title(f"Cellpose {model}: {labels.max()} cells")
    axes[1].axis("off")

    colors = np.random.rand(labels.max() + 1, 3)
    colors[0] = [0.5, 0.5, 0.5]
    axes[2].imshow(colors[labels])
    axes[2].set_title("Cell labels")
    axes[2].axis("off")

    plt.tight_layout()
    out = fname.rsplit(".", 1)[0] + f"_cellpose_{model}.png"
    plt.savefig(out, dpi=150)
    plt.show()
    print(f"Saved: {out}")