#!/opt/homebrew/bin/python3.10
"""Generate PNG assets used by the GitHub README."""

from __future__ import annotations

import argparse
import copy
import logging
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import patheffects
from scipy import ndimage

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from force_inference import geometry, segmentation, solvers, visualization
from force_inference.split_four_way import split_high_degree_vertices
from force_inference.topology_label import extract_topology_label


LOGGER = logging.getLogger("readme-assets")


def _segment_image(
    img_path: Path,
    model_type: str,
    prefer_cellpose: bool,
) -> tuple[np.ndarray, np.ndarray, str]:
    """Prefer Cellpose when available, otherwise fall back to watershed."""
    if prefer_cellpose:
        try:
            labels, gray = segmentation.segment_cellpose(
                str(img_path),
                model_type=model_type,
                gpu=True,
            )
            method = f"Cellpose ({model_type})"
            return labels, gray, method
        except Exception as exc:  # pragma: no cover - environment-dependent
            LOGGER.warning("Cellpose unavailable, using grayscale fallback: %s", exc)

    labels, gray = segmentation.segment_grayscale(
        str(img_path),
        h_depth=2.0,
        min_cell_size=5,
    )
    method = "grayscale fallback"

    return labels, gray, method


def _build_pipeline(labels: np.ndarray) -> tuple:
    tissue_raw = extract_topology_label(
        labels,
        min_edge_len=1,
        use_skeleton_geometry=False,
        collapse_stubs=False,
        collapse_tiny_twins=False,
    )
    if tissue_raw is None:
        raise RuntimeError("Topology extraction failed.")

    tissue_split = split_high_degree_vertices(copy.deepcopy(tissue_raw), split_length=4.0)
    tissue_geom = geometry.compute_curvature(copy.deepcopy(tissue_split))

    bayes = solvers.solve_bayesian(tissue_geom, mu=1e-2)
    laplace = solvers.solve_laplace(tissue_geom, regularization=1.0)
    if bayes is None or laplace is None:
        raise RuntimeError("Solver stage failed.")

    stress = geometry.calculate_batchelor_stress(tissue_geom, copy.deepcopy(bayes))
    return tissue_raw, tissue_split, tissue_geom, bayes, laplace, stress


def _label_rgb(labels: np.ndarray) -> np.ndarray:
    n_labels = int(labels.max())
    palette = np.ones((max(n_labels + 1, 2), 3), dtype=float)
    palette[0] = np.array([0.09, 0.10, 0.12])

    if n_labels > 0:
        rng = np.random.default_rng(7)
        hues = np.linspace(0.0, 1.0, n_labels, endpoint=False)
        rng.shuffle(hues)
        sat = np.full(n_labels, 0.65)
        val = np.full(n_labels, 0.88)
        hsv = np.column_stack([hues, sat, val])
        rgb = matplotlib.colors.hsv_to_rgb(hsv)
        palette[1 : n_labels + 1] = rgb

    return palette[labels]


def _fill_label_gaps_for_display(labels: np.ndarray) -> np.ndarray:
    """Fill zero-valued display gaps from the nearest labeled pixel.

    This is only for the README segmentation panel so thin Cellpose slivers or
    tiny unlabeled cracks do not render as black holes.
    """
    labels = np.asarray(labels)
    if labels.ndim != 2 or not np.any(labels == 0) or not np.any(labels > 0):
        return labels

    filled = labels.copy()
    _, (iy, ix) = ndimage.distance_transform_edt(labels == 0, return_indices=True)
    zero_mask = filled == 0
    filled[zero_mask] = filled[iy[zero_mask], ix[zero_mask]]
    return filled


def _vertex_degrees(edges: np.ndarray, n_vertices: int) -> np.ndarray:
    deg = np.zeros(n_vertices, dtype=int)
    for v1, v2 in edges:
        deg[int(v1)] += 1
        deg[int(v2)] += 1
    return deg


def _pick_zoom_center(tissue) -> np.ndarray:
    deg = _vertex_degrees(tissue.E, len(tissue.V))
    idx = int(np.argmax(deg))
    return tissue.V[idx, :2]


def _apply_crop(ax: plt.Axes, center: np.ndarray, half_size: int = 80) -> None:
    cx, cy = float(center[0]), float(center[1])
    ax.set_xlim(cx - half_size, cx + half_size)
    ax.set_ylim(cy + half_size, cy - half_size)


def _style_axes(ax: plt.Axes) -> None:
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_frame_on(False)


def _plot_split_overlay(ax: plt.Axes, tissue, zoom_center: np.ndarray) -> None:
    visualization.plot_topology_check(ax, tissue)
    synthetic = getattr(tissue, "E_synthetic", None)
    if synthetic is None:
        return

    for edge_idx, is_synth in enumerate(synthetic):
        if not is_synth:
            continue
        v1, v2 = tissue.E[edge_idx]
        p1 = tissue.V[v1, :2]
        p2 = tissue.V[v2, :2]
        ax.plot(
            [p1[0], p2[0]],
            [p1[1], p2[1]],
            linestyle=(0, (4, 2)),
            color="#ff8c42",
            linewidth=2.0,
            zorder=30,
        )

    txt = ax.text(
        0.03,
        0.03,
        "dashed: synthetic edge added by 4-way split",
        transform=ax.transAxes,
        color="white",
        ha="left",
        va="bottom",
        fontsize=8,
        bbox=dict(boxstyle="round,pad=0.25", facecolor=(0, 0, 0, 0.45), edgecolor="none"),
    )
    txt.set_path_effects([patheffects.withStroke(linewidth=2, foreground="black")])


def _make_mock_z_stack(shape: tuple[int, int], z_dim: int = 60) -> np.ndarray:
    h, w = shape
    yy, xx = np.indices((h, w))
    dome = 50.0 * np.exp(-((xx - w / 2.0) ** 2 + (yy - h / 2.0) ** 2) / (w / 3.0) ** 2)
    z_idx = np.clip(np.rint(dome), 0, z_dim - 1).astype(int)

    stack = np.zeros((z_dim, h, w), dtype=np.uint8)
    stack[z_idx, yy, xx] = 255
    return stack


def _save_pipeline_overview(
    out_path: Path,
    gray: np.ndarray,
    labels: np.ndarray,
    method: str,
    tissue,
    bayes,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)
    display_labels = _fill_label_gaps_for_display(labels)

    axes[0, 0].imshow(gray, cmap="gray")
    axes[0, 0].set_title("1. Membrane image")

    axes[0, 1].imshow(_label_rgb(display_labels))
    axes[0, 1].set_title("2. Segmentation labels")

    axes[1, 0].imshow(gray, cmap="gray", alpha=0.55)
    visualization.plot_topology_check(axes[1, 0], tissue)
    axes[1, 0].set_title(
        f"3. Label-driven topology\n{len(tissue.V)} vertices, {len(tissue.E)} edges"
    )

    axes[1, 1].imshow(gray, cmap="gray", alpha=0.4)
    visualization.plot_tensions(axes[1, 1], tissue, bayes, cmap="turbo", width=1.8)
    n_valid = int(np.sum(~np.isnan(bayes.tensions)))
    axes[1, 1].set_title(
        f"4. Bayesian tensions\n{n_valid} solved interior edges"
    )

    for ax in axes.flat:
        _style_axes(ax)

    fig.savefig(out_path, dpi=180, facecolor="white")
    plt.close(fig)


def _save_topology_zoom(
    out_path: Path,
    gray: np.ndarray,
    tissue_raw,
    tissue_split,
) -> None:
    zoom_center = _pick_zoom_center(tissue_raw)
    fig, axes = plt.subplots(1, 2, figsize=(12, 6), constrained_layout=True)

    axes[0].imshow(gray, cmap="gray", alpha=0.55)
    visualization.plot_topology_check(axes[0], tissue_raw)
    axes[0].set_title("Before split: extracted topology")
    _apply_crop(axes[0], zoom_center)

    axes[1].imshow(gray, cmap="gray", alpha=0.55)
    _plot_split_overlay(axes[1], tissue_split, zoom_center)
    axes[1].set_title("After split: only 3-way junctions remain")
    _apply_crop(axes[1], zoom_center)

    for ax in axes:
        _style_axes(ax)

    fig.savefig(out_path, dpi=200, facecolor="white")
    plt.close(fig)


def _save_solver_outputs(
    out_path: Path,
    gray: np.ndarray,
    tissue,
    bayes,
    laplace,
    stress,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5), constrained_layout=True)

    axes[0].imshow(gray, cmap="gray", alpha=0.4)
    visualization.plot_tensions(axes[0], tissue, bayes, cmap="turbo", width=1.7)
    axes[0].set_title("Bayesian solver: tensions")

    axes[1].imshow(gray, cmap="gray", alpha=0.28)
    visualization.plot_pressures(axes[1], tissue, laplace, cmap="coolwarm")
    axes[1].set_title("Young-Laplace solver: pressures")

    axes[2].imshow(gray, cmap="gray", alpha=0.28)
    visualization.plot_cell_stress_crosses(axes[2], tissue, stress, scale=60.0, min_mag=0.01)
    axes[2].set_title("Batchelor cell stress")

    for ax in axes:
        _style_axes(ax)

    fig.savefig(out_path, dpi=180, facecolor="white")
    plt.close(fig)


def _save_support_25d(out_path: Path, gray: np.ndarray, tissue, bayes) -> None:
    stack = _make_mock_z_stack(gray.shape[:2])
    tissue_25d = geometry.map_z_to_vertices(copy.deepcopy(tissue), stack)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5), constrained_layout=True)

    axes[0].imshow(gray, cmap="gray", alpha=0.35)
    visualization.plot_tensions(axes[0], tissue_25d, bayes, cmap="turbo", width=1.6)
    axes[0].set_title("Same 2D topology, carried into 2.5D")

    axes[1].scatter(
        tissue_25d.V[:, 0],
        tissue_25d.V[:, 2],
        c=tissue_25d.V[:, 2],
        cmap="magma",
        s=12,
        linewidths=0,
    )
    axes[1].set_title("Vertex heights in XZ")
    axes[1].set_xlabel("x")
    axes[1].set_ylabel("z")
    axes[1].grid(alpha=0.15)

    sc = axes[2].scatter(
        tissue_25d.V[:, 0],
        tissue_25d.V[:, 1],
        c=tissue_25d.V[:, 2],
        cmap="magma",
        s=10,
        linewidths=0,
    )
    axes[2].imshow(gray, cmap="gray", alpha=0.22)
    axes[2].set_title("Height-mapped vertices in XY")
    cbar = fig.colorbar(sc, ax=axes[2], fraction=0.046, pad=0.02)
    cbar.set_label("z index")

    _style_axes(axes[0])
    _style_axes(axes[2])

    fig.savefig(out_path, dpi=180, facecolor="white")
    plt.close(fig)


def generate_assets(
    img_path: Path,
    out_dir: Path,
    model_type: str,
    prefer_cellpose: bool,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    labels, gray, method = _segment_image(
        img_path,
        model_type=model_type,
        prefer_cellpose=prefer_cellpose,
    )
    tissue_raw, tissue_split, tissue_geom, bayes, laplace, stress = _build_pipeline(labels)

    _save_pipeline_overview(
        out_dir / "pipeline_overview.png",
        gray,
        labels,
        method,
        tissue_geom,
        bayes,
    )
    _save_topology_zoom(
        out_dir / "topology_zoom.png",
        gray,
        tissue_raw,
        tissue_split,
    )
    _save_solver_outputs(
        out_dir / "solver_outputs.png",
        gray,
        tissue_geom,
        bayes,
        laplace,
        stress,
    )
    _save_support_25d(
        out_dir / "support_25d.png",
        gray,
        tissue_geom,
        bayes,
    )

    print(f"Generated README assets in {out_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--image",
        type=Path,
        default=REPO_ROOT / "data" / "test.tif",
        help="Input microscopy image.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=REPO_ROOT / "docs" / "readme_assets",
        help="Output directory for PNGs.",
    )
    parser.add_argument(
        "--model-type",
        default="cyto3",
        help="Cellpose model type when Cellpose is installed.",
    )
    parser.add_argument(
        "--skip-cellpose",
        action="store_true",
        help="Use the grayscale fallback directly.",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
    generate_assets(
        args.image,
        args.out_dir,
        model_type=args.model_type,
        prefer_cellpose=not args.skip_cellpose,
    )


if __name__ == "__main__":
    main()
