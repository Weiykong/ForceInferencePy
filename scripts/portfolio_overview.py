#!/opt/homebrew/bin/python3.10
"""
Portfolio-quality pipeline overview — 4-panel figure.

Usage:
    python scripts/portfolio_overview.py [--image data/test.tif] [--out /tmp/portfolio.png]
"""

from __future__ import annotations

import argparse
import copy
import logging
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.colors as mcolors
import matplotlib.patheffects as pe
from matplotlib.collections import LineCollection
from matplotlib import font_manager
import numpy as np
from scipy import ndimage

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from force_inference import segmentation, solvers, visualization, geometry
from force_inference.split_four_way import split_high_degree_vertices
from force_inference.topology_label import extract_topology_label
from force_inference.visualization import _fix_edge_pixels

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
LOGGER = logging.getLogger("portfolio")

# ── colour palette ────────────────────────────────────────────────────────────
BG          = "#0d0d0d"   # figure background
PANEL_BG    = "#111111"   # axes background
TITLE_COL   = "#f0f0f0"
SUB_COL     = "#888888"
EDGE_COL    = "#ffffff"   # topology edges
DOT_COL     = "#ffffff"   # junction dots
CELL_DIM    = 0.38        # label-image brightness for overlay panels
# ─────────────────────────────────────────────────────────────────────────────


def _label_rgb(labels: np.ndarray, seed: int = 7) -> np.ndarray:
    n = int(labels.max())
    rng = np.random.default_rng(seed)
    hues = np.linspace(0.0, 1.0, max(n, 1), endpoint=False)
    rng.shuffle(hues)
    sat = np.full(n, 0.60)
    val = np.full(n, 0.90)
    hsv = np.column_stack([hues, sat, val])
    rgb = mcolors.hsv_to_rgb(hsv)
    palette = np.zeros((n + 1, 3))
    palette[0]  = [0.08, 0.08, 0.08]
    palette[1:] = rgb
    return palette[labels]


def _fill_gaps(labels: np.ndarray) -> np.ndarray:
    if not np.any(labels == 0) or not np.any(labels > 0):
        return labels
    filled = labels.copy()
    _, (iy, ix) = ndimage.distance_transform_edt(labels == 0, return_indices=True)
    m = filled == 0
    filled[m] = filled[iy[m], ix[m]]
    return filled


def _style_ax(ax: plt.Axes, bg: str = PANEL_BG) -> None:
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_facecolor(bg)
    for sp in ax.spines.values():
        sp.set_visible(True)
        sp.set_edgecolor("#2a2a2a")
        sp.set_linewidth(0.8)


def _title(ax: plt.Axes, main: str, sub: str = "") -> None:
    """Draw a bold panel title + dimmed subtitle inside the top-left of the panel."""
    ax.text(
        0.015, 0.978, main,
        transform=ax.transAxes,
        color=TITLE_COL, fontsize=11, fontweight="bold",
        va="top", ha="left", zorder=10,
    ).set_path_effects([
        pe.withStroke(linewidth=3, foreground=BG),
    ])
    if sub:
        ax.text(
            0.015, 0.944, sub,
            transform=ax.transAxes,
            color=SUB_COL, fontsize=8.0, fontweight="normal",
            va="top", ha="left", zorder=10,
        ).set_path_effects([
            pe.withStroke(linewidth=2, foreground=BG),
        ])


def _draw_topology_edges(ax: plt.Axes, tissue, linewidth: float = 1.1) -> None:
    """Draw edge paths in white — no cell colour fill, no numbers."""
    segs = []
    for i, (v1, v2) in enumerate(tissue.E):
        p1 = tissue.V[v1, :2]
        p2 = tissue.V[v2, :2]
        px = tissue.E_pixels[i] if tissue.E_pixels is not None else None
        if px is not None and len(px) > 1:
            try:
                px = _fix_edge_pixels(px, v1_pos=p1, v2_pos=p2)
                segs.append(px)
                continue
            except Exception:
                pass
        segs.append(np.array([p1, p2]))

    lc = LineCollection(
        segs,
        colors=EDGE_COL,
        linewidths=linewidth,
        alpha=0.90,
        capstyle="round",
        joinstyle="round",
        zorder=2,
    )
    ax.add_collection(lc)

    # Junction dots
    n_inner = getattr(tissue, "num_inner_vertices", len(tissue.V))
    vx = tissue.V[:n_inner, 0]
    vy = tissue.V[:n_inner, 1]
    dot_s = np.pi * linewidth ** 2 * 2.2
    ax.scatter(vx, vy, s=dot_s, c=DOT_COL, edgecolors="none",
               linewidths=0, zorder=3, alpha=0.85)


def make_portfolio(
    img_path: Path,
    out_path: Path,
    dpi: int = 220,
) -> None:
    # ── 1. Segment ────────────────────────────────────────────────────────────
    LOGGER.info("Segmenting %s …", img_path)
    labels, gray = segmentation.segment_grayscale(
        str(img_path), h_depth=2.0, min_cell_size=5
    )
    display_labels = _fill_gaps(labels)
    label_rgb      = _label_rgb(display_labels)

    # ── 2. Topology + solver ──────────────────────────────────────────────────
    LOGGER.info("Extracting topology …")
    tissue_raw = extract_topology_label(
        labels, min_edge_len=1,
        use_skeleton_geometry=False,
        collapse_stubs=False,
        collapse_tiny_twins=False,
    )
    tissue = split_high_degree_vertices(copy.deepcopy(tissue_raw), split_length=4.0)
    tissue = geometry.compute_curvature(copy.deepcopy(tissue))

    LOGGER.info("Solving Bayesian tensions …")
    bayes = solvers.solve_bayesian(tissue, mu=1e-2)
    n_valid = int(np.sum(~np.isnan(bayes.tensions)))

    H, W = gray.shape

    # ── 3. Layout ─────────────────────────────────────────────────────────────
    # Size the figure so each 2×2 panel cell has exactly the same aspect ratio
    # as the source image (W/H), avoiding any stretch or letterboxing.
    #
    # With GridSpec margins left=0.015, right=0.960, top=0.905, bottom=0.015
    # and wspace=0.05, hspace=0.07:
    #   panel_w ∝ 0.945 × W_fig / (2 + 0.05)   →  factor ≈ 0.4610
    #   panel_h ∝ 0.890 × H_fig / (2 + 0.07)   →  factor ≈ 0.4300
    # For panel_w/panel_h == W/H:
    #   W_fig / H_fig = (W/H) × (0.4300 / 0.4610) = (W/H) × 0.9328
    img_aspect = W / H          # source image pixel aspect
    fig_aspect = img_aspect * 0.9328
    W_fig = 15.0
    H_fig = W_fig / fig_aspect
    fig = plt.figure(figsize=(W_fig, H_fig), facecolor=BG)

    gs = gridspec.GridSpec(
        2, 2,
        figure=fig,
        hspace=0.07,
        wspace=0.05,
        left=0.015, right=0.960,
        top=0.905,  bottom=0.015,
    )
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, 0])
    ax4 = fig.add_subplot(gs[1, 1])

    def _show(ax: plt.Axes, img: np.ndarray, **kw) -> None:
        """imshow with correct aspect — no stretch, no letterboxing."""
        ax.imshow(img, origin="upper", aspect="equal", **kw)
        ax.set_xlim(-0.5, W - 0.5)
        ax.set_ylim(H - 0.5, -0.5)

    for ax in (ax1, ax2, ax3, ax4):
        _style_ax(ax)

    # ── Panel 1 — membrane image ──────────────────────────────────────────────
    p1_img = gray.astype(float)
    lo, hi = np.percentile(p1_img, [1, 99])
    p1_img = np.clip((p1_img - lo) / (hi - lo + 1e-6), 0, 1)
    _show(ax1, p1_img, cmap="gray", vmin=0, vmax=1)
    _title(ax1, "Membrane image", f"{W} × {H} px  ·  fluorescence")

    # ── Panel 2 — segmentation ────────────────────────────────────────────────
    _show(ax2, label_rgb)
    n_cells = int(labels.max())
    _title(ax2, "Cell segmentation", f"{n_cells} cells detected")

    # ── Panel 3 — topology skeleton ───────────────────────────────────────────
    _show(ax3, label_rgb * CELL_DIM)
    _draw_topology_edges(ax3, tissue, linewidth=1.4)
    ax3.set_xlim(-0.5, W - 0.5); ax3.set_ylim(H - 0.5, -0.5)
    _title(
        ax3,
        "Junction topology",
        f"{len(tissue.V)} vertices  ·  {len(tissue.E)} edges",
    )

    # ── Panel 4 — Bayesian tensions (no colorbar) ─────────────────────────────
    _show(ax4, label_rgb * CELL_DIM)
    visualization.plot_tensions(
        ax4, tissue, bayes, cmap="turbo", width=2.2, show_colorbar=False
    )
    ax4.set_xlim(-0.5, W - 0.5); ax4.set_ylim(H - 0.5, -0.5)
    _title(
        ax4,
        "Bayesian membrane tensions",
        f"{n_valid} solved edges",
    )

    # ── Global title ──────────────────────────────────────────────────────────
    fig.text(
        0.015, 0.960,
        "Cell Force Inference Pipeline",
        color=TITLE_COL, fontsize=15, fontweight="bold", va="bottom",
    )
    fig.text(
        0.015, 0.940,
        f"{img_path.name}  ·  label-driven topology  ·  Bayesian tension solver  ·  {n_cells} cells · {len(tissue.E)} edges",
        color=SUB_COL, fontsize=8.5, va="bottom",
    )

    # ── Save ──────────────────────────────────────────────────────────────────
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, facecolor=BG, bbox_inches="tight")
    plt.close(fig)
    LOGGER.info("Saved → %s", out_path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, default=REPO_ROOT / "data" / "test.tif")
    parser.add_argument("--out",   type=Path, default=Path("/tmp/portfolio_pipeline.png"))
    parser.add_argument("--dpi",   type=int,  default=220)
    args = parser.parse_args()
    make_portfolio(args.image, args.out, dpi=args.dpi)


if __name__ == "__main__":
    main()
