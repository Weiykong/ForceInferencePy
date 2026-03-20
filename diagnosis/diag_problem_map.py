"""
Diagnostic 1: Problem Map
Shows short edges (red), zero-curvature edges (blue), and high-degree vertices (orange/magenta)
with zoomed insets of problematic regions.

Requires:
  - test.tif in ../data/ or adjust filename below
  - force_inference package on PYTHONPATH
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from collections import Counter
import logging

logging.basicConfig(level=logging.WARNING)

from force_inference import geometry, segmentation, solvers
from force_inference.topology_label import extract_topology_label


def run(filename="../data/test.tif", out="diagnostic_test_tif.png"):
    # ── 1. Segment ──────────────────────────────────────────────────────
    labels, img_smooth = segmentation.segment_grayscale(
        filename, h_depth=2.0, min_cell_size=5
    )

    # ── 2. Topology ─────────────────────────────────────────────────────
    tissue = extract_topology_label(
        labels,
        use_skeleton_geometry=False,
        collapse_stubs=True,
        collapse_tiny_twins=False,
    )
    tissue = geometry.compute_curvature(tissue)
    result = solvers.solve_laplace(tissue, regularization=1.0)

    V = tissue.V[:, :2]
    E = tissue.E
    T = result.tensions

    # ── helpers ──────────────────────────────────────────────────────────
    degree = Counter()
    for v1, v2 in E:
        degree[v1] += 1
        degree[v2] += 1

    lengths = np.array(
        [np.linalg.norm(V[E[i, 1]] - V[E[i, 0]]) for i in range(len(E))]
    )

    # ── figure ───────────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 3, figsize=(24, 16))

    # ── Panel (0,0): Full tension map ────────────────────────────────────
    ax = axes[0, 0]
    ax.imshow(img_smooth, cmap="gray", alpha=0.5)
    for i, (v1, v2) in enumerate(E):
        p1, p2 = V[v1], V[v2]
        t_norm = (T[i] - T.min()) / (T.max() - T.min() + 1e-9)
        color = plt.cm.turbo(t_norm)
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], "-", color=color, lw=0.8, alpha=0.8)
    ax.set_title("Inferred Tensions", fontsize=14)
    ax.axis("off")

    # ── Panel (0,1): Problem map ─────────────────────────────────────────
    ax = axes[0, 1]
    ax.imshow(img_smooth, cmap="gray", alpha=0.4)
    # all edges faint gray
    for i, (v1, v2) in enumerate(E):
        p1, p2 = V[v1], V[v2]
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], "-", color="gray", lw=0.3, alpha=0.3)
    # short edges RED
    for i in range(len(E)):
        if lengths[i] < 5:
            v1, v2 = E[i]
            p1, p2 = V[v1], V[v2]
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]], "r-", lw=2, alpha=0.9)
    # zero-curvature edges BLUE
    zero_curv = np.where(tissue.E_curvature == 0)[0]
    for i in zero_curv:
        v1, v2 = E[i]
        p1, p2 = V[v1], V[v2]
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], "b-", lw=2.5, alpha=0.9)
    # 4-way / 5-way vertices
    deg4_verts = [v for v, d in degree.items() if d == 4]
    deg5_verts = [v for v, d in degree.items() if d >= 5]
    if deg4_verts:
        pts4 = V[deg4_verts]
        ax.scatter(
            pts4[:, 0], pts4[:, 1], c="orange", s=30, zorder=5,
            edgecolors="k", linewidths=0.5,
            label=f"4-way ({len(deg4_verts)})",
        )
    if deg5_verts:
        pts5 = V[deg5_verts]
        ax.scatter(
            pts5[:, 0], pts5[:, 1], c="magenta", s=50, zorder=5,
            edgecolors="k", linewidths=0.5,
            label=f"5-way ({len(deg5_verts)})",
        )
    ax.legend(fontsize=11, loc="upper right")
    ax.set_title(
        f"Problem Map\n"
        f"Red: edges<5px ({int(np.sum(lengths < 5))}), "
        f"Blue: zero curv ({len(zero_curv)}), "
        f"Orange/Magenta: deg≥4",
        fontsize=12,
    )
    ax.axis("off")

    # ── Panel (0,2): Edge-length histogram ───────────────────────────────
    ax = axes[0, 2]
    ax.hist(lengths, bins=50, color="steelblue", edgecolor="k", alpha=0.8)
    ax.axvline(3, color="red", ls="--", lw=2, label="3 px")
    ax.axvline(5, color="orange", ls="--", lw=2, label="5 px")
    ax.set_xlabel("Edge length (px)", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title(
        f"Edge Length Distribution\n"
        f"(min={lengths.min():.1f}, median={np.median(lengths):.1f})",
        fontsize=14,
    )
    ax.legend(fontsize=11)

    # ── Panels (1,0)–(1,2): Zoomed problem regions ──────────────────────
    problem_zones = []
    for v_idx in deg4_verts[:3]:
        problem_zones.append((V[v_idx, 0], V[v_idx, 1], f"4-way V{v_idx}"))
    short_idx = np.where(lengths < 5)[0]
    for si in short_idx:
        v1, v2 = E[si]
        mx = (V[v1, 0] + V[v2, 0]) / 2
        my = (V[v1, 1] + V[v2, 1]) / 2
        if 50 < mx < labels.shape[1] - 50 and 50 < my < labels.shape[0] - 50:
            problem_zones.append((mx, my, f"Short edge {si} ({lengths[si]:.1f}px)"))
            if len(problem_zones) >= 6:
                break

    for idx, (cx, cy, title) in enumerate(problem_zones[:3]):
        ax = axes[1, idx]
        hw = 60
        x0, x1 = int(cx - hw), int(cx + hw)
        y0, y1 = int(cy - hw), int(cy + hw)

        ax.imshow(img_smooth, cmap="gray", alpha=0.6)
        ax.set_xlim(x0, x1)
        ax.set_ylim(y1, y0)

        for i, (v1, v2) in enumerate(E):
            p1, p2 = V[v1], V[v2]
            in_view = (x0 < p1[0] < x1 and y0 < p1[1] < y1) or (
                x0 < p2[0] < x1 and y0 < p2[1] < y1
            )
            if not in_view:
                continue
            if lengths[i] < 5:
                color, lw = "red", 2.5
            elif tissue.E_curvature[i] == 0:
                color, lw = "blue", 2.5
            else:
                color, lw = "cyan", 1.0
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]], "-", color=color, lw=lw, alpha=0.9)
            if lengths[i] < 5:
                mx = (p1[0] + p2[0]) / 2
                my = (p1[1] + p2[1]) / 2
                ax.annotate(
                    f"{lengths[i]:.1f}",
                    (mx, my),
                    fontsize=7,
                    color="yellow",
                    ha="center",
                    bbox=dict(boxstyle="round,pad=0.1", facecolor="black", alpha=0.7),
                )

        for v_idx in range(len(V)):
            if x0 < V[v_idx, 0] < x1 and y0 < V[v_idx, 1] < y1:
                d = degree.get(v_idx, 0)
                if d >= 4:
                    ax.scatter(
                        [V[v_idx, 0]], [V[v_idx, 1]],
                        c="orange", s=80, zorder=5, edgecolors="k", linewidths=1,
                    )
                    ax.annotate(
                        f"d={d}",
                        (V[v_idx, 0] + 3, V[v_idx, 1] - 3),
                        fontsize=8, color="orange", weight="bold",
                    )
                elif d == 3:
                    ax.scatter(
                        [V[v_idx, 0]], [V[v_idx, 1]],
                        c="lime", s=25, zorder=5, edgecolors="k", linewidths=0.5,
                    )
                elif d <= 2:
                    ax.scatter(
                        [V[v_idx, 0]], [V[v_idx, 1]],
                        c="red", s=40, zorder=5, marker="x",
                    )
        ax.set_title(title, fontsize=12)
        ax.axis("off")

    n_4 = sum(1 for d in degree.values() if d == 4)
    plt.suptitle(
        f"test.tif — Force Inference Diagnostics\n"
        f"{len(V)} vertices, {len(E)} edges, {n_4} four-way, "
        f"{int(np.sum(lengths < 5))} short edges",
        fontsize=16,
    )
    plt.tight_layout()
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved → {out}")


if __name__ == "__main__":
    run()
