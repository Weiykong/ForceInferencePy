"""
Diagnostic 2: Twin Junction Comparison
Side-by-side zoomed views comparing topology with and without tiny-twin collapse.
Top row = default (stubs only), bottom row = stubs + tiny twin collapse.

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

from force_inference import geometry, segmentation
from force_inference.topology_label import extract_topology_label


def run(filename="../data/test.tif", out="twin_junction_comparison.png"):
    # ── 1. Segment ──────────────────────────────────────────────────────
    labels, img_smooth = segmentation.segment_grayscale(
        filename, h_depth=2.0, min_cell_size=5
    )

    # ── 2. Two topology configs ─────────────────────────────────────────
    t1 = extract_topology_label(
        labels,
        use_skeleton_geometry=False,
        collapse_stubs=True,
        collapse_tiny_twins=False,
    )
    t1 = geometry.compute_curvature(t1)

    t2 = extract_topology_label(
        labels,
        use_skeleton_geometry=False,
        collapse_stubs=True,
        collapse_tiny_twins=True,
        tiny_twin_threshold=3.0,
    )
    t2 = geometry.compute_curvature(t2)

    V1, E1 = t1.V[:, :2], t1.E
    V2, E2 = t2.V[:, :2], t2.E

    # ── 3. Find short-edge locations in t1 to zoom on ───────────────────
    lengths1 = np.array(
        [np.linalg.norm(V1[E1[i, 1]] - V1[E1[i, 0]]) for i in range(len(E1))]
    )
    H, W = labels.shape
    candidates = []
    for si in np.where(lengths1 < 3.0)[0]:
        v1, v2 = E1[si]
        mx = (V1[v1, 0] + V1[v2, 0]) / 2
        my = (V1[v1, 1] + V1[v2, 1]) / 2
        if 80 < mx < W - 80 and 80 < my < H - 80:
            candidates.append((si, mx, my, lengths1[si]))
    candidates.sort(key=lambda x: x[3])

    if len(candidates) < 4:
        # pad with 4-way vertices from t1
        deg1 = Counter()
        for v1, v2 in E1:
            deg1[v1] += 1
            deg1[v2] += 1
        for v_idx in sorted(deg1, key=lambda v: -deg1[v]):
            if deg1[v_idx] >= 4 and 80 < V1[v_idx, 0] < W - 80 and 80 < V1[v_idx, 1] < H - 80:
                candidates.append((-1, V1[v_idx, 0], V1[v_idx, 1], 0))
                if len(candidates) >= 4:
                    break

    # ── 4. Build figure ──────────────────────────────────────────────────
    n_cols = min(4, len(candidates))
    fig, axes = plt.subplots(2, n_cols, figsize=(6 * n_cols, 12))
    if n_cols == 1:
        axes = axes[:, np.newaxis]

    configs = [
        (t1, V1, E1, "Default (stub collapse only)"),
        (t2, V2, E2, "+ tiny twin collapse"),
    ]

    for row, (tissue, V, E, label_text) in enumerate(configs):
        deg = Counter()
        for v1, v2 in E:
            deg[v1] += 1
            deg[v2] += 1
        edge_lengths = np.array(
            [np.linalg.norm(V[E[i, 1]] - V[E[i, 0]]) for i in range(len(E))]
        )

        for col, (si, cx, cy, elen) in enumerate(candidates[:n_cols]):
            ax = axes[row, col]
            hw = 50
            x0, x1 = int(cx - hw), int(cx + hw)
            y0, y1 = int(cy - hw), int(cy + hw)

            ax.imshow(img_smooth, cmap="gray", alpha=0.6)
            ax.set_xlim(x0, x1)
            ax.set_ylim(y1, y0)

            for i, (v1, v2) in enumerate(E):
                p1, p2 = V[v1], V[v2]
                in_view = (
                    x0 - 10 < p1[0] < x1 + 10 and y0 - 10 < p1[1] < y1 + 10
                ) or (
                    x0 - 10 < p2[0] < x1 + 10 and y0 - 10 < p2[1] < y1 + 10
                )
                if not in_view:
                    continue

                is_short = edge_lengths[i] < 5
                is_zero_curv = tissue.E_curvature[i] == 0

                if is_short:
                    color, lw = "red", 3.0
                elif is_zero_curv:
                    color, lw = "blue", 2.5
                else:
                    color, lw = "cyan", 1.2
                ax.plot(
                    [p1[0], p2[0]], [p1[1], p2[1]],
                    "-", color=color, lw=lw, alpha=0.9,
                )
                if is_short:
                    mx = (p1[0] + p2[0]) / 2
                    my = (p1[1] + p2[1]) / 2
                    ax.annotate(
                        f"{edge_lengths[i]:.1f}px",
                        (mx + 2, my),
                        fontsize=8,
                        color="yellow",
                        bbox=dict(
                            boxstyle="round,pad=0.1",
                            facecolor="black",
                            alpha=0.7,
                        ),
                    )

            for v_idx in range(len(V)):
                if x0 - 5 < V[v_idx, 0] < x1 + 5 and y0 - 5 < V[v_idx, 1] < y1 + 5:
                    d = deg.get(v_idx, 0)
                    if d >= 4:
                        ax.scatter(
                            [V[v_idx, 0]], [V[v_idx, 1]],
                            c="orange", s=100, zorder=5,
                            edgecolors="k", linewidths=1.5,
                        )
                        ax.annotate(
                            f"{d}",
                            (V[v_idx, 0] + 4, V[v_idx, 1] - 4),
                            fontsize=10, color="orange", weight="bold",
                        )
                    elif d == 3:
                        ax.scatter(
                            [V[v_idx, 0]], [V[v_idx, 1]],
                            c="lime", s=30, zorder=5,
                            edgecolors="k", linewidths=0.5,
                        )
                    elif d == 1:
                        ax.scatter(
                            [V[v_idx, 0]], [V[v_idx, 1]],
                            c="white", s=30, zorder=5,
                            edgecolors="red", linewidths=1,
                        )

            n_short = int(np.sum(edge_lengths < 5))
            n_4way = sum(1 for d in deg.values() if d == 4)
            ax.set_title(
                f"{label_text}\nE={len(E)}, short<5={n_short}, 4-way={n_4way}",
                fontsize=11,
            )
            ax.axis("off")

    plt.suptitle(
        "Twin Junction Handling: Before vs After Collapse\n"
        "Red = short edges, Orange = 4-way vertices, Green = 3-way vertices",
        fontsize=14,
    )
    plt.tight_layout()
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved → {out}")


if __name__ == "__main__":
    run()
