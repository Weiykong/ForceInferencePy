"""
Diagnostic 3: Solver Diagnostics
Four-panel figure: tension map, pressure map, per-vertex force-balance
residual, and edge-length vs tension scatter.

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


def run(filename="../data/test.tif", out="solver_diagnostics.png"):
    # ── 1. Full pipeline ────────────────────────────────────────────────
    labels, img_smooth = segmentation.segment_grayscale(
        filename, h_depth=2.0, min_cell_size=5
    )
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
    vertex_edges = {}
    for i, (v1, v2) in enumerate(E):
        degree[v1] += 1
        degree[v2] += 1
        vertex_edges.setdefault(v1, []).append(i)
        vertex_edges.setdefault(v2, []).append(i)

    lengths = np.array(
        [np.linalg.norm(V[E[i, 1]] - V[E[i, 0]]) for i in range(len(E))]
    )

    # ── per-vertex force-balance residual ────────────────────────────────
    force_res = np.full(len(V), np.nan)
    for v_idx in range(len(V)):
        if v_idx not in vertex_edges:
            continue
        edges = vertex_edges[v_idx]
        if len(edges) < 2:
            continue
        fx, fy = 0.0, 0.0
        for ei in edges:
            v1, v2 = E[ei]
            t_val = T[ei]
            if np.isnan(t_val):
                continue
            if v_idx == v1:
                t_vec = tissue.E_tangents[ei, 0]
            else:
                t_vec = tissue.E_tangents[ei, 1]
            fx += t_val * t_vec[0]
            fy += t_val * t_vec[1]
        force_res[v_idx] = np.sqrt(fx ** 2 + fy ** 2)

    # ── figure ───────────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))

    # Panel (0,0): Tension map
    ax = axes[0, 0]
    ax.imshow(img_smooth, cmap="gray", alpha=0.4)
    for i, (v1, v2) in enumerate(E):
        p1, p2 = V[v1], V[v2]
        t_norm = (T[i] - T.min()) / (T.max() - T.min() + 1e-9)
        color = plt.cm.turbo(t_norm)
        ax.plot(
            [p1[0], p2[0]], [p1[1], p2[1]],
            "-", color=color, lw=max(0.3, T[i] * 1.5), alpha=0.85,
        )
    ax.set_title(f"Inferred Tensions\n[{T.min():.3f}, {T.max():.3f}]", fontsize=13)
    ax.axis("off")

    # Panel (0,1): Pressure map
    ax = axes[0, 1]
    ax.imshow(img_smooth, cmap="gray", alpha=0.3)
    pressure_img = np.full_like(labels, np.nan, dtype=float)
    for c in range(1, labels.max() + 1):
        if c - 1 < len(result.pressures):
            pressure_img[labels == c] = result.pressures[c - 1]
    valid = ~np.isnan(pressure_img)
    vmin, vmax = np.percentile(pressure_img[valid], [2, 98])
    masked = np.ma.masked_where(~valid, pressure_img)
    ax.imshow(masked, cmap="coolwarm", vmin=vmin, vmax=vmax, alpha=0.7)
    ax.set_title(
        f"Inferred Pressures\n"
        f"[{result.pressures.min():.3f}, {result.pressures.max():.3f}]",
        fontsize=13,
    )
    ax.axis("off")

    # Panel (1,0): Force-balance residual per vertex
    ax = axes[1, 0]
    ax.imshow(img_smooth, cmap="gray", alpha=0.4)
    valid_v = ~np.isnan(force_res)
    vmax_r = np.percentile(force_res[valid_v], 95)
    for v_idx in range(len(V)):
        if np.isnan(force_res[v_idx]):
            continue
        r_norm = min(1.0, force_res[v_idx] / (vmax_r + 1e-9))
        d = degree.get(v_idx, 0)
        color = plt.cm.hot(r_norm)
        size = 15 if d == 3 else (40 if d == 4 else 60)
        marker = "o" if d == 3 else ("s" if d == 4 else "D")
        ax.scatter(
            [V[v_idx, 0]], [V[v_idx, 1]],
            c=[color], s=size, zorder=5,
            edgecolors="none", marker=marker, alpha=0.8,
        )
    ax.set_title(
        "Force Balance Residual\n(squares = 4-way, diamonds = 5-way)", fontsize=13
    )
    ax.axis("off")

    # Panel (1,1): Edge length vs tension scatter
    ax = axes[1, 1]
    max_deg_per_edge = [
        max(degree.get(E[i, 0], 0), degree.get(E[i, 1], 0))
        for i in range(len(E))
    ]
    colors = [
        "red" if d >= 4 else ("orange" if d == 3 else "gray")
        for d in max_deg_per_edge
    ]
    ax.scatter(lengths, T, c=colors, s=10, alpha=0.5)
    ax.set_xlabel("Edge length (px)", fontsize=12)
    ax.set_ylabel("Inferred tension", fontsize=12)
    ax.axvline(5, color="red", ls="--", alpha=0.5, label="5 px")
    ax.legend()
    ax.set_title("Edge Length vs Tension\n(red = has 4-way endpoint)", fontsize=13)

    plt.suptitle(
        f"test.tif — Solver Results\nResidual: {result.residual:.2f}", fontsize=15
    )
    plt.tight_layout()
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved → {out}")


if __name__ == "__main__":
    run()
