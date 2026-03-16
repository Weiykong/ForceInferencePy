# Changelog

All notable changes to ForceInferencePy are documented here.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

---

## [0.1.0] — 2024 (Initial Release)

### Added

**Core data structures** (`force_inference/core.py`)
- `Tissue` dataclass — unified container for 2D/2.5D tissue topology (vertices, edges, cell neighbours, label mask).
- `ForceResult` dataclass — stores inferred tensions, pressures, residual, and optional stress tensors.

**Segmentation** (`force_inference/segmentation.py`)
- `segment_grayscale` — h-minima watershed pipeline for membrane-labelled images (TIFF, PNG, JPEG).

**Topology extraction** (`force_inference/topology.py`, `force_inference/topology_label.py`)
- Skeleton-based topology extraction (`extract_topology`).
- Label-driven topology extraction (`extract_topology_label`) — handles twin junctions without skeleton merging artefacts; supports stub collapse, tiny-twin promotion, spline resampling, and sub-pixel vertex snapping.

**Geometry** (`force_inference/geometry.py`)
- `compute_curvature` — pixel tracing + circle fitting + analytical tangent computation.
- `map_z_to_vertices` — maps brightest-Z position onto tissue vertices for 2.5D stacks.
- `calculate_batchelor_stress` — per-cell 2×2 stress tensor via Batchelor formula.
- `interpolate_stress_to_grid` — Gaussian-weighted coarse-graining of cell stress onto a regular grid.

**Solvers** (`force_inference/solvers.py`)
- `solve_bayesian` — Bayesian force inference with automatic μ selection via log-evidence maximisation.
- `solve_laplace` — Laplace-pressure solver with border-cell atmosphere treatment.
- `BayesianScanResult` — structured result for μ-scan output.

**Visualization** (`force_inference/visualization.py`)
- Tension overlay, pressure map, stress ellipse, and topology diagnostic plots.

**Examples** (`examples/`)
- `demo_2d_bayesian.py`, `demo_laplace.py`, `demo_stress_analysis.py`, `demo_25d_stack.py`.
- Diagnostic scripts: `diagnose_junctions.py`, `diagnose_tif_vs_jpg.py`, `compare_methods.py`.

**Documentation**
- `README.md`, `QUICK_START.md`, `LABEL_DRIVEN_TOPOLOGY_README.md`.
