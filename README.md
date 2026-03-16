# ForceInferencePy

A robust Python package for force inference in 2D and 2.5D tissues.

## Overview

This package provides tools for extracting tissue topology from segmentation masks and inferring mechanical forces (tensions and pressures) using Bayesian and Young-Laplace solvers.

## Key Features

- **Label-Driven Topology Extraction**: Robustly handles complex junctions (twin junctions) without skeletonization artifacts.
- **Multiple Solvers**: 
  - **Bayesian Solver**: For robust inference even with noisy data.
  - **Laplace Solver**: For curved interfaces.
- **Geometry Processing**: Curvature calculation and analytical tangents.
- **Visualization**: Tools for plotting tensions, pressures, and stress tensors.

## Installation

```bash
pip install .
```

For development:
```bash
pip install -e .
```

## Quick Start

See [QUICK_START.md](QUICK_START.md) for a fast introduction.

For more details on the topology extraction algorithm, see [LABEL_DRIVEN_TOPOLOGY_README.md](LABEL_DRIVEN_TOPOLOGY_README.md).

## Project Structure

- `force_inference/`: Core library code.
- `examples/`: Demonstration scripts and diagnostics.
- `tests/`: Unit and integration tests (using `pytest`).
- `data/`: Example tissue images.

## Testing

Run tests with `pytest`:
```bash
pytest tests/
```

## License

MIT
