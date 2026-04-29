import numpy as np
from dataclasses import dataclass
from typing import List, Optional

@dataclass
class Tissue:
    """Unified graph representation for a 2D or 2.5D epithelial tissue.

    Attributes:
        V: (N, 3) vertex positions [x, y, z]. z=0 for 2-D tissues.
        E: (M, 2) edge vertex indices [v1_idx, v2_idx].
        E_cells: (M, 2) cell label pairs bordering each edge [c1, c2].
        C_centroids: (C, 2) cell centroids [x, y].
        C_v: vertex indices per cell in counter-clockwise order.
        labels: 2-D segmentation mask (H, W) with integer cell labels.
        E_pixels: optional per-edge pixel paths [(K_i, 2), ...].
        E_curvature: optional (M,) curvature values κ; populated by
            ``compute_curvature``.
        E_synthetic: optional bool (M,) mask; True for short edges
            inserted by ``split_high_degree_vertices``.
        E_tangents: optional (M, 2, 2) unit tangent vectors at each
            endpoint; populated by ``compute_curvature``.
        E_circles: optional list of circle-fit metadata per edge;
            populated by ``compute_curvature``.
    """
    V: np.ndarray
    E: np.ndarray
    E_cells: np.ndarray
    C_centroids: np.ndarray
    C_v: List[List[int]]
    labels: np.ndarray

    E_pixels: Optional[List[np.ndarray]] = None
    E_curvature: Optional[np.ndarray] = None
    E_synthetic: Optional[np.ndarray] = None

    @property
    def n_vertices(self) -> int:
        """Number of vertices in the tissue graph."""
        return len(self.V)

    @property
    def n_edges(self) -> int:
        """Number of edges (cell-cell interfaces) in the tissue graph."""
        return len(self.E)

    @property
    def n_cells(self) -> int:
        """Number of cells (polygons) in the tissue graph."""
        return len(self.C_v)

    def to_2d(self) -> np.ndarray:
        """Return vertex coordinates as an (N, 2) array (x, y only).

        Returns:
            Array of shape (N, 2) containing the x and y columns of ``V``.
        """
        return self.V[:, :2]

    def __repr__(self) -> str:
        return (
            f"Tissue(n_vertices={self.n_vertices}, n_edges={self.n_edges}, "
            f"n_cells={self.n_cells}, labels={self.labels.shape})"
        )


@dataclass
class ForceResult:
    """Inferred mechanical quantities for a tissue.

    Attributes:
        tensions: (M,) inferred line tensions per edge. Edges excluded
            from the solve (e.g. border edges) carry ``nan``.
        pressures: (C,) inferred relative pressures per cell, mean-centred.
        residual: RMS residual of the force-balance system.
        stress_tensors: optional (C, 2, 2) per-cell stress tensors;
            populated by ``calculate_batchelor_stress``.
    """
    tensions: np.ndarray
    pressures: np.ndarray
    residual: float
    stress_tensors: Optional[np.ndarray] = None

    def summary(self) -> str:
        """Return a human-readable summary of the inferred forces.

        Returns:
            Multi-line string with statistics on tensions and pressures.
        """
        valid = self.tensions[np.isfinite(self.tensions)]
        lines = [
            f"ForceResult summary",
            f"  edges solved : {len(valid)} / {len(self.tensions)}",
            f"  tension      : mean={valid.mean():.4g}  std={valid.std():.4g}"
            f"  min={valid.min():.4g}  max={valid.max():.4g}" if len(valid) else "  tension      : (no valid edges)",
            f"  pressure     : mean={self.pressures.mean():.4g}  std={self.pressures.std():.4g}"
            f"  range=[{self.pressures.min():.4g}, {self.pressures.max():.4g}]",
            f"  residual     : {self.residual:.4g}",
        ]
        if self.stress_tensors is not None:
            lines.append(f"  stress tensors: computed for {len(self.stress_tensors)} cells")
        return "\n".join(lines)