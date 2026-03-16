import numpy as np
from dataclasses import dataclass
from typing import List, Optional

@dataclass
class Tissue:
    """
    Unified Data Structure for 2D and 2.5D Tissues.
    """
    V: np.ndarray             # (N, 3) Vertices [x, y, z]. For 2D, z=0.
    E: np.ndarray             # (M, 2) Edges [v1_idx, v2_idx]
    E_cells: np.ndarray       # (M, 2) Cell Neighbors [c1_idx, c2_idx]
    C_centroids: np.ndarray   # (C, 2) Cell Centroids [x, y]
    C_v: List[List[int]]      # List of vertex indices for each cell (CCW)
    labels: np.ndarray        # The segmentation mask image
    
    # Optional fields for advanced features
    E_pixels: Optional[List[np.ndarray]] = None  # Pixel paths for curvy edges
    E_curvature: Optional[np.ndarray] = None     # Curvature values (kappa)
    
    def to_2d(self) -> np.ndarray:
        """Return vertex coordinates as an (N, 2) array (x, y only).

        Convenience method for 2-D plotting where the z-coordinate is
        not needed.

        Returns:
            Array of shape (N, 2) containing the x and y columns of ``V``.
        """
        return self.V[:, :2]

@dataclass
class ForceResult:
    tensions: np.ndarray      # (M,)
    pressures: np.ndarray     # (C,)
    residual: float
    stress_tensors: Optional[np.ndarray] = None # (C, 2, 2)