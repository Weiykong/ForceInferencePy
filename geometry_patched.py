import numpy as np
from skimage import measure, morphology
import logging
from typing import Tuple, Optional
from scipy.spatial import cKDTree

# Relative import
from .core import Tissue
from .core import ForceResult

logger = logging.getLogger("ForceInference.Geometry")


def map_z_to_vertices(
    tissue: Tissue,
    stack: np.ndarray,
    xy_radius: int = 1,
    z_fallback: float = 0.0,
) -> Tissue:
    """
    Map a 3D image stack (Z, Y, X) onto tissue vertices by sampling the
    brightest Z index near each vertex (x, y).

    This updates ``tissue.V[:, 2]`` in-place and returns the same tissue object.
    """
    if stack.ndim != 3:
        raise ValueError(
            f"Expected stack with shape (Z, Y, X), got shape {stack.shape}"
        )
    if tissue.V.ndim != 2 or tissue.V.shape[1] < 3:
        raise ValueError("tissue.V must have shape (N, 3)")

    z_dim, h, w = stack.shape
    if z_dim == 0 or h == 0 or w == 0:
        tissue.V[:, 2] = z_fallback
        return tissue

    verts_xy = tissue.V[:, :2]
    z_values = np.full(len(verts_xy), float(z_fallback), dtype=float)

    for i, (x, y) in enumerate(verts_xy):
        xi = int(round(float(x)))
        yi = int(round(float(y)))

        if xi < 0 or xi >= w or yi < 0 or yi >= h:
            continue

        x0 = max(0, xi - xy_radius)
        x1 = min(w, xi + xy_radius + 1)
        y0 = max(0, yi - xy_radius)
        y1 = min(h, yi + xy_radius + 1)

        column = stack[:, y0:y1, x0:x1]
        if column.size == 0:
            continue

        # Aggregate the local patch so sparse membrane voxels still produce a
        # stable Z estimate for the vertex.
        profile = column.max(axis=(1, 2))
        if np.max(profile) <= 0:
            continue

        z_values[i] = float(np.argmax(profile))

    tissue.V[:, 2] = z_values
    return tissue


def calculate_batchelor_stress(tissue: Tissue, result: ForceResult) -> ForceResult:
    """
    Compute a per-cell 2x2 stress tensor using a simple Batchelor-style estimate:

        sigma_c = -P_c I + (1 / A_c) * sum_e (T_e * L_e * u_e outer u_e)

    where the edge sum runs over edges touching cell c.
    """
    n_cells = len(tissue.C_v)
    if n_cells == 0:
        result.stress_tensors = np.zeros((0, 2, 2), dtype=float)
        return result

    Vxy = tissue.V[:, :2]
    areas = _compute_cell_areas(tissue)
    stresses = np.zeros((n_cells, 2, 2), dtype=float)

    # Pressure term (pressures are stored label-indexed: label 1 -> index 0)
    pressures = getattr(result, "pressures", None)
    if pressures is not None:
        n_p = min(len(pressures), n_cells)
        for ci in range(n_p):
            stresses[ci] -= float(pressures[ci]) * np.eye(2)

    # Edge contribution: add to each neighboring cell.
    for e_idx, (v1, v2) in enumerate(tissue.E):
        p1 = Vxy[v1]
        p2 = Vxy[v2]
        d = p2 - p1
        L = float(np.linalg.norm(d))
        if L <= 1e-12:
            continue

        u = d / L
        contrib = float(result.tensions[e_idx]) * L * np.outer(u, u)

        for lbl in tissue.E_cells[e_idx]:
            ci = int(lbl) - 1
            if ci < 0 or ci >= n_cells:
                continue
            A = max(float(areas[ci]), 1e-12)
            stresses[ci] += contrib / A

    result.stress_tensors = stresses
    return result


def interpolate_stress_to_grid(
    tissue: Tissue,
    result: ForceResult,
    grid_size: int = 50,
    smoothing_sigma: Optional[float] = None,
) -> Tuple[Optional[Tuple[np.ndarray, np.ndarray]], Optional[np.ndarray]]:
    """
    Coarse-grain cell stress tensors onto a regular XY grid by Gaussian-weighted
    averaging of nearby cell-centered tensors.
    """
    if result is None or result.stress_tensors is None:
        return None, None

    tensors = np.asarray(result.stress_tensors, dtype=float)
    if tensors.ndim != 3 or tensors.shape[1:] != (2, 2) or len(tensors) == 0:
        return None, None

    H, W = tissue.labels.shape
    step = max(int(grid_size), 1)
    sigma = float(smoothing_sigma if smoothing_sigma is not None else 1.5 * step)
    sigma = max(sigma, 1e-6)

    xs = np.arange(step / 2.0, W, step, dtype=float)
    ys = np.arange(step / 2.0, H, step, dtype=float)
    if len(xs) == 0 or len(ys) == 0:
        return None, None

    grid_x, grid_y = np.meshgrid(xs, ys)
    grid_tensors = np.zeros(grid_x.shape + (2, 2), dtype=float)

    centroids = np.asarray(tissue.C_centroids, dtype=float)
    if len(centroids) == 0:
        return (grid_x, grid_y), grid_tensors

    # Keep only cells with meaningful centroids and finite tensors.
    valid_centroid = np.isfinite(centroids).all(axis=1)
    valid_centroid &= np.any(centroids != 0, axis=1)
    valid_tensor = np.isfinite(tensors).all(axis=(1, 2))
    valid = valid_centroid & valid_tensor
    if not np.any(valid):
        return None, None

    pts = centroids[valid]
    vals = tensors[valid]
    tree = cKDTree(pts)
    radius = max(2.0 * sigma, step)

    for i in range(grid_x.shape[0]):
        for j in range(grid_x.shape[1]):
            q = np.array([grid_x[i, j], grid_y[i, j]])
            qx = int(round(float(q[0])))
            qy = int(round(float(q[1])))
            if qx < 0 or qx >= W or qy < 0 or qy >= H:
                continue
            # Do not place coarse-grained stress on background/outside tissue.
            if tissue.labels[qy, qx] == 0:
                continue
            idxs = tree.query_ball_point(q, r=radius)
            if len(idxs) == 0:
                continue

            neigh_pts = pts[idxs]
            d2 = np.sum((neigh_pts - q) ** 2, axis=1)
            w = np.exp(-0.5 * d2 / (sigma ** 2))
            w_sum = float(np.sum(w))
            if w_sum <= 1e-12:
                continue

            neigh_tensors = vals[idxs]
            grid_tensors[i, j] = np.tensordot(w, neigh_tensors, axes=(0, 0)) / w_sum

    return (grid_x, grid_y), grid_tensors

def compute_curvature(tissue: Tissue) -> Tissue:
    """
    Geometry Pipeline:
    1. Trace Pixels
    2. Fit Circle (Get Curvature & Center)
    3. Compute ANALYTICAL Tangents from that Circle (Geometric Consistency)
    """
    tissue = _trace_and_measure_curvature(tissue)
    tissue = _compute_analytical_tangents(tissue)
    return tissue

def _trace_and_measure_curvature(tissue: Tissue) -> Tissue:
    """
    Pixel tracing + Circle Fitting.

    Uses pre-existing E_pixels from the topology extractor when available
    (these are the actual boundary pixels found during topology extraction).
    Falls back to dilation-based re-tracing only when E_pixels is absent.

    This fixes the problem where short edges between close junctions get
    zero boundary pixels from ``dilation(cell1, disk(1)) & cell2`` — the
    topology extractor already found those pixels but the old code threw
    them away and re-traced from scratch.
    """
    logger.info("Tracing pixels and fitting circles...")
    labels = tissue.labels
    H, W = labels.shape
    curvature_list = []

    # Preserve existing E_pixels from topology extraction
    existing_pixels = getattr(tissue, 'E_pixels', None)
    has_existing = (
        existing_pixels is not None
        and len(existing_pixels) == len(tissue.E)
    )

    tissue.E_circles = []
    new_E_pixels = []

    objects = measure.regionprops(labels)
    bboxes = {obj.label: obj.bbox for obj in objects}

    for i, (v1, v2) in enumerate(tissue.E):
        c1, c2 = tissue.E_cells[i]

        # --- 1. Get boundary pixels ---
        # Priority: use existing E_pixels from topology extraction
        points = None
        if has_existing and len(existing_pixels[i]) >= 3:
            points = np.asarray(existing_pixels[i], dtype=float)

        # Fallback: dilation-based re-tracing
        if points is None:
            if c1 in bboxes and c2 in bboxes:
                b1, b2 = bboxes[c1], bboxes[c2]
                min_r = max(0, min(b1[0], b2[0]) - 2)
                max_r = min(H, max(b1[2], b2[2]) + 2)
                min_c = max(0, min(b1[1], b2[1]) - 2)
                max_c = min(W, max(b1[3], b2[3]) + 2)
                sub_lbl = labels[min_r:max_r, min_c:max_c]
                m1 = (sub_lbl == c1)
                m2 = (sub_lbl == c2)
                boundary_mask = morphology.dilation(m1, morphology.disk(1)) & m2
                pts_y, pts_x = np.where(boundary_mask)
                if len(pts_y) >= 3:
                    points = np.column_stack(
                        (pts_x + min_c, pts_y + min_r)
                    ).astype(float)

        # Still nothing → zero curvature, keep whatever pixels we have
        if points is None or len(points) < 3:
            curvature_list.append(0.0)
            kept = (
                existing_pixels[i] if has_existing and len(existing_pixels[i]) > 0
                else np.zeros((0, 2))
            )
            new_E_pixels.append(kept)
            tissue.E_circles.append(None)
            continue

        new_E_pixels.append(points)

        # --- 2. Fit Circle ---
        kappa, center, radius = _fit_circle_parameters_full(points)

        # --- 3. Sign (Distance Method) ---
        if c1 > 0 and c1 <= len(tissue.C_centroids):
            cent = tissue.C_centroids[c1 - 1]
            sign = _get_curvature_sign_distance(points, cent)
        else:
            sign = 1.0

        curvature_list.append(kappa * sign)
        tissue.E_circles.append(
            {'center': center, 'radius': radius, 'sign': sign}
        )

    tissue.E_curvature = np.array(curvature_list)
    tissue.E_pixels = new_E_pixels
    return tissue

def _compute_analytical_tangents(tissue: Tissue) -> Tissue:
    """
    Computes tangents consistent with the fitted circle.
    Tangent is perpendicular to the radius vector (Center -> Vertex).
    """
    logger.info("Computing analytical circle tangents...")
    n_edges = len(tissue.E)
    tangents = np.zeros((n_edges, 2, 2))
    
    for i, (v1_idx, v2_idx) in enumerate(tissue.E):
        # Fallback to straight line if fit failed
        p1 = tissue.V[v1_idx, :2]
        p2 = tissue.V[v2_idx, :2]
        vec_straight = p2 - p1
        len_s = np.linalg.norm(vec_straight) + 1e-9
        u_straight = vec_straight / len_s
        
        circ = tissue.E_circles[i]
        
        # CHECK: If curvature is tiny, use straight line (Circle math unstable for R -> inf)
        kappa = abs(tissue.E_curvature[i])
        if circ is None or kappa < 0.01 or circ['radius'] > 500:
            tangents[i, 0] = u_straight    # p1 -> p2
            tangents[i, 1] = -u_straight   # p2 -> p1
            continue
            
        # Circle Logic
        # Tangent is perpendicular to Radius vector (Center -> Vertex)
        # Direction depends on which way the arc goes (Clockwise vs CCW)
        
        # 1. Vector Center -> P1
        R1 = p1 - circ['center']
        # Tangent is orthogonal: (-y, x) or (y, -x)
        # We need the one that points TOWARDS P2 (along the arc)
        t1_cand_a = np.array([-R1[1], R1[0]]) 
        t1_cand_b = np.array([R1[1], -R1[0]])
        
        # Normalize
        t1_cand_a /= (np.linalg.norm(t1_cand_a) + 1e-9)
        t1_cand_b /= (np.linalg.norm(t1_cand_b) + 1e-9)
        
        # Pick the one pointing somewhat towards P2
        if np.dot(t1_cand_a, vec_straight) > 0:
            tangents[i, 0] = t1_cand_a
        else:
            tangents[i, 0] = t1_cand_b
            
        # 2. Vector Center -> P2
        R2 = p2 - circ['center']
        t2_cand_a = np.array([-R2[1], R2[0]])
        t2_cand_b = np.array([R2[1], -R2[0]])
        t2_cand_a /= (np.linalg.norm(t2_cand_a) + 1e-9)
        t2_cand_b /= (np.linalg.norm(t2_cand_b) + 1e-9)
        
        # At P2, we need vector pointing TOWARDS P1 (inward tension)
        # Vector P2->P1 is -vec_straight
        if np.dot(t2_cand_a, -vec_straight) > 0:
            tangents[i, 1] = t2_cand_a
        else:
            tangents[i, 1] = t2_cand_b
            
    tissue.E_tangents = tangents
    return tissue

def _fit_circle_parameters_full(
    points: np.ndarray,
) -> Tuple[float, np.ndarray, float]:
    """Fit a circle to a set of 2-D points using least squares.

    Args:
        points: (N, 2) array of x/y coordinates.

    Returns:
        Tuple of (kappa, center, radius) where kappa = 1/radius.
        Returns (0.0, zeros(2), 0.0) when the fit fails or is degenerate.
    """
    x = points[:, 0]
    y = points[:, 1]
    A = np.column_stack((x, y, np.ones_like(x)))
    b = -(x**2 + y**2)
    try:
        C, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
        cx, cy = -C[0] / 2, -C[1] / 2
        R = np.sqrt(max(cx**2 + cy**2 - C[2], 0.0))
        if R < 1e-3:
            return 0.0, np.zeros(2), 0.0
        return 1.0 / R, np.array([cx, cy]), R
    except (np.linalg.LinAlgError, ValueError):
        return 0.0, np.zeros(2), 0.0


def _get_curvature_sign_distance(points: np.ndarray, cell_cent: np.ndarray) -> float:
    """Determine the sign of curvature using a midpoint-distance heuristic.

    Positive when the arc midpoint is farther from the cell centroid than the
    chord midpoint (i.e. the membrane bulges away from the cell centre).

    Args:
        points: (N, 2) ordered boundary pixel coordinates.
        cell_cent: (2,) centroid of the adjacent cell.

    Returns:
        +1.0 or -1.0.
    """
    mid_arc = points[len(points) // 2]
    mid_chord = (points[0] + points[-1]) / 2
    dist_arc = np.linalg.norm(mid_arc - cell_cent)
    dist_chord = np.linalg.norm(mid_chord - cell_cent)
    return 1.0 if dist_arc > dist_chord else -1.0


def _compute_cell_areas(tissue: Tissue) -> np.ndarray:
    """Estimate cell areas from polygons; fall back to label-pixel counts."""
    n_cells = len(tissue.C_v)
    areas = np.zeros(n_cells, dtype=float)
    Vxy = tissue.V[:, :2]

    for ci, verts in enumerate(tissue.C_v):
        if verts is None or len(verts) < 3:
            continue
        poly = Vxy[np.asarray(verts, dtype=int)]
        x = poly[:, 0]
        y = poly[:, 1]
        areas[ci] = 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))

    # Fallback for cells without valid polygon area.
    if getattr(tissue, "labels", None) is not None and np.any(areas <= 1e-12):
        labels = tissue.labels
        for ci in np.where(areas <= 1e-12)[0]:
            lbl = ci + 1
            areas[ci] = float(np.count_nonzero(labels == lbl))

    return np.maximum(areas, 1e-12)
