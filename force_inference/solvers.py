import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg
import logging
from typing import Optional, List, Union
from dataclasses import dataclass

from .core import Tissue, ForceResult

logger = logging.getLogger("ForceInference.Solvers")

@dataclass
class BayesianScanResult:
    best_mu: float
    best_result: ForceResult
    mu_values: np.ndarray
    log_evidences: np.ndarray
    residuals: np.ndarray

def solve_bayesian(tissue: Tissue,
                   mu: Union[float, List[float], np.ndarray] = None,
                   exclude_border_edges: bool = True,
                   border_margin: int = 5) -> Optional[Union[ForceResult, BayesianScanResult]]:
    """
    Bayesian force inference solver.

    Args:
        tissue: Tissue object with topology
        mu: Regularization parameter (scalar or 1-D array of values to scan).
            If None, a log-spaced range is scanned automatically.
            **Do not pass a label image here** — labels are read from
            ``tissue.labels`` which is set during topology extraction.
        exclude_border_edges: If True, excludes edges connected to artificial
                              boundary vertices (recommended for cleaner results).
        border_margin: Distance in pixels from the image edge within which a
                       vertex is classified as a border vertex and excluded from
                       force balance.  The topology extractor uses 2 px; the
                       default here (5 px) is intentionally more conservative
                       to avoid artefacts from partially-visible cells at the
                       image boundary.  Set equal to the topology margin (2) to
                       include near-edge vertices in the force balance.
    """
    # Guard: mu must be a scalar, 1-D array, or None — not a 2-D image array.
    if mu is not None and isinstance(mu, np.ndarray) and mu.ndim > 1:
        raise ValueError(
            "solve_bayesian() received a 2-D array as `mu`. "
            "The API no longer takes a label image as the second positional "
            "argument — labels are read from tissue.labels automatically. "
            "Call: solve_bayesian(tissue) or solve_bayesian(tissue, mu=0.01)"
        )

    matrices = _build_bayesian_matrices(tissue, exclude_border_edges=exclude_border_edges,
                                        border_margin=border_margin)
    if matrices is None:
        return None
    A, B, g, n_eq, n_vars, real_edge_indices = matrices
    n_real_edges = len(real_edge_indices)

    if isinstance(mu, (float, int)):
        return _solve_single_mu(A, B, g, mu, n_eq, tissue, real_edge_indices)

    else:
        # Scan Range
        if mu is None:
            mu_exponents = np.arange(-10.0, -5.1, 0.25)  # 10^-10 to 10^-5.25, 19 values
            mu_values = 10.0 ** mu_exponents
        else:
            mu_values = np.array(mu)

        logger.info(f"Scanning {len(mu_values)} mu values...")

        results = []
        log_evidences = []
        residuals = []

        # Pre-compute ATA for speed
        use_exact = n_vars < 4000
        ATA = None
        if use_exact:
            A_dense = A.toarray()
            ATA = A_dense.T @ A_dense

        for val in mu_values:
            res = _solve_single_mu(A, B, g, val, n_eq, tissue, real_edge_indices)

            # Use the Robust (Marginalized Variance) metric
            ev = _compute_log_evidence_robust(A, B, g, val, res, ATA, n_eq, n_real_edges) if use_exact else \
                 _compute_log_evidence_approx(A, B, val, res)

            results.append(res)
            log_evidences.append(ev)
            residuals.append(res.residual)

        best_idx = np.argmax(log_evidences)
        best_mu = mu_values[best_idx]
        best_res = results[best_idx]

        logger.info(f"Optimal mu found: {best_mu:.4g}")

        return BayesianScanResult(
            best_mu=best_mu,
            best_result=best_res,
            mu_values=mu_values,
            log_evidences=np.array(log_evidences),
            residuals=np.array(residuals)
        )

def _build_bayesian_matrices(tissue: Tissue, exclude_border_edges: bool = True,
                             border_margin: int = 5):
    """
    Build matrices for Bayesian inference.

    Args:
        tissue: Tissue object with topology
        exclude_border_edges: If True, completely excludes edges connected to
                              border vertices OR belonging to border cells
        border_margin: Pixel distance from image edge for border-vertex classification.
    """
    V = tissue.V
    E = tissue.E
    E_cells = tissue.E_cells
    C_v = tissue.C_v
    if len(V) == 0:
        return None
    H_img, W_img = tissue.labels.shape
    margin = border_margin

    # Identify border vertices (vertices within `margin` pixels of any image edge)
    is_border_v = (V[:, 0] < margin) | (V[:, 0] > W_img - margin) | \
                  (V[:, 1] < margin) | (V[:, 1] > H_img - margin)

    # Identify which edges are "real interior" edges.
    #
    # Previous logic excluded ALL edges whose adjacent cell touched the image
    # margin (border_cells).  This was overly conservative: a border cell can
    # share an edge with an interior cell at a fully-interior vertex, and the
    # force balance at that vertex IS complete — it should contribute an
    # equation to the system.
    #
    # Correct criterion: exclude only edges where at least one endpoint vertex
    # is on the image margin.  The "fully_interior_v" check below then ensures
    # we only build force-balance equations at vertices where every incident
    # edge is included, so the system remains consistent.
    if exclude_border_edges:
        is_real_edge = np.array([
            (not is_border_v[v1]) and (not is_border_v[v2])
            for v1, v2 in E
        ])

        real_edge_indices = np.where(is_real_edge)[0]
        real_edge_set = set(real_edge_indices)
        e_map = {old: new for new, old in enumerate(real_edge_indices)}
        n_edges = len(real_edge_indices)
        logger.info(f"Using {n_edges}/{len(E)} interior edges (excluding border-vertex edges)")

        # CRITICAL: Only balance forces at vertices where ALL incident edges are included
        # Build vertex -> incident edges mapping
        vertex_edges = {v: [] for v in range(len(V))}
        for e_idx, (v1, v2) in enumerate(E):
            vertex_edges[v1].append(e_idx)
            vertex_edges[v2].append(e_idx)

        # A vertex is "fully interior" if ALL its edges are real edges
        fully_interior_v = []
        for v_idx in range(len(V)):
            if is_border_v[v_idx]:
                continue
            incident = vertex_edges[v_idx]
            if len(incident) > 0 and all(e in real_edge_set for e in incident):
                fully_interior_v.append(v_idx)

        v_map = {v: i for i, v in enumerate(fully_interior_v)}
        logger.info(f"Using {len(fully_interior_v)} fully interior vertices for force balance")
    else:
        real_edge_indices = np.arange(len(E))
        e_map = {i: i for i in range(len(E))}
        n_edges = len(E)
        idx_int = np.where(~is_border_v)[0]
        v_map = {v: i for i, v in enumerate(idx_int)}

    if len(v_map) == 0:
        logger.warning("No fully interior vertices found!")
        return None

    n_eq = 2 * len(v_map)
    n_vars = n_edges + len(C_v)

    # Diagnostic: Check if system is well-determined
    logger.info(f"System: {n_eq} equations, {n_vars} unknowns ({n_edges} tensions + {len(C_v)} pressures)")
    logger.info(f"Ratio equations/unknowns: {n_eq/n_vars:.2f} (should be >1 for overdetermined)")

    A = sp.lil_matrix((n_eq, n_vars))

    # Only include real edges in force balance at fully interior vertices
    for old_idx in real_edge_indices:
        v1, v2 = E[old_idx]
        new_idx = e_map[old_idx]
        d_vec = V[v2, :2] - V[v1, :2]
        length = np.linalg.norm(d_vec) + 1e-9
        u = d_vec / length
        if v1 in v_map:
            r = 2 * v_map[v1]
            A[r, new_idx] += u[0]
            A[r+1, new_idx] += u[1]
        if v2 in v_map:
            r = 2 * v_map[v2]
            A[r, new_idx] -= u[0]
            A[r+1, new_idx] -= u[1]

    for c_idx, verts in enumerate(C_v):
        n_verts = len(verts)
        if n_verts < 3:
            continue
        for i, vc in enumerate(verts):
            if vc not in v_map:
                continue
            vp = verts[i - 1]
            vn = verts[(i + 1) % n_verts]
            r = 2 * v_map[vc]
            A[r, n_edges + c_idx] += 0.5 * (V[vp, 1] - V[vn, 1])
            A[r+1, n_edges + c_idx] += 0.5 * (V[vn, 0] - V[vp, 0])

    B = sp.lil_matrix((n_vars, n_vars))
    g = np.zeros(n_vars)
    for i in range(n_edges):
        B[i, i] = 1.0
        g[i] = 1.0

    # Store mapping for later reconstruction
    return A, B, g, n_eq, n_vars, real_edge_indices

def _solve_single_mu(A, B, g, mu, n_eq, tissue, real_edge_indices) -> ForceResult:
    tau = np.sqrt(mu)
    A_aug = sp.vstack([A, tau * B])
    b_aug = np.concatenate([np.zeros(n_eq), tau * g])
    # Increase iterations for stability
    res = scipy.sparse.linalg.lsqr(A_aug, b_aug, atol=1e-8, btol=1e-8, iter_lim=10000)
    x = res[0]

    n_real_edges = len(real_edge_indices)

    # Reconstruct full tension array (NaN for excluded border edges)
    full_tensions = np.full(len(tissue.E), np.nan)
    full_tensions[real_edge_indices] = x[:n_real_edges]
    # Center pressures
    pressures = x[n_real_edges:]
    if len(pressures) > 0:
        pressures = pressures - np.mean(pressures)
    return ForceResult(tensions=full_tensions, pressures=pressures, residual=res[3])

def _compute_log_evidence_robust(A, B, g, mu, result, ATA, n_eq, n_real_edges):
    """
    Computes Log Evidence assuming UNKNOWN noise variance (Marginalized).
    This formulation is scale-invariant and typically finds a sharp peak.

    L(mu) ~ - (N_eq / 2) * log( Chi2 ) - 0.5 * log_det_H + 0.5 * N_prior * log(mu)
    """
    # Only use real (non-NaN) tensions for evidence calculation
    real_tensions = result.tensions[~np.isnan(result.tensions)]
    x = np.concatenate([real_tensions, result.pressures])
    n_edges = n_real_edges

    # 1. Total Squared Error (Data + Prior)
    # Note: We use the 'Augmented' residual sum of squares
    res_phys = A.dot(x)
    E_data = np.sum(res_phys**2)
    
    res_prior = real_tensions - 1.0
    E_prior = np.sum(res_prior**2)
    
    # Chi2 = E_data + mu * E_prior
    # This represents the total "energy" of the solution
    total_chi2 = E_data + mu * E_prior
    
    if total_chi2 < 1e-12:
        total_chi2 = 1e-12

    # 2. Complexity Penalty (Log Determinant)
    # H = A'A + mu * B'B
    H = ATA.copy()
    idx = np.arange(n_edges)
    H[idx, idx] += mu 
    
    # Eigenvalues for robust determinant
    evals = np.linalg.eigvalsh(H)
    max_eval = np.max(evals)
    valid_evals = evals[evals > max_eval * 1e-12] # Filter zero modes
    log_det_H = np.sum(np.log(valid_evals))
    
    # 3. Normalization Terms
    # This term penalizes the "tightness" of the prior
    prior_vol = 0.5 * n_edges * np.log(mu)
    
    # 4. Final Evidence
    # The term -(N_eq / 2) * log(total_chi2) is the key. 
    # It replaces (-0.5 * total_chi2) which assumed variance=1.
    evidence = - (n_eq / 2.0) * np.log(total_chi2) - 0.5 * log_det_H + prior_vol
    
    return evidence

def _compute_log_evidence_approx(A, B, mu, result):
    """Fallback proxy if matrix is too huge for dense ops (N_vars > 5000)."""
    real_tensions = result.tensions[~np.isnan(result.tensions)]
    x = np.concatenate([real_tensions, result.pressures])
    res_phys = A.dot(x)
    E_data = np.sum(res_phys**2)
    E_prior = np.sum((real_tensions - 1.0)**2)
    return - (A.shape[0]/2.0) * np.log(E_data + mu*E_prior + 1e-12)

def solve_laplace(tissue: Tissue,
                  regularization: float = 1.0,
                  tension_val: float = 1.0,
                  detrend: bool = False,
                  zero_center: bool = False,
                  border_margin: int = 5) -> Optional[ForceResult]:
    """
    Young-Laplace solver: infers tensions from force balance at junctions,
    then infers pressures via ΔP = T·κ (2-D Young-Laplace law).

    Border cells (those touching the image margin) are treated as atmosphere
    (P = 0) to prevent artefacts at the tissue boundary.

    Args:
        tissue: Tissue object; must have had ``geometry.compute_curvature()``
                applied first.
        regularization: Strength of the Tikhonov regularisation pulling
                        tensions toward ``tension_val``.
        tension_val: Prior target tension (default 1.0).
        border_margin: Distance in pixels from the image edge within which a
                       vertex is classified as belonging to the border.  Cells
                       that own any such vertex are treated as atmosphere.
                       Should match the value used in ``solve_bayesian`` for
                       consistent comparisons (default 5 px).
    """
    if not hasattr(tissue, 'E_tangents'):
        logger.error("Tangents missing. Run geometry.compute_curvature() first.")
        return None

    H, W = tissue.labels.shape
    margin = border_margin
    
    # --- 1. IDENTIFY BORDER CELLS (The Atmosphere) ---
    is_border_v = (tissue.V[:, 0] < margin) | (tissue.V[:, 0] > W - margin) | \
                  (tissue.V[:, 1] < margin) | (tissue.V[:, 1] > H - margin)
    
    border_cell_indices = set()
    for c_idx, verts in enumerate(tissue.C_v):
        for v in verts:
            if is_border_v[v]:
                border_cell_indices.add(c_idx + 1)
                break
                
    border_cell_indices.add(0)

    # --- STEP 1: SOLVE TENSIONS ---
    logger.info("Step 1: Solving Tensions...")
    
    internal_v_indices = np.where(~is_border_v)[0]
    if len(internal_v_indices) < 3:
        internal_v_indices = np.arange(len(tissue.V))

    v_map = {v_idx: i for i, v_idx in enumerate(internal_v_indices)}
    n_balance = 2 * len(internal_v_indices)
    n_vars = len(tissue.E)
    
    rows, cols, data = [], [], []
    for e_idx, (v1, v2) in enumerate(tissue.E):
        t1 = tissue.E_tangents[e_idx, 0]
        t2 = tissue.E_tangents[e_idx, 1]
        
        if v1 in v_map:
            r = 2 * v_map[v1]
            rows.extend([r, r+1])
            cols.extend([e_idx, e_idx])
            data.extend(t1)
        if v2 in v_map:
            r = 2 * v_map[v2]
            rows.extend([r, r+1])
            cols.extend([e_idx, e_idx])
            data.extend(t2)
            
    M = sp.csr_matrix((data, (rows, cols)), shape=(n_balance, n_vars))
    
    # Regularization
    reg_w = regularization * np.sqrt(n_balance / max(1, n_vars))
    R_rows = np.arange(n_vars)
    R_cols = np.arange(n_vars)
    R_data = np.full(n_vars, reg_w)
    R = sp.csr_matrix((R_data, (R_rows, R_cols)), shape=(n_vars, n_vars))
    b_reg = np.full(n_vars, reg_w * float(tension_val))
    
    M_total = sp.vstack([M, R])
    b_total = np.concatenate([np.zeros(n_balance), b_reg])
    
    res_T = scipy.sparse.linalg.lsqr(M_total, b_total)
    inferred_tensions = np.maximum(res_T[0], 0.01)

    # --- STEP 2: SOLVE PRESSURES (With P_border = 0) ---
    logger.info("Step 2: Solving Pressures (Border Cells Fixed to 0)...")
    
    active_cells = np.unique(tissue.E_cells)
    valid_cells = [c for c in active_cells if c > 0]
    c_map = {c: i for i, c in enumerate(valid_cells)}
    
    rows_p, cols_p, data_p, b_p = [], [], [], []
    eq_idx = 0
    
    is_synthetic = (
        tissue.E_synthetic
        if tissue.E_synthetic is not None
        else np.zeros(len(tissue.E), dtype=bool)
    )

    for i in range(len(tissue.E)):
        # Skip synthetic short edges created by split_four_way: their kappa is
        # 0 by construction (only 2 pixel path), so including them would
        # incorrectly force P[c1] - P[c2] = 0 across the split junction.
        if is_synthetic[i]:
            continue

        c1, c2 = tissue.E_cells[i]
        kappa = tissue.E_curvature[i]
        T_val = inferred_tensions[i]
        # Young-Laplace law for a 2-D interface: ΔP = T · κ
        # (The 3-D spherical form ΔP = 2Tκ does NOT apply to a 2-D monolayer
        # cross-section; that factor-of-2 overcounts the out-of-plane curvature.)
        laplace_val = T_val * kappa

        if c1 in c_map:
            rows_p.append(eq_idx)
            cols_p.append(c_map[c1])
            data_p.append(1.0)
        if c2 in c_map:
            rows_p.append(eq_idx)
            cols_p.append(c_map[c2])
            data_p.append(-1.0)

        b_p.append(laplace_val)
        eq_idx += 1

    weight_border = 10.0
    for c in valid_cells:
        if c in border_cell_indices:
            rows_p.append(eq_idx)
            cols_p.append(c_map[c])
            data_p.append(weight_border)
            b_p.append(0.0)
            eq_idx += 1

    if eq_idx == 0:
        return None

    L_mat = sp.csr_matrix((data_p, (rows_p, cols_p)), shape=(eq_idx, len(c_map)))
    B_vec = np.array(b_p)
    
    res_P = scipy.sparse.linalg.lsqr(L_mat, B_vec)
    rel_P = res_P[0]
    
    full_P = np.zeros(tissue.labels.max())
    for c, idx in c_map.items():
        if c-1 < len(full_P): 
            full_P[c-1] = rel_P[idx]
            
    return ForceResult(tensions=inferred_tensions, pressures=full_P, residual=res_P[3])


# =============================================================================
# 3-D Bayesian force-balance solver
# =============================================================================

def solve_bayesian_3d(
    tissue: Tissue,
    mu: Union[float, List[float], np.ndarray] = None,
    exclude_border_edges: bool = True,
    border_margin: int = 5,
) -> Optional[Union[ForceResult, BayesianScanResult]]:
    """
    True 3-D Bayesian force-balance solver.

    Extends :func:`solve_bayesian` to use the full (x, y, z) vertex coordinates
    stored in ``tissue.V[:, 2]``, which must have been set beforehand via
    :func:`~force_inference.geometry.map_z_to_vertices`.

    Physics
    -------
    Force balance at each fully-interior junction vertex in 3-D:

        Σ_e  T_e · û_e(v)  +  Σ_c  P_c · a_c(v)  =  0     (x, y, z)

    where:
    - **û_e(v)** is the 3-D unit tangent of edge *e* at vertex *v*,
      computed from ``V[v2] − V[v1]`` in full 3-D.
    - **a_c(v)** is the area-gradient of cell *c* at vertex *v*:

          a_c(v_k) = ½ (r_{k−1} − r_{k+1}) × n̂_c

      with **n̂_c** the unit surface normal of cell *c*, estimated via
      Newell's method from the 3-D vertex polygon.

    For a flat tissue (all Z = 0) this reduces to exactly the 2-D system:
    the z-force equations collapse to 0 = 0 and are skipped automatically.

    Parameters
    ----------
    tissue : Tissue
        Must have ``tissue.V.shape[1] == 3`` and non-zero Z values.
        Call :func:`~force_inference.geometry.map_z_to_vertices` first.
    mu, exclude_border_edges, border_margin :
        Same semantics as :func:`solve_bayesian`.

    Returns
    -------
    ForceResult or BayesianScanResult
        Same structure as the 2-D solver.

    Notes
    -----
    * The Bayesian evidence, regularisation (tensions ~ 1 prior), and
      lsqr back-end are identical to the 2-D solver; only the force-balance
      matrix A is different (3 rows per vertex, 3-D tangents, cross-product
      pressure terms).
    * If all Z coordinates are zero, a warning is issued and the standard
      2-D solver is called instead.
    """
    # Sanity-check: do we actually have 3-D data?
    if tissue.V.shape[1] < 3 or not np.any(tissue.V[:, 2] != 0):
        logger.warning(
            "solve_bayesian_3d: Z coordinates are all zero — falling back to "
            "2-D solver.  Call geometry.map_z_to_vertices() first."
        )
        return solve_bayesian(tissue, mu=mu,
                              exclude_border_edges=exclude_border_edges,
                              border_margin=border_margin)

    matrices = _build_bayesian_matrices_3d(
        tissue,
        exclude_border_edges=exclude_border_edges,
        border_margin=border_margin,
    )
    if matrices is None:
        return None

    A, B, g, n_eq, n_vars, real_edge_indices = matrices
    n_real_edges = len(real_edge_indices)

    if isinstance(mu, (float, int)):
        return _solve_single_mu(A, B, g, mu, n_eq, tissue, real_edge_indices)

    # --- Scan mu ---
    if mu is None:
        mu_exponents = np.arange(-10.0, -5.1, 0.25)  # 10^-10 to 10^-5.25, 19 values
        mu_values = 10.0 ** mu_exponents
    else:
        mu_values = np.array(mu)

    logger.info(f"[3D] Scanning {len(mu_values)} mu values …")

    results, log_evidences, residuals = [], [], []
    use_exact = n_vars < 4000
    ATA = None
    if use_exact:
        ATA = A.toarray().T @ A.toarray()

    for val in mu_values:
        res = _solve_single_mu(A, B, g, val, n_eq, tissue, real_edge_indices)
        ev = (
            _compute_log_evidence_robust(A, B, g, val, res, ATA, n_eq, n_real_edges)
            if use_exact
            else _compute_log_evidence_approx(A, B, val, res)
        )
        results.append(res)
        log_evidences.append(ev)
        residuals.append(res.residual)

    best_idx = int(np.argmax(log_evidences))
    logger.info(f"[3D] Optimal mu = {mu_values[best_idx]:.4g}")

    return BayesianScanResult(
        best_mu=float(mu_values[best_idx]),
        best_result=results[best_idx],
        mu_values=mu_values,
        log_evidences=np.array(log_evidences),
        residuals=np.array(residuals),
    )


def _compute_cell_normals(tissue: Tissue) -> np.ndarray:
    """
    Estimate the 3-D surface normal for every cell using Newell's method.

    Newell's method computes the normal of an (optionally non-planar) polygon
    as the sum of cross products of consecutive edge pairs, which is numerically
    robust for polygons on a slightly curved surface.

    For a flat tissue (Z = 0) all normals are [0, 0, +1].

    Returns
    -------
    normals : (n_cells, 3) float array of unit normal vectors.
    """
    V = tissue.V          # (N, 3)
    n_cells = len(tissue.C_v)
    normals = np.zeros((n_cells, 3), dtype=float)

    for ci, verts in enumerate(tissue.C_v):
        if len(verts) < 3:
            normals[ci] = [0.0, 0.0, 1.0]
            continue

        pts = V[np.asarray(verts, dtype=int), :3]   # (n_v, 3)
        n_v = len(pts)

        # Newell's method: accumulate cross-products from a fan at pts[0]
        normal = np.zeros(3)
        for i in range(1, n_v - 1):
            a = pts[i]     - pts[0]
            b = pts[i + 1] - pts[0]
            normal += np.cross(a, b)

        norm_len = float(np.linalg.norm(normal))
        if norm_len > 1e-12:
            normals[ci] = normal / norm_len
        else:
            normals[ci] = [0.0, 0.0, 1.0]   # degenerate polygon → default to +Z

    return normals


def _build_bayesian_matrices_3d(
    tissue: Tissue,
    exclude_border_edges: bool = True,
    border_margin: int = 5,
):
    """
    Build the force-balance matrix for the 3-D Bayesian solver.

    Identical to :func:`_build_bayesian_matrices` except:
    - Edge tangents û_e are full 3-D unit vectors.
    - 3 force-balance equations per vertex (x, y, z).
    - Pressure contribution uses the cross-product formula with the
      per-cell surface normal (Newell's method).

    For flat tissue (Z = 0):
        û_e  →  (u_x, u_y, 0)
        n̂_c  →  (0, 0, 1)
        a_c(v) = ½(r_prev−r_next)×(0,0,1) = ½(y_prev−y_next, x_next−x_prev, 0)

    which is exactly the 2-D formula in the xy rows plus a trivially-zero z row.
    """
    V = tissue.V          # (N, 3)
    E = tissue.E
    E_cells = tissue.E_cells
    C_v = tissue.C_v

    if len(V) == 0:
        return None

    H_img, W_img = tissue.labels.shape
    margin = border_margin

    # --- Border classification (same as 2-D, based on XY only) ---
    is_border_v = (
        (V[:, 0] < margin) | (V[:, 0] > W_img - margin) |
        (V[:, 1] < margin) | (V[:, 1] > H_img - margin)
    )

    if exclude_border_edges:
        is_real_edge = np.array([
            (not is_border_v[v1]) and (not is_border_v[v2])
            for v1, v2 in E
        ])
        real_edge_indices = np.where(is_real_edge)[0]
        real_edge_set = set(real_edge_indices.tolist())
        e_map = {int(old): new for new, old in enumerate(real_edge_indices)}
        n_edges = len(real_edge_indices)

        vertex_edges: dict = {v: [] for v in range(len(V))}
        for e_idx, (v1, v2) in enumerate(E):
            vertex_edges[v1].append(e_idx)
            vertex_edges[v2].append(e_idx)

        fully_interior_v = [
            v for v in range(len(V))
            if not is_border_v[v]
            and len(vertex_edges[v]) > 0
            and all(e in real_edge_set for e in vertex_edges[v])
        ]
        v_map = {v: i for i, v in enumerate(fully_interior_v)}
        logger.info(f"[3D] {n_edges}/{len(E)} interior edges, "
                    f"{len(fully_interior_v)} interior vertices")
    else:
        real_edge_indices = np.arange(len(E))
        e_map = {i: i for i in range(len(E))}
        n_edges = len(E)
        v_map = {int(v): i for i, v in enumerate(np.where(~is_border_v)[0])}

    if len(v_map) == 0:
        logger.warning("[3D] No fully interior vertices found!")
        return None

    # --- Determine active equations ---
    # For a flat tissue the z-force equation is trivially 0 = 0.
    # We keep all 3 components but the lsqr solver handles rank-deficiency fine.
    ndim = 3
    n_eq   = ndim * len(v_map)
    n_vars = n_edges + len(C_v)

    logger.info(f"[3D] System: {n_eq} equations ({ndim}D × {len(v_map)} vertices), "
                f"{n_vars} unknowns ({n_edges} tensions + {len(C_v)} pressures)")
    logger.info(f"[3D] Ratio eq/unknowns: {n_eq/n_vars:.2f}")

    A = sp.lil_matrix((n_eq, n_vars))

    # --- Tension block: 3-D unit tangent vectors ---
    for old_idx in real_edge_indices:
        v1, v2 = E[old_idx]
        new_idx = e_map[int(old_idx)]
        d_vec = V[v2, :3] - V[v1, :3]           # full 3-D displacement
        length = float(np.linalg.norm(d_vec)) + 1e-9
        u = d_vec / length                        # unit tangent (3-D)

        if v1 in v_map:
            r = ndim * v_map[v1]
            A[r,     new_idx] += u[0]
            A[r + 1, new_idx] += u[1]
            A[r + 2, new_idx] += u[2]
        if v2 in v_map:
            r = ndim * v_map[v2]
            A[r,     new_idx] -= u[0]
            A[r + 1, new_idx] -= u[1]
            A[r + 2, new_idx] -= u[2]

    # --- Pressure block: cross-product with cell surface normal ---
    cell_normals = _compute_cell_normals(tissue)   # (n_cells, 3) unit normals

    for c_idx, verts in enumerate(C_v):
        n_verts = len(verts)
        if n_verts < 3:
            continue
        n_c = cell_normals[c_idx]      # unit normal for this cell

        for i, vc in enumerate(verts):
            if vc not in v_map:
                continue
            vp = verts[i - 1]
            vn = verts[(i + 1) % n_verts]
            r  = ndim * v_map[vc]

            # a_c(v_k) = ½ (r_{k-1} − r_{k+1}) × n̂_c
            # For Z=0, n̂_c=(0,0,1):
            #   → (½(y_p−y_n), ½(x_n−x_p), 0)   — same as the 2-D formula ✓
            dr = V[vp, :3] - V[vn, :3]
            f  = 0.5 * np.cross(dr, n_c)          # (3,) area-gradient vector

            A[r,     n_edges + c_idx] += f[0]
            A[r + 1, n_edges + c_idx] += f[1]
            A[r + 2, n_edges + c_idx] += f[2]

    # --- Regularisation prior: tensions ~ 1 ---
    B = sp.lil_matrix((n_vars, n_vars))
    g = np.zeros(n_vars)
    for i in range(n_edges):
        B[i, i] = 1.0
        g[i]    = 1.0

    return A, B, g, n_eq, n_vars, real_edge_indices
