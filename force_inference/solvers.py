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
                   exclude_border_edges: bool = True) -> Optional[Union[ForceResult, BayesianScanResult]]:
    """Solve for tensions and pressures using Bayesian force inference.

    Minimises ``‖A x‖² + μ ‖x − 1‖²`` where x concatenates edge tensions
    and cell pressures, and A encodes force balance at interior vertices.

    When *mu* is ``None`` (default) a log-spaced range is scanned and the
    value maximising the marginalised log-evidence is selected automatically,
    returning a :class:`BayesianScanResult`.  Pass a scalar *mu* to skip the
    scan and get a :class:`ForceResult` directly.

    Args:
        tissue: Tissue object produced by ``extract_topology_label`` (or
            ``extract_topology``).  Labels must be stored in
            ``tissue.labels``.
        mu: Regularization strength.

            - ``None`` — auto-scan a log-spaced range and pick the best μ.
            - scalar ``float`` — solve once with this μ value.
            - 1-D array — scan this exact set of μ values.

            **Do not pass a label image here.**  Labels are read from
            ``tissue.labels`` automatically.
        exclude_border_edges: Exclude edges whose endpoints lie within
            5 px of the image margin (recommended; avoids incomplete
            force-balance equations at the boundary).

    Returns:
        :class:`BayesianScanResult` when *mu* is None or an array, or a
        :class:`ForceResult` when *mu* is a scalar.  Returns ``None`` if
        the tissue has no interior vertices.
    """
    # Guard: mu must be a scalar, 1-D array, or None — not a 2-D image array.
    if mu is not None and isinstance(mu, np.ndarray) and mu.ndim > 1:
        raise ValueError(
            "solve_bayesian() received a 2-D array as `mu`. "
            "The API no longer takes a label image as the second positional "
            "argument — labels are read from tissue.labels automatically. "
            "Call: solve_bayesian(tissue) or solve_bayesian(tissue, mu=0.01)"
        )

    matrices = _build_bayesian_matrices(tissue, exclude_border_edges=exclude_border_edges)
    if matrices is None:
        return None
    A, B, g, n_eq, n_vars, real_edge_indices = matrices
    n_real_edges = len(real_edge_indices)

    if isinstance(mu, (float, int)):
        return _solve_single_mu(A, B, g, mu, n_eq, tissue, real_edge_indices)

    else:
        # Scan Range
        if mu is None:
            mu_exponents = np.arange(-10.0, -5.1, 0.25) # 10^-3 to 10^3
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

def _build_bayesian_matrices(tissue: Tissue, exclude_border_edges: bool = True):
    """
    Build matrices for Bayesian inference.

    Args:
        tissue: Tissue object with topology
        exclude_border_edges: If True, completely excludes edges connected to
                              border vertices OR belonging to border cells
    """
    V = tissue.V
    E = tissue.E
    E_cells = tissue.E_cells
    C_v = tissue.C_v
    if len(V) == 0:
        return None
    H_img, W_img = tissue.labels.shape
    margin = 5

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
                  zero_center: bool = False) -> Optional[ForceResult]:
    """Solve for tensions and pressures using Young-Laplace force balance.

    Treats all border cells as atmosphere (P=0) to prevent artificial
    pressure jumps across image boundaries.

    ``compute_curvature(tissue)`` **must** be called before this function
    to populate ``tissue.E_tangents`` and ``tissue.E_curvature``.

    Args:
        tissue: Tissue object with curvature already computed.
        regularization: Tikhonov regularization weight for tension solve.
        tension_val: Prior mean for tensions (default 1.0).
        detrend: Unused; reserved for future pressure detrending.
        zero_center: Unused; reserved for future pressure centering.

    Returns:
        ForceResult with tensions and pressures, or None if the system
        is degenerate (fewer than 3 interior vertices).

    Raises:
        ValueError: If ``tissue.E_tangents`` or ``tissue.E_curvature``
            are missing — call ``compute_curvature(tissue)`` first.
    """
    if not hasattr(tissue, 'E_tangents') or tissue.E_tangents is None:
        raise ValueError(
            "tissue.E_tangents is missing. "
            "Call compute_curvature(tissue) before solve_laplace()."
        )
    if tissue.E_curvature is None:
        raise ValueError(
            "tissue.E_curvature is missing. "
            "Call compute_curvature(tissue) before solve_laplace()."
        )
    if len(tissue.E) == 0:
        logger.warning("solve_laplace: tissue has no edges, returning None.")
        return None

    H, W = tissue.labels.shape
    margin = 5
    
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
        laplace_val = 2.0 * T_val * kappa

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
