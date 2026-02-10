"""
eglobal/algebra.py

Core algebra utilities for global epistasis.

This module focuses on:
  - Constructing g, H and M = <DF DF^T> under the 2nd-order Walsh–Hadamard truncation.
  - Optionally computing C = M - g g^T.
  - Computing spectral properties of M.
  - Computing null/uncertainty bootstrap ensembles:
      * method="second_order": from WH bootstrap coefficient samples stored in WalshResults.meta
      * method="df": via stats_noise.noise_models wild bootstrap on replicate columns (VarianceProject-consistent)
  - Providing a "general" (non-WH) computation of M from discrete gradients DF defined via neighbor differences,
    with automatic detection of full-hypercube vs empirical averaging.

All docstrings use an Inputs/Outputs format.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Literal, Tuple, Dict, Sequence, Iterator, Mapping

import numpy as np
import pandas as pd

from epistasia.stats_noise.noise_models import bootstrap_null, bootstrap_uncertainty

Which = Literal["null", "uncertainty"]
Store = Literal["eigvals", "eigvecs", "M", "C", "gh", "full"]
Solver = Literal["eigh", "eig"]
Method = Literal["second_order", "df"]


#################################################
#                    HELPERS                    #
#################################################

@dataclass(frozen=True)
class GHPlan:
    """
    Inputs
    ------
    N : int
        Number of loci.
    idx_order1 : np.ndarray, shape (N,)
        idx_order1[i] is the WH mode index corresponding to the 1st-order coefficient f_i.
    pairs_order2 : np.ndarray, shape (P, 3)
        Each row is (i, j, mode_idx) with i < j, mapping 2nd-order coefficient f_{ij}.

    Outputs
    -------
    GHPlan
        A lightweight plan used to extract g and H from a full WH coefficient vector fs
        without repeated searches.
    """
    N: int
    idx_order1: np.ndarray          # (N,)
    pairs_order2: np.ndarray        # (P,3) rows: (i,j,mode_idx), i<j


def build_gh_plan(s_bits: np.ndarray, orders: np.ndarray) -> GHPlan:
    """
    Build index mappings to extract (g, H) from WH coefficients.

    Inputs
    ------
    s_bits : np.ndarray, shape (K, N)
        Binary indicator of which loci are present in each WH mode.
        Row k corresponds to mode index k.
    orders : np.ndarray, shape (K,)
        Hamming weight |s| (order) for each WH mode.

    Outputs
    -------
    plan : GHPlan
        Contains:
          - idx_order1[i] = mode index for f_i
          - pairs_order2 rows (i, j, k) for f_{ij} located at mode index k
    """
    s_bits = np.asarray(s_bits)
    orders = np.asarray(orders)

    if s_bits.ndim != 2:
        raise ValueError("s_bits must be 2D (K,N).")
    if orders.ndim != 1 or orders.shape[0] != s_bits.shape[0]:
        raise ValueError("orders must be 1D (K,) aligned with s_bits.")

    K, N = s_bits.shape

    idx_order1 = np.full(N, -1, dtype=int)
    for k in np.where(orders == 1)[0]:
        ones = np.flatnonzero(s_bits[k])
        if ones.size == 1:
            idx_order1[int(ones[0])] = int(k)

    if np.any(idx_order1 < 0):
        missing = np.where(idx_order1 < 0)[0].tolist()
        raise ValueError(f"Missing order-1 WH modes for loci: {missing}")

    pairs = []
    for k in np.where(orders == 2)[0]:
        ones = np.flatnonzero(s_bits[k])
        if ones.size == 2:
            i, j = int(ones[0]), int(ones[1])
            if i > j:
                i, j = j, i
            pairs.append((i, j, int(k)))

    pairs_order2 = np.array(pairs, dtype=int) if pairs else np.zeros((0, 3), dtype=int)
    return GHPlan(N=N, idx_order1=idx_order1, pairs_order2=pairs_order2)


def gh_from_fs(fs: np.ndarray, plan: GHPlan, *, diag_zero: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract first- and second-order components (g, H) from a full WH coefficient vector.

    Unified-epistasis identification (2nd order truncation):
      g_i = f_i
      H_ij = f_{ij}, with H_ii = 0

    Inputs
    ------
    fs : np.ndarray, shape (K,)
        Full WH coefficient vector (all orders).
    plan : GHPlan
        Precomputed mapping from WH modes to indices (i) and (i,j).
    diag_zero : bool
        If True, enforce H_ii = 0.

    Outputs
    -------
    g : np.ndarray, shape (N,)
        First-order coefficient vector (g_i = f_i).
    H : np.ndarray, shape (N, N)
        Second-order coefficient matrix (H_ij = f_{ij}), symmetric, with H_ii = 0 if requested.
    """
    fs = np.asarray(fs, dtype=float)

    g = fs[plan.idx_order1].copy()
    H = np.zeros((plan.N, plan.N), dtype=float)

    for i, j, k in plan.pairs_order2:
        H[i, j] = fs[k]
        H[j, i] = fs[k]

    if diag_zero:
        np.fill_diagonal(H, 0.0)

    return g, H

###############################################################
#                   COMPUTE GRADIENTS DF                      #
###############################################################

@dataclass(frozen=True)
class DFResult:
    """
    Container for discrete-gradient samples DF(z).

    Fields
    ------
    D : np.ndarray, shape (n_used, N)
        Discrete gradient vectors DF evaluated at selected states.
    states_used : np.ndarray, shape (n_used, N)
        Binary states (subset of obj.states) at which DF is evaluated.
    rows_used : np.ndarray, shape (n_used,)
        Absolute row indices into obj.states/obj.values corresponding to states_used.
    abs_idx_plus : np.ndarray, shape (n_used, N)
        abs_idx_plus[k,i] is the absolute row index of the neighbor with bit i set to 1.
    abs_idx_minus : np.ndarray, shape (n_used, N)
        abs_idx_minus[k,i] is the absolute row index of the neighbor with bit i set to 0.
    F_mean : np.ndarray, shape (M,)
        Mean landscape value per state used to compute D.
    df_info : dict
        Diagnostics: mode ("df_full" or "df_empirical"), is_full_hypercube, n_valid, coverage.
    boot_unc_D : Optional[np.ndarray], shape (B_uncertainty, n_used, N)
        Uncertainty bootstrap draws of DF (if requested).
    boot_null_D : Optional[np.ndarray], shape (B_null, n_used, N)
        Null bootstrap draws of DF (if requested).
    boot_info : dict
        Metadata describing bootstrap parameters.
    """
    D: np.ndarray
    states_used: np.ndarray
    rows_used: np.ndarray
    abs_idx_plus: np.ndarray
    abs_idx_minus: np.ndarray
    F_mean: np.ndarray
    df_info: Dict[str, Any]
    boot_unc_D: Optional[np.ndarray] = None
    boot_null_D: Optional[np.ndarray] = None
    boot_info: Optional[Dict[str, Any]] = None

def _DF_empirical_with_rows(
    states01: np.ndarray,
    F: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Empirical DF evaluation on states with complete neighbor information.

    Outputs
    -------
    D : (n_valid, N)
    rows_used : (n_valid,)
        Absolute indices into the original states01/F arrays.
    abs_idx_plus : (n_valid, N)
        For each valid row and dimension i, absolute row index of the + neighbor (bit=1).
    abs_idx_minus : (n_valid, N)
        Absolute row index of the - neighbor (bit=0).
    info : dict
    """
    states01 = np.asarray(states01, dtype=np.uint8)
    F = np.asarray(F, dtype=float)

    if states01.ndim != 2:
        raise ValueError("states01 must be 2D (M,N).")
    if F.ndim != 1 or F.shape[0] != states01.shape[0]:
        raise ValueError("F must be 1D aligned with states01 rows.")

    M, N = states01.shape
    codes = _states_to_int_codes(states01).astype(np.uint64, copy=False)
    code_to_row = {int(c): int(i) for i, c in enumerate(codes)}

    D_list = []
    rows_used = []
    plus_list = []
    minus_list = []

    for row, code in enumerate(codes):
        DF = np.empty(N, dtype=float)
        idx_p = np.empty(N, dtype=int)
        idx_m = np.empty(N, dtype=int)

        valid = True
        for i in range(N):
            mask = (np.uint64(1) << np.uint64(N - 1 - i))
            code_plus = int(code | mask)
            code_minus = int(code & (~mask))

            rp = code_to_row.get(code_plus, -1)
            rm = code_to_row.get(code_minus, -1)
            if rp < 0 or rm < 0:
                valid = False
                break

            idx_p[i] = rp
            idx_m[i] = rm
            DF[i] = 0.5 * (F[rp] - F[rm])

        if valid:
            D_list.append(DF)
            rows_used.append(row)
            plus_list.append(idx_p)
            minus_list.append(idx_m)

    if not D_list:
        raise ValueError("No states with complete neighbor information; cannot compute empirical DF.")

    D = np.vstack(D_list)
    rows_used = np.asarray(rows_used, dtype=int)
    abs_idx_plus = np.vstack(plus_list).astype(int)
    abs_idx_minus = np.vstack(minus_list).astype(int)

    info = {
        "mode": "df_empirical",
        "is_full_hypercube": False,
        "n_valid": int(D.shape[0]),
        "coverage": float(D.shape[0]) / float(M),
    }
    return D, rows_used, abs_idx_plus, abs_idx_minus, info

def compute_DF(
    obj: Any,
    *,
    use_mean_over_replicates: bool = True,
    df_auto: bool = True,
    # --- bootstrap ---
    B_uncertainty: int = 0,
    uncertainty_flavor: str = "wildcluster",
    B_null: int = 0,
    preserve_residual_corr: bool = True,
    multipliers: str = "rademacher",
    rng: Optional[np.random.Generator] = None,
) -> DFResult:
    """
    Compute DF(z) from neighbor differences, optionally with bootstrap ensembles.

    Inputs
    ------
    obj :
        Must provide:
          - obj.states : (M,N) in {0,1}
          - obj.values : (M,R) replicate matrix (float) OR (M,) single column
    use_mean_over_replicates :
        If True, compute DF from the per-state mean F_mean.
        If False, currently not supported (would require DF per replicate).
    df_auto :
        If True, use full-hypercube exact method when possible, else empirical-valid subset.
    B_uncertainty, uncertainty_flavor, multipliers :
        Passed to epistasia.stats_noise.noise_models.bootstrap_uncertainty.
    B_null, preserve_residual_corr, multipliers :
        Passed to bootstrap_null.
    rng :
        Random generator.

    Outputs
    -------
    DFResult
    """
    if rng is None:
        rng = np.random.default_rng()

    if not hasattr(obj, "states") or not hasattr(obj, "values"):
        raise TypeError("obj must provide .states and .values")

    states01 = np.asarray(obj.states, dtype=np.uint8)
    values = np.asarray(obj.values, dtype=float)
    if values.ndim == 1:
        values = values[:, None]

    M, N = states01.shape
    R = values.shape[1]

    if not use_mean_over_replicates:
        raise NotImplementedError(
            "use_mean_over_replicates=False not implemented yet. "
            "We can add a replicate-aware DF pipeline if needed."
        )

    F_mean = np.nanmean(values, axis=1)

    # ------------------------------------------------------------
    # Compute D and neighbor index maps (plus/minus) on used states
    # ------------------------------------------------------------
    if df_auto and is_full_hypercube(states01):
        # Full cube: we can build plus/minus indices vectorized per i.
        codes = _states_to_int_codes(states01).astype(np.uint64, copy=False)

        code_to_row = np.empty((1 << N), dtype=int)
        code_to_row.fill(-1)
        code_to_row[codes] = np.arange(M, dtype=int)

        abs_idx_plus = np.empty((M, N), dtype=int)
        abs_idx_minus = np.empty((M, N), dtype=int)
        D = np.empty((M, N), dtype=float)

        for i in range(N):
            mask = (np.uint64(1) << np.uint64(N - 1 - i))
            codes_plus = codes | mask
            codes_minus = codes & (~mask)
            rp = code_to_row[codes_plus]
            rm = code_to_row[codes_minus]
            abs_idx_plus[:, i] = rp
            abs_idx_minus[:, i] = rm
            D[:, i] = 0.5 * (F_mean[rp] - F_mean[rm])

        rows_used = np.arange(M, dtype=int)
        states_used = states01
        df_info = {
            "mode": "df_full",
            "is_full_hypercube": True,
            "n_valid": int(M),
            "coverage": 1.0,
        }

    else:
        D, rows_used, abs_idx_plus, abs_idx_minus, df_info = _DF_empirical_with_rows(states01, F_mean)
        states_used = states01[rows_used]

    # ------------------------------------------------------------
    # Bootstraps (optional): generate F_boot and push through ± maps
    # ------------------------------------------------------------
    boot_unc_D = None
    boot_null_D = None
    boot_info: Dict[str, Any] = {}

    if (B_uncertainty > 0 or B_null > 0) and R < 2:
        raise ValueError("At least 2 replicates required for bootstrap.")

    if B_uncertainty > 0 or B_null > 0:
        # Build DataFrame exactly like epistasis.py does (states columns + replicate columns)
        F_df = pd.DataFrame(
            np.hstack([states01.astype(int, copy=False), values]),
            columns=[f"s{j}" for j in range(N)] + [f"rep{r}" for r in range(R)],
        )

    if B_uncertainty > 0:
        Fb_unc = bootstrap_uncertainty(
            F_df,
            B=B_uncertainty,
            multipliers=multipliers,
            flavor=uncertainty_flavor,
            rng=rng,
        )
        boots = Fb_unc.values[:, N:]  # (M, B)

        n_used = D.shape[0]
        boot_unc_D = np.empty((B_uncertainty, n_used, N), dtype=float)

        # Vectorize over used rows; for each i compute 0.5*(F[+]-F[-])
        for b in range(B_uncertainty):
            F_b = boots[:, b]
            for i in range(N):
                rp = abs_idx_plus[:, i]
                rm = abs_idx_minus[:, i]
                boot_unc_D[b, :, i] = 0.5 * (F_b[rp] - F_b[rm])

        boot_info["uncertainty"] = {
            "B": B_uncertainty,
            "flavor": uncertainty_flavor,
            "multipliers": multipliers,
        }

    if B_null > 0:
        Fb_null = bootstrap_null(
            F_df,
            B=B_null,
            multipliers=multipliers,
            preserve_residual_corr=preserve_residual_corr,
            rng=rng,
        )
        boots = Fb_null.values[:, N:]  # (M, B)

        n_used = D.shape[0]
        boot_null_D = np.empty((B_null, n_used, N), dtype=float)

        for b in range(B_null):
            F_b = boots[:, b]
            for i in range(N):
                rp = abs_idx_plus[:, i]
                rm = abs_idx_minus[:, i]
                boot_null_D[b, :, i] = 0.5 * (F_b[rp] - F_b[rm])

        boot_info["null"] = {
            "B": B_null,
            "preserve_residual_corr": preserve_residual_corr,
            "multipliers": multipliers,
        }

    return DFResult(
        D=D,
        states_used=states_used,
        rows_used=rows_used,
        abs_idx_plus=abs_idx_plus,
        abs_idx_minus=abs_idx_minus,
        F_mean=F_mean,
        df_info=df_info,
        boot_unc_D=boot_unc_D,
        boot_null_D=boot_null_D,
        boot_info=boot_info if boot_info else None,
    )

def sample_DF_second_order(
    g: np.ndarray,
    H: np.ndarray,
    n_samples: int,
    rng: Optional[np.random.Generator] = None,
    *,
    spins: str = "pm1",
    return_Z: bool = False,
    chunk_size: Optional[int] = None,
) -> np.ndarray | Tuple[np.ndarray, np.ndarray]:
    """
    Sample DF(z) under the 2nd-order truncation DF(z) ≈ g + H z.

    Inputs
    ------
    g : (N,)
    H : (N,N)
    n_samples : int
    rng : np.random.Generator
    spins : {"pm1","01"}
        - "pm1": z in {-1,+1}^N  (recommended / matches WH convention)
        - "01":  x in {0,1}^N, internally mapped to z = 2x - 1
    return_Z : bool
        If True, also return the sampled spin vectors Z (in pm1 convention).
    chunk_size : Optional[int]
        If set, generate samples in chunks to reduce peak memory.

    Outputs
    -------
    D : (n_samples, N)
    Z : (n_samples, N)   only if return_Z=True (pm1 convention)
    """
    if rng is None:
        rng = np.random.default_rng()

    g = np.asarray(g, dtype=float)
    H = np.asarray(H, dtype=float)
    if g.ndim != 1:
        raise ValueError("g must be 1D (N,).")
    N = int(g.shape[0])
    if H.shape != (N, N):
        raise ValueError("H must have shape (N,N) matching g.")

    if chunk_size is None:
        chunk_size = n_samples

    D_out = np.empty((n_samples, N), dtype=float)
    Z_out = np.empty((n_samples, N), dtype=float) if return_Z else None

    start = 0
    while start < n_samples:
        end = min(n_samples, start + int(chunk_size))
        m = end - start

        if spins == "pm1":
            Z = rng.choice([-1.0, 1.0], size=(m, N))
        elif spins == "01":
            X = rng.integers(0, 2, size=(m, N)).astype(float)
            Z = 2.0 * X - 1.0
        else:
            raise ValueError("spins must be 'pm1' or '01'.")

        D_chunk = g[None, :] + (Z @ H.T)
        D_out[start:end, :] = D_chunk

        if return_Z:
            Z_out[start:end, :] = Z

        start = end

    return (D_out, Z_out) if return_Z else D_out


###############################################################
#     COMPUTE SECOND MOMENT MATRIX OF THE GRADIENT FIELD      #
###############################################################

def M_second_order_from_gh(g: np.ndarray, H: np.ndarray) -> np.ndarray:
    """
    Second-order approximation of M.

    Under the 2nd-order truncation DF(z) ≈ g + H z (uniform z in {-1,+1}^N),
    the second moment is:
        M = <DF DF^T> ≈ g g^T + H H^T
    Since H is symmetric in our construction, H H^T = H @ H.

    Inputs
    ------
    g : np.ndarray, shape (N,)
    H : np.ndarray, shape (N,N)

    Outputs
    -------
    M : np.ndarray, shape (N,N)
        Symmetric real matrix.
    """
    g = np.asarray(g, dtype=float)
    H = np.asarray(H, dtype=float)
    M = np.outer(g, g) + (H @ H)
    return 0.5 * (M + M.T)


def C_from_Mg(M: np.ndarray, g: np.ndarray) -> np.ndarray:
    """
    Compute C = M - g g^T.

    Inputs
    ------
    M : np.ndarray, shape (N,N)
        Second moment matrix.
    g : np.ndarray, shape (N,)
        Mean gradient / first-order vector.

    Outputs
    -------
    C : np.ndarray, shape (N,N)
        Matrix C = M - g g^T (symmetric if M is symmetric).
    """
    M = np.asarray(M, dtype=float)
    g = np.asarray(g, dtype=float)
    C = M - np.outer(g, g)
    return 0.5 * (C + C.T)


###############################################################
#              HELPERS FOR SPECTRAL PROPERTIES                #
###############################################################

def eigensolve(
    M: np.ndarray,
    *,
    solver: Solver = "eigh",
    k: Optional[int] = None,
    descending: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Eigen-decomposition with an option to return only the dominant (top-k) components.

    Notes
    -----
    - For symmetric real M, 'eigh' is the correct/stable solver.
    - 'eig' is provided only if you explicitly want it.

    Inputs
    ------
    M : np.ndarray, shape (N,N)
        Matrix to diagonalize.
    solver : {"eigh","eig"}
        - "eigh": for symmetric/Hermitian matrices (recommended).
        - "eig": for general matrices (not recommended here).
    k : int or None
        If k is provided, return only the k largest-eigenvalue eigenpairs ("top-k").
        If None, return the full spectrum.
    descending : bool
        If True, sort eigenvalues from largest to smallest.

    Outputs
    -------
    vals : np.ndarray, shape (N,) or (k,)
        Eigenvalues.
    vecs : np.ndarray, shape (N,N) or (N,k)
        Eigenvectors as columns corresponding to vals.
    """
    M = np.asarray(M, dtype=float)

    if solver == "eigh":
        vals, vecs = np.linalg.eigh(M)
    elif solver == "eig":
        vals, vecs = np.linalg.eig(M)
    else:
        raise ValueError("solver must be 'eigh' or 'eig'")

    idx = np.argsort(vals)
    if descending:
        idx = idx[::-1]
    vals = vals[idx]
    vecs = vecs[:, idx]

    if k is not None:
        vals = vals[:k]
        vecs = vecs[:, :k]

    return vals, vecs


def _align_vecs_to_ref(V: np.ndarray, Vref: np.ndarray) -> np.ndarray:
    """
    Align eigenvector signs column-wise to a reference basis.

    Inputs
    ------
    V : np.ndarray, shape (N,k)
        Candidate eigenvectors.
    Vref : np.ndarray, shape (N,k)
        Reference eigenvectors.

    Outputs
    -------
    V_aligned : np.ndarray, shape (N,k)
        Same vectors with sign flips so that dot(V[:,j], Vref[:,j]) >= 0.
    """
    V = V.copy()
    for j in range(V.shape[1]):
        if float(np.dot(V[:, j], Vref[:, j])) < 0.0:
            V[:, j] *= -1.0
    return V

def orient_u1_with_popcount(u1: np.ndarray, states01: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    x = np.asarray(states01, dtype=float)
    pop = x.sum(axis=1)
    nu  = x @ u1
    cov = float(np.mean((nu - nu.mean()) * (pop - pop.mean())))

    if cov < -eps:
        return -u1
    if cov > eps:
        return u1

    # deterministic fallback
    j = int(np.argmax(np.abs(u1)))
    return u1 if u1[j] >= 0.0 else -u1



def align_u1_to_ref(u1_boot: np.ndarray, u1_ref: np.ndarray) -> np.ndarray:
    return u1_boot if float(u1_boot @ u1_ref) >= 0.0 else -u1_boot


###############################################################
#          GENERAL (DF-BASED) COMPUTATION OF M                #
###############################################################

def _states_to_int_codes(states01: np.ndarray) -> np.ndarray:
    """
    Inputs
    ------
    states01 : np.ndarray, shape (X,N)
        Binary states in {0,1}^N.

    Outputs
    -------
    codes : np.ndarray, shape (X,)
        Integer code for each state, using MSB-first convention.
    """
    states01 = np.asarray(states01, dtype=np.uint8)
    X, N = states01.shape
    weights = (1 << np.arange(N - 1, -1, -1, dtype=np.uint64))
    return (states01.astype(np.uint64) * weights).sum(axis=1)


def is_full_hypercube(states01: np.ndarray) -> bool:
    """
    Inputs
    ------
    states01 : np.ndarray, shape (X,N)
        Binary states in {0,1}^N.

    Outputs
    -------
    is_full : bool
        True iff X == 2^N and all states are present exactly once.
    """
    states01 = np.asarray(states01, dtype=np.uint8)
    if states01.ndim != 2:
        return False
    X, N = states01.shape
    if X != (1 << N):
        return False
    codes = _states_to_int_codes(states01)
    if np.unique(codes).size != X:
        return False
    return (codes.min() == 0) and (codes.max() == (1 << N) - 1)


def M_from_DF_neighbors(states01: np.ndarray, F: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute M from the definition M = <DF DF^T> using neighbor differences.
    This requires the FULL hypercube (all 2^N states).

    Inputs
    ------
    states01 : np.ndarray, shape (X,N)
        All genotypes/states x in {0,1}^N (X must equal 2^N).
    F : np.ndarray, shape (X,)
        Landscape values F(x) aligned with states01 rows.

    Outputs
    -------
    M : np.ndarray, shape (N,N)
        Second moment matrix M = <DF DF^T> over the FULL hypercube.
    D : np.ndarray, shape (X,N)
        Discrete gradient field DF evaluated at each state (row order matches states01).
    """
    states01 = np.asarray(states01, dtype=np.uint8)
    F = np.asarray(F, dtype=float)

    if states01.ndim != 2:
        raise ValueError("states01 must be 2D (X,N).")
    if F.ndim != 1 or F.shape[0] != states01.shape[0]:
        raise ValueError("F must be 1D aligned with states01 rows.")

    X, N = states01.shape
    if not is_full_hypercube(states01):
        raise ValueError("DF full method requires the full hypercube (X=2^N and all states present once).")

    codes = _states_to_int_codes(states01).astype(np.uint64, copy=False)

    code_to_row = np.empty((1 << N), dtype=int)
    code_to_row.fill(-1)
    code_to_row[codes] = np.arange(X, dtype=int)

    D = np.empty((X, N), dtype=float)
    for i in range(N):
        mask = (np.uint64(1) << np.uint64(N - 1 - i))
        codes_plus = codes | mask
        codes_minus = codes & (~mask)
        rows_plus = code_to_row[codes_plus]
        rows_minus = code_to_row[codes_minus]
        D[:, i] = 0.5 * (F[rows_plus] - F[rows_minus])

    M = (D.T @ D) / float(X)
    M = 0.5 * (M + M.T)
    return M, D


def M_from_DF_empirical(states01: np.ndarray, F: np.ndarray) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    Empirical estimation of M = <DF DF^T> over accessible states only.

    A state contributes to the average only if all its neighbor values needed to compute
    D_iF are available for every i.

    Inputs
    ------
    states01 : np.ndarray, shape (X,N)
        Observed genotypes x in {0,1}^N.
    F : np.ndarray, shape (X,)
        Landscape values aligned with states01.

    Outputs
    -------
    M : np.ndarray, shape (N,N)
        Empirical second moment matrix.
    D : np.ndarray, shape (n_valid,N)
        Discrete gradients DF for valid states only.
    n_valid : int
        Number of states used in the empirical average.
    """
    states01 = np.asarray(states01, dtype=np.uint8)
    F = np.asarray(F, dtype=float)

    if states01.ndim != 2:
        raise ValueError("states01 must be 2D (X,N).")
    if F.ndim != 1 or F.shape[0] != states01.shape[0]:
        raise ValueError("F must be 1D aligned with states01 rows.")

    X, N = states01.shape
    codes = _states_to_int_codes(states01).astype(np.uint64, copy=False)
    code_to_row = {int(c): int(i) for i, c in enumerate(codes)}

    D_list = []
    for code in codes:
        DF = np.empty(N, dtype=float)
        valid = True
        for i in range(N):
            mask = (np.uint64(1) << np.uint64(N - 1 - i))
            code_plus = int(code | mask)
            code_minus = int(code & (~mask))
            if (code_plus not in code_to_row) or (code_minus not in code_to_row):
                valid = False
                break
            DF[i] = 0.5 * (F[code_to_row[code_plus]] - F[code_to_row[code_minus]])
        if valid:
            D_list.append(DF)

    if len(D_list) == 0:
        raise ValueError("No states with complete neighbor information; cannot compute empirical DF-based M.")

    D = np.vstack(D_list)
    M = (D.T @ D) / float(D.shape[0])
    M = 0.5 * (M + M.T)
    return M, D, int(D.shape[0])


def M_from_DF_auto(states01: np.ndarray, F: np.ndarray) -> Dict[str, Any]:
    """
    Compute DF-based M, automatically choosing between:
      - full hypercube averaging (exact) if all 2^N states are present
      - empirical averaging over valid states otherwise

    Inputs
    ------
    states01 : np.ndarray, shape (X,N)
        Observed states x in {0,1}^N.
    F : np.ndarray, shape (X,)
        Values F(x) aligned with states01 rows.

    Outputs
    -------
    out : dict with keys
        M : np.ndarray, shape (N,N)
            Estimated second moment matrix.
        D : np.ndarray
            Gradient samples used in the average:
              - shape (X,N) for full hypercube
              - shape (n_valid,N) for empirical
        is_full_hypercube : bool
        n_valid : int
            Number of states used (== X for full case).
        coverage : float
            n_valid / X
        mode : str
            "df_full" or "df_empirical"
    """
    states01 = np.asarray(states01, dtype=np.uint8)
    F = np.asarray(F, dtype=float)
    X, _ = states01.shape

    if is_full_hypercube(states01):
        M, D = M_from_DF_neighbors(states01, F)
        return {
            "M": M,
            "D": D,
            "is_full_hypercube": True,
            "n_valid": int(X),
            "coverage": 1.0,
            "mode": "df_full",
        }

    M, D, n_valid = M_from_DF_empirical(states01, F)
    return {
        "M": M,
        "D": D,
        "is_full_hypercube": False,
        "n_valid": int(n_valid),
        "coverage": float(n_valid) / float(X),
        "mode": "df_empirical",
    }


###############################################################
#                    BOOTSTRAP CONTAINERS                     #
###############################################################

@dataclass
class BootstrapEnsemble:
    """
    Outputs
    -------
    eigvals : np.ndarray or None
        Bootstrap eigenvalues. Shape:
          - (B, N) if full spectrum
          - (B, k) if top-k requested
    eigvecs : np.ndarray or None
        Bootstrap eigenvectors. Shape (B, N, k) if requested.
    M : np.ndarray or None
        Bootstrap matrices. Shape (B, N, N) if requested.
    C : np.ndarray or None
        Bootstrap C matrices. Shape (B, N, N) if requested.
    """
    eigvals: Optional[np.ndarray] = None
    eigvecs: Optional[np.ndarray] = None
    M: Optional[np.ndarray] = None
    C: Optional[np.ndarray] = None
    F0: Optional[np.ndarray] = None      # (B,)
    g: Optional[np.ndarray] = None       # (B,N)
    H: Optional[np.ndarray] = None       # (B,N,N)


@dataclass
class BootstrapMResult(Mapping[str, BootstrapEnsemble]):
    """
    A dict-like wrapper holding bootstrap ensembles for M.

    This object is *iterable* and supports:
        "null" in res.bootstrap
        res.bootstrap["null"]
        res.bootstrap.get("null")

    Outputs
    -------
    boot : dict[str, BootstrapEnsemble]
        Bootstrap ensembles.
    meta : dict
        Metadata.
    """
    boot: Dict[str, BootstrapEnsemble]
    meta: Dict[str, Any]

    # --- Mapping interface (for clean plotting) ---
    def __getitem__(self, k: str) -> BootstrapEnsemble:
        return self.boot[k]

    def __iter__(self) -> Iterator[str]:
        return iter(self.boot)

    def __len__(self) -> int:
        return len(self.boot)

    def __contains__(self, k: object) -> bool:
        return k in self.boot

    def keys(self):
        return self.boot.keys()

    def items(self):
        return self.boot.items()

    def values(self):
        return self.boot.values()

    def get(self, k: str, default=None):
        return self.boot.get(k, default)


###############################################################
#     BOOTSTRAP (DF-BASED) FROM stats_noise BOOT DATAFRAMES   #
###############################################################

@dataclass(frozen=True)
class BootstrapDFMResult:
    """
    Outputs
    -------
    B : int
        Number of bootstrap landscapes.
    N : int
        Number of loci.
    store : {"eigvals","M","eigvecs"}
        Which object(s) were stored.
    compute_C : bool
        Whether C = M - g g^T was computed and stored (if store supports it).
    eigvals : np.ndarray or None
        Eigenvalues per bootstrap draw, shape (B,N) or (B,k).
    eigvecs : np.ndarray or None
        Eigenvectors per bootstrap draw, shape (B,N,k).
    M : np.ndarray or None
        Bootstrapped matrices, shape (B,N,N).
    C : np.ndarray or None
        Bootstrapped C matrices, shape (B,N,N).
    """
    B: int
    N: int
    store: Store
    compute_C: bool
    eigvals: Optional[np.ndarray] = None
    eigvecs: Optional[np.ndarray] = None
    M: Optional[np.ndarray] = None
    C: Optional[np.ndarray] = None


def _infer_state_cols_and_N(df: pd.DataFrame) -> Tuple[list[str], int]:
    """
    Infer genotype/state columns.

    Priority:
      1) s1, s2, ..., sN
      2) all columns before first "boot_" column that are binary {0,1}

    Outputs
    -------
    state_cols : list[str]
    N : int
    """
    cols = list(df.columns)

    s_cols = [c for c in cols if str(c).lower().startswith("s") and str(c)[1:].isdigit()]
    if len(s_cols) > 0:
        s_cols = sorted(s_cols, key=lambda x: int(str(x)[1:]))
        return s_cols, len(s_cols)

    boot_cols = [c for c in cols if str(c).startswith("boot_")]
    if len(boot_cols) == 0:
        raise ValueError("No 's*' columns and no 'boot_*' columns found to infer states.")

    first_boot_idx = cols.index(boot_cols[0])
    candidate = cols[:first_boot_idx]

    state_cols = []
    for c in candidate:
        vals = df[c].dropna().unique()
        if len(vals) == 0:
            continue
        if np.all(np.isin(vals, [0, 1, "0", "1", True, False])):
            state_cols.append(c)

    if len(state_cols) == 0:
        raise ValueError("Could not infer state columns (binary) before first 'boot_*' column.")

    return state_cols, len(state_cols)


def compute_M_from_boot_df(
    boot_df: pd.DataFrame,
    *,
    store: Store = "eigvals",
    compute_C: bool = False,
    k: Optional[int] = None,
    solver: Solver = "eigh",
) -> BootstrapDFMResult:
    """
    Compute DF-based M (and optionally C) for each bootstrap landscape in a bootstrap DataFrame.

    Inputs
    ------
    boot_df : pd.DataFrame
        Output of bootstrap_null/bootstrap_uncertainty:
          - genotype columns (ideally s1..sN)
          - bootstrap columns boot_1, ..., boot_B with F^{(b)}(x)
    store : {"eigvals","M","eigvecs"}
    compute_C : bool
        If True, compute g = <DF> and C = M - g g^T for each draw (stored if store == "M").
    k : int or None
        If store="eigvecs", store only top-k eigenvectors/eigenvalues.
    solver : {"eigh","eig"}

    Outputs
    -------
    out : BootstrapDFMResult
    """
    if not isinstance(boot_df, pd.DataFrame):
        raise TypeError("boot_df must be a pandas DataFrame.")

    state_cols, N = _infer_state_cols_and_N(boot_df)

    boot_cols = [c for c in boot_df.columns if str(c).startswith("boot_")]
    if len(boot_cols) == 0:
        raise ValueError("No 'boot_*' columns found in boot_df.")

    def _boot_key(c: str) -> int:
        try:
            return int(str(c).split("_", 1)[1])
        except Exception:
            return 10**9

    boot_cols = sorted(boot_cols, key=_boot_key)
    B = len(boot_cols)

    states01 = boot_df[state_cols].to_numpy(dtype=int)

    if store == "eigvecs" and (k is None or k <= 0):
        raise ValueError("store='eigvecs' requires a positive k (top-k).")

    eigvals_out = None
    eigvecs_out = None
    M_out = None
    C_out = None
    Vref = None
    if store == "eigvals":
        out_dim = N if k is None else int(k)
        eigvals_out = np.empty((B, out_dim), dtype=float)
    elif store == "M":
        M_out = np.empty((B, N, N), dtype=float)
        if compute_C:
            C_out = np.empty((B, N, N), dtype=float)
    elif store == "eigvecs":
        eigvals_out = np.empty((B, int(k)), dtype=float)
        eigvecs_out = np.empty((B, N, int(k)), dtype=float)
        # Build a reference landscape (mean across bootstrap draws)
        F_ref = boot_df[boot_cols].to_numpy(dtype=float).mean(axis=1)

        out_ref = M_from_DF_auto(states01, F_ref)
        M_ref = out_ref["M"]

        _, Vref = eigensolve(M_ref, solver=solver, k=k, descending=True)

        # Enforce your u1 convention (choose ONE of the two)
        Vref[:, 0] = orient_u1_with_popcount(Vref[:, 0], states01)

    else:
        raise ValueError("store must be 'eigvals', 'M', or 'eigvecs'.")

    for b_idx, cboot in enumerate(boot_cols):
        F_b = boot_df[cboot].to_numpy(dtype=float)
        out = M_from_DF_auto(states01, F_b)
        M_b = out["M"]

        if store == "M":
            M_out[b_idx] = M_b
            if compute_C:
                D = np.asarray(out["D"], dtype=float)
                g = np.nanmean(D, axis=0)
                C_out[b_idx] = C_from_Mg(M_b, g)
        else:
            vals, vecs = eigensolve(M_b, solver=solver, k=k, descending=True)
            if store == "eigvals":
                eigvals_out[b_idx] = vals
            else:
                assert Vref is not None
                vecs = _align_vecs_to_ref(vecs, Vref)
                eigvals_out[b_idx] = vals
                eigvecs_out[b_idx] = vecs

    return BootstrapDFMResult(
        B=B,
        N=N,
        store=store,
        compute_C=compute_C,
        eigvals=eigvals_out,
        eigvecs=eigvecs_out,
        M=M_out,
        C=C_out,
    )


def _F_df_for_stats_bootstrap(states01: np.ndarray, values: np.ndarray) -> pd.DataFrame:
    """
    Build the wide DataFrame expected by stats_noise.bootstrap_*.

    Inputs
    ------
    states01 : np.ndarray, shape (M,N)
        States (0/1).
    values : np.ndarray, shape (M,R) or (M,)
        Replicate measurements. If 1D, treated as a single replicate.

    Outputs
    -------
    F_df : pd.DataFrame
        Columns:
          - s1..sN  (int 0/1)
          - r1..rR  (float replicates)
    """
    states01 = np.asarray(states01, dtype=int)
    values = np.asarray(values, dtype=float)
    if values.ndim == 1:
        values = values[:, None]
    if values.ndim != 2:
        raise ValueError("values must be (M,) or (M,R).")
    if states01.shape[0] != values.shape[0]:
        raise ValueError("states01 and values must have same number of rows.")

    M, N = states01.shape
    R = values.shape[1]

    data: Dict[str, Any] = {}
    for i in range(N):
        data[f"s{i+1}"] = states01[:, i].astype(int)

    for r in range(R):
        data[f"r{r+1}"] = values[:, r].astype(float)

    return pd.DataFrame(data)


###############################################################
#     BOOTSTRAP (WH-BASED) FOR SECOND-ORDER M/C               #
###############################################################

def _get_fs_samples_from_meta(W: Any, which: Which) -> np.ndarray:
    """
    Fetch WH bootstrap coefficient samples stored in W.meta.

    Inputs
    ------
    W : WalshResults-like
        Must contain W.meta produced by walsh_analysis.
    which : {"null","uncertainty"}

    Outputs
    -------
    fs_samples : np.ndarray, shape (K, B)
        Bootstrap samples of WH coefficients.
    """
    if not hasattr(W, "meta"):
        raise ValueError("W has no .meta; cannot fetch bootstrap samples.")
    meta = W.meta

    if which == "null":
        try:
            return np.asarray(meta["null"]["fs_null_b"], dtype=float)
        except Exception as e:
            raise ValueError("Missing W.meta['null']['fs_null_b']. Run walsh_analysis with B_null>0.") from e

    if which == "uncertainty":
        try:
            return np.asarray(meta["uncertainty"]["fs_unc_b"], dtype=float)
        except Exception as e:
            raise ValueError("Missing W.meta['uncertainty']['fs_unc_b']. Run walsh_analysis with B_uncertainty>0.") from e

    raise ValueError("which must be 'null' or 'uncertainty'")


def _get_F0_from_fs(fs: np.ndarray, W: Any) -> float:
    orders = np.asarray(W.orders)
    idx0 = np.where(orders == 0)[0]
    if idx0.size != 1:
        raise ValueError("Could not uniquely identify the order-0 (empty set) coefficient.")
    return float(fs[int(idx0[0])])


def bootstrap_second_order_from_walsh(
    W: Any,
    *,
    plan: Optional[GHPlan] = None,
    which: Sequence[Which] = ("null",),
    store: Store = "eigvals",
    compute_C: bool = False,
    solver: Solver = "eigh",
    k: Optional[int] = None,
    align_eigvecs: bool = True,
) -> BootstrapMResult:
    """
    Compute bootstrap distributions under the second-order approximation.

    Input
    -----
    W : WalshResults-like
        Must expose:
          - W.mean   : (K,)
          - W.s_bits : (K,N)   (observed states; used only for popcount orientation)
          - W.orders : (K,)
          - W.meta[...] with fs samples for requested `which`.
    plan : GHPlan or None
        Precomputed mapping; built from W if None.
    which : sequence of {"null","uncertainty"}
        Which bootstrap ensembles to compute.
    store : {"eigvals","eigvecs","M","gh","full"}
        - "eigvals": store eigenvalues per draw
        - "eigvecs": store top-k eigenpairs per draw
        - "M": store M (and optionally C) per draw
        - "gh": store g and H per draw (needed for regional CI propagation)
        - "full": store g, H, and M (and optionally C) per draw
    compute_C : bool
        If True, store C only when store in {"M","full"}.
    solver : {"eigh","eig"}
        Eigen-solver for spectrum modes.
    k : int or None
        Top-k eigenpairs when store in {"eigvals","eigvecs"}.
        Required if store == "eigvecs".
    align_eigvecs : bool
        If True and store=="eigvecs", align eigenvector signs to the observed eigenvectors.

    Output
    ------
    out : BootstrapMResult
        Dict-like wrapper with keys in `which`.
        Each entry is a BootstrapEnsemble containing the requested per-draw outputs.
    """
    # -----------------------
    # Validate / plan
    # -----------------------
    if plan is None:
        plan = build_gh_plan(W.s_bits, W.orders)

    if store not in ("eigvals", "eigvecs", "M", "gh", "full"):
        raise ValueError("store must be 'eigvals', 'eigvecs', 'M', 'gh', or 'full'.")

    if store == "eigvecs" and k is None:
        raise ValueError("store='eigvecs' requires k.")

    # C only meaningful if we store full matrices
    compute_C_store = bool(compute_C and (store in ("M", "full")))

    # Observed states (used only to orient u1 in reference eigvecs)
    states01 = np.asarray(W.s_bits, dtype=int)

    # -----------------------
    # Reference eigvecs for sign alignment
    # -----------------------
    Vref = None
    if store == "eigvecs":
        g_obs, H_obs = gh_from_fs(np.asarray(W.mean, dtype=float), plan, diag_zero=True)
        M_obs = M_second_order_from_gh(g_obs, H_obs)
        _, Vref = eigensolve(M_obs, solver=solver, k=int(k), descending=True)
        # Orient observed u1 with popcount convention
        Vref[:, 0] = orient_u1_with_popcount(Vref[:, 0], states01)

    # -----------------------
    # Flags
    # -----------------------
    need_M = store in ("M", "full", "eigvals", "eigvecs")
    need_gh = store in ("gh", "full")
    need_spectrum = store in ("eigvals", "eigvecs")
    need_vecs = store == "eigvecs"

    boot_dict: Dict[str, BootstrapEnsemble] = {}

    # -----------------------
    # Main loop over ensembles
    # -----------------------
    Wmean = np.asarray(W.mean, dtype=float)
    K_expected = int(Wmean.shape[0])

    for key in which:
        fs_samples = _get_fs_samples_from_meta(W, key)  # expected (K,B) but we accept (B,K) too
        fs_samples = np.asarray(fs_samples, dtype=float)

        if fs_samples.ndim != 2:
            raise ValueError(f"Expected fs_samples as 2D, got shape={fs_samples.shape}.")

        # Accept both orientations: (K,B) or (B,K)
        if fs_samples.shape[0] == K_expected:
            K, B = fs_samples.shape
            get_fs = lambda b: fs_samples[:, b]
        elif fs_samples.shape[1] == K_expected:
            B, K = fs_samples.shape
            get_fs = lambda b: fs_samples[b, :]
        else:
            raise ValueError(
                f"Inconsistent fs_samples shape={fs_samples.shape} with K_expected={K_expected}."
            )

        ens = BootstrapEnsemble()

        # -----------------------
        # Allocate storage
        # -----------------------
        if store in ("M", "full"):
            ens.M = np.empty((B, plan.N, plan.N), dtype=float)
            if compute_C_store:
                ens.C = np.empty((B, plan.N, plan.N), dtype=float)

        if store == "eigvals":
            out_dim = plan.N if k is None else int(k)
            ens.eigvals = np.empty((B, out_dim), dtype=float)

        if store == "eigvecs":
            kk = int(k)
            ens.eigvals = np.empty((B, kk), dtype=float)
            ens.eigvecs = np.empty((B, plan.N, kk), dtype=float)

        if need_gh:
            ens.F0 = np.empty((B,), dtype=float)
            ens.g  = np.empty((B, plan.N), dtype=float)
            ens.H  = np.empty((B, plan.N, plan.N), dtype=float)

        # -----------------------
        # Draw loop
        # -----------------------
        for b in range(B):
            fs_b = get_fs(b)  # (K,) vector of WH coefficients for this draw
            g_b, H_b = gh_from_fs(fs_b, plan, diag_zero=True)

            if need_gh:
                ens.F0[b] = float(_get_F0_from_fs(fs_b, W))
                ens.g[b] = g_b
                ens.H[b] = H_b

            if not need_M:
                # store="gh" only: do not build M or spectrum
                continue

            M_b = M_second_order_from_gh(g_b, H_b)

            if store in ("M", "full"):
                ens.M[b] = M_b
                if compute_C_store:
                    ens.C[b] = C_from_Mg(M_b, g_b)

            if store == "eigvals":
                vals_b, _ = eigensolve(M_b, solver=solver, k=k, descending=True)
                ens.eigvals[b] = vals_b

            elif store == "eigvecs":
                vals_b, vecs_b = eigensolve(M_b, solver=solver, k=int(k), descending=True)
                if align_eigvecs and (Vref is not None):
                    vecs_b = _align_vecs_to_ref(vecs_b, Vref)
                ens.eigvals[b] = vals_b
                ens.eigvecs[b] = vecs_b

        boot_dict[str(key)] = ens

    return BootstrapMResult(
        boot=boot_dict,
        meta={
            "method": "second_order",
            "store": store,
            "solver": solver,
            "k": k,
            "compute_C": compute_C,
            "compute_C_store": compute_C_store,
            "which": tuple(which),
        },
    )


# ============================================================
# Backwards-compatible helpers required by qinference.py
# ============================================================

def _get_fs_bootstrap_samples(W: Any, which: Which) -> np.ndarray:
    """
    Fetch WH bootstrap coefficient samples stored in W.meta.

    Inputs
    ------
    W : WalshResults-like
        Must expose W.meta populated by walsh_analysis.
    which : {"null","uncertainty"}
        Which bootstrap ensemble to fetch.

    Outputs
    -------
    fs_samples : np.ndarray, shape (K, B)
        Bootstrap samples of WH coefficients.
    """
    if not hasattr(W, "meta") or W.meta is None:
        raise ValueError("W has no .meta; cannot fetch bootstrap samples.")

    meta = W.meta

    # ---- NULL ----
    if which == "null":
        # preferred keys
        if isinstance(meta, dict) and "null" in meta and isinstance(meta["null"], dict):
            for key in ("fs_null_b", "fs_boot", "fs_b", "fs_null"):
                if key in meta["null"]:
                    arr = np.asarray(meta["null"][key], dtype=float)
                    return arr.T if (arr.ndim == 2 and arr.shape[0] != W.mean.shape[0]) else arr

        # fallbacks (older/other conventions)
        for key in ("fs_null_b", "fs_null", "null_fs", "fsb_null"):
            if isinstance(meta, dict) and key in meta:
                arr = np.asarray(meta[key], dtype=float)
                return arr.T if (arr.ndim == 2 and arr.shape[0] != W.mean.shape[0]) else arr

        raise ValueError("Missing null bootstrap in W.meta (expected meta['null']['fs_null_b'] or similar). "
                         "Run walsh_analysis with B_null>0.")

    # ---- UNCERTAINTY ----
    if which == "uncertainty":
        if isinstance(meta, dict) and "uncertainty" in meta and isinstance(meta["uncertainty"], dict):
            for key in ("fs_unc_b", "fs_boot", "fs_b", "fs_unc", "fs_uncertainty_b"):
                if key in meta["uncertainty"]:
                    arr = np.asarray(meta["uncertainty"][key], dtype=float)
                    return arr.T if (arr.ndim == 2 and arr.shape[0] != W.mean.shape[0]) else arr

        for key in ("fs_unc_b", "fs_unc", "unc_fs", "fsb_unc"):
            if isinstance(meta, dict) and key in meta:
                arr = np.asarray(meta[key], dtype=float)
                return arr.T if (arr.ndim == 2 and arr.shape[0] != W.mean.shape[0]) else arr

        raise ValueError("Missing uncertainty bootstrap in W.meta (expected meta['uncertainty']['fs_unc_b'] or similar). "
                         "Run walsh_analysis with B_uncertainty>0.")

    raise ValueError("which must be 'null' or 'uncertainty'.")


# qinference old import name (some code expects this)
# If your file defines bootstrap_second_order_from_walsh, alias it:
if "bootstrap_second_order_from_walsh" in globals() and "bootstrap_second_moment_from_walsh" not in globals():
    bootstrap_second_moment_from_walsh = bootstrap_second_order_from_walsh


###############################################################
#                    PUBLIC WRAPPER: compute_M                #
###############################################################

@dataclass
class ComputeMResult:
    """
    Outputs
    -------
    method : str
        "second_order" or "df".
    M : np.ndarray, shape (N,N)
        Observed M.
    C : np.ndarray or None, shape (N,N)
        Observed C = M - g g^T, if requested and available.
    g : np.ndarray or None
        Observed g (available if method="second_order" OR method="df" with compute_C=True).
    H : np.ndarray or None
        Observed H (only for method="second_order").
    eigvals : np.ndarray or None
        Eigenvalues of observed M (full or top-k).
    eigvecs : np.ndarray or None
        Eigenvectors of observed M (N,k) if requested.
    df_info : dict or None
        For method="df", diagnostics: {"mode","coverage","n_valid","is_full_hypercube"}.
    bootstrap : BootstrapMResult or None
        Dict-like wrapper for bootstrap ensembles. Keys: "null" and/or "uncertainty".
    meta : dict
        Extra metadata.
    """
    method: str
    M: np.ndarray
    C: Optional[np.ndarray]
    g: Optional[np.ndarray]
    H: Optional[np.ndarray]
    eigvals: Optional[np.ndarray]
    eigvecs: Optional[np.ndarray]
    df_info: Optional[Dict[str, Any]]
    bootstrap: Optional[BootstrapMResult]
    meta: Dict[str, Any]


def compute_M(
    obj: Any,
    *,
    method: Method = "second_order",
    # --- observed extras ---
    compute_C: bool = False,
    compute_spectrum: bool = True,
    store_obs_eigvecs: bool = False,
    solver: Solver = "eigh",
    k: Optional[int] = None,
    # --- second-order inputs ---
    W: Any = None,
    plan: Optional[GHPlan] = None,
    walsh_analysis_fn: Any = None,
    walsh_kwargs: Optional[Dict[str, Any]] = None,
    # --- DF inputs ---
    df_auto: bool = True,
    # --- bootstrap ---
    bootstrap: bool = False,
    which: Sequence[Which] = ("null",),
    store_boot: Store = "eigvals",  # (1) lighter default than "gh"
    B_null: int = 200,
    B_uncertainty: int = 200,
    multipliers: str = "rademacher",
    uncertainty_flavor: str = "wildcluster",
    preserve_residual_corr: bool = True,
    rng: Optional[np.random.Generator] = None,
) -> ComputeMResult:
    """
    Compute the second moment matrix of the discrete gradient field:
        M = <DF DF^T>.

    Inputs
    ------
    obj :
        Landscape-like object exposing:
          - obj.states : (M,N) in {0,1}
          - obj.values : (M,R) replicates (float) OR (M,) single measurement
    method : {"second_order","df"}
        - "second_order": M ≈ g g^T + H H^T using WH coefficients (g=f_i, H=f_ij)
        - "df": compute M from finite differences DF over observed states
    compute_C : bool
        If True, return C = M - g g^T (requires g).
    compute_spectrum : bool
        If True, compute eigenvalues (and optionally eigenvectors) of observed M.
    store_obs_eigvecs : bool
        If True, store eigenvectors of observed M.
    bootstrap : bool
        If True, compute bootstrap ensembles for requested `which`:
          - second_order: via WalshResults.meta (requires walsh_analysis_fn)
          - df: via stats_noise.bootstrap_null/bootstrap_uncertainty (requires >= 2 replicates)
        If df has < 2 replicates, bootstrap is skipped silently (bootstrap=None).

    Outputs
    -------
    res : ComputeMResult
    """
    import warnings

    if rng is None:
        rng = np.random.default_rng()

    states01 = np.asarray(getattr(obj, "states"), dtype=int)
    values = np.asarray(getattr(obj, "values"), dtype=float)
    if values.ndim == 1:
        values = values[:, None]

    # ---------------------------------------------------------
    # Validate store_boot depending on method (IMPORTANT)
    # ---------------------------------------------------------
    _valid_store_second_order = {"eigvals", "eigvecs", "M", "gh", "full"}
    _valid_store_df = {"eigvals", "eigvecs", "M"}

    if method == "df":
        if store_boot not in _valid_store_df:
            raise ValueError(
                f"method='df' does not support store_boot={store_boot!r}. "
                f"Use one of {_valid_store_df}."
            )
    elif method == "second_order":
        if store_boot not in _valid_store_second_order:
            raise ValueError(
                f"method='second_order' store_boot must be one of {_valid_store_second_order}, "
                f"got {store_boot!r}."
            )
    else:
        raise ValueError("method must be 'second_order' or 'df'.")

    # (2) Early, explicit check for eigvec bootstrap needing k
    if bootstrap and store_boot == "eigvecs" and k is None:
        raise ValueError("bootstrap with store_boot='eigvecs' requires k (top-k eigenvectors).")

    # =========================================================
    # 1) OBSERVED M (and g/H when available)
    # =========================================================
    M_obs: np.ndarray
    g_obs: Optional[np.ndarray] = None
    H_obs: Optional[np.ndarray] = None
    df_info: Optional[Dict[str, Any]] = None

    if method == "second_order":
        # If bootstrap requested, we must run walsh_analysis with B_* > 0
        if W is None:
            if walsh_analysis_fn is None:
                raise ValueError("method='second_order' requires W or walsh_analysis_fn.")
            kw = {} if walsh_kwargs is None else dict(walsh_kwargs)
            kw.setdefault("as_dataframe", False)
            if bootstrap:
                kw["B_null"] = int(B_null) if ("null" in set(which)) else 0
                kw["B_uncertainty"] = int(B_uncertainty) if ("uncertainty" in set(which)) else 0
            W = walsh_analysis_fn(obj, **kw)

        if not hasattr(W, "mean") or not hasattr(W, "s_bits") or not hasattr(W, "orders"):
            raise ValueError("W must have .mean/.s_bits/.orders for method='second_order'.")

        if plan is None:
            plan = build_gh_plan(W.s_bits, W.orders)

        fs = np.asarray(W.mean, dtype=float)
        g_obs, H_obs = gh_from_fs(fs, plan, diag_zero=True)
        M_obs = M_second_order_from_gh(g_obs, H_obs)

    else:  # method == "df"
        if hasattr(obj, "mean_over_replicates"):
            F_mean = np.asarray(obj.mean_over_replicates(), dtype=float)
        else:
            F_mean = np.nanmean(values, axis=1)

        out = M_from_DF_auto(states01, F_mean) if df_auto else M_from_DF_auto(states01, F_mean)
        M_obs = out["M"]
        df_info = {kk: vv for kk, vv in out.items() if kk not in ("M", "D")}

        if compute_C:
            D = np.asarray(out["D"], dtype=float)
            g_obs = np.nanmean(D, axis=0)

    # =========================================================
    # 2) OBSERVED C
    # =========================================================
    C_obs: Optional[np.ndarray] = None
    if compute_C:
        if g_obs is None:
            raise ValueError("compute_C=True requires g to be available.")
        C_obs = C_from_Mg(M_obs, g_obs)

    # ---------------------------------------------------------
    # Bootstrap-specific handling of C storage
    # ---------------------------------------------------------
    if method == "df":
        compute_C_boot = bool(compute_C and (store_boot == "M"))
    else:  # second_order
        compute_C_boot = bool(compute_C and (store_boot in ("M", "full")))

    # (3) Warn if user asked for C but bootstrap won't store it
    if bootstrap and compute_C and not compute_C_boot:
        warnings.warn(
            "compute_C=True: computed C_obs, but bootstrap C will not be stored "
            f"(method={method}, store_boot={store_boot}).",
            RuntimeWarning,
        )

    # =========================================================
    # 3) OBSERVED SPECTRUM
    # =========================================================
    eigvals: Optional[np.ndarray] = None
    eigvecs: Optional[np.ndarray] = None
    if compute_spectrum:
        vals, vecs = eigensolve(M_obs, solver=solver, k=k, descending=True)
        eigvals = vals
        eigvecs = vecs if store_obs_eigvecs else None
        u1 = vecs[:, 0].copy()
        u1 = orient_u1_with_popcount(u1, states01)
        vecs[:, 0] = u1

    # =========================================================
    # 4) BOOTSTRAP
    # =========================================================
    boot_res: Optional[BootstrapMResult] = None
    if bootstrap:
        which_set = set(which)

        if method == "df":
            # stats_noise expects [state cols | replicate cols]
            F_df = _F_df_for_stats_bootstrap(states01, values)
            R = int(values.shape[1])
            if R < 2:
                boot_res = None
            else:
                boot_dict: Dict[str, BootstrapEnsemble] = {}

                if ("uncertainty" in which_set) and (B_uncertainty > 0):
                    boot_df_unc = bootstrap_uncertainty(
                        F_df,
                        B=int(B_uncertainty),
                        multipliers=multipliers,
                        flavor=uncertainty_flavor,
                        rng=rng,
                    )
                    out_unc = compute_M_from_boot_df(
                        boot_df_unc,
                        store=store_boot,
                        compute_C=compute_C_boot,
                        k=(k if store_boot == "eigvecs" else None),
                        solver=solver,
                    )
                    boot_dict["uncertainty"] = BootstrapEnsemble(
                        eigvals=out_unc.eigvals,
                        eigvecs=out_unc.eigvecs,
                        M=out_unc.M,
                        C=out_unc.C,
                    )

                if ("null" in which_set) and (B_null > 0):
                    boot_df_null = bootstrap_null(
                        F_df,
                        B=int(B_null),
                        multipliers=multipliers,
                        preserve_residual_corr=preserve_residual_corr,
                        rng=rng,
                    )
                    out_null = compute_M_from_boot_df(
                        boot_df_null,
                        store=store_boot,
                        compute_C=compute_C_boot,
                        k=(k if store_boot == "eigvecs" else None),
                        solver=solver,
                    )
                    boot_dict["null"] = BootstrapEnsemble(
                        eigvals=out_null.eigvals,
                        eigvecs=out_null.eigvecs,
                        M=out_null.M,
                        C=out_null.C,
                    )

                boot_res = (
                    BootstrapMResult(
                        boot=boot_dict,
                        meta={
                            "method": "df",
                            "store": store_boot,
                            "solver": solver,
                            "k": k,
                            "compute_C": compute_C_boot,
                            "which": tuple(which),
                            "B_null": int(B_null),
                            "B_uncertainty": int(B_uncertainty),
                        },
                    )
                    if len(boot_dict) > 0
                    else None
                )

        else:  # second_order
            if W is None:
                raise RuntimeError("Internal error: W is None in second_order bootstrap path.")

            boot_res = bootstrap_second_order_from_walsh(
                W,
                plan=plan,
                which=tuple(which),
                store=store_boot,  # supports "gh"/"full"
                compute_C=compute_C_boot,
                solver=solver,
                k=k,
            )

            # Robust emptiness check (BootstrapMResult is dict-like: .boot is the dict)
            if (
                (boot_res is None)
                or (not hasattr(boot_res, "boot"))
                or (boot_res.boot is None)
                or (len(boot_res.boot) == 0)
            ):
                boot_res = None

    return ComputeMResult(
        method=str(method),
        M=M_obs,
        C=C_obs,
        g=g_obs,
        H=H_obs,
        eigvals=eigvals,
        eigvecs=eigvecs,
        df_info=df_info,
        bootstrap=boot_res,
        meta={},
    )


#############################################
#                   PCAS                    #
#############################################

Spins = Literal["01", "pm1"]

def project_states_on_pcs(
    states01: np.ndarray,
    eigvecs: np.ndarray,
    *,
    k: int = 2,
    spins: Spins = "01",
    center: bool = False,
    normalize_pcs: bool = True,
) -> np.ndarray:
    """
    Project binary states onto the first k principal components (PCs).

    This is a pure coordinate transform:
        nu = X @ U_k
    where U_k are the first k eigenvectors (columns) of M.

    Inputs
    ------
    states01 : np.ndarray, shape (M, N)
        Binary states in {0,1}^N.
    eigvecs : np.ndarray, shape (N, >=k)
        Eigenvectors of M as columns (e.g., output of eigensolve / compute_M with store_obs_eigvecs=True).
    k : int
        Number of PCs to project onto.
    spins : {"01", "pm1"}
        - "01": use X in {0,1} (as provided).
        - "pm1": internally map to Z = 2X - 1 before projection.
    center : bool
        If True, subtract the mean state vector before projection (in the chosen spin convention).
    normalize_pcs : bool
        If True, normalize each PC column to unit norm (robust to accidental scaling).

    Outputs
    -------
    nu : np.ndarray, shape (M, k)
        Latent coordinates nu[:,j] = <state, u_j>.
    """
    X = np.asarray(states01, dtype=float)
    if X.ndim != 2:
        raise ValueError("states01 must be 2D (M,N).")

    U = np.asarray(eigvecs, dtype=float)
    if U.ndim != 2:
        raise ValueError("eigvecs must be 2D (N,K).")

    M, N = X.shape
    if U.shape[0] != N:
        raise ValueError(f"eigvecs has incompatible shape {U.shape}; expected first dim N={N}.")

    if k <= 0:
        raise ValueError("k must be a positive integer.")
    if U.shape[1] < k:
        raise ValueError(f"eigvecs has only {U.shape[1]} columns, but k={k} was requested.")

    U = U[:, :k].copy()

    if normalize_pcs:
        norms = np.linalg.norm(U, axis=0)
        if np.any(norms == 0.0):
            raise ValueError("At least one requested PC has zero norm.")
        U /= norms[None, :]

    if spins == "pm1":
        X = 2.0 * X - 1.0
    elif spins != "01":
        raise ValueError("spins must be '01' or 'pm1'.")

    if center:
        X = X - X.mean(axis=0, keepdims=True)

    nu = X @ U
    return nu

def explain_latent_regions_by_presence(
    *,
    nu: np.ndarray,
    pc: int = 0,
    method: str = "quantile",
    q: tuple[float, float] = (25, 75),
    return_masks: bool = True,
    L=None,
    states01: np.ndarray | None = None,
    feature_names: list[str] | None = None,
):
    """
    Identify which binary features (species/genes) best explain
    the separation between two regions in latent (PC) space.

    The function splits the dataset into two regions along a chosen
    latent coordinate (PC) and compares presence probabilities of
    each feature between the regions.

    -----------------
    Input
    -----------------
    nu : ndarray, shape (X, K)
        Latent coordinates of the X samples (e.g. PCA scores,
        learned latent variables).

    pc : int, default=1
        Index of the latent coordinate used to define the split.
        Example: pc=1 corresponds to the second PC (nu[:, 1]).

    method : {"median", "quantile"}, default="quantile"
        Strategy to define the two regions in latent space.
        - "median": split at the median of nu[:, pc]
        - "quantile": compare lower and upper quantiles

    q : tuple(float, float), default=(25, 75)
        Percentiles used if method="quantile".
        Points between the two percentiles are ignored.

    return_masks : bool, default=True
        Whether to include boolean masks identifying region A,
        region B, and ignored points.

    L : Landscape-like object, optional
        Object providing:
        - L.states : ndarray (X, N), binary presence/absence matrix
        - L.feature_names : list[str], names of the N features
        If provided, states01 and feature_names are ignored.

    states01 : ndarray, shape (X, N), optional
        Binary presence/absence matrix (0/1).
        Required if L is not provided.

    feature_names : list[str], optional
        Names of the N binary features.
        If not provided, generic names "x1", "x2", ... are used.

    -----------------
    Output
    -----------------
    result : dict with keys

        delta_p : ndarray, shape (N,)
            Difference in presence probability between regions:
            delta_p = pA - pB.

        pA : ndarray, shape (N,)
            Mean presence probability of each feature in region A.

        pB : ndarray, shape (N,)
            Mean presence probability of each feature in region B.

        ranking : ndarray, shape (N,)
            Indices of features sorted by decreasing |delta_p|.

        top_idx : int
            Index of the feature with the largest |delta_p|.

        top_name : str
            Name of the feature with the largest |delta_p|.

        pc : int
            Latent coordinate used for the split.

        method : str
            Method used to define the regions ("median" or "quantile").

        thresholds : tuple
            Threshold(s) defining the regions in latent space.

        region : ndarray of shape (X,)
            Labels {"A", "B", "ignored"} for each sample.

        mask_A, mask_B, mask_ignored : ndarray of bool, optional
            Boolean masks for the two regions and ignored points
            (only if return_masks=True).

    -----------------
    Notes
    -----------------
    - This is a diagnostic / interpretability tool, not an inference
      procedure.
    - Large |delta_p| values indicate features that strongly
      distinguish the two latent regions.
    - If several features have comparable |delta_p|, the separation
      is multivariate rather than driven by a single feature.
    """

     # -----------------------
    # Resolve inputs
    # -----------------------
    nu = np.asarray(nu, dtype=float)
    if nu.ndim != 2:
        raise ValueError("nu must be a 2D array (X, K)")

    if L is not None:
        # Landscape in your core has: L.states and L.feature_names :contentReference[oaicite:1]{index=1}
        states01 = np.asarray(L.states, dtype=float)
        feature_names = list(getattr(L, "feature_names", None) or [])
        if len(feature_names) == 0:
            # fallback
            feature_names = [f"x{i+1}" for i in range(states01.shape[1])]
    else:
        if states01 is None:
            raise ValueError("Provide either L or states01.")
        states01 = np.asarray(states01, dtype=float)
        if states01.ndim != 2:
            raise ValueError("states01 must be a 2D array (X, N)")
        if feature_names is None:
            feature_names = [f"x{i+1}" for i in range(states01.shape[1])]
        else:
            if len(feature_names) != states01.shape[1]:
                raise ValueError("feature_names length must match states01.shape[1]")

    if states01.shape[0] != nu.shape[0]:
        raise ValueError("states01 and nu must have the same number of rows")

    X, N = states01.shape

    # -----------------------
    # Define regions in PC space
    # -----------------------
    coord = nu[:, pc]

    if method == "median":
        thr = float(np.median(coord))
        mask_A = coord <= thr
        mask_B = coord > thr
        thresholds = (thr,)
        mask_ignored = np.zeros(X, dtype=bool)

    elif method == "quantile":
        qlow, qhigh = np.percentile(coord, q)
        mask_A = coord <= qlow
        mask_B = coord >= qhigh
        thresholds = (float(qlow), float(qhigh))
        mask_ignored = ~(mask_A | mask_B)

    else:
        raise ValueError("method must be 'median' or 'quantile'")

    if mask_A.sum() == 0 or mask_B.sum() == 0:
        raise RuntimeError("One of the regions is empty. Adjust thresholds.")

    # -----------------------
    # Presence probabilities
    # -----------------------
    pA = states01[mask_A].mean(axis=0)
    pB = states01[mask_B].mean(axis=0)
    delta_p = pA - pB

    ranking = np.argsort(np.abs(delta_p))[::-1]
    top_idx = int(ranking[0])
    top_name = feature_names[top_idx]

    # Region labels for plotting
    region = np.full(X, fill_value="ignored", dtype=object)
    region[mask_A] = "A"
    region[mask_B] = "B"

    result = {
        "delta_p": delta_p,
        "pA": pA,
        "pB": pB,
        "ranking": ranking,
        "top_idx": top_idx,
        "top_name": top_name,
        "pc": pc,
        "method": method,
        "thresholds": thresholds,
        "region": region,
    }

    if return_masks:
        result["mask_A"] = mask_A
        result["mask_B"] = mask_B
        result["mask_ignored"] = mask_ignored

    return result

def plot_nu_regions_with_top_feature(
    *,
    nu: np.ndarray,
    split_info: dict,
    pcx: int = 0,
    pcy: int = 1,
    s: float = 12.0,
    alpha: float = 0.8,
    show_ignored: bool = True,
    show_plot: bool = True,
    return_plot_data: bool = False,
):
    """
    Scatter plot of latent coordinates colored by region,
    with optional return of plotting data and display control.

    -----------------
    Input
    -----------------
    nu : ndarray, shape (X, K)
        Latent coordinates of the samples.

    split_info : dict
        Output of explain_latent_regions_by_presence().
        Must contain at least:
        "region", "top_name", "top_idx", "delta_p".

    pcx, pcy : int
        Indices of the latent coordinates plotted on x- and y-axis.

    s : float
        Marker size.

    alpha : float
        Marker transparency.

    show_ignored : bool, default=True
        Whether to show samples in the ignored (middle) region.

    show_plot : bool, default=True
        If True, display the plot using matplotlib.
        If False, the figure is created but not shown.

    return_plot_data : bool, default=False
        If True, also return a dictionary containing the data
        used for plotting.

    -----------------
    Output
    -----------------
    fig : matplotlib.figure.Figure
        The created figure.

    ax : matplotlib.axes.Axes
        The axes containing the scatter plot.

    plot_data : dict, optional
        Returned only if return_plot_data=True.
        Contains coordinates by region and metadata.

    -----------------
    Notes
    -----------------
    - This function performs no analysis; it only visualizes
      the diagnostic split.
    - Suitable for interactive exploration and batch figure
      generation.
    """

    nu = np.asarray(nu, float)
    region = split_info["region"]

    top_idx = split_info["top_idx"]
    top_name = split_info["top_name"]
    delta = split_info["delta_p"][top_idx]

    mask_A = region == "A"
    mask_B = region == "B"
    mask_I = region == "ignored"

    # -----------------
    # Prepare plot data
    # -----------------
    plot_data = {
        "A": {"x": nu[mask_A, pcx], "y": nu[mask_A, pcy]},
        "B": {"x": nu[mask_B, pcx], "y": nu[mask_B, pcy]},
        "ignored": {"x": nu[mask_I, pcx], "y": nu[mask_I, pcy]},
        "pcx": pcx,
        "pcy": pcy,
        "top_name": top_name,
        "delta_p": float(delta),
    }

    # -----------------
    # Create plot
    # -----------------
    fig, ax = plt.subplots()

    ax.scatter(plot_data["A"]["x"], plot_data["A"]["y"],
               s=s, alpha=alpha, label="Region A")

    ax.scatter(plot_data["B"]["x"], plot_data["B"]["y"],
               s=s, alpha=alpha, label="Region B")

    if show_ignored and mask_I.any():
        ax.scatter(plot_data["ignored"]["x"], plot_data["ignored"]["y"],
                   s=s, alpha=0.2, label="Ignored")

    ax.set_xlabel(f"nu_{pcx+1}")
    ax.set_ylabel(f"nu_{pcy+1}")
    ax.set_title(f"Top separator: {top_name} (Δp={delta:+.3f})")
    ax.legend()

    if show_plot:
        plt.show()

    if return_plot_data:
        return fig, ax, plot_data

    return fig, ax


######################################################
#             PCAS: STRONG REGINALITY 2D             #
######################################################

@dataclass
class StrongRegionalityResult:
    """
    Strong regionality gate (rank-2 gradient geometry) using (u1, u2).

    Attributes
    ----------
    j_gate : int
        Index of the gating feature j maximizing |Δb_j|.
    gate_name : str
        Human-readable feature name (from L.feature_names[j_gate]).
    delta_b : ndarray, shape (N,)
        Δb_j = E[b(z) | z_j=+1] - E[b(z) | z_j=-1] for each feature j.
    delta_b_gate : float
        Δb_{j_gate}.
    mask_present : ndarray, shape (X,), dtype=bool
        Boolean mask for states with x_j=1 (equivalently z_j=+1).
    mask_absent : ndarray, shape (X,), dtype=bool
        Boolean mask for states with x_j=0 (equivalently z_j=-1).
    u1 : ndarray, shape (N,)
        First eigenvector of M used as dominant collective direction.
    u2 : ndarray, shape (N,)
        Second eigenvector of M used as "regional" direction.
    a_of_z : ndarray, shape (X,), optional
        a(z) = u1^T DF(z), returned if return_arrays=True.
    b_of_z : ndarray, shape (X,), optional
        b(z) = u2^T DF(z), returned if return_arrays=True.
    """
    j_gate: int
    gate_name: str
    delta_b: np.ndarray
    delta_b_gate: float
    mask_present: np.ndarray
    mask_absent: np.ndarray
    u1: np.ndarray
    u2: np.ndarray
    a_of_z: Optional[np.ndarray] = None
    b_of_z: Optional[np.ndarray] = None


def strong_regionality_gate(
    L,
    M_res,
    *,
    pc_u1: int = 0,
    pc_u2: int = 1,
    min_group_size: int = 8,
    return_arrays: bool = False,
) -> StrongRegionalityResult:
    """
    Identify a single gating feature j that best explains strong regionality in
    rank-2 gradient geometry.

    The idea:
      - Use second-order identity DF(z) ≈ g + H z
      - Use u2 (2nd eigenvector of M) to define b(z) = u2^T DF(z)
      - For each feature j, compute Δb_j = E[b | z_j=+1] - E[b | z_j=-1]
      - Gate is argmax_j |Δb_j|

    Input
    -----
    L : ep.core.Landscape
        Landscape with attributes:
          - states: (X,N) binary matrix in {0,1}
          - feature_names: list-like of length N
    M_res : ComputeMResult
        Output of ep.eglobal.compute_M with:
          - method == "second_order"
          - g: (N,)
          - H: (N,N)
          - eigvecs: (N,k) (must include u1,u2 columns requested)
    pc_u1 : int, default=0
        Column index in M_res.eigvecs used as u1.
    pc_u2 : int, default=1
        Column index in M_res.eigvecs used as u2.
    min_group_size : int, default=8
        Minimum number of samples required in each region (z_j=+1 and z_j=-1).
        If a feature doesn't meet this, its Δb_j is set to NaN.
    return_arrays : bool, default=False
        If True, return a_of_z and b_of_z arrays as well.

    Output
    ------
    StrongRegionalityResult
        Dataclass containing the gate feature, masks, u1/u2 and diagnostics.
    """
    # --- Validate inputs ---
    if getattr(M_res, "method", None) != "second_order":
        raise ValueError("strong_regionality_gate requires M_res.method == 'second_order'.")

    if M_res.g is None or M_res.H is None:
        raise ValueError("strong_regionality_gate requires M_res.g and M_res.H (second_order).")

    if M_res.eigvecs is None:
        raise ValueError("strong_regionality_gate requires M_res.eigvecs. "
                         "Call compute_M(..., compute_spectrum=True, store_obs_eigvecs=True).")

    Xstates = np.asarray(L.states, dtype=float)
    if Xstates.ndim != 2:
        raise ValueError("L.states must be a 2D array of shape (X,N).")

    X, N = Xstates.shape

    g = np.asarray(M_res.g, dtype=float)
    H = np.asarray(M_res.H, dtype=float)
    if g.shape != (N,):
        raise ValueError(f"M_res.g has shape {g.shape}, expected {(N,)}.")
    if H.shape != (N, N):
        raise ValueError(f"M_res.H has shape {H.shape}, expected {(N, N)}.")

    eigvecs = np.asarray(M_res.eigvecs, dtype=float)
    if eigvecs.shape[0] != N:
        raise ValueError(f"M_res.eigvecs has shape {eigvecs.shape}, expected (N,k) with N={N}.")

    if pc_u1 < 0 or pc_u1 >= eigvecs.shape[1]:
        raise ValueError(f"pc_u1={pc_u1} out of bounds for eigvecs with k={eigvecs.shape[1]}.")
    if pc_u2 < 0 or pc_u2 >= eigvecs.shape[1]:
        raise ValueError(f"pc_u2={pc_u2} out of bounds for eigvecs with k={eigvecs.shape[1]}.")

    u1 = eigvecs[:, pc_u1].copy()
    u2 = eigvecs[:, pc_u2].copy()

    # --- Build z-states and DF(z) ---
    Zstates = 2.0 * Xstates - 1.0  # z in {-1,+1}

    # DF(z) ≈ g + H z. Vectorized: (X,N) = (1,N) + (X,N) @ (N,N)^T
    DF = g[None, :] + (Zstates @ H.T)

    # a(z), b(z) amplitudes
    a_of_z = DF @ u1
    b_of_z = DF @ u2

    # --- Compute Δb_j for each feature ---
    delta_b = np.full(N, np.nan, dtype=float)
    for j in range(N):
        mask_plus = Zstates[:, j] > 0   # z_j = +1  <-> x_j = 1
        mask_minus = ~mask_plus         # z_j = -1  <-> x_j = 0

        if mask_plus.sum() < min_group_size or mask_minus.sum() < min_group_size:
            continue

        delta_b[j] = float(b_of_z[mask_plus].mean() - b_of_z[mask_minus].mean())

    if not np.any(np.isfinite(delta_b)):
        raise RuntimeError(
            "No valid Δb_j could be computed (all NaN). "
            "Try lowering min_group_size or check that both states (0/1) exist."
        )

    j_gate = int(np.nanargmax(np.abs(delta_b)))
    gate_name = str(L.feature_names[j_gate]) if hasattr(L, "feature_names") else str(j_gate)
    delta_b_gate = float(delta_b[j_gate])

    mask_present = Xstates[:, j_gate].astype(bool)
    mask_absent = ~mask_present

    res = StrongRegionalityResult(
        j_gate=j_gate,
        gate_name=gate_name,
        delta_b=delta_b,
        delta_b_gate=delta_b_gate,
        mask_present=mask_present,
        mask_absent=mask_absent,
        u1=u1,
        u2=u2,
    )

    if return_arrays:
        res.a_of_z = a_of_z
        res.b_of_z = b_of_z

    return res

#######################################################
#               k switching regionality               #
#######################################################

def explain_latent_regions_by_presence_k(
    *,
    nu: np.ndarray,
    pc: int = 0,
    method: str = "quantile",
    q: tuple[float, float] = (25, 75),
    k: int = 2,
    return_masks: bool = True,
    L: Any = None,
    states01: Optional[np.ndarray] = None,
    feature_names: Optional[list[str]] = None,
):
    """
    Generalize explain_latent_regions_by_presence to select top-k features and
    produce 2^k gating-region labels (bit patterns) for plotting in (nu1, nu2).

    Steps:
      1) Define two latent regions A/B along nu[:, pc] using median or quantiles.
      2) Rank features by |Δp_j| between A and B.
      3) Select top-k features and label each sample by its k-bit pattern.

    Notes:
      - This is an interpretability tool, not an inference procedure.
      - For k=1..2, the 2^k regions are easy to visualize and robust.
    """

    # -----------------------
    # Resolve inputs
    # -----------------------
    nu = np.asarray(nu, dtype=float)
    if nu.ndim != 2:
        raise ValueError("nu must be a 2D array (X, K)")

    if L is not None:
        states01 = np.asarray(L.states, dtype=float)
        feature_names = list(getattr(L, "feature_names", None) or [])
        if len(feature_names) == 0:
            feature_names = [f"x{i+1}" for i in range(states01.shape[1])]
    else:
        if states01 is None:
            raise ValueError("Provide either L or states01.")
        states01 = np.asarray(states01, dtype=float)
        if states01.ndim != 2:
            raise ValueError("states01 must be a 2D array (X, N)")
        if feature_names is None:
            feature_names = [f"x{i+1}" for i in range(states01.shape[1])]
        else:
            if len(feature_names) != states01.shape[1]:
                raise ValueError("feature_names length must match states01.shape[1]")

    if states01.shape[0] != nu.shape[0]:
        raise ValueError("states01 and nu must have the same number of rows")

    X, N = states01.shape
    if k <= 0 or k > N:
        raise ValueError("k must be in [1, N]")

    # -----------------------
    # Define regions A/B in latent space (same as your function)
    # -----------------------
    coord = nu[:, pc]

    if method == "median":
        thr = float(np.median(coord))
        mask_A = coord <= thr
        mask_B = coord > thr
        thresholds = (thr,)
        mask_ignored = np.zeros(X, dtype=bool)

    elif method == "quantile":
        qlow, qhigh = np.percentile(coord, q)
        mask_A = coord <= qlow
        mask_B = coord >= qhigh
        thresholds = (float(qlow), float(qhigh))
        mask_ignored = ~(mask_A | mask_B)

    else:
        raise ValueError("method must be 'median' or 'quantile'")

    if mask_A.sum() == 0 or mask_B.sum() == 0:
        raise RuntimeError("One of the regions is empty. Adjust thresholds.")

    # -----------------------
    # Rank features by |Δp|
    # -----------------------
    pA = states01[mask_A].mean(axis=0)
    pB = states01[mask_B].mean(axis=0)
    delta_p = pA - pB

    ranking = np.argsort(np.abs(delta_p))[::-1]
    top_idx = ranking[:k].astype(int)
    top_names = [feature_names[i] for i in top_idx]

    # -----------------------
    # Build 2^k gating labels (bit patterns) for each sample
    # -----------------------
    # Bits: b0 = state of top_idx[0], b1 = state of top_idx[1], ...
    # Pattern string: "00", "01", "10", "11" for k=2.
    Xk = (states01[:, top_idx] > 0.5).astype(int)  # (X, k) in {0,1}

    # Convert each row to an integer code in [0, 2^k - 1]
    # MSB = first selected feature (top_idx[0])
    weights = (1 << np.arange(k - 1, -1, -1))
    code = (Xk * weights).sum(axis=1).astype(int)

    pattern = np.array([format(c, f"0{k}b") for c in code], dtype=object)

    # Optionally, keep ignored points labeled explicitly
    region = np.array(pattern, dtype=object)
    region[mask_ignored] = "ignored"

    # Masks per pattern (useful for plotting)
    pattern_masks = None
    if return_masks:
        pattern_masks = {}
        for c in range(1 << k):
            label = format(c, f"0{k}b")
            pattern_masks[label] = (pattern == label) & (~mask_ignored)

    result = {
        "delta_p": delta_p,
        "pA": pA,
        "pB": pB,
        "ranking": ranking,
        "top_idx": top_idx,
        "top_names": top_names,
        "pc": pc,
        "method": method,
        "thresholds": thresholds,
        "mask_A": mask_A if return_masks else None,
        "mask_B": mask_B if return_masks else None,
        "mask_ignored": mask_ignored if return_masks else None,
        "code": code,              # integer in [0, 2^k-1]
        "pattern": pattern,        # e.g. "01"
        "region": region,          # pattern or "ignored"
        "pattern_masks": pattern_masks,  # dict[str, bool array] or None
    }

    return result
