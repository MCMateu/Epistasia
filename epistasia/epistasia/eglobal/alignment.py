"""eglobal.alignment

Alignment / orientational-order utilities for global epistasis.

This module is intentionally lightweight and model-agnostic:
it operates directly on collections of local response vectors
(`DF(z)` or any other N-dimensional vectors) and provides

  1) a nematic-like order parameter Q (0 = isotropic, 1 = perfectly aligned)
  2) the distribution of pairwise angles (via cos^2 of angles)
  3) a geometric null model for angle statistics in R^N

The intended use is:
  - compute DF(z) across configurations z (or across datasets)
  - normalize directions
  - quantify global alignment via Q
  - quantify microscopic alignment via pairwise angle distributions

"""

# eglobal/alignment.py
from __future__ import annotations

import pandas as pd
from math import lgamma
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, Literal

import numpy as np

# We only depend on DFResult as a container (computed in algebra.py)
try:
    from .algebra import DFResult  # type: ignore
except Exception:  # pragma: no cover
    DFResult = Any  # fallback for type checkers


U1Estimator = Literal["directional", "raw_moment"]
AngleMode = Literal["pairwise", "to_u1"]

###############################################
#                  CLASSES                    #
###############################################

@dataclass(frozen=True)
class U1Result:
    """Principal-direction extraction result."""
    u1: np.ndarray
    eigvals: np.ndarray
    eigvecs: Optional[np.ndarray]
    frac_u1: float


@dataclass(frozen=True)
class AlignmentResult:
    """Outputs for Q + angles, optionally with bootstrap ensembles."""
    Q: float
    u1: np.ndarray
    u1_info: Dict[str, Any]
    angles: Optional[np.ndarray]  # cos^2 angles
    angles_info: Optional[Dict[str, Any]]
    boot_unc_Q: Optional[np.ndarray]
    boot_null_Q: Optional[np.ndarray]
    boot_unc_angles: Optional[np.ndarray]
    boot_null_angles: Optional[np.ndarray]
    boot_info: Optional[Dict[str, Any]]


############################################
#       Core linear-algebra primitives     #
############################################

def compute_u1_from_M(
    M: np.ndarray,
    *,
    store_eigvecs: bool = False,
    eps: float = 1e-12,
) -> U1Result:
    """
    Compute principal direction u1 from a (symmetric) second-moment matrix M.

    Parameters
    ----------
    M : (N,N)
        Second-moment matrix.
    store_eigvecs : bool
        Whether to include full eigenvector matrix.
    eps : float
        Numerical tolerance.

    Returns
    -------
    U1Result
        u1 is the top eigenvector (unit norm).
        frac_u1 = lambda1 / tr(M).
    """
    M = np.asarray(M, dtype=float)
    if M.ndim != 2 or M.shape[0] != M.shape[1]:
        raise ValueError("M must be a square matrix (N,N).")
    N = M.shape[0]

    # Numerical safeguard only (M should be symmetric by construction)
    M = 0.5 * (M + M.T)

    eigvals, eigvecs = np.linalg.eigh(M)
    idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]

    u1 = eigvecs[:, 0]
    nrm = float(np.linalg.norm(u1))
    if nrm < eps:
        raise ValueError("Top eigenvector has near-zero norm; cannot define u1.")
    u1 = u1 / nrm

    tr = float(np.trace(M))
    frac_u1 = float(eigvals[0] / tr) if tr > eps else float("nan")

    return U1Result(
        u1=u1,
        eigvals=eigvals,
        eigvecs=eigvecs if store_eigvecs else None,
        frac_u1=frac_u1,
    )


def _row_normalize(D: np.ndarray, eps: float = 1e-12) -> Tuple[np.ndarray, np.ndarray]:
    """
    Normalize rows of D to unit length.

    Returns
    -------
    Dh : (K,N)
        Normalized rows for which norm > eps.
    keep : (K,) bool
        Mask selecting rows kept.
    """
    D = np.asarray(D, dtype=float)
    norms = np.linalg.norm(D, axis=1)
    keep = norms > eps
    Dh = D[keep] / norms[keep][:, None]
    return Dh, keep


def build_M_directional(
    D: np.ndarray,
    *,
    eps: float = 1e-12,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build M_dir = < d_hat d_hat^T > from DF samples D.

    Returns
    -------
    M_dir : (N,N)
    keep_mask : (K,) bool
        Which rows of D were used (non-zero vectors).
    """
    Dh, keep = _row_normalize(D, eps=eps)
    if Dh.shape[0] == 0:
        raise ValueError("All DF vectors have near-zero norm; cannot build directional M.")
    M = (Dh.T @ Dh) / float(Dh.shape[0])
    M = 0.5 * (M + M.T)
    return M, keep


# -------------------------------
# Q and angle statistics
# -------------------------------

def compute_Q_from_DF(
    D: np.ndarray,
    u1: np.ndarray,
    *,
    eps: float = 1e-12,
) -> float:
    """
    Compute Q using the user's definition:

    Q = < (N/(N-1)) * ( (DF·u1)^2 / ||DF||^2 ) - 1/(N-1) >_z

    Notes
    -----
    - Rows with ||DF|| <= eps are excluded.
    - u1 is normalized internally.
    """
    D = np.asarray(D, dtype=float)
    if D.ndim != 2:
        raise ValueError("D must be 2D (K,N).")
    K, N = D.shape

    u1 = np.asarray(u1, dtype=float)
    if u1.ndim != 1 or u1.shape[0] != N:
        raise ValueError(f"u1 must have shape ({N},).")
    u1 = u1 / max(float(np.linalg.norm(u1)), eps)

    norms2 = np.sum(D * D, axis=1)
    keep = norms2 > (eps * eps)
    if not np.any(keep):
        raise ValueError("All DF vectors have near-zero norm; cannot compute Q.")

    proj = D[keep] @ u1
    ratio = (proj * proj) / norms2[keep]  # (DF·u1)^2 / ||DF||^2
    S = float(np.mean(ratio))

    Q = (N / (N - 1.0)) * S - (1.0 / (N - 1.0))
    return float(np.clip(Q, 0.0, 1.0))


def cos2_angles_pairwise(
    D: np.ndarray,
    *,
    max_pairs: Optional[int] = None,
    rng: Optional[np.random.Generator] = None,
    eps: float = 1e-12,
) -> np.ndarray:
    """
    Return cos^2(angle) between pairs of DF directions.

    - We normalize each DF row to direction.
    - Return cos^2 to remove sign (nematic symmetry).
    """
    D = np.asarray(D, dtype=float)
    Dh, _ = _row_normalize(D, eps=eps)
    K = Dh.shape[0]
    if K < 2:
        return np.array([], dtype=float)

    if max_pairs is None:
        G = Dh @ Dh.T
        iu = np.triu_indices(K, k=1)
        y = G[iu] ** 2
        return np.clip(y.astype(float, copy=False), 0.0, 1.0)

    if rng is None:
        rng = np.random.default_rng()

    i = rng.integers(0, K, size=max_pairs)
    j = rng.integers(0, K, size=max_pairs)
    mask = i != j
    i, j = i[mask], j[mask]
    if i.size == 0:
        return np.array([], dtype=float)

    y = np.sum(Dh[i] * Dh[j], axis=1) ** 2
    return np.clip(y.astype(float, copy=False), 0.0, 1.0)


def cos2_angles_to_u1(
    D: np.ndarray,
    u1: np.ndarray,
    *,
    eps: float = 1e-12,
) -> np.ndarray:
    """
    Return cos^2(angle) between each DF direction and u1 (nematic).
    """
    D = np.asarray(D, dtype=float)
    K, N = D.shape

    u1 = np.asarray(u1, dtype=float)
    if u1.ndim != 1 or u1.shape[0] != N:
        raise ValueError(f"u1 must have shape ({N},).")
    u1 = u1 / max(float(np.linalg.norm(u1)), eps)

    Dh, keep = _row_normalize(D, eps=eps)
    if Dh.shape[0] == 0:
        return np.array([], dtype=float)

    y = (Dh @ u1) ** 2
    return np.clip(y.astype(float, copy=False), 0.0, 1.0)


# -------------------------------
# High-level: use DFResult
# -------------------------------

def compute_alignment(
    dfres: DFResult,
    *,
    u1_estimator: U1Estimator = "directional",
    angles: Optional[AngleMode] = None,
    max_pairs: Optional[int] = None,
    rng: Optional[np.random.Generator] = None,
    eps: float = 1e-12,
    bootstrap: bool = True,
) -> AlignmentResult:
    """
    Compute u1, Q, and optional angle distributions from a DFResult.

    Parameters
    ----------
    dfres : DFResult
        Output of algebra.compute_DF(...) containing D and optional bootstrap arrays.
    u1_estimator : {"directional","raw_moment"}
        How to estimate u1:
          - "directional": u1 = top eigvec of M_dir = < d_hat d_hat^T >
          - "raw_moment":  u1 = top eigvec of M_raw = < DF DF^T >
        Note: Q is ALWAYS computed from the user's definition using DF and u1.
    angles : {"pairwise","to_u1"} or None
        Which angle statistic to compute (cos^2 values).
    max_pairs : int or None
        If angles="pairwise": sample up to max_pairs pairs (None = all pairs).
    bootstrap : bool
        If True and dfres has bootstrap arrays, compute bootstrap distributions for Q and angles.

    Returns
    -------
    AlignmentResult
    """
    if rng is None:
        rng = np.random.default_rng()

    D = np.asarray(dfres.D, dtype=float)
    if D.ndim != 2:
        raise ValueError("dfres.D must be 2D (n_used, N).")
    n_used, N = D.shape

    # --- estimate u1 ---
    if u1_estimator == "directional":
        M_dir, keep = build_M_directional(D, eps=eps)
        u1res = compute_u1_from_M(M_dir, store_eigvecs=False, eps=eps)
        u1_info = {
            "u1_estimator": "directional",
            "n_used": int(n_used),
            "n_used_nonzero": int(np.sum(keep)),
            "frac_u1": u1res.frac_u1,
            "eigvals": u1res.eigvals,
        }
        u1 = u1res.u1
    elif u1_estimator == "raw_moment":
        M_raw = (D.T @ D) / float(n_used)
        M_raw = 0.5 * (M_raw + M_raw.T)
        u1res = compute_u1_from_M(M_raw, store_eigvecs=False, eps=eps)
        u1_info = {
            "u1_estimator": "raw_moment",
            "n_used": int(n_used),
            "frac_u1": u1res.frac_u1,
            "eigvals": u1res.eigvals,
        }
        u1 = u1res.u1
    else:
        raise ValueError("u1_estimator must be 'directional' or 'raw_moment'.")

    # --- Q (always from user's definition) ---
    Q = compute_Q_from_DF(D, u1, eps=eps)

    # --- angles (optional) ---
    ang = None
    ang_info = None
    if angles is not None:
        if angles == "pairwise":
            ang = cos2_angles_pairwise(D, max_pairs=max_pairs, rng=rng, eps=eps)
            ang_info = {
                "mode": "pairwise",
                "max_pairs": max_pairs,
                "n_angles": int(ang.size),
                "null_beta": cos2_null_beta_params(N),
            }
        elif angles == "to_u1":
            ang = cos2_angles_to_u1(D, u1, eps=eps)
            ang_info = {
                "mode": "to_u1",
                "n_angles": int(ang.size),
                "null_beta": cos2_null_beta_params(N),
            }
        else:
            raise ValueError("angles must be None, 'pairwise', or 'to_u1'.")

    # --- bootstrap ---
    boot_unc_Q = None
    boot_null_Q = None
    boot_unc_ang = None
    boot_null_ang = None
    boot_info = None

    if bootstrap:
        boot_info = dfres.boot_info if getattr(dfres, "boot_info", None) is not None else None

        def _process_boot(D_boot: np.ndarray) -> Tuple[np.ndarray, Optional[np.ndarray]]:
            # D_boot: (B, n_used, N)
            B = D_boot.shape[0]
            Qb = np.empty(B, dtype=float)
            Ab = None

            # Optionally store angles per bootstrap draw:
            # - pairwise: variable length if max_pairs=None; enforce max_pairs for bootstrap.
            # - to_u1: fixed length n_used_nonzero; OK.
            if angles is not None:
                if angles == "pairwise":
                    if max_pairs is None:
                        raise ValueError("For bootstrap with angles='pairwise', set max_pairs to a fixed value.")
                    Ab = np.empty((B, max_pairs), dtype=float)
                elif angles == "to_u1":
                    # length depends on how many nonzero DF rows there are in that draw;
                    # for simplicity we store variable-length as object arrays.
                    Ab = np.empty(B, dtype=object)

            for b in range(B):
                Db = D_boot[b]

                # Estimate u1 for this bootstrap draw using the same estimator.
                if u1_estimator == "directional":
                    M_dir_b, _ = build_M_directional(Db, eps=eps)
                    u1b = compute_u1_from_M(M_dir_b, store_eigvecs=False, eps=eps).u1
                else:
                    Mb = (Db.T @ Db) / float(Db.shape[0])
                    Mb = 0.5 * (Mb + Mb.T)
                    u1b = compute_u1_from_M(Mb, store_eigvecs=False, eps=eps).u1

                Qb[b] = compute_Q_from_DF(Db, u1b, eps=eps)

                if angles is not None:
                    if angles == "pairwise":
                        Ab[b, :] = cos2_angles_pairwise(Db, max_pairs=max_pairs, rng=rng, eps=eps)
                    elif angles == "to_u1":
                        Ab[b] = cos2_angles_to_u1(Db, u1b, eps=eps)

            return Qb, Ab

        if getattr(dfres, "boot_unc_D", None) is not None:
            boot_unc_Q, boot_unc_ang = _process_boot(dfres.boot_unc_D)

        if getattr(dfres, "boot_null_D", None) is not None:
            boot_null_Q, boot_null_ang = _process_boot(dfres.boot_null_D)

    return AlignmentResult(
        Q=Q,
        u1=u1,
        u1_info=u1_info,
        angles=ang,
        angles_info=ang_info,
        boot_unc_Q=boot_unc_Q,
        boot_null_Q=boot_null_Q,
        boot_unc_angles=boot_unc_ang,
        boot_null_angles=boot_null_ang,
        boot_info=boot_info,
    )

#################################################
#             ANGLES AND PLOTTINGS              #
#################################################

def cos2_null_beta_params(N: int) -> Tuple[float, float]:
    if N < 2:
        raise ValueError("N must be >= 2.")
    return 0.5, 0.5 * (N - 1)

def beta_pdf(y: np.ndarray, a: float, b: float, eps: float = 1e-12) -> np.ndarray:
    y = np.asarray(y, dtype=float)
    y = np.clip(y, eps, 1 - eps)
    logB = lgamma(a) + lgamma(b) - lgamma(a + b)
    return np.exp((a - 1) * np.log(y) + (b - 1) * np.log(1 - y) - logB)

def cos2_null_pdf(y: np.ndarray, N: int, eps: float = 1e-12) -> np.ndarray:
    a, b = cos2_null_beta_params(N)
    return beta_pdf(y, a, b, eps=eps)

def cos2_to_dataframe(
    cos2: np.ndarray,
    *,
    N: int,
    kind: str = "data",
    extra: Optional[Dict[str, Any]] = None,
) -> pd.DataFrame:
    cos2 = np.asarray(cos2, dtype=float)
    cos2 = cos2[np.isfinite(cos2)]
    cos2 = cos2[(cos2 >= 0) & (cos2 <= 1)]
    df = pd.DataFrame({"cos2": cos2})
    df["kind"] = kind
    df["N"] = int(N)
    if extra:
        for k, v in extra.items():
            df[k] = v
    return df

def null_cos2_curve_dataframe(
    *,
    N: int,
    grid_size: int = 400,
    extra: Optional[Dict[str, Any]] = None,
) -> pd.DataFrame:
    x = np.linspace(0, 1, grid_size)
    y = cos2_null_pdf(x, N)
    df = pd.DataFrame({"cos2": x, "density": y})
    df["kind"] = "null_pdf"
    df["N"] = int(N)
    if extra:
        for k, v in extra.items():
            df[k] = v
    return df


def plot_cos2_vs_null(
    cos2: np.ndarray,
    *,
    N: int,
    bins: int = 50,
    grid_size: int = 400,
    density: bool = True,
    show: bool = True,
    return_data: bool = False,
    title: Optional[str] = None,
    label_data: str = "data",
    label_null: str = "null",
):
    """
    Compare empirical cos^2(angle) distribution to the null Beta distribution.

    Parameters
    ----------
    cos2 : array-like
        Empirical cos^2(angle) values in [0,1].
    N : int
        Ambient dimension. Null is Beta(1/2, (N-1)/2).
    bins : int
        Number of histogram bins.
    grid_size : int
        Number of points for the null pdf grid.
    density : bool
        Whether to normalize the histogram to a density.
    show : bool
        If True, produce a matplotlib plot.
    return_data : bool
        If True, return the data used for plotting.
    title : str, optional
        Plot title.
    label_data : str
        Label for empirical data.
    label_null : str
        Label for null curve.

    Returns
    -------
    out : dict or None
        If return_data=True, returns a dictionary with:
            - 'hist_x' : bin centers
            - 'hist_y' : histogram values
            - 'null_x' : grid for null pdf
            - 'null_y' : null pdf values
            - 'N'      : dimension
    """
    import numpy as np

    # -------------------------
    # sanitize data
    # -------------------------
    y = np.asarray(cos2, dtype=float)
    y = y[np.isfinite(y)]
    y = y[(y >= 0.0) & (y <= 1.0)]
    if y.size == 0:
        raise ValueError("cos2 array is empty after filtering.")

    if N < 2:
        raise ValueError("N must be >= 2.")

    # -------------------------
    # histogram
    # -------------------------
    hist_y, edges = np.histogram(y, bins=bins, range=(0, 1), density=density)
    hist_x = 0.5 * (edges[:-1] + edges[1:])

    # -------------------------
    # null Beta pdf
    # -------------------------
    from math import lgamma

    def beta_pdf(x, a, b, eps=1e-12):
        x = np.clip(x, eps, 1 - eps)
        logB = lgamma(a) + lgamma(b) - lgamma(a + b)
        return np.exp((a - 1) * np.log(x) + (b - 1) * np.log(1 - x) - logB)

    a = 0.5
    b = 0.5 * (N - 1)
    null_x = np.linspace(0.0, 1.0, grid_size)
    null_y = beta_pdf(null_x, a, b)

    # -------------------------
    # plot (optional)
    # -------------------------
    if show:
        import matplotlib.pyplot as plt

        plt.figure()
        plt.bar(
            hist_x,
            hist_y,
            width=(edges[1] - edges[0]),
            alpha=0.6,
            label=label_data,
        )
        plt.plot(null_x, null_y, linewidth=2.0, label=label_null)
        plt.xlabel(r"$\cos^2(\omega)$")
        plt.ylabel("density" if density else "count")
        if title is None:
            title = rf"$\cos^2$ distribution vs null ($N={N}$)"
        plt.title(title)
        plt.legend()
        plt.tight_layout()
        plt.show()

    # -------------------------
    # return data (optional)
    # -------------------------
    if return_data:
        return {
            "hist_x": hist_x,
            "hist_y": hist_y,
            "null_x": null_x,
            "null_y": null_y,
            "N": N,
        }

    return None

