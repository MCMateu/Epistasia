"""
qinference.py

Inference utilities for the 2nd-order (quadratic) global-epistasis limit:
    F(z) ≈ F0 + g^T z + 1/2 z^T H z
Under a rank-1 (or dominant-mode) approximation of M, infer:
    q(ν) = F0 + a ν + b ν^2,   with ν = u1 · z
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

from math import lgamma

import numpy as np

from .algebra import (
    build_gh_plan,
    gh_from_fs,
    eigensolve,
    bootstrap_second_order_from_walsh,
    _get_fs_bootstrap_samples,
    M_second_order_from_gh, 
    orient_u1_with_popcount,
    StrongRegionalityResult,
)


from epistasia.walshhadamard import walsh_analysis

############################################
#                                          #
#           SECOND ORDER INFRENCE          #
#                                          #
############################################

@dataclass(frozen=True)
class QSecondOrderResult:
    """Container for q(ν)=F0 + a ν + b ν^2 inference results."""
    F0: float
    a: float
    b: float
    u1: np.ndarray

    # Bootstrap draws (if available)
    boot_a: Optional[np.ndarray] = None
    boot_b: Optional[np.ndarray] = None
    boot_F0: Optional[np.ndarray] = None
    boot_u1: Optional[np.ndarray] = None   # shape (B, N)

    # Diagnostics / extra outputs
    lambda1: Optional[float] = None # (u^T M u) 
    rayleigh:  Optional[float] = None  # (u^T M u) / tr(M)
    b_abs: Optional[float] = None
    sign_proxy: Optional[float] = None  # u^T H u (observed)


def _signed_b_from_M_g_H(lam1: float, a: float, u1: np.ndarray, H: np.ndarray) -> Tuple[float, float, float]:
    """
    Compute signed b from:
        |b| = sqrt(max(0, lam1 - a^2))
    and choose sign(b) = sign(u^T H u).

    Returns
    -------
    (b, b_abs, sign_proxy)
    """
    b_abs = float(np.sqrt(max(0.0, lam1 - a * a)))
    sign_proxy = float(u1 @ (H @ u1))
    sgn = float(np.sign(sign_proxy))
    if sgn == 0.0:
        sgn = 1.0
    b = sgn * b_abs
    return b, b_abs, sign_proxy


def compute_q_second_order(
    L: Any,
    *,
    which: str = "uncertainty",
    k: int = 1,
    align_eigvecs: bool = True,
    return_bootstrap: bool = True,
    walsh_kwargs: Optional[Dict[str, Any]] = None,
) -> QSecondOrderResult:
    """
    Infer quadratic q(ν)=F0 + a ν + b ν^2 in the second-order (rank-1) limit
    directly from a Landscape object.

    This function:
      1) runs walsh_analysis(L, **walsh_kwargs)
      2) constructs (g, H) from WH coefficients (2nd-order)
      3) builds M = g g^T + H^2 (observed)
      4) infers (F0, a, b, u1) from the dominant eigendirection of M
      5) optionally propagates uncertainty via bootstrap samples in W.meta,
         if available.

    Notes on robustness
    -------------------
    - If bootstrap samples are not present (e.g., B_uncertainty=0 or single replicate),
      the function returns point estimates and sets boot_* = None.
    - The sign of b is determined using sign(u^T H u), while |b| comes from
      sqrt(max(0, lambda1 - a^2)).

    Parameters
    ----------
    L : Landscape
        Epistasia Landscape object.
    which : {"uncertainty", "null"}
        Bootstrap ensemble to use (if available).
    k : int
        Number of eigenpairs to compute (only k=1 supported).
    align_eigvecs : bool
        Align eigenvectors across bootstrap draws to avoid sign flips.
    return_bootstrap : bool
        If True, returns bootstrap arrays for a, b, and F0 when possible.
    walsh_kwargs : dict, optional
        Keyword arguments passed to walsh_analysis.
        Defaults to:
            as_dataframe=False, B_null=0, B_uncertainty=0

    Returns
    -------
    QSecondOrderResult
        Point estimates and (optionally) bootstrap draws.
    """
    if k != 1:
        raise ValueError("compute_q_second_order supports only k=1 (dominant-mode inference).")
    if which not in ("uncertainty", "null"):
        raise ValueError("which must be 'uncertainty' or 'null'.")

    # -------------------------------------------------
    # Walsh analysis (internal)
    # -------------------------------------------------
    if walsh_kwargs is None:
        walsh_kwargs = {}
    # sensible, safe defaults (do NOT force bootstrap here)
    walsh_kwargs = {
        "as_dataframe": False,
        "B_null": 0,
        "B_uncertainty": 0,
        **walsh_kwargs,
    }

    # If the user requests bootstrap but the dataset has no replicates,
    # bootstrap may be meaningless. We degrade gracefully by disabling it.
    vals = np.asarray(getattr(L, "values", []))
    has_replicates = bool(vals.ndim == 2 and vals.shape[1] > 1)

    if return_bootstrap and not has_replicates:
        # Turn off bootstrap requests (graceful fallback)
        walsh_kwargs = dict(walsh_kwargs)
        walsh_kwargs["B_uncertainty"] = 0
        walsh_kwargs["B_null"] = 0
        return_bootstrap = False

    W = walsh_analysis(L, **walsh_kwargs)

    # -------------------------------------------------
    # Build plan and compute observed g, H
    # -------------------------------------------------
    plan = build_gh_plan(W.s_bits, W.orders)
    g_obs, H_obs = gh_from_fs(W.mean, plan)

    # -------------------------------------------------
    # Observed M and its dominant mode (always available)
    # -------------------------------------------------
    M_obs = M_second_order_from_gh(g_obs, H_obs)
    vals_obs, vecs_obs = eigensolve(M_obs, k=1)
    lam1_obs = float(vals_obs[0])
    rayleigh_obs = lam1_obs / np.trace(M_obs)
    u1_obs = vecs_obs[:, 0]

    states01 = np.asarray(W.s_bits, dtype=int)
    u1_obs = orient_u1_with_popcount(u1_obs, states01)

    # -------------------------------------------------
    # Infer a, signed b, F0 (point estimates)
    # -------------------------------------------------
    a_obs = float(g_obs @ u1_obs)
    b_obs, b_abs_obs, sign_proxy_obs = _signed_b_from_M_g_H(lam1_obs, a_obs, u1_obs, H_obs)
    F0_obs = float(W.mean[0])

    # -------------------------------------------------
    # Bootstrap distributions (only if samples exist)
    # -------------------------------------------------
    boot_a = boot_b = boot_F0 = None
    boot_u1 = None

    if return_bootstrap:
        meta = getattr(W, "meta", {}) or {}

        # Detect whether the requested ensemble exists
        if which == "uncertainty":
            has_samples = bool("uncertainty" in meta and "fs_unc_b" in meta["uncertainty"])
        else:  # which == "null"
            has_samples = bool("null" in meta and "fs_null_b" in meta["null"])

        if has_samples:
            # Use existing bootstrap machinery for eigenspectra (with alignment)
            out = bootstrap_second_order_from_walsh(
                W,
                which=(which,),
                store="eigvecs",
                k=1,
                compute_C=False,
                align_eigvecs=align_eigvecs,
            )

            fs_samples = _get_fs_bootstrap_samples(W, which)  # (K,B)
            ens = out.boot[which]              
            boot_u1 = ens.eigvecs[:,:,0]
            B = fs_samples.shape[1]

            boot_a = np.empty(B, dtype=float)
            boot_b = np.empty(B, dtype=float)
            boot_F0 = fs_samples[0, :].astype(float, copy=False)

            for t in range(B):
                fs_t = fs_samples[:, t]
                g_t, H_t = gh_from_fs(fs_t, plan)

                lam1_t = float(ens.eigvals[t, 0])
                u1_t = ens.eigvecs[t, :, 0]

                a_t = float(g_t @ u1_t)
                b_t, _, _ = _signed_b_from_M_g_H(lam1_t, a_t, u1_t, H_t)

                boot_a[t] = a_t
                boot_b[t] = b_t
        else:
            # No samples -> keep boot_* as None (graceful)
            boot_a = boot_b = boot_F0 = None
            boot_u1 = None

    return QSecondOrderResult(
        F0=F0_obs,
        a=a_obs,
        b=b_obs,
        u1=u1_obs,
        boot_a=boot_a,
        boot_b=boot_b,
        boot_u1 = boot_u1,
        boot_F0=boot_F0,
        lambda1=lam1_obs,
        rayleigh=rayleigh_obs,
        b_abs=b_abs_obs,
        sign_proxy=sign_proxy_obs,
    )

def compute_q_second_order_given_u1(
    L: Any,
    u1: np.ndarray,
    *,
    which: str = "uncertainty",
    align_u_with_popcount: bool = True,
    return_bootstrap: bool = True,
    walsh_kwargs: Optional[Dict[str, Any]] = None,
) -> QSecondOrderResult:
    """
    Infer q(ν)=F0 + a ν + b ν^2 in the 2nd-order limit CONDITIONAL on a user-supplied direction u1.

    This is the "fixed-u" companion of compute_q_second_order:
      - We still build g, H and M from Walsh coefficients,
      - but we do NOT compute eigenvectors of M.
      - Instead we set ν = u1 · z (with u1 fixed) and infer (a,b) from projections:
            a = g^T u
            λ_u = u^T M u
            |b| = sqrt(max(0, λ_u - a^2))
            sign(b) = sign(u^T H u)

    Notes
    -----
    - u1 is normalized internally.
    - Optionally we orient u1 with popcount for a consistent sign convention.

    Returns
    -------
    QSecondOrderResult with u1 = (possibly oriented/normalized) u1.
    """
    if which not in ("uncertainty", "null"):
        raise ValueError("which must be 'uncertainty' or 'null'.")

    # -------------------------
    # Walsh analysis
    # -------------------------
    if walsh_kwargs is None:
        walsh_kwargs = {}
    walsh_kwargs = {
        "as_dataframe": False,
        "B_null": 0,
        "B_uncertainty": 0,
        **walsh_kwargs,
    }

    vals = np.asarray(getattr(L, "values", []))
    has_replicates = bool(vals.ndim == 2 and vals.shape[1] > 1)
    if return_bootstrap and not has_replicates:
        walsh_kwargs = dict(walsh_kwargs)
        walsh_kwargs["B_uncertainty"] = 0
        walsh_kwargs["B_null"] = 0
        return_bootstrap = False

    W = walsh_analysis(L, **walsh_kwargs)

    # -------------------------
    # Build g, H, M (observed)
    # -------------------------
    plan = build_gh_plan(W.s_bits, W.orders)
    g_obs, H_obs = gh_from_fs(W.mean, plan)
    M_obs = M_second_order_from_gh(g_obs, H_obs)

    # -------------------------
    # Prepare u (normalize + optional orientation)
    # -------------------------
    u = np.asarray(u1, dtype=float).copy()
    nu = float(np.linalg.norm(u))
    if nu == 0.0:
        raise ValueError("u1 has zero norm.")
    u /= nu

    if align_u_with_popcount:
        states01 = np.asarray(W.s_bits, dtype=int)
        u = orient_u1_with_popcount(u, states01)

    # -------------------------
    # Point estimates (conditional on u)
    # -------------------------
    lam_u = float(u @ (M_obs @ u))
    rayleigh = lam_u / np.trace(M_obs)
    a_obs = float(g_obs @ u)
    b_obs, b_abs_obs, sign_proxy_obs = _signed_b_from_M_g_H(lam_u, a_obs, u, H_obs)
    F0_obs = float(W.mean[0])

    # -------------------------
    # Bootstrap (optional): project each bootstrap (g_t, H_t) onto the SAME fixed u
    # -------------------------
    boot_a = boot_b = boot_F0 = None
    boot_u1 = None

    if return_bootstrap:
        meta = getattr(W, "meta", {}) or {}
        if which == "uncertainty":
            has_samples = bool("uncertainty" in meta and "fs_unc_b" in meta["uncertainty"])
        else:
            has_samples = bool("null" in meta and "fs_null_b" in meta["null"])

        if has_samples:
            fs_samples = _get_fs_bootstrap_samples(W, which)  # (K,B)
            B = fs_samples.shape[1]

            boot_a = np.empty(B, dtype=float)
            boot_b = np.empty(B, dtype=float)
            boot_F0 = fs_samples[0, :].astype(float, copy=False)

            # keep u fixed; store it for convenience
            boot_u1 = np.tile(u[None, :], (B, 1))

            for t in range(B):
                fs_t = fs_samples[:, t]
                g_t, H_t = gh_from_fs(fs_t, plan)
                M_t = M_second_order_from_gh(g_t, H_t)

                lam_u_t = float(u @ (M_t @ u))
                a_t = float(g_t @ u)
                b_t, _, _ = _signed_b_from_M_g_H(lam_u_t, a_t, u, H_t)

                boot_a[t] = a_t
                boot_b[t] = b_t

    return QSecondOrderResult(
        F0=F0_obs,
        a=a_obs,
        b=b_obs,
        u1=u,
        boot_a=boot_a,
        boot_b=boot_b,
        boot_F0=boot_F0,
        boot_u1=boot_u1,
        lambda1=lam_u,      
        rayleigh=rayleigh,  
        b_abs=b_abs_obs,
        sign_proxy=sign_proxy_obs,
    )


##############################################
#       PIECEWISE OR STRONG REGIONALITY      #
##############################################

def Qz_to_Qx_params(
    F0z: float,
    az: np.ndarray,
    Bz: np.ndarray,
    U: np.ndarray,
) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    Convert a K-dimensional quadratic manifold expressed in nu_z = U^T z
    to the coordinate nu_x = U^T x, where z = 2x - 1.

    nu_z = 2 nu_x - c, with c = U^T 1 (shape (K,)).

    If:
        Qz(nu_z) = F0z + az^T nu_z + nu_z^T Bz nu_z
    then:
        Qx(nu_x) = F0x + ax^T nu_x + nu_x^T Bx nu_x

    where:
        Bx = 4 Bz
        ax = 2 az - 4 Bz c
        F0x = F0z - az^T c + c^T Bz c

    Parameters
    ----------
    F0z : float
        Constant term in z-space.
    az : np.ndarray
        Linear coefficients in z-space, shape (K,).
    Bz : np.ndarray
        Quadratic coefficients in z-space, shape (K,K), symmetric.
    U : np.ndarray
        Latent basis used to define nu_z and nu_x, shape (N,K).

    Returns
    -------
    (F0x, ax, Bx)
        Parameters in x-space.
    """
    az = np.asarray(az, dtype=float)
    Bz = np.asarray(Bz, dtype=float)
    U = np.asarray(U, dtype=float)

    if az.ndim != 1:
        raise ValueError("az must be a 1D array of shape (K,).")
    K = az.shape[0]
    if Bz.shape != (K, K):
        raise ValueError(f"Bz must have shape {(K,K)}.")
    if U.ndim != 2 or U.shape[1] != K:
        raise ValueError(f"U must have shape (N,{K}).")

    ones = np.ones(U.shape[0], dtype=float)   # (N,)
    c = U.T @ ones                            # (K,)

    Bx = 4.0 * Bz
    ax = 2.0 * az - 4.0 * (Bz @ c)
    F0x = float(F0z - az @ c + c @ (Bz @ c))

    # Defensive symmetrization (numerical)
    Bx = 0.5 * (Bx + Bx.T)

    return F0x, ax, Bx

def q_params_second_order_given_u_from_gh(
    F0z: float,
    g: np.ndarray,
    H: np.ndarray,
    u: np.ndarray,
) -> Tuple[float, float, float]:
    """
    Compute theoretical fixed-u quadratic parameters q(ν) in z-space from (F0,g,H).

    In the second-order interaction limit:
        F(z) ≈ F0 + g^T z + 1/2 z^T H z
    and the gradient-cov identity gives:
        M ≈ g g^T + H^2.

    For a fixed direction u (normalized internally), define ν = u·z and the
    one-dimensional quadratic approximation:
        q(ν) = F0 + a ν + b ν^2.

    This function returns (F0, a, b) with:
        a = g·u
        |b| = sqrt( u^T M u - a^2 )
    and sign(b) chosen from the curvature proxy sign(u^T H u).

    Input
    -----
    F0z : float
        Constant term in z-space.
    g : ndarray, shape (N,)
        First-order coefficient vector.
    H : ndarray, shape (N,N)
        Second-order coefficient matrix (symmetric; typically hollow off-diagonal).
    u : ndarray, shape (N,)
        Direction defining ν = u·z (need not be normalized).

    Output
    ------
    (F0, a, b) : tuple(float,float,float)
        Quadratic parameters in z-space.
    """
    u = np.asarray(u, dtype=float).copy()
    nu = float(np.linalg.norm(u))
    if nu == 0.0:
        raise ValueError("u has zero norm.")
    u /= nu

    g = np.asarray(g, dtype=float)
    H = np.asarray(H, dtype=float)

    # M = g g^T + H^2
    M = np.outer(g, g) + (H @ H)
    M = 0.5 * (M + M.T)

    lam_u = float(u @ (M @ u))
    a = float(g @ u)

    b_abs = float(np.sqrt(max(0.0, lam_u - a * a)))
    sign_proxy = float(u @ (H @ u))
    sgn = float(np.sign(sign_proxy)) if sign_proxy != 0.0 else 1.0
    b = sgn * b_abs

    return float(F0z), float(a), float(b)


def qz_to_qx_params(
    F0z: float,
    az: float,
    bz: float,
    u_sum: float,
) -> Tuple[float, float, float]:
    """
    Convert a quadratic q expressed in νz = u·z to the νx = u·x coordinate.

    With z = 2x - 1, we have:
        νz = u·z = 2(u·x) - sum(u) = 2 νx - c,  where c = sum(u).

    If:
        qz(νz) = F0z + az νz + bz νz^2
    then:
        qx(νx) = F0x + ax νx + bx νx^2

    where:
        bx = 4 bz
        ax = 2 az - 4 bz c
        F0x = F0z - az c + bz c^2

    Input
    -----
    F0z, az, bz : float
        Quadratic parameters in z-space.
    u_sum : float
        c = sum_i u_i (using the same u as in νz and νx).

    Output
    ------
    (F0x, ax, bx) : tuple(float,float,float)
        Quadratic parameters in x-space.
    """
    c = float(u_sum)
    bx = 4.0 * bz
    ax = 2.0 * az - 4.0 * bz * c
    F0x = F0z - az * c + bz * c * c
    return float(F0x), float(ax), float(bx)

def compute_q_restricted_to_gate_from_gh(
    *,
    F0z: float,
    g: np.ndarray,
    H: np.ndarray,
    u1: np.ndarray,
    j_gate: int,
) -> Dict[str, Any]:
    """
    Compute region-specific theoretical q(ν) by restricting the global second-order
    quadratic form to the subcube defined by a gating coordinate z_j = s.

    This implements the "no refitting" strong-regionality notion:
      - Start from a global quadratic approximation in z-space:
            F(z) ≈ F0z + g^T z + 1/2 z^T H z
      - Fix z_j = s (s ∈ {-1,+1}) and restrict to remaining coordinates z_rest
      - Obtain an induced quadratic form on the (N-1)-dimensional subcube:
            F_s(z_rest) ≈ F0_s + g_s^T z_rest + 1/2 z_rest^T H_rest z_rest
        with:
            F0_s = F0z + s g_j
            g_s  = g_rest + s H_rest,j   (i.e., column j restricted to rest)
            H_rest = H_{rest,rest}
      - Define the restricted direction u_sub = normalize(u1_rest),
        where u1_rest is u1 with coordinate j removed.
      - Return the fixed-u quadratic parameters (F0_s, a_s, b_s) along ν_sub = u_sub·z_rest.

    Input
    -----
    F0z : float
        Global constant term in z-space.
    g : ndarray, shape (N,)
        Global first-order coefficient vector in z-space.
    H : ndarray, shape (N,N)
        Global second-order coefficient matrix (symmetric).
    u1 : ndarray, shape (N,)
        Global dominant direction defining νz = u1·z and νx = u1·x.
        (Need not be normalized; the function uses it consistently as provided.)
    j_gate : int
        Index of the gating coordinate (0 <= j_gate < N).

    Output
    ------
    result : dict
        Dictionary with keys:
          - "absent": dict for region z_j=-1 (x_j=0)
          - "present": dict for region z_j=+1 (x_j=1)
          - "meta": mapping constants to evaluate regional curves on a νx-grid

        Each region dict contains:
          - "s": float, either -1.0 or +1.0
          - "F0": float (z-space constant for q_s(ν_sub))
          - "a": float  (linear coefficient in ν_sub)
          - "b": float  (quadratic coefficient in ν_sub)
          - "u_sub": ndarray, shape (N-1,) normalized restricted direction
          - "idx_rest": ndarray, shape (N-1,) indices kept

        meta contains:
          - "j_gate": int
          - "idx_rest": ndarray, shape (N-1,)
          - "u1j": float, u1[j_gate]
          - "norm_rest": float, ||u1_rest||
          - "c": float, sum(u1)
          - "map_nux_to_nusub": callable(grid, s) -> ν_sub values
            where νz = 2 νx - c, and ν_sub = (νz - u1j*s) / norm_rest

    Notes
    -----
    - This is fully theoretical: it does NOT run Walsh on subsets.
    - It assumes the global quadratic approximation (F0z,g,H) is meaningful on the full cube.
    - The returned coefficients parameterize q_s as a function of ν_sub, not directly νx.
      Use meta["map_nux_to_nusub"](nu_x_grid, s) to evaluate curves on the νx axis.
    """
    g = np.asarray(g, dtype=float)
    H = np.asarray(H, dtype=float)
    u1 = np.asarray(u1, dtype=float)

    if g.ndim != 1:
        raise ValueError("g must be 1D (N,).")
    if H.ndim != 2 or H.shape[0] != H.shape[1]:
        raise ValueError("H must be 2D square (N,N).")
    if u1.ndim != 1:
        raise ValueError("u1 must be 1D (N,).")

    N = g.shape[0]
    if H.shape != (N, N):
        raise ValueError(f"H has shape {H.shape}, expected {(N, N)}.")
    if u1.shape != (N,):
        raise ValueError(f"u1 has shape {u1.shape}, expected {(N,)}.")
    if not (0 <= j_gate < N):
        raise ValueError(f"j_gate={j_gate} out of bounds for N={N}.")

    # Indices of the remaining coordinates
    idx_rest = np.array([i for i in range(N) if i != j_gate], dtype=int)

    # Restricted direction (remove coordinate j, then normalize)
    u1_rest = u1[idx_rest]
    norm_rest = float(np.linalg.norm(u1_rest))
    if norm_rest == 0.0:
        raise ValueError("u1 restricted to rest has zero norm; cannot define u_sub.")
    u_sub = u1_rest / norm_rest

    # Restricted interaction matrix (same for both regions)
    H_rest = H[np.ix_(idx_rest, idx_rest)]

    # Useful mapping constants for νx axis
    u1j = float(u1[j_gate])
    c = float(np.sum(u1))

    def map_nux_to_nusub(nu_x_grid: np.ndarray, s: float) -> np.ndarray:
        """
        Map νx = u·x to ν_sub = u_sub·z_rest on region z_j=s.

        νz_full = u·z = 2 νx - c
        νz_full = u1j*s + u1_rest·z_rest = u1j*s + norm_rest*(u_sub·z_rest)

        => ν_sub = (νz_full - u1j*s) / norm_rest
        """
        nu_x_grid = np.asarray(nu_x_grid, dtype=float)
        nu_z_full = 2.0 * nu_x_grid - c
        return (nu_z_full - u1j * float(s)) / norm_rest

    # Build region-specific coefficients by analytic restriction
    def region_coeffs(s: float) -> Dict[str, Any]:
        s = float(s)
        F0_s = float(F0z + s * g[j_gate])
        g_rest_s = g[idx_rest] + s * H[idx_rest, j_gate]

        # Fixed-u quadratic parameters along ν_sub in the (N-1)-space
        F0_out, a_out, b_out = q_params_second_order_given_u_from_gh(
            F0z=F0_s,
            g=g_rest_s,
            H=H_rest,
            u=u_sub,
        )
        return dict(
            s=s,
            F0=float(F0_out),
            a=float(a_out),
            b=float(b_out),
            u_sub=u_sub.copy(),
            idx_rest=idx_rest.copy(),
        )

    res_absent = region_coeffs(-1.0)   # x_j=0 <-> z_j=-1
    res_present = region_coeffs(+1.0)  # x_j=1 <-> z_j=+1

    return dict(
        absent=res_absent,
        present=res_present,
        meta=dict(
            j_gate=int(j_gate),
            idx_rest=idx_rest.copy(),
            u1j=u1j,
            norm_rest=norm_rest,
            c=c,
            map_nux_to_nusub=map_nux_to_nusub,
        ),
    )


def bootstrap_ci_q_restricted_to_gate(
    *,
    boot_F0z: np.ndarray,          # (B,)
    boot_g: np.ndarray,            # (B,N)
    boot_H: np.ndarray,            # (B,N,N)
    u1: np.ndarray,                # (N,)
    j_gate: int,
    nu_x_grid0: np.ndarray,        # grid for region x_j=0  (z_j=-1)
    nu_x_grid1: np.ndarray,        # grid for region x_j=1  (z_j=+1)
    ci: Tuple[float, float] = (2.5, 97.5),
) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """
    Compute pointwise CI bands for region-specific q(ν) curves in "restrict" mode
    by propagating bootstrap uncertainty of the GLOBAL (F0,g,H) coefficients.

    Input
    -----
    boot_F0z : (B,)
        Bootstrap draws of F0 in z-space.
    boot_g : (B,N)
        Bootstrap draws of g in z-space.
    boot_H : (B,N,N)
        Bootstrap draws of H in z-space.
    u1 : (N,)
        Fixed direction for ν (same convention as in plotting).
    j_gate : int
        Gate coordinate.
    nu_x_grid0 : (G0,)
        νx grid for region x_j=0 (z_j=-1).
    nu_x_grid1 : (G1,)
        νx grid for region x_j=1 (z_j=+1).
    ci : (lo,hi)
        Percentiles.

    Output
    ------
    bands : dict
        {"absent": (lo0, hi0), "present": (lo1, hi1)}
    """
    boot_F0z = np.asarray(boot_F0z, float)
    boot_g = np.asarray(boot_g, float)
    boot_H = np.asarray(boot_H, float)
    u1 = np.asarray(u1, float)

    B = boot_F0z.shape[0]
    if boot_g.shape[0] != B or boot_H.shape[0] != B:
        raise ValueError("Bootstrap arrays must share the same first dimension B.")

    # Preallocate evaluations
    Q0 = np.empty((B, nu_x_grid0.size), dtype=float)
    Q1 = np.empty((B, nu_x_grid1.size), dtype=float)

    for b in range(B):
        restrict = compute_q_restricted_to_gate_from_gh(
            F0z=float(boot_F0z[b]),
            g=boot_g[b],
            H=boot_H[b],
            u1=u1,
            j_gate=int(j_gate),
        )

        # Map νx -> ν_sub and evaluate q_s(ν_sub)
        nu_sub0 = restrict["meta"]["map_nux_to_nusub"](nu_x_grid0, s=-1.0)
        nu_sub1 = restrict["meta"]["map_nux_to_nusub"](nu_x_grid1, s=+1.0)

        Q0[b] = restrict["absent"]["F0"] + restrict["absent"]["a"] * nu_sub0 + restrict["absent"]["b"] * (nu_sub0 ** 2)
        Q1[b] = restrict["present"]["F0"] + restrict["present"]["a"] * nu_sub1 + restrict["present"]["b"] * (nu_sub1 ** 2)

    lo0 = np.percentile(Q0, ci[0], axis=0)
    hi0 = np.percentile(Q0, ci[1], axis=0)
    lo1 = np.percentile(Q1, ci[0], axis=0)
    hi1 = np.percentile(Q1, ci[1], axis=0)

    return {"absent": (lo0, hi0), "present": (lo1, hi1)}


def plot_strong_regionality_q(
    L: Any,
    gate: "StrongRegionalityResult",
    *,
    M_res: Any,
    dataset_name: Optional[str] = None,
    which: str = "uncertainty",
    walsh_kwargs: Optional[Dict[str, Any]] = None,
    min_group_size: int = 8,
    show_plot: bool = True,
    return_data: bool = False,
    show_bootstrap: bool = True,
    ci: Tuple[float, float] = (2.5, 97.5),
    ax=None,
    verbose: bool = False,
    pc: int = 0,
    u_source: str = "M_res",          # "M_res" | "gate"
    u_override: Optional[np.ndarray] = None,  # if given, overrides pc/u_source
):
    """
    Plot F(x) vs (u_pc · x) together with theoretical global and regional q(ν) curves
    under the strong-regionality gate (RESTRICT mode; no refit on subsets).

    Projection choice
    -----------------
    - By default, uses the eigenvector indexed by `pc` from M_res.U (or M_res.eigvecs).
    - If u_source='gate', uses gate.u1 (only compatible with pc=0).
    - If u_override is provided, it is used directly (highest priority).

    Robust behavior
    ---------------
    - If bootstrap is unavailable (no replicas / no stored bootstraps / B=0), bands are skipped silently.
    - Global curve always plotted; regional curves plotted if min_group_size satisfied.
    - Requires M_res.g and M_res.H for restricted (regional) curves.

    Input / Output
    --------------
    If return_data=True, returns regions including optional CI bands (may be None).
    """
    import numpy as np
    import matplotlib.pyplot as plt

    if which not in ("uncertainty", "null"):
        raise ValueError("which must be 'uncertainty' or 'null'.")
    if u_source not in ("M_res", "gate"):
        raise ValueError("u_source must be 'M_res' or 'gate'.")
    if not isinstance(pc, int) or pc < 0:
        raise ValueError("pc must be a non-negative integer.")

    if getattr(M_res, "g", None) is None or getattr(M_res, "H", None) is None:
        raise ValueError("plot_strong_regionality_q requires M_res.g and M_res.H (second_order).")

    # -------------------------
    # Observed data
    # -------------------------
    X01 = np.asarray(L.states, dtype=float)
    if X01.ndim != 2:
        raise ValueError("L.states must be a 2D array (X,N).")

    vals = np.asarray(getattr(L, "values", []))
    if vals.ndim == 2:
        Fobs = vals.mean(axis=1).astype(float)
    else:
        Fobs = vals.astype(float)

    N = int(X01.shape[1])

    # -------------------------
    # Choose projection vector u (normalized)
    # -------------------------
    if u_override is not None:
        u = np.asarray(u_override, dtype=float).copy()
        if u.ndim != 1 or u.shape[0] != N:
            raise ValueError(f"u_override must have shape ({N},).")
    else:
        u = None

        if u_source == "M_res":
            if hasattr(M_res, "U") and M_res.U is not None:
                U_all = np.asarray(M_res.U, dtype=float)
                if U_all.ndim != 2 or U_all.shape[0] != N:
                    raise ValueError(f"M_res.U must have shape ({N},K).")
                if pc >= U_all.shape[1]:
                    raise ValueError(f"pc out of range: pc={pc}, available K={U_all.shape[1]}")
                u = U_all[:, pc].copy()

            elif hasattr(M_res, "eigvecs") and M_res.eigvecs is not None:
                V_all = np.asarray(M_res.eigvecs, dtype=float)
                if V_all.ndim != 2 or V_all.shape[0] != N:
                    raise ValueError(f"M_res.eigvecs must have shape ({N},K).")
                if pc >= V_all.shape[1]:
                    raise ValueError(f"pc out of range: pc={pc}, available K={V_all.shape[1]}")
                u = V_all[:, pc].copy()

            else:
                # fallback: gate.u1 only for pc=0
                if pc != 0:
                    raise ValueError(
                        "M_res has no U/eigvecs. Provide them or set u_source='gate' with pc=0."
                    )
                u = np.asarray(gate.u1, dtype=float).copy()

        else:  # u_source == "gate"
            if pc != 0:
                raise ValueError("u_source='gate' only supports pc=0 (gate.u1). Use u_source='M_res' for pc>0.")
            u = np.asarray(gate.u1, dtype=float).copy()

    u = np.asarray(u, dtype=float).copy()
    norm_u = float(np.linalg.norm(u))
    if norm_u == 0.0:
        raise ValueError("Selected projection vector u has zero norm.")
    u /= norm_u

    # νx = u·x (x in {0,1})
    nu_x = X01 @ u

    # ---------------------------------------------------------
    # Decide whether we should even TRY bootstrap
    # ---------------------------------------------------------
    try_bootstrap = bool(show_bootstrap)

    # -------------------------
    # Global q(ν) (full landscape) given fixed u
    # -------------------------
    q_global = compute_q_second_order_given_u1(
        L,
        u,
        which=which,
        return_bootstrap=try_bootstrap,
        walsh_kwargs=walsh_kwargs,
    )

    # Keep u aligned with qinference internal conventions
    u = np.asarray(q_global.u1, dtype=float)
    u /= float(np.linalg.norm(u))
    c = float(np.sum(u))

    # Convert global (F0,a,b) z->x to evaluate on νx
    F0x_g, ax_g, bx_g = qz_to_qx_params(float(q_global.F0), float(q_global.a), float(q_global.b), c)

    nu_grid = np.linspace(float(nu_x.min()), float(nu_x.max()), 400)
    qg_grid = F0x_g + ax_g * nu_grid + bx_g * (nu_grid ** 2)

    # -------------------------
    # Global CI band (best-effort)
    # -------------------------
    qg_band = None
    bootstrap_ok_global = (
        try_bootstrap
        and (getattr(q_global, "boot_a", None) is not None)
        and (getattr(q_global, "boot_b", None) is not None)
        and (getattr(q_global, "boot_F0", None) is not None)
    )

    if bootstrap_ok_global:
        Ba = np.asarray(q_global.boot_a, float)
        Bb = np.asarray(q_global.boot_b, float)
        BF0 = np.asarray(q_global.boot_F0, float)

        bxB = 4.0 * Bb
        axB = 2.0 * Ba - 4.0 * Bb * c
        F0xB = BF0 - Ba * c + Bb * c * c

        Q = F0xB[:, None] + axB[:, None] * nu_grid[None, :] + bxB[:, None] * (nu_grid[None, :] ** 2)
        lo = np.percentile(Q, ci[0], axis=0)
        hi = np.percentile(Q, ci[1], axis=0)
        qg_band = (lo, hi)

    if verbose:
        print("DEBUG bootstrap_ok_global:", bootstrap_ok_global)

    # -------------------------
    # Region curves via restriction (no refit)
    # -------------------------
    do_regions = (gate.mask_present.sum() >= min_group_size) and (gate.mask_absent.sum() >= min_group_size)

    reg = dict(
        grid0=None, grid1=None,
        q0=None, q1=None,
        band0=None, band1=None,
        restrict=None,
        bootstrap_ok_regional=False,
    )

    if do_regions:
        g = np.asarray(M_res.g, dtype=float)
        H = np.asarray(M_res.H, dtype=float)

        restrict = compute_q_restricted_to_gate_from_gh(
            F0z=float(q_global.F0),
            g=g,
            H=H,
            u1=u,
            j_gate=int(gate.j_gate),
        )
        reg["restrict"] = restrict

        nu0 = nu_x[gate.mask_absent]
        nu1_ = nu_x[gate.mask_present]
        grid0 = np.linspace(float(nu0.min()), float(nu0.max()), 250)
        grid1 = np.linspace(float(nu1_.min()), float(nu1_.max()), 250)

        nu_sub0 = restrict["meta"]["map_nux_to_nusub"](grid0, s=-1.0)
        nu_sub1 = restrict["meta"]["map_nux_to_nusub"](grid1, s=+1.0)

        q0 = restrict["absent"]["F0"] + restrict["absent"]["a"] * nu_sub0 + restrict["absent"]["b"] * (nu_sub0 ** 2)
        q1 = restrict["present"]["F0"] + restrict["present"]["a"] * nu_sub1 + restrict["present"]["b"] * (nu_sub1 ** 2)

        reg["grid0"], reg["grid1"] = grid0, grid1
        reg["q0"], reg["q1"] = q0, q1

        # -------------------------
        # Regional CI bands (best-effort)
        # -------------------------
        if try_bootstrap:
            boot_F0z = getattr(q_global, "boot_F0", None)

            boot = getattr(M_res, "bootstrap", None)
            pack = None
            if boot is not None and hasattr(boot, "boot") and boot.boot is not None:
                pack = boot.boot.get(which, None)

            boot_g = getattr(pack, "g", None) if pack is not None else None
            boot_H = getattr(pack, "H", None) if pack is not None else None

            bootstrap_ok_regional = (boot_F0z is not None) and (boot_g is not None) and (boot_H is not None)
            reg["bootstrap_ok_regional"] = bootstrap_ok_regional

            if verbose:
                print("DEBUG bootstrap_ok_regional:", bootstrap_ok_regional)

            if bootstrap_ok_regional:
                bands = bootstrap_ci_q_restricted_to_gate(
                    boot_F0z=np.asarray(boot_F0z, float),
                    boot_g=np.asarray(boot_g, float),
                    boot_H=np.asarray(boot_H, float),
                    u1=u,
                    j_gate=int(gate.j_gate),
                    nu_x_grid0=grid0,
                    nu_x_grid1=grid1,
                    ci=ci,
                )
                reg["band0"] = bands["absent"]
                reg["band1"] = bands["present"]

    # -------------------------
    # Plot
    # -------------------------
    if ax is None:
        fig, ax = plt.subplots(figsize=(7.4, 5.8))
    else:
        fig = ax.figure

    ax.scatter(nu_x[gate.mask_absent],  Fobs[gate.mask_absent],  s=24, alpha=0.85, color="#F7C59F",
               label=f"{gate.gate_name} absent (x=0)")
    ax.scatter(nu_x[gate.mask_present], Fobs[gate.mask_present], s=24, alpha=0.85, color="#48CAE4",
               label=f"{gate.gate_name} present (x=1)")

    ax.plot(nu_grid, qg_grid, linewidth=3.0, label=r"global $q(\nu)$", color="#dd2d4a")
    if qg_band is not None:
        ax.fill_between(nu_grid, qg_band[0], qg_band[1], alpha=0.20, linewidth=0.0, label="global CI", color="#dd2d4a")

    if do_regions and (reg["q0"] is not None) and (reg["q1"] is not None):
        ax.plot(reg["grid0"], reg["q0"], linestyle="--", linewidth=2.5, zorder=2,
                label=rf"$q(\nu)$ | {gate.gate_name}=0", color="#ff6700")
        ax.plot(reg["grid1"], reg["q1"], linestyle="--", linewidth=2.5, zorder=2,
                label=rf"$q(\nu)$ | {gate.gate_name}=1", color="#0077B6")

        if reg["band0"] is not None:
            ax.fill_between(reg["grid0"], reg["band0"][0], reg["band0"][1], alpha=0.2, zorder=2, linewidth=0.0, color="#ff6700")
        if reg["band1"] is not None:
            ax.fill_between(reg["grid1"], reg["band1"][0], reg["band1"][1], alpha=0.2, zorder=2, linewidth=0.0, color="#0077B6")

    title = dataset_name if dataset_name is not None else "Strong regionality q(ν)"
    ax.set_title(f"{title}\nGate: {gate.gate_name} (Δb={gate.delta_b_gate:+.3f})")

    ax.set_xlabel(rf"$\mathbf{{u}}_{{{pc+1}}}\cdot \mathbf{{x}}$")
    ax.set_ylabel(r"$F(\mathbf{x})$")
    ax.legend(frameon=False, loc="best")

    ax.tick_params(which="both", length=9, width=2.5, labelsize=13)
    for spine in ["top", "bottom", "left", "right"]:
        ax.spines[spine].set_linewidth(2)

    plt.tight_layout()
    if show_plot:
        plt.show()

    if return_data:
        data = dict(
            nu_x=nu_x,
            Fobs=Fobs,
            gate=gate,
            u=u,
            pc=pc,
            eglobal=dict(
                nu_grid=nu_grid,
                q=qg_grid,
                band=qg_band,
                qres=q_global,
                bootstrap_ok=bootstrap_ok_global,
            ),
            regions=reg,
            do_regions=do_regions,
        )
        return fig, ax, data

    return fig, ax



#####################################################
#                                                   #
#           QMANIFOLD IN SECOND ODER APPROACH       #
#                                                   #
#####################################################

@dataclass(frozen=True)
class QManifoldSecondOrderResult:
    """
    Container for K-dimensional quadratic manifold inference:

        Q(nu) = F0 + a^T nu + nu^T B nu,
        nu = U^T z,   U = [u1,...,uK].

    Conventions
    ----------
    - U has orthonormal columns (N,K).
    - a has shape (K,).
    - B is symmetric with shape (K,K).
    - lambdas are the top-K eigenvalues of M in descending order.

    Bootstrap fields (if available) follow the same convention:
    - boot_U: (B, N, K)
    - boot_a: (B, K)
    - boot_B: (B, K, K)
    - boot_F0: (B,)
    - boot_lambdas: (B, K)
    """
    # Point estimates
    F0: float
    a: np.ndarray          # (K,)
    B: np.ndarray          # (K,K)
    U: np.ndarray          # (N,K)
    lambdas: np.ndarray    # (K,)

    # Bootstrap draws (optional)
    boot_F0: Optional[np.ndarray] = None        # (B,)
    boot_a: Optional[np.ndarray] = None         # (B,K)
    boot_B: Optional[np.ndarray] = None         # (B,K,K)
    boot_U: Optional[np.ndarray] = None         # (B,N,K)
    boot_lambdas: Optional[np.ndarray] = None   # (B,K)

    # Diagnostics / extra outputs (optional)
    rayleighs: Optional[np.ndarray] = None      # (K,)  lambdas / tr(M)
    explained: Optional[float] = None           # sum(lambdas)/tr(M)
    # Optionally store observed g, H, M if you want downstream plots/analysis
    g: Optional[np.ndarray] = None              # (N,)
    H: Optional[np.ndarray] = None              # (N,N)
    M: Optional[np.ndarray] = None              # (N,N)




def _procrustes_align(U: np.ndarray, Uref: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Align basis U to Uref (both N x K with orthonormal columns) by solving:
        R = argmin_{R in O(K)} || U R - Uref ||_F
    Returns aligned_U = U R and R.
    """
    G = Uref.T @ U  # (K,K)
    P, _, Qt = np.linalg.svd(G, full_matrices=False)
    R = Qt.T @ P.T  # (K,K), orthogonal

    # Optional: enforce det(R)=+1 to avoid reflections (usually harmless, but stabilizes)
    if np.linalg.det(R) < 0:
        Qt[-1, :] *= -1.0
        R = Qt.T @ P.T

    return U @ R, R


def Qmanifold_second_order(
    L: Any,
    *,
    which: str = "uncertainty",
    K: int = 2,
    align_eigvecs: bool = True,
    align_basis: str = "procrustes",   # "procrustes" | "none"
    return_bootstrap: bool = True,
    walsh_kwargs: Optional[Dict[str, Any]] = None,
) -> QManifoldSecondOrderResult:
    """
    Infer K-dimensional quadratic manifold:
        Q(nu) = F0 + a^T nu + nu^T B nu,
    where nu = U^T z and U are the top-K eigenvectors of M = g g^T + H^2.

    Steps
    -----
    1) walsh_analysis(L, **walsh_kwargs)
    2) build (g,H) from WH coefficients (2nd order truncation)
    3) build M = g g^T + H^2
    4) compute top-K eigenpairs (lambdas, U)
    5) set a = U^T g,   B = U^T H U
    6) optional bootstrap: propagate uncertainty from W.meta bootstrap fs samples

    Robustness
    ----------
    - If bootstrap samples are not present, returns point estimates and boot_* = None.
    - If dataset has no replicates, disables bootstrap requests (best-effort).

    Parameters
    ----------
    which : {"uncertainty","null"}
    K : int
        Number of latent dimensions (top eigenpairs).
    align_eigvecs : bool
        Align eigenvector signs across bootstrap draws (uses existing machinery).
    align_basis : {"procrustes","none"}
        Additional alignment for K>1 to prevent arbitrary rotations within the
        top-K subspace across bootstrap draws (recommended: "procrustes").
    return_bootstrap : bool
        If True, returns bootstrap arrays when possible.
    walsh_kwargs : dict
        Passed to walsh_analysis; defaults: as_dataframe=False, B_null=0, B_uncertainty=0
    """
    if which not in ("uncertainty", "null"):
        raise ValueError("which must be 'uncertainty' or 'null'.")
    if not isinstance(K, int) or K < 1:
        raise ValueError("K must be a positive integer.")
    if align_basis not in ("procrustes", "none"):
        raise ValueError("align_basis must be 'procrustes' or 'none'.")

    # -------------------------------------------------
    # Walsh analysis (internal)
    # -------------------------------------------------
    if walsh_kwargs is None:
        walsh_kwargs = {}
    walsh_kwargs = {
        "as_dataframe": False,
        "B_null": 0,
        "B_uncertainty": 0,
        **walsh_kwargs,
    }

    vals = np.asarray(getattr(L, "values", []))
    has_replicates = bool(vals.ndim == 2 and vals.shape[1] > 1)
    if return_bootstrap and not has_replicates:
        walsh_kwargs = dict(walsh_kwargs)
        walsh_kwargs["B_uncertainty"] = 0
        walsh_kwargs["B_null"] = 0
        return_bootstrap = False

    W = walsh_analysis(L, **walsh_kwargs)

    # -------------------------------------------------
    # Build plan and compute observed g, H
    # -------------------------------------------------
    plan = build_gh_plan(W.s_bits, W.orders)
    g_obs, H_obs = gh_from_fs(W.mean, plan)

    # -------------------------------------------------
    # Observed M and its top-K subspace
    # -------------------------------------------------
    M_obs = M_second_order_from_gh(g_obs, H_obs)
    vals_obs, vecs_obs = eigensolve(M_obs, k=int(K))
    lambdas_obs = np.asarray(vals_obs, dtype=float)
    U_obs = np.asarray(vecs_obs, dtype=float)  # (N,K)

    # Orientation convention: at least fix the sign of u1 with popcount
    states01 = np.asarray(W.s_bits, dtype=int)
    U_obs[:, 0] = orient_u1_with_popcount(U_obs[:, 0], states01)

    # Point estimates
    a_obs = (U_obs.T @ g_obs).astype(float, copy=False)     # (K,)
    B_obs = (U_obs.T @ (H_obs @ U_obs)).astype(float, copy=False)  # (K,K)
    # Symmetrize defensively
    B_obs = 0.5 * (B_obs + B_obs.T)

    trM = float(np.trace(M_obs))
    rayleighs_obs = lambdas_obs / trM if trM > 0 else None
    explained_obs = float(lambdas_obs.sum() / trM) if trM > 0 else None
    F0_obs = float(W.mean[0])

    # -------------------------------------------------
    # Bootstrap (optional)
    # -------------------------------------------------
    boot_F0 = boot_a = boot_B = boot_U = boot_lambdas = None

    if return_bootstrap:
        meta = getattr(W, "meta", {}) or {}
        if which == "uncertainty":
            has_samples = bool("uncertainty" in meta and "fs_unc_b" in meta["uncertainty"])
        else:
            has_samples = bool("null" in meta and "fs_null_b" in meta["null"])

        if has_samples:
            out = bootstrap_second_order_from_walsh(
                W,
                which=(which,),
                store="eigvecs",
                k=int(K),
                compute_C=False,
                align_eigvecs=align_eigvecs,
            )
            ens = out.boot[which]  # has eigvals (B,K) and eigvecs (B,N,K)

            fs_samples = _get_fs_bootstrap_samples(W, which)  # (Kwh, B)
            Bn = int(fs_samples.shape[1])

            boot_F0 = fs_samples[0, :].astype(float, copy=False)
            boot_a = np.empty((Bn, int(K)), dtype=float)
            boot_B = np.empty((Bn, int(K), int(K)), dtype=float)
            boot_U = np.empty((Bn, U_obs.shape[0], int(K)), dtype=float)
            boot_lambdas = np.asarray(ens.eigvals, dtype=float).copy()  # (B,K)

            # Reference basis for alignment
            Uref = U_obs.copy()

            for t in range(Bn):
                fs_t = fs_samples[:, t]
                g_t, H_t = gh_from_fs(fs_t, plan)

                Ut = np.asarray(ens.eigvecs[t, :, :], dtype=float)  # (N,K)

                # Optional: procrustes alignment to the observed subspace basis
                if int(K) > 1 and align_basis == "procrustes":
                    Ut, R = _procrustes_align(Ut, Uref)
                    # rotate coefficients consistently
                    at = (Ut.T @ g_t)
                    Bt = (Ut.T @ (H_t @ Ut))
                    at = (R.T @ at)
                    Bt = (R.T @ Bt @ R)
                else:
                    at = (Ut.T @ g_t)
                    Bt = (Ut.T @ (H_t @ Ut))

                Bt = 0.5 * (Bt + Bt.T)

                boot_U[t, :, :] = Ut
                boot_a[t, :] = at
                boot_B[t, :, :] = Bt

    return QManifoldSecondOrderResult(
        F0=F0_obs,
        a=a_obs,
        B=B_obs,
        U=U_obs,
        lambdas=lambdas_obs,
        boot_F0=boot_F0,
        boot_a=boot_a,
        boot_B=boot_B,
        boot_U=boot_U,
        boot_lambdas=boot_lambdas,
        rayleighs=rayleighs_obs,
        explained=explained_obs,
    )

def compute_Qmanifold_restricted_to_gate_from_gh(
    F0z: float,
    g: np.ndarray,
    H: np.ndarray,
    U: np.ndarray,
    j_gate: int,
) -> Dict[str, Any]:
    """
    Restrict a 2D (or K-dim) quadratic manifold to a strong-regionality gate
    by fixing z_j = s with s in {-1, +1} (RESTRICT mode; no refit).

    Model in z-space
    ----------------
    F(z) ≈ F0z + g^T z + 1/2 z^T H z,   with z ∈ {-1,+1}^N.

    Global manifold coordinates
    ---------------------------
    ν = U^T z,   U ∈ R^{N×K} with orthonormal columns.

    Gate restriction
    ----------------
    Fix z_j = s. Let "rest" be indices i != j.
    Then:
      F_s(z_rest) ≈ F0s + g_s^T z_rest + 1/2 z_rest^T H_rest z_rest

    where:
      F0s = F0z + s g_j
      g_s = g_rest + s H_rest,j
      H_rest = H_rest,rest

    Relationship to global ν
    ------------------------
    ν = U_rest^T z_rest + s U_j,:^T  = ν' + δ_s,
    where δ_s = s * U[j,:] and ν' = U_rest^T z_rest.

    This function returns regional quadratic forms in the *global* ν coordinates:
      Q_s(ν) = F0s + a_s^T (ν - δ_s) + (ν - δ_s)^T B_s (ν - δ_s)

    with:
      a_s = U_rest^T g_s
      B_s = U_rest^T H_rest U_rest   (note: consistent with your convention
                                      Q = F0 + a^T ν + ν^T B ν; i.e. no 1/2 here)

    Output
    ------
    dict with keys:
      - "meta": { "j_gate", "K", "idx_rest", "U_j", "delta_absent", "delta_present" }
      - "absent":  params for s=-1
      - "present": params for s=+1

    Each region params contain:
      F0, a (K,), B (K,K), delta (K,), s

    Notes
    -----
    - Assumes H is symmetric (we symmetrize defensively).
    - Works for any K>=1 (including K=2).
    - "absent" corresponds to z_j = -1, "present" to z_j = +1.
    """
    g = np.asarray(g, dtype=float)
    H = np.asarray(H, dtype=float)
    U = np.asarray(U, dtype=float)

    if g.ndim != 1:
        raise ValueError("g must be 1D (N,).")
    if H.ndim != 2 or H.shape[0] != H.shape[1]:
        raise ValueError("H must be square (N,N).")
    if U.ndim != 2:
        raise ValueError("U must be 2D (N,K).")

    N = g.shape[0]
    if H.shape != (N, N):
        raise ValueError(f"H shape mismatch: expected ({N},{N}), got {H.shape}.")
    if U.shape[0] != N:
        raise ValueError(f"U shape mismatch: expected ({N},K), got {U.shape}.")
    if not (0 <= int(j_gate) < N):
        raise ValueError(f"j_gate out of range: {j_gate} for N={N}.")

    K = int(U.shape[1])

    # Symmetrize defensively
    H = 0.5 * (H + H.T)

    # Indices excluding the gate
    idx_rest = np.arange(N, dtype=int)
    idx_rest = idx_rest[idx_rest != int(j_gate)]

    # Partition objects
    g_rest = g[idx_rest]                        # (N-1,)
    H_rest_rest = H[np.ix_(idx_rest, idx_rest)] # (N-1,N-1)
    H_rest_j = H[idx_rest, int(j_gate)]         # (N-1,)

    U_rest = U[idx_rest, :]                     # (N-1,K)
    U_j = U[int(j_gate), :].copy()              # (K,)

    def _region_params(s: float) -> Dict[str, Any]:
        s = float(s)
        F0s = float(F0z) + s * float(g[int(j_gate)])
        g_s = g_rest + s * H_rest_j             # (N-1,)

        # Project to latent subspace using U_rest
        a_s = (U_rest.T @ g_s).astype(float, copy=False)            # (K,)
        B_s = (U_rest.T @ (H_rest_rest @ U_rest)).astype(float, copy=False)  # (K,K)
        B_s = 0.5 * (B_s + B_s.T)

        delta_s = (s * U_j).astype(float, copy=False)              # (K,)

        return dict(F0=F0s, a=a_s, B=B_s, delta=delta_s, s=s)

    out = dict(
        meta=dict(
            j_gate=int(j_gate),
            K=K,
            idx_rest=idx_rest,
            U_j=U_j,
            delta_absent=(-1.0 * U_j),
            delta_present=(+1.0 * U_j),
        ),
        absent=_region_params(-1.0),
        present=_region_params(+1.0),
    )
    return out
