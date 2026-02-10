# epistasia/fees.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Literal, Optional, Tuple, Sequence

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from ..core import Landscape
from ..epistasis import focal_effect
from .qinference import compute_q_second_order, QSecondOrderResult

import warnings

YMode = Literal["deltaF", "DF", "Fprime"]

NuStarMode = Literal["mean_nu", "mean_states", "custom"]


####################################################
#                    DATACLASS                     #
####################################################

@dataclass
class FEEData:
    i: int
    feature_name: str

    # replicate-level data (M', R)
    F0: np.ndarray
    F1: np.ndarray
    delta: np.ndarray

    # backgrounds (M', N-1)
    backgrounds: np.ndarray

    # x/y summaries (M',)
    x_mean: np.ndarray
    y_mean: np.ndarray
    x_std: Optional[np.ndarray] = None
    y_std: Optional[np.ndarray] = None

    # optional: bootstrap CI (M',)
    x_ci_low: Optional[np.ndarray] = None
    x_ci_high: Optional[np.ndarray] = None
    y_ci_low: Optional[np.ndarray] = None
    y_ci_high: Optional[np.ndarray] = None

    # optional fit
    slope: Optional[float] = None
    intercept: Optional[float] = None
    r2: Optional[float] = None

    meta: Optional[Dict[str, Any]] = None

    def to_dataframe(self) -> pd.DataFrame:
        df = pd.DataFrame({
            "x_mean": self.x_mean,
            "y_mean": self.y_mean,
        })
        if self.x_std is not None:
            df["x_std"] = self.x_std
        if self.y_std is not None:
            df["y_std"] = self.y_std
        if self.x_ci_low is not None:
            df["x_ci_low"] = self.x_ci_low
            df["x_ci_high"] = self.x_ci_high
        if self.y_ci_low is not None:
            df["y_ci_low"] = self.y_ci_low
            df["y_ci_high"] = self.y_ci_high
        return df

###########################################################
#               FUNCTIONAL EFFECT EQUATION                #
###########################################################

def fee_data(
    L,
    i: int,
    *,
    y_mode: YMode = "deltaF",
    missing_policy: str = "error",
    nan_policy: str = "omit",
    # pass-through bootstrap settings for focal_effect
    B_uncertainty: int = 0,
    uncertainty_flavor: str = "iid",
    multipliers: str = "rademacher",
    ci_level: float = 0.95,
    rng: np.random.Generator | None = None,
    fit: bool = True,
) -> FEEData:
    """
    Build FEE scatter data for locus i.

    Uses epistasis.focal_effect() for pairing/ΔF computation.
    Requires focal_effect to store abs_idx0/abs_idx1 in E.meta for reconstructing F0/F1.
    """
    E = focal_effect(
        L, i,
        missing_policy=missing_policy,
        nan_policy=nan_policy,
        B_uncertainty=B_uncertainty,
        uncertainty_flavor=uncertainty_flavor,
        multipliers=multipliers,
        ci_level=ci_level,
        rng=rng,
        as_dataframe=False,
    )

    if E.meta is None or "abs_idx0" not in E.meta or "abs_idx1" not in E.meta:
        raise ValueError(
            "focal_effect() must store 'abs_idx0' and 'abs_idx1' in E.meta "
            "so fees can reconstruct F0/F1 without recomputing pairing."
        )

    abs_idx0 = np.asarray(E.meta["abs_idx0"], dtype=int)
    abs_idx1 = np.asarray(E.meta["abs_idx1"], dtype=int)

    F0 = np.asarray(L.values[abs_idx0], dtype=float)  # (M', R)
    F1 = np.asarray(L.values[abs_idx1], dtype=float)  # (M', R)
    delta = np.asarray(E.values, dtype=float)         # (M', R)
    #delta = F1 - F0   # (M', R)

    # X axis: typically F0
    x_rep=F0
    
    # Y axis
    if y_mode == "deltaF":
        y_rep = delta
    elif y_mode == "DF":
        y_rep = 0.5 * delta
    elif y_mode == "Fprime":
        y_rep = F1
    else:
        raise ValueError("y_mode must be 'deltaF', 'DF', or 'Fprime'")

    x_mean = np.nanmean(x_rep, axis=1)
    y_mean = np.nanmean(y_rep, axis=1)
    x_std = np.nanstd(x_rep, axis=1, ddof=1) if x_rep.shape[1] > 1 else None
    y_std = np.nanstd(y_rep, axis=1, ddof=1) if y_rep.shape[1] > 1 else None

    out = FEEData(
        i=i,
        feature_name=L.feature_names[i],
        F0=F0,
        F1=F1,
        delta=delta,
        backgrounds=np.asarray(E.backgrounds, dtype=int),
        x_mean=x_mean,
        y_mean=y_mean,
        x_std=x_std,
        y_std=y_std,
        meta={
            "y_mode": y_mode,
            "missing_policy": missing_policy,
            "nan_policy": nan_policy,
        },
    )

    # Optional: use bootstrap CI from focal_effect for y if available
    # (Only for deltaF case; later puedes extender a x también)
    if y_mode in ("deltaF", "DF") and hasattr(E, "ci_low") and E.ci_low is not None:
        y_ci_low = np.asarray(E.ci_low, dtype=float)
        y_ci_high = np.asarray(E.ci_high, dtype=float)
        if y_mode == "DF":
            y_ci_low = 0.5 * y_ci_low
            y_ci_high = 0.5 * y_ci_high
        out.y_ci_low = y_ci_low
        out.y_ci_high = y_ci_high

    # Fit lineal + R2 sobre medias
    if fit:
        ok = np.isfinite(x_mean) & np.isfinite(y_mean)
        if np.sum(ok) >= 2:
            x = x_mean[ok]
            y = y_mean[ok]
            A = np.vstack([x, np.ones_like(x)]).T
            slope, intercept = np.linalg.lstsq(A, y, rcond=None)[0]
            yhat = slope * x + intercept
            ss_res = np.sum((y - yhat) ** 2)
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
            out.slope = float(slope)
            out.intercept = float(intercept)
            out.r2 = float(r2)

    return out

###############################################################
#                   SECOND ORDER THEORY                       #
###############################################################

@dataclass(frozen=True)
class QSecondOrderX:
    F0x: float
    ax: float
    bx: float
    u1: np.ndarray
    c: float
    # original (z) por si lo quieres guardar
    F0: float
    a: float
    b: float

@dataclass(frozen=True)
class QSecondOrderX:
    """
    Quadratic q in x-coordinates:
        F(x) = F0x + ax * nu_x + bx * nu_x^2
    where nu_x = u1^T x (same u1 as in z-space).

    We keep also the original z-space params for debugging.
    """
    # x-space point estimates
    F0x: float
    ax: float
    bx: float

    # direction + affine shift between nu_z and nu_x
    u1: np.ndarray
    c: float

    # z-space point estimates (optional but useful)
    F0: float
    a: float
    b: float

    # bootstrap draws in x-space (optional)
    boot_F0x: Optional[np.ndarray] = None  # (B,)
    boot_ax: Optional[np.ndarray] = None   # (B,)
    boot_bx: Optional[np.ndarray] = None   # (B,)
    boot_c: Optional[np.ndarray] = None    # (B,)
    boot_u1: Optional[np.ndarray] = None   # (B,N)

    # carry-through diagnostics if you want
    meta: Optional[Dict[str, Any]] = None


def _qz_to_qx_scalar_params(
    *,
    F0: float,
    a: float,
    b: float,
    u1: np.ndarray,
) -> Tuple[float, float, float, float]:
    """
    Convert scalar rank-1 quadratic from z to x.

    nu_z = u1^T z
    nu_x = u1^T x
    z = 2x - 1  ->  nu_z = 2 nu_x - c,  c = sum_i u1_i

    If q(nu_z) = F0 + a nu_z + b nu_z^2, then
        bx = 4 b
        ax = 2 a - 4 b c
        F0x = F0 - a c + b c^2
    """
    u1 = np.asarray(u1, float)
    c = float(np.sum(u1))

    bx = 4.0 * float(b)
    ax = 2.0 * float(a) - 4.0 * float(b) * c
    F0x = float(F0) - float(a) * c + float(b) * (c ** 2)

    return F0x, ax, bx, c


def compute_q_second_order_x(
    L,
    *,
    which: str = "uncertainty",
    k: int = 1,
    align_eigvecs: bool = True,
    return_bootstrap: bool = True,
    walsh_kwargs: Optional[dict] = None,
) -> QSecondOrderX:
    """
    Call qinference in z, then convert coefficients to x-coordinates.
    Now propagates bootstrap draws (including boot_u1) draw-by-draw.
    """
    Q: QSecondOrderResult = compute_q_second_order(
        L,
        which=which,
        k=k,
        align_eigvecs=align_eigvecs,
        return_bootstrap=return_bootstrap,
        walsh_kwargs=walsh_kwargs,
    )

    u1 = np.asarray(Q.u1, dtype=float)
    F0, a, b = float(Q.F0), float(Q.a), float(Q.b)

    F0x, ax, bx, c = _qz_to_qx_scalar_params(F0=F0, a=a, b=b, u1=u1)

    # -------------------------
    # Bootstrap propagation
    # -------------------------
    boot_F0x = boot_ax = boot_bx = boot_c = boot_u1 = None

    if return_bootstrap and (Q.boot_a is not None) and (Q.boot_b is not None) and (Q.boot_F0 is not None):
        boot_a = np.asarray(Q.boot_a, float)
        boot_b = np.asarray(Q.boot_b, float)
        boot_F0 = np.asarray(Q.boot_F0, float)

        # Prefer boot_u1 if present (best), else fall back to fixed u1
        if Q.boot_u1 is not None:
            boot_u1 = np.asarray(Q.boot_u1, float)  # (B,N)
        else:
            boot_u1 = np.tile(u1[None, :], (boot_a.shape[0], 1))

        B = int(boot_a.shape[0])
        if boot_b.shape[0] != B or boot_F0.shape[0] != B or boot_u1.shape[0] != B:
            raise ValueError("Bootstrap arrays have inconsistent lengths in QSecondOrderResult.")

        boot_F0x = np.empty(B, dtype=float)
        boot_ax = np.empty(B, dtype=float)
        boot_bx = np.empty(B, dtype=float)
        boot_c = np.empty(B, dtype=float)

        for t in range(B):
            F0x_t, ax_t, bx_t, c_t = _qz_to_qx_scalar_params(
                F0=float(boot_F0[t]),
                a=float(boot_a[t]),
                b=float(boot_b[t]),
                u1=boot_u1[t],
            )
            boot_F0x[t] = F0x_t
            boot_ax[t] = ax_t
            boot_bx[t] = bx_t
            boot_c[t] = c_t

    return QSecondOrderX(
        F0x=F0x, ax=ax, bx=bx,
        u1=u1, c=c,
        F0=F0, a=a, b=b,
        boot_F0x=boot_F0x, boot_ax=boot_ax, boot_bx=boot_bx, boot_c=boot_c, boot_u1=boot_u1,
        meta={"which": which, "k": k, "align_eigvecs": align_eigvecs},
    )


# ------------------------------------------------------------
# FEE theory: exact curve + linearization + bootstrap bands
# ------------------------------------------------------------

def fee_theory_curve_deltaF_vs_F0(
    F0_vals: np.ndarray,
    *,
    u_i: float,
    F0x: float,
    ax: float,
    bx: float,
    branch: int = +1,   # +1 or -1 selects the sqrt branch
) -> np.ndarray:
    """
    Exact quadratic-model FEE (x-coordinates):
        Δ_i F  as a function of  F0 = F(x with x_i=0)

    Uses inversion of:
        F0 = F0x + ax nu + bx nu^2    (nu = u^T x)
    and then:
        Δ_i F = u_i * q'(nu_z) + (bx * u_i^2)   (in your derived closed form)
    Implemented as:
        Δ_i F(F0) = branch * u_i * sqrt(ax^2 + 4 bx (F0 - F0x)) + bx u_i^2
    """
    F0_vals = np.asarray(F0_vals, float)
    disc = ax * ax + 4.0 * bx * (F0_vals - F0x)
    disc = np.maximum(disc, 0.0)  # numerical safety
    return (float(branch) * float(u_i) * np.sqrt(disc)) + (float(bx) * float(u_i) ** 2)


def fee_linear_alpha_beta(
    *,
    u_i: float,
    F0x: float,
    ax: float,
    bx: float,
    eps: float = 1e-12,
) -> Tuple[float, float]:
    """
    Linearized FEE around the vertex / small-curvature regime:
        Δ_i F ≈ alpha_i + beta_i * F0

    beta = (2 bx / ax) u_i
    alpha = (ax u_i) + (bx u_i^2) - beta * F0x

    Guard: if ax ~ 0, slope explodes -> return (nan, nan).
    """
    if abs(ax) < eps:
        return (float("nan"), float("nan"))

    beta = (2.0 * bx / ax) * u_i
    alpha = (ax * u_i) + (bx * u_i * u_i) - beta * F0x
    return float(alpha), float(beta)


def fee_theory_curve_deltaF_vs_F0_band(
    F0_vals: np.ndarray,
    *,
    u_i: float,
    Qx: QSecondOrderX,
    branch: int = +1,
    ci_level: float = 0.95,
) -> Dict[str, np.ndarray]:
    """
    Bootstrap band for the exact curve Δ_iF(F0).
    Returns dict with keys: "mean", "low", "high".
    If no bootstraps, returns the point curve as all three.
    """
    F0_vals = np.asarray(F0_vals, float)

    point = fee_theory_curve_deltaF_vs_F0(
        F0_vals, u_i=u_i, F0x=Qx.F0x, ax=Qx.ax, bx=Qx.bx, branch=branch
    )

    if (Qx.boot_F0x is None) or (Qx.boot_ax is None) or (Qx.boot_bx is None):
        return {"mean": point, "low": point, "high": point}

    B = int(Qx.boot_ax.shape[0])
    Y = np.empty((B, F0_vals.size), dtype=float)
    for t in range(B):
        Y[t] = fee_theory_curve_deltaF_vs_F0(
            F0_vals,
            u_i=u_i,
            F0x=float(Qx.boot_F0x[t]),
            ax=float(Qx.boot_ax[t]),
            bx=float(Qx.boot_bx[t]),
            branch=branch,
        )

    alpha = 0.5 * (1.0 - ci_level)
    low = np.quantile(Y, alpha, axis=0)
    high = np.quantile(Y, 1.0 - alpha, axis=0)
    mean = np.nanmean(Y, axis=0)

    return {"mean": mean, "low": low, "high": high}


def fee_linear_alpha_beta_bootstrap(
    *,
    u_i: float,
    Qx: QSecondOrderX,
    ci_level: float = 0.95,
) -> Dict[str, float]:
    """
    Bootstrap summary for (alpha_i, beta_i) using the linearized formulas.

    Returns:
      {
        "alpha": point_alpha, "beta": point_beta,
        "alpha_low", "alpha_high", "beta_low", "beta_high"
      }
    """
    alpha0, beta0 = fee_linear_alpha_beta(u_i=u_i, F0x=Qx.F0x, ax=Qx.ax, bx=Qx.bx)

    out: Dict[str, float] = {"alpha": float(alpha0), "beta": float(beta0)}

    if (Qx.boot_F0x is None) or (Qx.boot_ax is None) or (Qx.boot_bx is None):
        out.update(
            alpha_low=float(alpha0), alpha_high=float(alpha0),
            beta_low=float(beta0), beta_high=float(beta0),
        )
        return out

    B = int(Qx.boot_ax.shape[0])
    A = np.empty(B, float)
    BETA = np.empty(B, float)
    for t in range(B):
        at, bt = fee_linear_alpha_beta(
            u_i=u_i,
            F0x=float(Qx.boot_F0x[t]),
            ax=float(Qx.boot_ax[t]),
            bx=float(Qx.boot_bx[t]),
        )
        A[t] = at
        BETA[t] = bt

    alpha = 0.5 * (1.0 - ci_level)
    out.update(
        alpha_low=float(np.nanquantile(A, alpha)),
        alpha_high=float(np.nanquantile(A, 1.0 - alpha)),
        beta_low=float(np.nanquantile(BETA, alpha)),
        beta_high=float(np.nanquantile(BETA, 1.0 - alpha)),
    )
    return out


def plot_fees_grid_from_landscape(
    L,
    *,
    y_mode: str = "deltaF",
    missing_policy: str = "error",
    nan_policy: str = "omit",
    # focal_effect bootstrap (for y errorbars/CI in fee_data)
    B_uncertainty: int = 0,
    uncertainty_flavor: str = "iid",
    multipliers: str = "rademacher",
    ci_level: float = 0.95,
    rng: np.random.Generator | None = None,
    # qinference bootstrap (for theoretical CI band)
    which: str = "uncertainty",
    k: int = 1,
    align_eigvecs: bool = True,
    return_bootstrap: bool = True,
    walsh_kwargs: dict | None = None,
    # plotting
    figsize_per_ax: tuple[float, float] = (3.3, 3.0),
    x_grid: int = 200,
    theory_ci_level: float | None = None,  # if None, uses ci_level
    show_legend: bool = True,
    sharex: bool = False,
    sharey: bool = False,
) -> tuple[plt.Figure, np.ndarray]:
    """
    Wrapper: given a Landscape L, plot all FEEs in a 2-row grid with ceil(N/2) columns.

    Each panel:
      - scatter of (F0, DeltaF) using fee_data() means (with optional y CI/errorbars)
      - linear fit from fee_data() (slope/intercept/R2)
      - theoretical prediction curve + bootstrap CI band (from compute_q_second_order_x)

    Notes:
      - X-axis is F0 (background with x_i=0), matching fee_data().
      - Theoretical band uses qinference bootstraps; if absent, band collapses to the point curve.
      - Chooses theory branch (+/-) per locus by SSE against y_mean.
    """
    if theory_ci_level is None:
        theory_ci_level = ci_level

    # Determine N robustly
    if hasattr(L, "N") and L.N is not None:
        N = int(L.N)
    else:
        N = len(L.feature_names)

    nrows = 2
    ncols = int(np.ceil(N / nrows))

    fig_w = ncols * float(figsize_per_ax[0])
    fig_h = nrows * float(figsize_per_ax[1])
    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(fig_w, fig_h),
        squeeze=False,
        sharex=sharex,
        sharey=sharey,
    )

    # Precompute quadratic theory in x-coordinates (with bootstraps if available)
    Qx = compute_q_second_order_x(
        L,
        which=which,
        k=k,
        align_eigvecs=align_eigvecs,
        return_bootstrap=return_bootstrap,
        walsh_kwargs=walsh_kwargs,
    )

    def ax_for(i: int):
        r = i // ncols
        c = i % ncols
        return axes[r, c]

    for i in range(N):
        ax = ax_for(i)

        # Build FEE data
        E = fee_data(
            L, i,
            y_mode=y_mode,
            missing_policy=missing_policy,
            nan_policy=nan_policy,
            B_uncertainty=B_uncertainty,
            uncertainty_flavor=uncertainty_flavor,
            multipliers=multipliers,
            ci_level=ci_level,
            rng=rng,
            fit=True,
        )

        x = np.asarray(E.x_mean, dtype=float)
        y = np.asarray(E.y_mean, dtype=float)
        ok = np.isfinite(x) & np.isfinite(y)
        x_ok = x[ok]
        y_ok = y[ok]

        # Title
        name = E.feature_name if E.feature_name is not None else f"feature_{i}"
        ax.set_title(name, fontsize=10)

        # Scatter (means)
        ax.scatter(x_ok, y_ok, s=12, alpha=0.8, label="data")

        # Optional y-CI from focal_effect bootstrap (stored in fee_data)
        if E.y_ci_low is not None and E.y_ci_high is not None:
            ylo = np.asarray(E.y_ci_low, dtype=float)[ok]
            yhi = np.asarray(E.y_ci_high, dtype=float)[ok]
            yerr = np.vstack([y_ok - ylo, yhi - y_ok])  # asymmetric
            ax.errorbar(x_ok, y_ok, yerr=yerr, fmt="none", alpha=0.35, linewidth=0.8)

        # Linear fit line (from fee_data)
        if E.slope is not None and E.intercept is not None and np.sum(ok) >= 2:
            xmin, xmax = float(np.nanmin(x_ok)), float(np.nanmax(x_ok))
            xfit = np.linspace(xmin, xmax, 50)
            yfit = float(E.slope) * xfit + float(E.intercept)
            lbl = "fit"
            if E.r2 is not None and np.isfinite(E.r2):
                lbl = f"fit ($R^2={E.r2:.2f}$)"
            ax.plot(xfit, yfit, linewidth=1.5, label=lbl)

        # Theoretical curve + CI band (choose branch by SSE)
        if np.sum(ok) >= 2:
            u_i = float(Qx.u1[i])

            xmin, xmax = float(np.nanmin(x_ok)), float(np.nanmax(x_ok))
            xgrid = np.linspace(xmin, xmax, int(x_grid))

            # choose branch by SSE on observed points
            yhat_p = fee_theory_curve_deltaF_vs_F0(
                x_ok, u_i=u_i, F0x=Qx.F0x, ax=Qx.ax, bx=Qx.bx, branch=+1
            )
            yhat_m = fee_theory_curve_deltaF_vs_F0(
                x_ok, u_i=u_i, F0x=Qx.F0x, ax=Qx.ax, bx=Qx.bx, branch=-1
            )
            sse_p = float(np.nansum((y_ok - yhat_p) ** 2))
            sse_m = float(np.nansum((y_ok - yhat_m) ** 2))
            branch = +1 if sse_p <= sse_m else -1

            band = fee_theory_curve_deltaF_vs_F0_band(
                xgrid, u_i=u_i, Qx=Qx, branch=branch, ci_level=float(theory_ci_level)
            )
            ax.plot(xgrid, band["mean"], linewidth=1.8, label="theory")
            ax.fill_between(xgrid, band["low"], band["high"], alpha=0.18)

        # Axis labels: only left column + bottom row
        if (i // ncols) == (nrows - 1):
            ax.set_xlabel(r"$F_0$")
        if (i % ncols) == 0:
            ax.set_ylabel(r"$\Delta_i F$")

    # Turn off unused axes
    total = nrows * ncols
    for j in range(N, total):
        axes[j // ncols, j % ncols].axis("off")

    # Legend only once to avoid clutter
    if show_legend:
        # pick first non-empty axis
        first_ax = None
        for i in range(N):
            first_ax = ax_for(i)
            break
        if first_ax is not None:
            handles, labels = first_ax.get_legend_handles_labels()
            if handles:
                fig.legend(
                    handles, labels,
                    loc="upper center",
                    ncol=min(4, len(labels)),
                    frameon=False,
                )
                fig.tight_layout(rect=(0, 0, 1, 0.93))
            else:
                fig.tight_layout()
    else:
        fig.tight_layout()

    return fig, axes
