#######################################################
#                      IMPORTS                        #
#######################################################

from __future__ import annotations

import numpy as np
from math import comb
import pandas as pd
import matplotlib.pyplot as plt

from typing import TYPE_CHECKING, Any, Dict, Optional

from dataclasses import dataclass 

from .stats_noise.noise_models import bootstrap_uncertainty, bootstrap_null

if TYPE_CHECKING:
    from .core import Landscape  # only for typing, no runtime import

#############################################################
#                                                           #
#                     WALSH-HADAMARD MODULE                 #
#                                                           #
#############################################################

######################################################
#                     DATACLASS                      #
######################################################

@dataclass
class WalshResults:
    """
    Container for Walsh–Hadamard coefficients (fs) with
    replicate-level statistics, uncertainty bands, and null tests.

    This mirrors the Epistasis dataclass but in the WH (mode) domain.

    Attributes
    ----------
    values : np.ndarray
        (K, R) replicate-level Walsh coefficients f_s^(r),
        where K = 2^N is the number of modes and R is the
        number of experimental replicates.

    mean : np.ndarray
        (K,) mean across replicates (biological estimate of f_s).

    std_experimental : np.ndarray
        (K,) standard deviation across replicates (experimental variability).

    std_biological : np.ndarray
        (K,) standard error of the mean = std_experimental / sqrt(R).

    s_bits : np.ndarray
        (K, N) binary representation of each mode s in {0,1}^N.

    orders : np.ndarray
        (K,) Hamming weight |s| for each mode.

    # --- Uncertainty bootstrap (around f̂_s) ---

    ci_low : Optional[np.ndarray]
        (K,) lower confidence bound for f_s under the uncertainty bootstrap.

    ci_high : Optional[np.ndarray]
        (K,) upper confidence bound for f_s under the uncertainty bootstrap.

    sign_prob_pos : Optional[np.ndarray]
        (K,) P(f_s > 0 | data) estimated from the uncertainty bootstrap.

    sign_prob_neg : Optional[np.ndarray]
        (K,) P(f_s < 0 | data) estimated from the uncertainty bootstrap.

    # --- Null bootstrap (distribution of fs under null landscapes) ---

    null_ci_low : Optional[np.ndarray]
        (K,) lower confidence bound for f_s under the null bootstrap.

    null_ci_high : Optional[np.ndarray]
        (K,) upper confidence bound for f_s under the null bootstrap.

    snr_null : Optional[np.ndarray]
        (K,) signal-to-null-noise ratio:
            |mean_s| / std_null_s,
        where std_null_s is the std of the null distribution of f_s.

    # Global null test (variance across modes) ---
    p_value_var : Optional[float]
        One-sided p-value for excess variance across modes under the null.

    var_obs : Optional[float]
        Observed variance of mean f_s across modes.

    var_null_median : Optional[float]
        Median of the null variance distribution across modes.

    effect_size_var : Optional[float]
        var_obs - var_null_median.

    # Optional per-mode p-values (two-sided, under null around 0)
    p_null_two_sided : Optional[np.ndarray]
        (K,) two-sided p-values per mode, based on |f_s| vs null.

    meta : Optional[Dict[str, Any]]
        Metadata: transform path, bootstrap sizes, flavors, CI level, seeds, etc.
    """

    # Replicate-level WH coefficients
    values: np.ndarray        # (K, R)
    mean: np.ndarray          # (K,)
    std_experimental: np.ndarray
    std_biological: np.ndarray

    # Geometry
    s_bits: np.ndarray        # (K, N)
    orders: np.ndarray        # (K,)

    # Uncertainty bootstrap
    ci_low: Optional[np.ndarray] = None
    ci_high: Optional[np.ndarray] = None
    sign_prob_pos: Optional[np.ndarray] = None
    sign_prob_neg: Optional[np.ndarray] = None

    # Null bootstrap
    null_ci_low: Optional[np.ndarray] = None
    null_ci_high: Optional[np.ndarray] = None
    snr_null: Optional[np.ndarray] = None

    # Global null test
    p_value_var: Optional[float] = None
    var_obs: Optional[float] = None
    var_null_median: Optional[float] = None
    effect_size_var: Optional[float] = None

    # Optional per-mode null p-values
    p_null_two_sided: Optional[np.ndarray] = None

    meta: Optional[Dict[str, Any]] = None

    # Optional feature names for each locus (for DataFrame outputs)
    feature_names: Optional[list[str]] = None

    # ---------------- Convenience methods ----------------

    def summary(self) -> np.ndarray:
        """
        Return a compact (K, 5) array summarizing mean, stds,
        and confidence intervals.

        Columns:
        [mean, std_experimental, std_biological, ci_low, ci_high]
        """
        ci_lo = self.ci_low if self.ci_low is not None else np.full_like(self.mean, np.nan)
        ci_hi = self.ci_high if self.ci_high is not None else np.full_like(self.mean, np.nan)
        return np.vstack(
            [self.mean, self.std_experimental, self.std_biological, ci_lo, ci_hi]
        ).T

    def to_dataframe(self) -> pd.DataFrame:
        """
        Return a user-friendly DataFrame for interpretation in the WH domain.
        One row per mode s.
        """
        K, N = self.s_bits.shape

        # Encode modes as strings, e.g. "0101"
        s_str = [''.join(map(str, row.tolist())) for row in self.s_bits]

        df = pd.DataFrame({
            "Mode bits": s_str,
            "Order": self.orders,
            "Walsh (mean)": self.mean,
            "Experimental SD": self.std_experimental,
            "Biological SD": self.std_biological,
            "Uncertainty CI (low)": (
                self.ci_low if self.ci_low is not None else np.full_like(self.mean, np.nan)
            ),
            "Uncertainty CI (high)": (
                self.ci_high if self.ci_high is not None else np.full_like(self.mean, np.nan)
            ),
        })

        # Add sign probabilities if available
        if self.sign_prob_pos is not None:
            df["Prob(fs > 0)"] = self.sign_prob_pos
            df["Prob(fs < 0)"] = self.sign_prob_neg

        # Add null confidence intervals
        if self.null_ci_low is not None:
            df["Null CI (low)"] = self.null_ci_low
            df["Null CI (high)"] = self.null_ci_high

        # Add SNR (null)
        if self.snr_null is not None:
            df["Signal-to-Null-Noise (SNR)"] = self.snr_null

        # Global statistics replicated on each row (for convenience)
        if self.var_obs is not None:
            df["Variance (observed)"] = self.var_obs
            df["Variance (null median)"] = self.var_null_median
            df["P-value (variance excess)"] = self.p_value_var

        # --- per-mode null p-values and detection probability ---
        if self.p_null_two_sided is not None:
            p_null = self.p_null_two_sided
            df["p-null"] = p_null
            #df["p-detection"] = 1.0 - p_null

        # Optional mapping from bits to feature names
        if self.feature_names is not None:
            mode_as_features = []
            for row in self.s_bits:
                features = [name for bit, name in zip(row, self.feature_names) if bit == 1]
                mode_as_features.append(",".join(features) if features else "∅")

            df.insert(
                1,                     # position 1 → second column
                "Mode (features)",
                mode_as_features,
            )

        return df


@dataclass
class EpistasisAmplitudeResults:
    # geometry
    orders: np.ndarray                    # S = 1..N
    amplitude: np.ndarray                 # <E^2>_k (point estimate)

    # uncertainty bootstrap
    ci_low: Optional[np.ndarray] = None
    ci_high: Optional[np.ndarray] = None

    # null bootstrap
    null_ci_low: Optional[np.ndarray] = None
    null_ci_high: Optional[np.ndarray] = None
    snr_null: Optional[np.ndarray] = None   # |A_obs| / std_null

    # ---- NEW: per-order variance statistics ----
    var_obs_order: Optional[np.ndarray] = None         # observed amplitude per order (= amplitude itself)
    var_null_median_order: Optional[np.ndarray] = None # median null amplitude per order

    # ---- NEW: per-order significance ----
    p_value_order: Optional[np.ndarray] = None         # P( null ≥ obs ) per order

    # ---- Global tests (optional, for compatibility) ----
    var_obs_global: Optional[float] = None             # total amplitude sum
    var_null_median_global: Optional[float] = None
    p_value_var: Optional[float] = None                # global null test
    effect_size_var: Optional[float] = None

    meta: Optional[Dict[str, Any]] = None

    def to_dataframe(self):
        """Return a tidy DataFrame with per-order epistasis amplitude statistics."""
        return pd.DataFrame({
            "Order": self.orders,
            "Epistasis amplitude <E^2>_k": self.amplitude,
            "CI low": self.ci_low,
            "CI high": self.ci_high,
            "Null CI low": self.null_ci_low,
            "Null CI high": self.null_ci_high,
            "SNR (null)": self.snr_null,
            "Variance (obs)": self.var_obs_order,
            "Variance (null median)": self.var_null_median_order,
            "P-value order": self.p_value_order,
            "P-value var (global)": (
                np.repeat(self.p_value_var, len(self.orders))
                if self.p_value_var is not None else None
            )
        })

######################################################
#                      HELPERS                       #
######################################################


def _clean_landscape_for_wh(
    L: "Landscape",
    *,
    missing_policy: str = "error",
    nan_policy: str = "omit",
) -> tuple[np.ndarray, np.ndarray, int]:
    """
    Minimal cleaning before applying Walsh–Hadamard.
    """
    values = np.asarray(L.values)   # (M,) or (M,R)
    states = np.asarray(L.states)   # (M,N) in {0,1}
    if values.ndim == 1:
        values = values[:, None]    # (M,1)

    if states.shape[0] != values.shape[0]:
        raise ValueError("states and values must have matching first dimension.")

    # NaN handling
    if nan_policy == "omit":
        mask = np.isfinite(values).all(axis=1)
        values = values[mask]
        states = states[mask]
    elif nan_policy == "propagate":
        pass
    else:
        raise ValueError(f"Unknown nan_policy={nan_policy!r}")

    # Missing policy hook (you can extend this)
    if missing_policy not in {"error", "drop"}:
        raise ValueError(f"Unknown missing_policy={missing_policy!r}")

    N = states.shape[1]
    return values, states, N


def _draw_multipliers(
    size: int,
    multipliers: str,
    rng: np.random.Generator,
) -> np.ndarray:
    """Draw bootstrap multipliers for wild bootstrap."""
    if multipliers == "rademacher":
        return rng.choice([-1.0, 1.0], size=size)
    elif multipliers == "normal":
        return rng.standard_normal(size=size)
    else:
        raise ValueError(f"Unknown multipliers={multipliers!r}")


def states_to_spins_shifted(states: np.ndarray) -> np.ndarray:
    """
    Map binary states {0,1}^N to shifted WH spins z_i(x) = (-1)^(x_i + 1).
    This is the base object to build any WH column phi_s(x) as product of spins
    over the indices where s_i = 1.

    Parameters
    ----------
    states : (M, N) array-like of {0,1}
        Each row is a genotype x in {0,1}^N.

    Returns
    -------
    Z : (M, N) float ndarray in {-1.0, +1.0}
        Z[m, i] = -1 if states[m, i] == 0; +1 if states[m, i] == 1.

    Notes
    -----
    - With this "shifted" convention: 0 -> -1, 1 -> +1.
    - Any WH mode s can be evaluated later as:
        phi_s = np.prod(Z[:, mask_s], axis=1)
      where mask_s is a boolean mask of the active coordinates in s.
    """
    states = np.asarray(states)
    if states.ndim != 2:
        raise ValueError("states must be a 2D array of shape (M, N).")
    # 0 -> -1, 1 -> +1 (shifted convention)
    return np.where(states == 1, 1.0, -1.0)


def fwht_shifted(a: np.ndarray, N: int) -> np.ndarray:
    """
    Fast Walsh–Hadamard Transform with 'shifted' correction built-in.
    - Input states must be in lexicographic order over {0,1}^N.
    - Works on shape (2^N,) or (2^N, R) and applies the transform column-wise.
    - Returns orthonormal coefficients f_s under φ_s(x)=(-1)^{s·(x+1)}.

    Steps:
      1) FWHT (non-normalized), in-place on a copy.
      2) Orthonormal normalization: divide by 2^N.
      3) Shifted correction: multiply each coefficient by (-1)^{|s|}.

    Parameters
    ----------
    a : np.ndarray
        Phenotype vector (2^N,) or matrix (2^N, R) without NaNs.
    N : int
        Number of loci (so len(a) == 2^N along axis 0).

    Returns
    -------
    np.ndarray
        f_s with same shape as input.
    """
    A = np.asarray(a)
    if A.ndim == 1:
        A = A[:, None]
        squeeze = True
    elif A.ndim == 2:
        squeeze = False
    else:
        raise ValueError("a must be 1D (2^N,) or 2D (2^N, R).")

    M, R = A.shape
    if M != (1 << N):
        raise ValueError("First dimension must be 2^N and match N.")
    if M & (M - 1):
        raise ValueError("Length must be power of two (2^N).")

    # --- FWHT (butterfly), in-place on a copy to avoid mutating user data ---
    A = A.copy()
    h = 1
    while h < M:
        step = h << 1
        for i in range(0, M, step):
            j = i + h
            x = A[i:j, :]
            y = A[j:i+step, :]
            A[i:j, :], A[j:i+step, :] = x + y, x - y
        h = step

    # --- Orthonormal normalization ---
    A /= float(1 << N)

    # --- Shifted correction: multiply each coefficient by (-1)^{|s|} ---
    idx = np.arange(M)[:, None]                      # coefficient indices = binary s
    hw = ((idx >> np.arange(N)) & 1).sum(axis=1)     # Hamming weight |s|
    sign = np.where(hw % 2 == 0, 1.0, -1.0)          # (-1)^{|s|}
    A *= sign[:, None]

    return A[:, 0] if squeeze else A

def _ensure_lex_and_sort(V, states):
    """Ensure lex order by sorting states as binary number and applying same index to V."""
    idx = _lex_index(states)                 # (M,)
    order = np.argsort(idx)
    reordered = not np.all(order == np.arange(len(order)))
    return V[order, :], reordered

def _lex_index(states):
    """Map each state row (b0..bN-1) to its integer code in lex (b0 MSB)."""
    N = states.shape[1]
    powers = 1 << np.arange(N-1, -1, -1)     # [2^(N-1), ..., 1]
    return (states @ powers)

def _binary_domain(N):
    """All s in {0,1}^N in lex order."""
    # Equivalent to: np.array(list(product([0,1], repeat=N)))
    grid = np.indices((2,) * N).reshape(N, -1).T
    return grid

def _fallback_projection(V, states, N):
    """
    Slow but general: project onto shifted basis using observed rows only.
    Works per column (replicate-safe). Returns (fs, mask_rows).
    """
    # Build spins Z (M,N) in {-1,+1}
    Z = np.where(states == 1, 1.0, -1.0)  # shifted spins
    M, R = V.shape
    K = 1 << N
    fs = np.zeros((K, R), dtype=float)

    # mask: rows without NaNs per column; here we require row-wise valid across all columns (simple choice)
    mask_rows = ~np.any(np.isnan(V), axis=1)
    Vc = V[mask_rows, :]; Zc = Z[mask_rows, :]

    # For each mode s (indexed 0..2^N-1), compute phi_s(x)=prod(Z_i^s_i) and project.
    # NOTE: This is O(M * 2^N); fine for tutorial/small N. You can vectorize later.
    for s_idx in range(K):
        s_bits = ((s_idx >> np.arange(N-1, -1, -1)) & 1).astype(bool)  # boolean mask of active loci
        phi_s = np.prod(Zc[:, s_bits], axis=1) if s_bits.any() else np.ones(Zc.shape[0])
        denom = float(Zc.shape[0])
        # Column-wise dot
        fs[s_idx, :] = (phi_s[:, None] * Vc).sum(axis=0) / denom

    return fs, mask_rows

def binary_domain(N: int) -> np.ndarray:
    """
    All s in {0,1}^N in lexicographic order.
    Returns
    -------
    np.ndarray, shape (2^N, N)
        Each row is a binary vector s.
    """
    # Equivalent to: np.array(list(product([0,1], repeat=N)))
    grid = np.indices((2,) * N).reshape(N, -1).T
    return grid


def mode_orders(N: int) -> np.ndarray:
    """
    Hamming weights |s| for all modes s in lex order.
    Returns
    -------
    np.ndarray, shape (2^N,)
        orders[k] = sum of bits of s_k.
    """
    s_bits = binary_domain(N)
    return s_bits.sum(axis=1)

def decode_mode(self, idx: int):
    """
    Return the list of feature names corresponding to the 1-bits of mode `idx`.
    """
    if self.feature_names is None:
        raise ValueError("feature_names not set in WalshResults.")
    bits = self.s_bits[idx]
    return [name for bit, name in zip(bits, self.feature_names) if bit == 1]

##############################################################
#                                                            #
#                   WH TRANSFORM FUNCTION                    #
#                                                            #
##############################################################

def wh_transform(values: np.ndarray, states: np.ndarray, N: int, strategy: str = "auto"):
    """
    Unified entry point: choose FWHT (complete, no NaNs) or fallback projection (incomplete/NaNs).
    values: (M,) or (M,R)
    states: (M,N) in {0,1}
    """
    V = np.asarray(values)
    if V.ndim == 1:
        V = V[:, None]  # (M,1)
        squeeze = True
    else:
        squeeze = False

    M = V.shape[0]
    meta = {"path": None, "reordered": False, "mask_rows": None}

    # Decide path
    complete = (M == (1 << N))
    has_nans = np.isnan(V).any()

    if strategy == "fwht":
        if not complete or has_nans:
            raise ValueError("FWHT requires complete landscape (2^N) and no NaNs.")
        V_sorted, reordered = _ensure_lex_and_sort(V, states)
        fs = fwht_shifted(V_sorted, N)
        meta.update({"path": "fwht", "reordered": reordered, "mask_rows": np.ones(M, bool)})
    elif strategy == "fallback":
        fs, mask = _fallback_projection(V, states, N)
        meta.update({"path": "fallback", "reordered": False, "mask_rows": mask})
    else:  # "auto"
        if complete and not has_nans:
            V_sorted, reordered = _ensure_lex_and_sort(V, states)
            fs = fwht_shifted(V_sorted, N)
            meta.update({"path": "fwht", "reordered": reordered, "mask_rows": np.ones(M, bool)})
        else:
            fs, mask = _fallback_projection(V, states, N)
            meta.update({"path": "fallback", "reordered": False, "mask_rows": mask})

    # Build s_bits and orders in lex order
    s_bits = _binary_domain(N)               # (2^N, N)
    orders = s_bits.sum(axis=1)              # (2^N,)

    return (fs[:, 0] if squeeze else fs), s_bits, orders, meta


#############################################################
#                                                           #
#                     WH STATISTICS FUNCTION                #   
#                                                           #
#############################################################

def walsh_analysis(
    L: "Landscape",
    *,
    # Robustness options
    missing_policy: str = "error",     # {"error","drop"} – currently only sanity-checked
    nan_policy: str = "omit",          # {"omit","propagate"} – forwarded to _clean_landscape_for_wh
    # Bootstrap options
    B_uncertainty: int = 0,
    uncertainty_flavor: str = "iid",   # {"iid","wildcluster"}
    B_null: int = 0,
    preserve_residual_corr: bool = False,   # null with/without biological correlations
    multipliers: str = "normal",       # {"rademacher","normal"}
    ci_level: float = 0.95,
    rng: np.random.Generator | None = None,
    # Output options
    as_dataframe: bool = True,
) -> WalshResults | pd.DataFrame:
    """
    Compute Walsh–Hadamard coefficients f_s over all modes s, with
    replicate-level statistics, bootstrap uncertainty bands, and
    null tests, mirroring the epistasis_k / focal_effect APIs.

    Additionally computes per-mode two-sided null p-values:
        p_s = P( |f_s^{(b)}| >= |f_s,obs| ) under the null bootstrap.
    """
    if rng is None:
        rng = np.random.default_rng()

    # ---------- cleaning and basic checks ----------
    values, states, N = _clean_landscape_for_wh(
        L,
        missing_policy=missing_policy,
        nan_policy=nan_policy,
    )
    M, R = values.shape
    if states.shape != (M, N):
        raise ValueError("states and values must have compatible shapes (M,N) and (M,R).")

    if R < 2 and (B_uncertainty > 0 or B_null > 0):
        raise ValueError("At least 2 replicates are required for bootstrap procedures.")

    # ---------- replicate-level WH coefficients ----------
    fs_all, s_bits, orders, meta = wh_transform(values=values, states=states, N=N)
    fs_all = np.asarray(fs_all)
    if fs_all.ndim == 1:
        fs_all = fs_all[:, None]
        R = 1
    K, R = fs_all.shape

    # Summary stats across replicates
    mean_fs = np.nanmean(fs_all, axis=1)              # (K,)
    std_exp = np.nanstd(fs_all, axis=1, ddof=1)       # (K,)
    std_bio = std_exp / np.sqrt(R)                    # (K,)

    W = WalshResults(
        values=fs_all,
        mean=mean_fs,
        std_experimental=std_exp,
        std_biological=std_bio,
        s_bits=s_bits,
        orders=orders,
        feature_names=list(L.feature_names),
        meta={"N": N, "missing_policy": missing_policy, "nan_policy": nan_policy, "transform_meta": meta},
    )

    # Early exit if no bootstrap requested
    if B_uncertainty <= 0 and B_null <= 0:
        return W.to_dataframe() if as_dataframe else W

    # ---------- DataFrame for bootstrap engines ----------
    F_df = pd.DataFrame(
        np.hstack([states.astype(int), values]),
        columns=[f"s{j}" for j in range(N)] + [f"rep{r}" for r in range(R)],
    )

    alpha = 1.0 - ci_level
    lo_q, hi_q = 100 * (alpha / 2.0), 100 * (1.0 - alpha / 2.0)

    # ==================== UNCERTAINTY BOOTSTRAP ====================
    if B_uncertainty > 0:
        Fb_unc = bootstrap_uncertainty(
            F_df,
            B=B_uncertainty,
            multipliers=multipliers,
            flavor=uncertainty_flavor,
            rng=rng,
        )
        boots_mat = Fb_unc.values[:, N:]          # (M, B_uncertainty)

        fs_unc_b = np.empty((K, B_uncertainty), dtype=float)
        for b in range(B_uncertainty):
            F_b = boots_mat[:, b]                 # (M,)
            fs_b, _, _, _ = wh_transform(
                values=F_b,
                states=states,
                N=N,
            )
            fs_b = np.asarray(fs_b)
            if fs_b.ndim > 1 and fs_b.shape[1] == 1:
                fs_b = fs_b[:, 0]
            fs_unc_b[:, b] = fs_b

        # CI per mode (uncertainty)
        ci_low = np.nanpercentile(fs_unc_b, lo_q, axis=1)   # (K,)
        ci_high = np.nanpercentile(fs_unc_b, hi_q, axis=1)  # (K,)

        # Sign probabilities with respect to 0
        sign_prob_pos = np.mean(fs_unc_b > 0, axis=1)       # (K,)
        sign_prob_neg = np.mean(fs_unc_b < 0, axis=1)       # (K,)

        if W.meta is None:
            W.meta = {}
        W.meta["uncertainty"] = {
            "B": B_uncertainty,
            "flavor": uncertainty_flavor,
            "multipliers": multipliers,
            "ci_level": ci_level,
            "sign_null_value": 0.0,
        }
        W.meta["uncertainty"]["fs_unc_b"] = fs_unc_b

        W.ci_low = ci_low
        W.ci_high = ci_high
        W.sign_prob_pos = sign_prob_pos
        W.sign_prob_neg = sign_prob_neg

    # ==================== NULL BOOTSTRAP ====================
    if B_null > 0:
        Fb_null = bootstrap_null(
            F_df,
            B=B_null,
            multipliers=multipliers,
            preserve_residual_corr=preserve_residual_corr,
            rng=rng,
        )
        boots_mat = Fb_null.values[:, N:]        # (M, B_null)

        fs_null_b = np.empty((K, B_null), dtype=float)
        for b in range(B_null):
            F_b = boots_mat[:, b]
            fs_b, _, _, _ = wh_transform(
                values=F_b,
                states=states,
                N=N,
            )
            fs_b = np.asarray(fs_b)
            if fs_b.ndim > 1 and fs_b.shape[1] == 1:
                fs_b = fs_b[:, 0]
            fs_null_b[:, b] = fs_b

        # Observed variance across modes (from point estimate)
        var_obs = float(np.nanvar(mean_fs, ddof=1))
        # Null variances distribution across modes
        var_null = np.nanvar(fs_null_b, axis=0, ddof=1)     # (B_null,)
        var_null = var_null[np.isfinite(var_null)]
        var_null_median = float(np.nanmedian(var_null)) if var_null.size else np.nan
        p_value = float(np.mean(var_null >= var_obs)) if var_null.size else np.nan
        effect_size = var_obs - var_null_median if np.isfinite(var_null_median) else np.nan

        # Null CI per mode
        null_ci_low = np.nanpercentile(fs_null_b, lo_q, axis=1)   # (K,)
        null_ci_high = np.nanpercentile(fs_null_b, hi_q, axis=1)  # (K,)

        # SNR with respect to null noise per mode
        noise_std_null = np.nanstd(fs_null_b, axis=1, ddof=1)     # (K,)
        with np.errstate(divide="ignore", invalid="ignore"):
            snr_null = np.abs(mean_fs) / noise_std_null
            snr_null[~np.isfinite(snr_null)] = np.nan

        # NEW: two-sided per-mode p-values under null around 0
        # p_s = P( |f_s^{(b)}| >= |f_s,obs| )
        abs_obs = np.abs(mean_fs)[:, None]          # (K,1)
        abs_null = np.abs(fs_null_b)                # (K,B_null)
        with np.errstate(invalid="ignore"):
            comp = abs_null >= abs_obs              # (K,B_null) boolean
            p_null_two_sided = np.mean(comp, axis=1)   # (K,)

        if W.meta is None:
            W.meta = {}
        W.meta["null"] = {
            "B": B_null,
            "preserve_residual_corr": preserve_residual_corr,
            "multipliers": multipliers,
            "var_null_distribution": var_null.tolist(),
            "ci_level": ci_level,
        }
        W.meta["null"]["fs_null_b"] = fs_null_b

        W.feature_names = L.feature_names
        W.var_obs = var_obs
        W.var_null_median = var_null_median
        W.p_value_var = p_value
        W.effect_size_var = effect_size
        W.null_ci_low = null_ci_low
        W.null_ci_high = null_ci_high
        W.snr_null = snr_null
        W.p_null_two_sided = p_null_two_sided

    # ---------- Final output ----------
    return W.to_dataframe() if as_dataframe else W

############################################
#           EPISTASIS AMPLITUDE            #
############################################
def epistasis_amplitude(
    L: "Landscape",
    *,
    # Cleaning policies
    missing_policy: str = "error",
    nan_policy: str = "omit",
    # Uncertainty bootstrap
    B_uncertainty: int = 0,
    uncertainty_flavor: str = "iid",   # {"iid","wildcluster"}
    # Null bootstrap
    B_null: int = 0,
    preserve_residual_corr: bool = False,
    multipliers: str = "normal",
    ci_level: float = 0.95,
    rng: np.random.Generator | None = None,
    # Output
    as_dataframe: bool = True,
    # How to build uncertainty CIs
    ci_method_uncertainty: str = "percentile",  # {"percentile","studentized","bca"}
    normalized: bool = False,
):
    """
    Compute the epistasis amplitude U(S) = <E^2>_k with uncertainty and null bootstrap
    statistics.

    Extended version with:
      - Per-order null variance
      - Per-order p-values
      - Per-order observed variance = U_obs(S)
      - Maintains global p-value for compatibility
    """

    if rng is None:
        rng = np.random.default_rng()

    # -----------------------------
    # 1. Clean landscape and extract data
    # -----------------------------
    values, states, N = _clean_landscape_for_wh(
        L,
        missing_policy=missing_policy,
        nan_policy=nan_policy,
    )
    M, R = values.shape

    if R < 2 and (B_uncertainty > 0 or B_null > 0):
        raise ValueError("At least 2 replicates are required for bootstrap procedures.")

    # -----------------------------
    # 2. Point estimate: F̂(x)
    # -----------------------------
    F_hat = np.nanmean(values, axis=1)  # (M,)

    # -----------------------------
    # 3. Walsh–Hadamard transform of F̂
    # -----------------------------
    fs_hat, s_bits, orders, _meta = wh_transform(
        values=F_hat,
        states=states,
        N=N,
    )
    fs_hat = np.asarray(fs_hat).ravel()

    # Total WH variance (excluding S=0)
    mask_nonzero = (orders > 0)
    Vtot = np.nansum(fs_hat[mask_nonzero] ** 2)

    # Unique orders S >= 1
    S_all = np.sort(np.unique(orders))
    S_unique = S_all[S_all > 0]
    K_ord = S_unique.size

    # -----------------------------
    # 4. Helper: epistasis amplitude
    # -----------------------------
    def amplitude_from_fs(fs_1d: np.ndarray) -> np.ndarray:
        """
        Compute U(S) = <E^2>_k from WH coefficients fs_1d.
        """
        U = np.zeros(K_ord, dtype=float)
        for i, S in enumerate(S_unique):
            mask = (orders == S)
            if not np.any(mask):
                U[i] = np.nan
                continue
            V_S = np.nansum(fs_1d[mask] ** 2)
            U[i] = (4.0 ** S) * V_S / comb(N, int(S))
        return U

    # Point estimate
    U_obs = amplitude_from_fs(fs_hat)

    if normalized:
        if Vtot > 0:
            U_obs = U_obs / Vtot
        else:
            U_obs = np.full_like(U_obs, np.nan)

    # Early exit
    if B_uncertainty <= 0 and B_null <= 0:
        if as_dataframe:
            return pd.DataFrame({
                "Order": S_unique,
                "Epistasis amplitude <E^2>_k": U_obs,
            })
        return {"orders": S_unique, "amplitude": U_obs}

    # -----------------------------
    # 5. Prepare DF for bootstrap engines
    # -----------------------------
    F_df = pd.DataFrame(
        np.hstack([states.astype(int), values]),
        columns=[f"s{j}" for j in range(N)] + [f"rep{r}" for r in range(R)],
    )

    alpha = 1.0 - ci_level
    lo_q = 100 * (alpha / 2)
    hi_q = 100 * (1 - alpha / 2)

    # Containers
    ci_low = ci_high = None
    null_ci_low = null_ci_high = None
    null_mean = None
    snr_null = None

    # per-order stats
    var_obs_order = U_obs.copy()
    var_null_median_order = None
    p_value_order = None

    # OLD global stats (kept for compatibility)
    var_obs_global = var_null_median_global = p_value_var = None

    # ==============================================================
    # 6. UNCERTAINTY BOOTSTRAP
    # ==============================================================
    U_unc_b = None  # (B_unc, K_ord)

    if B_uncertainty > 0:
        Fb_unc = bootstrap_uncertainty(
            F_df,
            B=B_uncertainty,
            multipliers=multipliers,
            flavor=uncertainty_flavor,
            rng=rng,
        )
        boots_mat = Fb_unc.values[:, N:]

        U_unc_b = np.empty((B_uncertainty, K_ord), dtype=float)
        for b in range(B_uncertainty):
            fs_b, _, _, _ = wh_transform(values=boots_mat[:, b], states=states, N=N)
            fs_b = np.asarray(fs_b).ravel()

            # NEW: compute amplitude and normalize per bootstrap sample if needed
            U_b = amplitude_from_fs(fs_b)
            if normalized:
                if Vtot > 0:
                    U_b = U_b / Vtot
                else:
                    U_b = np.full_like(U_b, np.nan)

            U_unc_b[b, :] = U_b

        # uncertainty CIs
        if ci_method_uncertainty == "percentile":
            ci_low = np.nanpercentile(U_unc_b, lo_q, axis=0)
            ci_high = np.nanpercentile(U_unc_b, hi_q, axis=0)
        else:
            # (simplified to percentile for robustness)
            ci_low = np.nanpercentile(U_unc_b, lo_q, axis=0)
            ci_high = np.nanpercentile(U_unc_b, hi_q, axis=0)

    # ==============================================================
    # 7. NULL BOOTSTRAP
    # ==============================================================
    if B_null > 0:
        Fb_null = bootstrap_null(
            F_df,
            B=B_null,
            multipliers=multipliers,
            preserve_residual_corr=preserve_residual_corr,
            rng=rng,
        )

        boots_mat = Fb_null.values[:, N:]
        U_null_b = np.empty((B_null, K_ord), dtype=float)

        for b in range(B_null):
            fs_b, _, _, _ = wh_transform(values=boots_mat[:, b], states=states, N=N)
            fs_b = np.asarray(fs_b).ravel()

            # compute amplitude and normalize per null sample if needed
            U_b = amplitude_from_fs(fs_b)
            if normalized:
                if Vtot > 0:
                    U_b = U_b / Vtot
                else:
                    U_b = np.full_like(U_b, np.nan)

            U_null_b[b, :] = U_b

        # Null CI per order
        null_ci_low = np.nanpercentile(U_null_b, lo_q, axis=0)
        null_ci_high = np.nanpercentile(U_null_b, hi_q, axis=0)
        null_mean = np.nanmean(U_null_b, axis=0)

        # SNR (ya en espacio normalizado si normalized=True)
        std_null = np.nanstd(U_null_b, axis=0, ddof=1)
        snr_null = np.where(std_null > 0, U_obs / std_null, np.nan)

        # variance_null_per_order (aquí es simplemente la mediana de U(S), normalizada si toca)
        var_null_median_order = np.nanmedian(U_null_b, axis=0)

        # p_value_per_order
        p_value_order = np.mean(U_null_b >= U_obs[None, :], axis=0)

        # Keep global test for compatibility (suma de amplitudes, normalizadas si normalized=True)
        var_obs_global = float(np.nansum(U_obs))
        total_null = np.nansum(U_null_b, axis=1)
        var_null_median_global = float(np.nanmedian(total_null))
        p_value_var = float(np.mean(total_null >= var_obs_global))

    else:
        null_mean = np.full(K_ord, np.nan)
        var_null_median_order = np.full(K_ord, np.nan)
        p_value_order = np.full(K_ord, np.nan)
        p_value_var = np.nan
        null_ci_low = null_ci_high = None
        snr_null = None
        var_obs_global = var_null_median_global = None


    # -----------------------------
    # 8. Final output DF
    # -----------------------------
    if as_dataframe:
        return pd.DataFrame({
            "Order": S_unique,
            "Epistasis amplitude <E^2>_k": U_obs,
            "CI low": ci_low if ci_low is not None else [np.nan] * K_ord,
            "CI high": ci_high if ci_high is not None else [np.nan] * K_ord,
            "Null CI low": null_ci_low if null_ci_low is not None else [np.nan] * K_ord,
            "Null CI high": null_ci_high if null_ci_high is not None else [np.nan] * K_ord,
            "SNR (null)": snr_null if snr_null is not None else [np.nan] * K_ord,
            "Null mean": null_mean if null_mean is not None else [np.nan] * K_ord,
            # per-order stats
            "Variance (obs)": var_obs_order,
            "Variance (null median)": var_null_median_order
                if var_null_median_order is not None else [np.nan] * K_ord,
            "P-value order": p_value_order
                if p_value_order is not None else [np.nan] * K_ord,

            # OLD global stats
            "P-value var": [p_value_var] * K_ord if p_value_var is not None else [np.nan] * K_ord,
        })

    return {
        "orders": S_unique,
        "amplitude": U_obs,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "null_mean": null_mean,
        "null_ci_low": null_ci_low,
        "null_ci_high": null_ci_high,
        "snr_null": snr_null,
        "var_obs_order": var_obs_order,
        "var_null_median_order": var_null_median_order,
        "p_value_order": p_value_order,
        "p_value_var": p_value_var,
    }




############################################
#            VARIANCE SPECTRUM             #
############################################

def variance_spectrum(fs: np.ndarray,
                      orders: np.ndarray,
                      aggregate: str = "sum",
                      exclude_order0: bool = True) -> dict[int, np.ndarray]:
    """
    Compute the epistatic variance spectrum V(S) from WH coefficients.

    Parameters
    ----------
    fs : np.ndarray
        WH coefficients. Shape (K,) or (K, R), with K=2^N.
        If 2D, each column is a replicate (spectrum returned per column).
    orders : np.ndarray, shape (K,)
        Hamming weights |s| for each mode (aligned with fs rows).
    aggregate : {"sum","mean"}
        Whether to sum f_s^2 within each order (default) or average.
    exclude_order0 : bool
        If True, ignore S=0 (global mean).

    Returns
    -------
    dict[int, np.ndarray]
        For each order S, an array of shape (R,) if fs is 2D, else shape (1,)
        with V(S) aggregated over modes of that order.
    """
    F = np.asarray(fs)
    if F.ndim == 1:
        F = F[:, None]  # (K,1)
    K, R = F.shape
    if orders.shape[0] != K:
        raise ValueError("orders must have length equal to fs.shape[0] (K=2^N).")

    V = {}
    S_all = np.unique(orders)
    for S in S_all:
        if exclude_order0 and S == 0:
            continue
        mask = (orders == S)
        # sum over modes of order S, per column
        vals = (F[mask, :] ** 2).sum(axis=0)
        if aggregate == "mean":
            vals = vals / mask.sum()
        V[int(S)] = vals
    return V


def variance_decomposition(fs: np.ndarray,
                           orders: np.ndarray,
                           N: int,
                           exclude_order0: bool = True) -> dict[int, dict[str, np.ndarray]]:
    """
    Decompose variance by order into (entropy H(S), geometry 4^{-S}, interaction U(S)).

    Definitions (see PDF):
      - fs = 2^{-S} * E_s  =>  E_s = 2^{S} * fs
      - V(S) = sum_{|s|=S} fs^2 = 4^{-S} * C(N,S) * <E_s^2>_S
      - H(S) = C(N,S)  (degeneracy/entropy term)
      - geom(S) = 4^{-S}
      - U(S) = <E_s^2>_S = average over modes with |s|=S of E_s^2

    Parameters
    ----------
    fs : (K,) or (K,R)
        WH coefficients (shifted, orthonormal). K = 2^N.
    orders : (K,)
        Hamming weights |s| for each mode row in fs.
    N : int
        Number of loci.
    exclude_order0 : bool
        If True, skip S=0 (global mean).

    Returns
    -------
    dict[int, dict[str, np.ndarray]]
        For each order S:
          {
            "H":   (R,) or (1,)   # C(N,S)
            "geom":(R,) or (1,)   # 4^{-S}
            "U":   (R,) or (1,)   # <E_s^2>_S
            "V":   (R,) or (1,)   # V(S) = geom * H * U
          }
    """
    F = np.asarray(fs)
    if F.ndim == 1:
        F = F[:, None]  # (K,1)
    K, R = F.shape
    if orders.shape[0] != K:
        raise ValueError("orders length must match fs rows (K=2^N).")

    out: dict[int, dict[str, np.ndarray]] = {}
    for S in np.unique(orders):
        if exclude_order0 and S == 0:
            continue
        mask = (orders == S)
        if not np.any(mask):
            continue

        # E_s = 2^S * f_s  ->  E_s^2 = 4^S * f_s^2
        Esq = ( (2.0 ** S) * F[mask, :] ) ** 2  # (num_modes_S, R)
        U = Esq.mean(axis=0)                    # <E_s^2>_S  shape (R,)

        H = float(comb(N, int(S)))              # scalar
        geom = 4.0 ** (-S)                      # scalar

        # Broadcast scalars to (R,)
        H_vec = np.full_like(U, H, dtype=float)
        geom_vec = np.full_like(U, geom, dtype=float)

        V = geom_vec * H_vec * U                # V(S)

        out[int(S)] = {"H": H_vec, "geom": geom_vec, "U": U, "V": V}
    return out

def variance_by_order_statistics(
    L: "Landscape",
    *,
    # Cleaning policies
    missing_policy: str = "error",
    nan_policy: str = "omit",
    # Uncertainty bootstrap (around F̂)
    B_uncertainty: int = 0,
    uncertainty_flavor: str = "iid",  # {"iid","wildcluster"}
    # Null bootstrap (noise-only landscapes)
    B_null: int = 0,
    preserve_residual_corr: bool = False,
    multipliers: str = "normal",
    ci_level: float = 0.95,
    rng: np.random.Generator | None = None,
    # What quantity to analyze per order
    as_fraction: bool = False,        # False -> V(S); True -> V(S)/sum_T V(T)
    as_dataframe: bool = True,
    # how to build uncertainty CIs
    ci_method_uncertainty: str = "percentile",  # {"percentile","studentized","bca"}
):
    """
    Compute variance by interaction order using WH coefficients, with
    uncertainty and null bootstrap statistics.

    The base quantity per order S is:
        V(S) = sum_{|s|=S} f_s^2

    If `as_fraction=False`, the analysis is done on V(S).
    If `as_fraction=True`, the analysis is done on the normalized spectrum:
        V_frac(S) = V(S) / sum_T V(T)

    For each order S >= 1, this function returns:
        - point estimate of V(S) or V_frac(S)
        - bootstrap uncertainty CI (method controlled by ci_method_uncertainty)
        - bootstrap null CI (percentile)
        - SNR vs null
        - a global variance test over orders S (P-value var, global)
        - a per-order null test (P-value order).

    Notes
    -----
    - When `as_fraction=False`, the "Variance (obs)" column is V(S).
    - When `as_fraction=True`, the "Variance (obs)" column is the fraction
      of variance V(S)/sum_T V(T).
    """

    import numpy as np
    import pandas as pd

    if rng is None:
        rng = np.random.default_rng()

    # -----------------------------
    # 1. Clean landscape and extract data
    # -----------------------------
    values, states, N = _clean_landscape_for_wh(
        L,
        missing_policy=missing_policy,
        nan_policy=nan_policy,
    )
    M, R = values.shape

    if R < 2 and (B_uncertainty > 0 or B_null > 0):
        raise ValueError("At least 2 replicates are required for bootstrap procedures.")

    # -----------------------------
    # 2. Point estimate: F̂(x)
    # -----------------------------
    F_hat = np.nanmean(values, axis=1)  # (M,)

    # -----------------------------
    # 3. Walsh–Hadamard transform of F̂
    # -----------------------------
    fs_hat, s_bits, orders, _meta = wh_transform(
        values=F_hat,
        states=states,
        N=N,
    )
    fs_hat = np.asarray(fs_hat)
    if fs_hat.ndim > 1 and fs_hat.shape[1] == 1:
        fs_hat = fs_hat[:, 0]

    # -----------------------------
    # 4. Helper: variance spectrum from a 1D fs
    # -----------------------------
    def _spectrum_from_fs(fs_1d: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute V(S) = sum_{|s|=S} f_s^2 for a single WH coefficient vector.

        Returns
        -------
        S_arr : (K_ord,) array of orders S >= 1
        V_arr : (K_ord,) array of V(S) for each S in S_arr
        """
        spec = variance_spectrum(
            fs_1d,
            orders=orders,
            aggregate="sum",
            exclude_order0=True,   # skip S=0 (global mean)
        )
        S_list = sorted(spec.keys())
        V_arr = np.array([spec[S][0] for S in S_list], dtype=float)  # each spec[S] has shape (1,)
        return np.asarray(S_list, dtype=int), V_arr

    # Point estimate of V(S)
    S_arr, V_obs = _spectrum_from_fs(fs_hat)  # S_arr: (K_ord,), V_obs: (K_ord,)
    K_ord = S_arr.size

    # Choose the target quantity Q(S)
    if as_fraction:
        total_var = V_obs.sum()
        if total_var > 0:
            Q_obs = V_obs / total_var
        else:
            Q_obs = np.full_like(V_obs, np.nan)
        quantity_label = "Fraction of variance V(S)/sum_T V(T)"
    else:
        Q_obs = V_obs.copy()
        quantity_label = "Variance V(S)"

    # Early exit if no bootstrap is requested
    if B_uncertainty <= 0 and B_null <= 0:
        if as_dataframe:
            return pd.DataFrame({
                "Order": S_arr,
                quantity_label: Q_obs,
            })
        return {
            "orders": S_arr,
            "quantity": Q_obs,
            "as_fraction": as_fraction,
        }

    # -----------------------------
    # 5. Prepare DataFrame for bootstrap engines
    # -----------------------------
    F_df = pd.DataFrame(
        np.hstack([states.astype(int), values]),
        columns=[f"s{j}" for j in range(N)] + [f"rep{r}" for r in range(R)],
    )

    alpha = 1.0 - ci_level
    lo_q = 100 * (alpha / 2.0)
    hi_q = 100 * (1.0 - alpha / 2.0)

    # Containers
    ci_low = ci_high = None
    null_ci_low = null_ci_high = None
    snr_null = None
    # Global stats (based on total quantity across orders)
    var_obs_global = var_null_median_global = p_value_var_global = np.nan
    # Per-order stats
    p_value_order = None
    var_obs_order = None
    var_null_median_order = None

    # -----------------------------
    # 6. Uncertainty bootstrap
    # -----------------------------
    Q_unc_b = None  # will store bootstrap realizations of Q(S)

    if B_uncertainty > 0:
        Fb_unc = bootstrap_uncertainty(
            F_df,
            B=B_uncertainty,
            multipliers=multipliers,
            flavor=uncertainty_flavor,
            rng=rng,
        )
        boots_mat = Fb_unc.values[:, N:]  # (M, B_uncertainty)

        Q_unc_b = np.empty((B_uncertainty, K_ord), dtype=float)

        for b in range(B_uncertainty):
            F_b = boots_mat[:, b]
            fs_b, _, _, _ = wh_transform(values=F_b, states=states, N=N)
            fs_b = np.asarray(fs_b)
            if fs_b.ndim > 1 and fs_b.shape[1] == 1:
                fs_b = fs_b[:, 0]
            S_b, V_b = _spectrum_from_fs(fs_b)

            # For safety, ensure same order set
            if not np.array_equal(S_b, S_arr):
                raise RuntimeError("Inconsistent orders S in uncertainty bootstrap.")

            if as_fraction:
                tot_b = V_b.sum()
                if tot_b > 0:
                    # normalize by observed total_var to keep statistics comparable
                    Q_b = V_b / total_var
                else:
                    Q_b = np.zeros_like(V_b)  # avoid NaNs
            else:
                Q_b = V_b

            Q_unc_b[b, :] = Q_b

        # ---- Build CIs for uncertainty according to ci_method_uncertainty ----
        if ci_method_uncertainty == "percentile":
            ci_low = np.nanpercentile(Q_unc_b, lo_q, axis=0)
            ci_high = np.nanpercentile(Q_unc_b, hi_q, axis=0)

        elif ci_method_uncertainty == "studentized":
            # Centered CIs around Q_obs (bootstrap-t style)
            T = Q_unc_b - Q_obs[None, :]      # shape (B_unc, K_ord)
            T_low = np.nanpercentile(T, lo_q, axis=0)
            T_high = np.nanpercentile(T, hi_q, axis=0)
            ci_low = Q_obs + T_low
            ci_high = Q_obs + T_high

        elif ci_method_uncertainty == "bca":
            # Bias-Corrected and Accelerated (BCa) intervals
            try:
                from scipy.stats import norm
            except ImportError as e:
                raise ImportError(
                    "ci_method_uncertainty='bca' requires scipy to be installed."
                ) from e

            # Jackknife over replicates to estimate acceleration a(S)
            # Q_jack has shape (R, K_ord): leave-one-replicate-out estimate of Q(S)
            Q_jack = np.empty((R, K_ord), dtype=float)
            for i in range(R):
                mask = np.ones(R, dtype=bool)
                mask[i] = False
                F_i = np.nanmean(values[:, mask], axis=1)  # leave-one-out mean
                fs_i, _, _, _ = wh_transform(values=F_i, states=states, N=N)
                fs_i = np.asarray(fs_i)
                if fs_i.ndim > 1 and fs_i.shape[1] == 1:
                    fs_i = fs_i[:, 0]
                S_i, V_i = _spectrum_from_fs(fs_i)
                if not np.array_equal(S_i, S_arr):
                    raise RuntimeError("Inconsistent orders S in jackknife.")
                if as_fraction:
                    tot_i = V_i.sum()
                    if tot_i > 0:
                        Q_i = V_i / total_var
                    else:
                        Q_i = np.zeros_like(V_i)
                else:
                    Q_i = V_i
                Q_jack[i, :] = Q_i

            # Compute BCa parameters and CIs per order
            ci_low = np.empty(K_ord, dtype=float)
            ci_high = np.empty(K_ord, dtype=float)

            z_alpha_low = norm.ppf(alpha / 2.0)
            z_alpha_high = norm.ppf(1.0 - alpha / 2.0)

            for k in range(K_ord):
                theta_hat = Q_obs[k]
                theta_boot = Q_unc_b[:, k]
                theta_jack = Q_jack[:, k]

                # Bias-correction term z0
                prop_less = np.mean(theta_boot < theta_hat)
                # Avoid 0 or 1 to keep finite z0
                eps = 1.0 / (2.0 * B_uncertainty)
                prop_less = np.clip(prop_less, eps, 1.0 - eps)
                z0 = norm.ppf(prop_less)

                # Acceleration a via jackknife
                theta_bar = np.mean(theta_jack)
                diffs = theta_bar - theta_jack
                num = np.sum(diffs ** 3)
                den = np.sum(diffs ** 2)
                if den > 0.0:
                    a = num / (6.0 * den ** 1.5)
                else:
                    a = 0.0

                # Adjusted quantiles
                z_low = z0 + (z0 + z_alpha_low) / (1.0 - a * (z0 + z_alpha_low))
                z_high = z0 + (z0 + z_alpha_high) / (1.0 - a * (z0 + z_alpha_high))

                p_low = norm.cdf(z_low)
                p_high = norm.cdf(z_high)

                # Convert to percentiles and clip to [0,100]
                q_low = 100.0 * np.clip(p_low, 0.0, 1.0)
                q_high = 100.0 * np.clip(p_high, 0.0, 1.0)

                ci_low[k] = np.nanpercentile(theta_boot, q_low)
                ci_high[k] = np.nanpercentile(theta_boot, q_high)

        else:
            raise ValueError(
                f"Unknown ci_method_uncertainty={ci_method_uncertainty!r}. "
                "Use 'percentile', 'studentized' or 'bca'."
            )

    # -----------------------------
    # 7. Null bootstrap (percentile CIs + tests)
    # -----------------------------
    if B_null > 0:
        Fb_null = bootstrap_null(
            F_df,
            B=B_null,
            multipliers=multipliers,
            preserve_residual_corr=preserve_residual_corr,
            rng=rng,
        )
        boots_mat = Fb_null.values[:, N:]  # (M, B_null)

        Q_null_b = np.empty((B_null, K_ord), dtype=float)

        for b in range(B_null):
            F_b = boots_mat[:, b]
            fs_b, _, _, _ = wh_transform(values=F_b, states=states, N=N)
            fs_b = np.asarray(fs_b)
            if fs_b.ndim > 1 and fs_b.shape[1] == 1:
                fs_b = fs_b[:, 0]
            S_b, V_b = _spectrum_from_fs(fs_b)

            if not np.array_equal(S_b, S_arr):
                raise RuntimeError("Inconsistent orders S in null bootstrap.")

            if as_fraction:
                tot_b = V_b.sum()
                if tot_b > 0:
                    # normalize by observed total_var to keep statistics comparable
                    Q_b = V_b / total_var
                else:
                    Q_b = np.zeros_like(V_b)
            else:
                Q_b = V_b

            Q_null_b[b, :] = Q_b

        # Null CIs per order (percentile)
        null_ci_low = np.nanpercentile(Q_null_b, lo_q, axis=0)
        null_ci_high = np.nanpercentile(Q_null_b, hi_q, axis=0)

        # SNR per order: |Q_obs| / std_null
        std_null = np.nanstd(Q_null_b, axis=0, ddof=1)
        snr_null = np.where(std_null > 0, Q_obs / std_null, np.nan)

        # Per-order "variance": just Q(S) itself
        var_obs_order = Q_obs.copy()
        var_null_median_order = np.nanmedian(Q_null_b, axis=0)

        # Per-order p-values
        # p_value_order[S] = P[ Q_null_b[:, S] >= Q_obs[S] ]
        p_value_order = np.mean(Q_null_b >= Q_obs[None, :], axis=0)

        # Optional: global test (based on total quantity across orders)
        total_obs = float(np.nansum(Q_obs))
        total_null = np.nansum(Q_null_b, axis=1)
        total_null = total_null[np.isfinite(total_null)]
        if total_null.size > 0:
            var_obs_global = total_obs
            var_null_median_global = float(np.nanmedian(total_null))
            p_value_var_global = float(np.mean(total_null >= total_obs))
        else:
            var_obs_global = np.nan
            var_null_median_global = np.nan
            p_value_var_global = np.nan

    else:
        # No null bootstrap -> no null-based stats
        Q_null_b = None
        null_ci_low = null_ci_high = None
        snr_null = None
        var_obs_order = Q_obs.copy()
        var_null_median_order = np.full(K_ord, np.nan)
        p_value_order = np.full(K_ord, np.nan)
        var_obs_global = np.nan
        var_null_median_global = np.nan
        p_value_var_global = np.nan

    # -----------------------------
    # 8. Final output
    # -----------------------------
    if as_dataframe:
        return pd.DataFrame({
            "Order": S_arr,
            quantity_label: Q_obs,
            "CI low": ci_low if ci_low is not None else [np.nan] * K_ord,
            "CI high": ci_high if ci_high is not None else [np.nan] * K_ord,
            "Null CI low": null_ci_low if null_ci_low is not None else [np.nan] * K_ord,
            "Null CI high": null_ci_high if null_ci_high is not None else [np.nan] * K_ord,
            "SNR (null)": snr_null if snr_null is not None else [np.nan] * K_ord,
            # Ahora por orden:
            "Variance (obs)": var_obs_order if var_obs_order is not None else [np.nan] * K_ord,
            "Variance (null median)": var_null_median_order if var_null_median_order is not None else [np.nan] * K_ord,
            # Test global (mismo valor repetido por orden, por compatibilidad)
            "P-value var": [p_value_var_global] * K_ord,
            # Test por orden:
            "P-value order": p_value_order if p_value_order is not None else [np.nan] * K_ord,
        })

    # Raw dict output
    return {
        "orders": S_arr,
        "quantity": Q_obs,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "null_ci_low": null_ci_low,
        "null_ci_high": null_ci_high,
        "snr_null": snr_null,
        "var_obs_order": var_obs_order,
        "var_null_median_order": var_null_median_order,
        "p_value_var_global": p_value_var_global,
        "p_value_order": p_value_order,
        "as_fraction": as_fraction,
    }





def plot_variance_and_amplitude(
    L: "Landscape",
    *,
    # Bootstrap options
    B_uncertainty: int = 300,
    B_null: int = 300,
    ci_level: float = 0.95,
    multipliers: str = "normal",
    uncertainty_flavor: str = "iid",
    preserve_residual_corr: bool = False,
    # Variance spectrum options
    as_fraction: bool = True,          # analyze fraction of variance by default
    # RNG
    rng: np.random.Generator | None = None,
    # Plot options
    figsize: tuple[float, float] = (12, 4.5),
    show: bool = True,
    # Extra options
    show_components: bool = False,     # plot combinatorial C(N,S) and dilution 4^{-S}
    return_data: bool = False,         # optionally return DataFrames and components
     # how to build uncertainty CIs
    ci_method_uncertainty: str = "bca",  # {"percentile","studentized","bca"}
):
    """
    High-level plotting helper for variance decomposition and epistasis amplitude.

    This function:
      1. Computes variance by interaction order using `variance_by_order_statistics`,
         either as absolute V(S) or as fraction V(S)/sum_T V(T).
      2. Computes the epistasis amplitude <E^2>_k using `epistasis_amplitude`.
      3. Creates a 1x2 matplotlib figure:
           - left: variance spectrum (with uncertainty + null CIs)
           - right: epistasis amplitude (with uncertainty + null CIs)
      4. Optionally overlays the combinatorial (C(N,S)) and geometric (4^{-S})
         contributions on the variance panel, normalized for visualization.
      5. Optionally returns the underlying data objects for further customization.

    Parameters
    ----------
    L : Landscape
        Input landscape (must provide .states and .values).
    B_uncertainty : int
        Number of bootstrap draws for uncertainty bands.
    B_null : int
        Number of bootstrap draws for null bands.
    ci_level : float
        Confidence level for both uncertainty and null CIs (e.g. 0.95).
    multipliers : {"rademacher","normal"}
        Multiplier distribution for wild bootstrap.
    uncertainty_flavor : {"iid","wildcluster"}
        Flavor of uncertainty bootstrap.
    preserve_residual_corr : bool
        If True, preserve biological correlations in the null bootstrap.
    as_fraction : bool
        If True, plot fractional variance V(S)/sum_T V(T).
        If False, plot absolute variance V(S).
    rng : np.random.Generator, optional
        Random number generator for reproducibility.
    figsize : (float, float)
        Figure size passed to matplotlib.
    show : bool
        If True, call plt.show() at the end.
    show_components : bool
        If True, overlay normalized combinatorial C(N,S) and dilution 4^{-S}
        on the variance-by-order panel (left subplot).
    return_data : bool
        If True, in addition to (fig, axes) also return a dict with:
            {
              "variance": df_var,
              "amplitude": df_amp,
              "components": {
                  "orders": orders_var,
                  "H": H,          # C(N,S)
                  "geom": geom     # 4^{-S}
              }  # only if show_components=True
            }

    Returns
    -------
    fig : matplotlib.figure.Figure
        The created figure.
    axes : np.ndarray of Axes
        The axes array with shape (2,).
    data : dict (optional)
        Only if return_data=True. See description above.
    """
    if rng is None:
        rng = np.random.default_rng()

    # ---------------------------------------
    # 1) Variance by order (left panel)
    # ---------------------------------------
    df_var = variance_by_order_statistics(
        L,
        missing_policy="error",
        nan_policy="omit",
        B_uncertainty=B_uncertainty,
        uncertainty_flavor=uncertainty_flavor,
        B_null=B_null,
        preserve_residual_corr=preserve_residual_corr,
        multipliers=multipliers,
        ci_level=ci_level,
        rng=rng,
        as_fraction=as_fraction,
        as_dataframe=True,
        ci_method_uncertainty=ci_method_uncertainty,
    )

    # Column 1 after "Order" contains the main quantity (variance or fraction)
    quantity_label = df_var.columns[1]

    orders_var = df_var["Order"].values
    Q_obs = df_var[quantity_label].values
    Q_ci_low = df_var["CI low"].values
    Q_ci_high = df_var["CI high"].values
    Q_null_low = df_var["Null CI low"].values
    Q_null_high = df_var["Null CI high"].values

    # ---------------------------------------
    # 2) Epistasis amplitude (right panel)
    # ---------------------------------------
    df_amp = epistasis_amplitude(
        L,
        missing_policy="error",
        nan_policy="omit",
        B_uncertainty=B_uncertainty,
        uncertainty_flavor=uncertainty_flavor,
        B_null=B_null,
        preserve_residual_corr=preserve_residual_corr,
        multipliers=multipliers,
        ci_level=ci_level,
        rng=rng,
        as_dataframe=True,
        normalized=as_fraction,
        ci_method_uncertainty=ci_method_uncertainty,
    )

    orders_amp = df_amp["Order"].values
    A_obs = df_amp["Epistasis amplitude <E^2>_k"].values
    A_ci_low = df_amp["CI low"].values
    A_ci_high = df_amp["CI high"].values
    A_null_low = df_amp["Null CI low"].values
    A_null_high = df_amp["Null CI high"].values
    A_null_mean = df_amp["Null mean"].values

    # ---------------------------------------
    # 3) Create figure and subplots
    # ---------------------------------------
    if show_components:
        fig, axes = plt.subplots(1, 4, figsize=figsize, sharex=False)
        ax_var, ax_geo, ax_dil, ax_amp = axes
    else:
        fig, axes = plt.subplots(1, 2, figsize=figsize, sharex=False)
        ax_var, ax_amp = axes
    

    # ---------- Left: variance by order ----------
    ax = ax_var

    # Null band
    ax.fill_between(
        orders_var,
        Q_null_low,
        Q_null_high,
        alpha=0.25,
        label="Null CI",
    )

    # Observed + uncertainty error bars
    yerr_low = np.clip(Q_obs - Q_ci_low, a_min=0, a_max=None)
    yerr_high = np.clip(Q_ci_high - Q_obs, a_min=0, a_max=None)

    ax.errorbar(
        orders_var,
        Q_obs,
        yerr=[yerr_low, yerr_high],
        fmt="o",
        capsize=4,
        label="Observed",
    )

    ax.set_xlabel(r"$\text{Interaction order } \mathcal{S}$",fontsize=14)
    if(as_fraction):
        ax.set_ylabel(r"$\frac{V(\mathcal{S})}{V_{\text{tot}}}$", rotation=0, labelpad=20,fontsize=22)
        #ax.set_title("Fraction of variance by order")
    else:
        ax.set_ylabel(r"$\text{Variance } V(\mathcal{S})$")
        #ax.set_title("Variance by order")

    # Optionally plot combinatorial C(N,S) and geometric dilution 4^{-S}
    components = None
    if show_components:
        # Number of loci
        N = L.states.shape[1]
        S = orders_var.astype(int)

        # Combinatorial factor and geometric (dilution) factor
        H = np.array([comb(N, int(s)) for s in S], dtype=float)   # C(N,S)
        geom = 4.0 ** (-S)                                        # 4^{-S}

        # Normalize both to [0, 1] for visualization on the same axis
        H_norm = H / H.max() if np.max(H) > 0 else H
        geom_norm = geom / geom.max() if np.max(geom) > 0 else geom

        ax_geo.plot(
            orders_var,
            H_norm,
            linestyle="-",
            marker="o",
        )
        ax_geo.set_xlabel(r"$\text{Interaction order }  \mathcal{S}$",fontsize=14)
        ax_geo.set_ylabel(r"$\text{Combinatorial } C(N,\mathcal{S})$",fontsize=14)

        ax_dil.plot(
            orders_var,
            geom_norm,
            linestyle="-",
            marker="o",
        )
        ax_dil.set_xlabel(r"$\text{Interaction order }  \mathcal{S}$",fontsize=14)
        ax_dil.set_ylabel(r"$\text{Dilution } 4^{-\mathcal{S}}$",fontsize=14)

        components = {
            "orders": S,
            "H": H,
            "geom": geom,
        }

    #ax.grid(alpha=0.3)
    ax.legend(frameon=False)

    # ---------- Right: epistasis amplitude ----------
    ax = ax_amp

    # Null band
    ax.fill_between(
        orders_amp,
        A_null_low,
        A_null_high,
        alpha=0.25,
        label="Null CI",
    )

    ax.plot(
        orders_amp,
        A_null_mean,
        linestyle="--",
        color="C0",
        label="Null mean",
    )

    # Observed + uncertainty error bars
    yerr_low = np.clip(A_obs - A_ci_low, a_min=0, a_max=None)
    yerr_high= np.clip(A_ci_high - A_obs, a_min=0, a_max=None)
    ax.errorbar(
        orders_amp,
        A_obs,
        yerr=[yerr_low, yerr_high],
        fmt="o",
        capsize=4,
        #label="Observed ± uncertainty CI",
    )

    ax.set_xlabel(r"$\text{Interaction order }  \mathcal{S}$",fontsize=14)
    if(as_fraction):
        ax.set_ylabel(r"$\frac{⟨\mathcal{E}^2⟩_\mathcal{S}}{V_{\text{tot}}}$",labelpad=30,fontsize=22,rotation=0)
    else:
        ax.set_ylabel(r"$\text{Epistasis amplitude } ⟨\mathcal{E}^2⟩_\mathcal{S}$",fontsize=14)
    #ax.set_title("Epistasis amplitude by order")
    ax.set_yscale("log")
    #ax.grid(alpha=0.3)
    ax_amp.legend(frameon=False,fontsize=14)

    # --- Panel labels (A, B, C, D) ---
    if show_components:
        panel_labels = ["A", "B", "C", "D"]  # var, combinatorial, dilution, amplitude
    else:
        panel_labels = ["A", "B"]            # var, amplitude

    for ax, label in zip(axes, panel_labels):
        ax.text(
            -0.3, 1.05, label,
            transform=ax.transAxes,
            fontsize=16,
            fontweight="bold",
            va="top",
            ha="left",
        )

    fig.tight_layout()

    if show:
        plt.show()
    else:
        plt.close(fig)   

    if return_data:
        data = {
            "variance": df_var,
            "amplitude": df_amp,
        }
        if components is not None:
            data["components"] = components
        return fig, axes, data

    return fig, axes


def plot_walsh_volcano(
    wh_df: pd.DataFrame,
    *,
    orders: Optional[Sequence[int]] = None,
    effect_col: str = "Walsh (mean)",
    p_col: str = "p-null",
    clip: float = 1e-12,
    return_data: bool = False,
    mode: str = "by-order",          # {"single","by-order"}
    alpha: float | None = 0.05,
    colormap: str = "default",       # {"default","custom", other->custom}
    ax: Optional["plt.Axes"] = None, # or array-like Axes if mode="by-order"
    fig_kwargs: Optional[dict] = None,
):
    """
    Volcano plot for Walsh–Hadamard coefficients (modes).

    Expects a DataFrame produced by walsh_analysis(...).to_dataframe(), containing:
      - "Order"
      - effect_col (default: "Walsh (mean)")
      - p_col      (default: "p-null")  [requires B_null>0 in walsh_analysis]
    """
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    from matplotlib.transforms import blended_transform_factory

    if mode not in {"single", "by-order"}:
        raise ValueError("mode must be 'single' or 'by-order'.")

    if "Order" not in wh_df.columns:
        raise ValueError("wh_df must contain an 'Order' column.")
    if effect_col not in wh_df.columns:
        raise ValueError(f"Column '{effect_col}' not found in wh_df.")
    if p_col not in wh_df.columns:
        raise ValueError(
            f"Column '{p_col}' not found in wh_df. "
            "If you want p-null, run walsh_analysis with B_null>0."
        )

    fig_kwargs = {} if fig_kwargs is None else dict(fig_kwargs)
    df = wh_df.copy()

    # Optional filter by interaction orders
    if orders is not None:
        orders = list(orders)
        df = df[df["Order"].isin(orders)]

    # Drop missing values
    df = df.dropna(subset=[effect_col, p_col]).copy()
    if df.empty:
        raise ValueError("No rows left after filtering and dropping NaNs.")

    # Compute -log10(p)
    p_raw = df[p_col].astype(float)
    p_clipped = p_raw.clip(lower=clip)
    df["neg_log10_p"] = -np.log10(p_clipped)

    y_label = r"$-\log_{10}(p)$"
    thr_label = r"$p \leq " + (f"{alpha:g}" if alpha is not None else "") + r"$"

    orders_unique = np.sort(df["Order"].unique())
    n_orders = len(orders_unique)
    if n_orders == 0:
        raise ValueError("No orders available after filtering.")

    # Colormaps (same spirit as your epistasis volcano)
    BASE_COLORS = ["#ff5400", "#ff9e00", "#00b4d8", "#0077b6", "#03045e"]
    CUSTOM_CMAP = LinearSegmentedColormap.from_list("walsh_volcano", BASE_COLORS, N=256)

    if colormap == "default":
        BASE_CMAP = plt.get_cmap("tab10")
    else:
        BASE_CMAP = CUSTOM_CMAP

    colors = [BASE_CMAP(t) for t in np.linspace(0, 1, n_orders)]

    # -------------------------
    # MODE: single panel
    # -------------------------
    if mode == "single":
        if ax is None:
            fig, ax = plt.subplots(figsize=(6, 5), **fig_kwargs)
        else:
            fig = ax.figure

        for j, o in enumerate(orders_unique):
            sub = df[df["Order"] == o]

            sig_mask = (sub[p_col] <= alpha) if alpha is not None else np.ones(len(sub), dtype=bool)

            # Non-significant in gray
            ax.scatter(
                sub.loc[~sig_mask, effect_col],
                sub.loc[~sig_mask, "neg_log10_p"],
                s=18,
                alpha=0.35,
                color="0.7",
            )
            # Significant colored by order
            ax.scatter(
                sub.loc[sig_mask, effect_col],
                sub.loc[sig_mask, "neg_log10_p"],
                s=20,
                alpha=0.75,
                color=colors[j],
                label=f"{o}",
            )

        if alpha is not None:
            y_thr = -np.log10(alpha)
            ax.axhline(y_thr, linestyle="--", linewidth=1.6, alpha=0.8, color="black")

            trans_thr = blended_transform_factory(ax.transAxes, ax.transData)
            ax.text(
                0.99, y_thr, thr_label,
                ha="right", va="bottom", fontsize=16, transform=trans_thr,
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.6,frameon=False),
            )

            # Clip line if relevant
            if clip < alpha:
                y_clip = -np.log10(clip)
                ax.axhline(y_clip, color="0.7", linestyle=":", linewidth=2, alpha=0.8)

                trans_clip = blended_transform_factory(ax.transAxes, ax.transData)
                ax.text(
                    0.99, y_clip, "clip",
                    ha="right", va="bottom", fontsize=16, transform=trans_clip,
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", frameon=False, alpha=0.6),
                    color="0.5",
                )

        ax.set_xlabel(r"$\hat f_{\mathbf{s}}$",fontsize=14)
        ax.set_ylabel(y_label)
        ax.legend(title="Order", frameon=False, loc="center left", bbox_to_anchor=(1.02, 0.5))
        plt.tight_layout()

    # -------------------------
    # MODE: by-order subplots
    # -------------------------
    else:
        if ax is None:
            if n_orders <= 4:
                nrows, ncols = 1, n_orders
            else:
                nrows, ncols = 2, int(np.ceil(n_orders / 2))

            fig, axes = plt.subplots(
                nrows, ncols, sharey=True,
                figsize=(5 * ncols, 4 * nrows),
                **fig_kwargs,
            )
        else:
            axes = np.atleast_1d(ax)
            if axes.ndim > 1:
                axes = axes.ravel()
            if len(axes) < n_orders:
                raise ValueError(f"Provided ax has {len(axes)} axes but need {n_orders}.")
            fig = axes[0].figure

        axes = np.atleast_1d(axes).ravel()

        for j, o in enumerate(orders_unique):
            sub = df[df["Order"] == o]
            ax_j = axes[j]
            ax_j.ticklabel_format(axis="x", style="sci", scilimits=(-2, 2))
            ax_j.xaxis.get_offset_text().set_fontsize(14)

            sig_mask = (sub[p_col] <= alpha) if alpha is not None else np.ones(len(sub), dtype=bool)

            ax_j.scatter(
                sub.loc[~sig_mask, effect_col],
                sub.loc[~sig_mask, "neg_log10_p"],
                s=18, alpha=0.35, color="0.7",
            )
            ax_j.scatter(
                sub.loc[sig_mask, effect_col],
                sub.loc[sig_mask, "neg_log10_p"],
                s=20, alpha=0.75, color=colors[j],
            )

            if alpha is not None:
                y_thr = -np.log10(alpha)
                ax_j.axhline(y_thr, linestyle="--", linewidth=2.0, alpha=0.8, color="black")

                trans_thr = blended_transform_factory(ax_j.transAxes, ax_j.transData)
                ax_j.text(
                    0.99, y_thr, thr_label,
                    ha="right", va="bottom", fontsize=16, transform=trans_thr,
                    bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.6),
                )

                if clip < alpha:
                    y_clip = -np.log10(clip)
                    ax_j.axhline(y_clip, color="0.7", linestyle=":", linewidth=2, alpha=0.8)

                # % significant
                n_total = len(sub)
                if n_total > 0:
                    n_sig = int(np.sum(sub[p_col] <= alpha))
                    frac_sig = 100.0 * n_sig / n_total
                    ax_j.text(
                        0.05, 0.93,
                        f"{frac_sig:.1f}% sig.",
                        transform=ax_j.transAxes,
                        va="top", ha="left", fontsize=16,
                    )

            ax_j.set_title(f"Order {int(o)}", fontsize=16)

            # axis labels (bottom row only if we created axes)
            if j >= len(orders_unique)/2:
                ax_j.set_xlabel(r"$\hat f_{\mathbf{s}}$",fontsize=16)
            else:
                ax_j.set_xlabel("")

            if j % (axes.size if ax is not None else 10) == 0:
                ax_j.set_ylabel(y_label)

        # Turn off unused axes
        for k in range(n_orders, len(axes)):
            axes[k].axis("off")

        plt.tight_layout()

    if return_data:
        keep = [c for c in df.columns if c in {"Order", effect_col, p_col, "neg_log10_p", "Mode bits", "Mode (features)"}]
        return df[keep].copy()
    return None
