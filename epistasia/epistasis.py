##########################################
#              EPISTASIS MODULE          #
##########################################

"""
Description:
This module provides generic tools to compute epistatic coefficients
of any order (k ≥ 1) from a binary functional landscape.

The main output object is the `Epistasis` dataclass, which stores both
replicate-level results and summary statistics (mean and uncertainties).

Compatible with: `core.Landscape`
                    `stats_noise` (bootstrap uncertainty),
                    `walshhadamard` (spectral projection)
"""

##########################################

# IMPORTS 
from __future__ import annotations
from .core import Landscape
from dataclasses import dataclass
import numpy as np
from typing import Optional, Tuple, Iterable, Sequence, Dict, Any, Union,  Literal
import pandas as pd

from matplotlib.colors import LinearSegmentedColormap

from itertools import product, combinations
from collections import namedtuple

from .stats_noise.noise_models import bootstrap_uncertainty, bootstrap_null

MissingPolicy = Literal["error", "drop"]

##################################################################
#                      EPISTASIS DATACLASS                       #
##################################################################

from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any
import numpy as np

@dataclass
class Epistasis:
    """
    Container for epistatic coefficients of arbitrary order.

    Attributes
    ----------
    values : np.ndarray
        Array of shape (M', R) with replicate-level epistatic values.
        Each row corresponds to a background configuration; each column to a replicate.
    mean : np.ndarray
        Array of shape (M',) with the mean epistatic effect across replicates
        for each background.
    std_experimental : np.ndarray
        Array of shape (M',) containing the standard deviation across replicates
        (experimental variability) for each background.
    std_biological : np.ndarray
        Array of shape (M',) with the standard deviation of the mean estimator,
        i.e. std_experimental / sqrt(R). This reflects the uncertainty of the
        inferred biological effect for each background.
    order : int
        Epistasis order (k), i.e. the size of the interacting locus set.
    indices : Tuple[int, ...]
        Tuple of locus indices involved in the interaction (e.g. (0, 2, 4)).
    backgrounds : np.ndarray
        Binary matrix of shape (M', N - k) representing the backgrounds over
        which the epistatic coefficients are evaluated. Each row is a background
        configuration for the loci not in `indices`.

    ci_low : Optional[np.ndarray]
        Array of shape (M',) with the lower confidence bound for each background,
        obtained from the uncertainty bootstrap around the estimated landscape F̂.
    ci_high : Optional[np.ndarray]
        Array of shape (M',) with the upper confidence bound for each background,
        obtained from the uncertainty bootstrap around F̂.

    null_ci_low : Optional[np.ndarray]
        Array of shape (M',) with the lower bound of the null confidence interval
        for each background, obtained from the null bootstrap (observed null
        landscapes). It characterizes the range of epistatic values expected
        purely from experimental noise.
    null_ci_high : Optional[np.ndarray]
        Array of shape (M',) with the upper bound of the null confidence interval
        for each background, obtained from the null bootstrap.

    p_value_var : Optional[float]
        One-sided global p-value for excess variance across backgrounds under
        the null model. It quantifies how likely it is to observe a variance
        of the epistatic effect across backgrounds as large as (or larger than)
        the observed one, if only noise were present.
    var_obs : Optional[float]
        Observed variance of the background-averaged epistatic effect (the variance
        of `mean` across backgrounds).
    var_null_median : Optional[float]
        Median of the null variance distribution across backgrounds, obtained
        from the null bootstrap (one scalar summarizing the typical null variance).
    effect_size_var : Optional[float]
        Difference var_obs - var_null_median, providing a simple effect-size
        measure for the excess variance signal.

    p_values_per_background : Optional[np.ndarray]
        Optional array of shape (M',) with per-background raw p-values, if additional
        background-wise tests are performed (e.g. testing each background against
        its null distribution). Not necessarily populated in all workflows.

    snr_null : Optional[np.ndarray]
        Optional array of shape (M',) with the signal-to-noise ratio for each
        background under the null model. Typically defined as
        |mean| / sigma_null, where sigma_null is estimated from the null
        bootstrap. Values much larger than 1 indicate epistatic effects that
        clearly stand out from the null noise level.

    var_null_per_background : Optional[np.ndarray]
        Optional array of shape (M',) with the null variance for each background,
        estimated from the null bootstrap (variance across bootstrap replicates
        of the epistatic coefficient for that background).

    sign_prob_pos : Optional[np.ndarray]
        Optional array of shape (M',) with the bootstrap-based probability
        that the epistatic effect is positive in each background,
        P(ε > 0 | data), computed from the uncertainty bootstrap.
    sign_prob_neg : Optional[np.ndarray]
        Optional array of shape (M',) with the bootstrap-based probability
        that the epistatic effect is negative in each background,
        P(ε < 0 | data).

    meta : Optional[dict]
        Optional metadata dictionary containing information about the
        computations: bootstrap sizes, flavors, confidence level, seeds,
        flags used in the construction, etc.

    indices_names : Optional[Tuple[str, ...]]
        Optional tuple of biological names for the loci in `indices`
        (e.g. gene names). If provided, indices_names[j] corresponds to
        indices[j].

    background_indices : Optional[Tuple[int, ...]]
        Optional tuple of locus indices that define the background coordinates
        (complement of `indices`). This is useful when reconstructing which
        column of `backgrounds` corresponds to which locus.
    background_names : Optional[Tuple[str, ...]]
        Optional tuple of biological names for the loci in `background_indices`.
        If provided, it offers a human-readable description of the background
        dimensions in `backgrounds`.
    """

    values: np.ndarray
    mean: np.ndarray
    std_experimental: np.ndarray
    std_biological: np.ndarray
    order: int
    indices: Tuple[int, ...]
    backgrounds: np.ndarray

    # Optional uncertainty bands (from bootstrap around F̂)
    ci_low: Optional[np.ndarray] = None
    ci_high: Optional[np.ndarray] = None

    # Null CI per background (from null bootstrap on Δ)
    null_ci_low: Optional[np.ndarray] = None
    null_ci_high: Optional[np.ndarray] = None

    # Null-test summary (optional)
    p_value_var: Optional[float] = None
    var_obs: Optional[float] = None
    var_null_median: Optional[float] = None
    effect_size_var: Optional[float] = None

    # Per-background p-values
    p_values_per_background: Optional[np.ndarray] = None

    # Signal to noise ratios (optional)
    snr_null: Optional[np.ndarray] = None

    # Null variance per background
    var_null_per_background: Optional[np.ndarray] = None

    # Optional sign probabilities (from uncertainty bootstrap)
    sign_prob_pos: Optional[np.ndarray] = None  # P(Δ > 0 | data)
    sign_prob_neg: Optional[np.ndarray] = None  # P(Δ < 0 | data)

    meta: Optional[Dict[str, Any]] = None
    indices_names: Optional[Tuple[str, ...]] = None

    background_indices: Optional[Tuple[int, ...]] = None   # indices of loci in backgrounds
    background_names: Optional[Tuple[str, ...]] = None      # biological names of those loci

    # ---------- Light-weight integrators (no heavy computation here) ----------

    def attach_uncertainty(
        self,
        ci_low: np.ndarray,
        ci_high: np.ndarray,
        *,
        ci_level: float,
        flavor: str,              # "iid" or "wildcluster"
        multipliers: str,         # "rademacher" or "normal"
        B_uncertainty: int,
        seed: Optional[int] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        # TODO: validate shapes; assign self.ci_low/self.ci_high; update self.meta
        return

    def attach_null_test(
        self,
        *,
        var_obs: float,
        var_null_median: float,
        p_value_var: float,
        effect_size_var: float,
        preserve_residual_corr: bool,
        multipliers: str,
        B_null: int,
        seed: Optional[int] = None,
        p_values_per_background: Optional[np.ndarray] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        # TODO: validate shapes; assign fields; update self.meta
        return

    def summary(self) -> np.ndarray:
        ci_low = self.ci_low if self.ci_low is not None else np.full_like(self.mean, np.nan)
        ci_high = self.ci_high if self.ci_high is not None else np.full_like(self.mean, np.nan)
        return np.vstack([
            self.mean,
            self.std_experimental,
            self.std_biological,
            ci_low,
            ci_high
        ]).T

    def to_dataframe(self):
        import pandas as pd
        import numpy as np

        bg_bits = [''.join(map(str, row.tolist())) for row in self.backgrounds]
        n_rows = len(bg_bits)

        if self.indices_names is not None:
            loci_names = [tuple(self.indices_names)] * n_rows
        else:
            loci_names = [None] * n_rows

        if getattr(self, "background_indices", None) is not None:
            bg_loci = [tuple(self.background_indices)] * n_rows
        else:
            bg_loci = [None] * n_rows

        if getattr(self, "background_names", None) is not None:
            bg_loci_names = [tuple(self.background_names)] * n_rows
        else:
            bg_loci_names = [None] * n_rows

        if getattr(self, "background_indices", None) is not None:
            bg_active = [
                tuple(idx for bit, idx in zip(row, self.background_indices) if bit == 1)
                for row in self.backgrounds
            ]
        else:
            bg_active = [None] * n_rows

        if getattr(self, "background_names", None) is not None:
            bg_active_names = [
                tuple(name for bit, name in zip(row, self.background_names) if bit == 1)
                for row in self.backgrounds
            ]
        else:
            bg_active_names = [None] * n_rows

        df = pd.DataFrame({
            "Background": bg_bits,
            "Background loci": bg_loci,
            "Background loci names": bg_loci_names,
            "Background active loci": bg_active,
            "Background active names": bg_active_names,
            "Order": self.order,
            "Loci involved": [self.indices] * n_rows,
            "Loci names": loci_names,
            "Epistasis (mean)": self.mean,
            "Experimental SD": self.std_experimental,
            "Biological SD": self.std_biological,
            "Uncertainty CI (low)": (
                self.ci_low if self.ci_low is not None
                else np.full_like(self.mean, np.nan)
            ),
            "Uncertainty CI (high)": (
                self.ci_high if self.ci_high is not None
                else np.full_like(self.mean, np.nan)
            ),
        })

        if getattr(self, "sign_prob_pos", None) is not None:
            df["Prob(Effect > 0)"] = self.sign_prob_pos
            df["Prob(Effect < 0)"] = self.sign_prob_neg

        if getattr(self, "null_ci_low", None) is not None:
            df["Null CI (low)"] = self.null_ci_low
            df["Null CI (high)"] = self.null_ci_high

        if getattr(self, "snr_null", None) is not None:
            df["Signal-to-Null-Noise (SNR)"] = self.snr_null

        if getattr(self, "var_null_per_background", None) is not None:
            df["Null variance (per background)"] = self.var_null_per_background

        if getattr(self, "var_obs", None) is not None:
            df["Variance (observed)"] = self.var_obs
            df["Variance (null median)"] = self.var_null_median
            df["p-var"] = self.p_value_var

        if getattr(self, "p_values_per_background", None) is not None:
            df["p-null"] = self.p_values_per_background

        return df



##############################################################################
#   Internal helpers
##############################################################################

def _bin_to_int(bits: Iterable[int]) -> int:
    """Convert a {0,1} vector into a non-negative integer (lex/MSB-first).

    Parameters
    ----------
    bits : iterable of ints {0,1}
        Binary vector x with x[0] as the most-significant bit.

    Returns
    -------
    int
        Integer index of the configuration.

    Notes
    -----
    - The mapping is idx(x) = Σ_{m=0}^{N-1} x[m] * 2^{N-1-m}.
    - This function does not assume a fixed N; it infers length from `bits`.
    """
    out = 0
    for b in bits:
        bi = int(b)
        if bi not in (0, 1):
            raise ValueError("_bin_to_int expects bits in {0,1}.")
        out = (out << 1) | bi
    return out


def _flip_indices(x: np.ndarray, flip_idx: Sequence[int], vals: Sequence[int]) -> np.ndarray:
    """Return a copy of `x` with positions `flip_idx` set to `vals`.

    Parameters
    ----------
    x : (N,) array-like of {0,1}
        Background configuration to copy and modify.
    flip_idx : sequence of int
        Indices to overwrite.
    vals : sequence of {0,1}
        Values to assign at `flip_idx`. Must have the same length as `flip_idx`.

    Returns
    -------
    out : (N,) ndarray of int
        Modified copy of `x` with requested positions set.
    """
    x = np.asarray(x, dtype=int)
    out = x.copy()
    flip_idx = np.asarray(flip_idx, dtype=int)
    vals = np.asarray(vals, dtype=int)
    if flip_idx.shape[0] != vals.shape[0]:
        raise ValueError("flip_idx and vals must have the same length.")
    if np.any((vals != 0) & (vals != 1)):
        raise ValueError("vals must be in {0,1}.")
    out[flip_idx] = vals
    return out


def _states_for_backgrounds(N: int, fixed_idx: Sequence[int]) -> np.ndarray:
    """Enumerate all backgrounds where `fixed_idx` are left free to vary later.

    This returns the 2^{N-k} configurations of the *other* N-k loci, laid out in
    lexicographic order (MSB-first). The positions in `fixed_idx` are placeholders
    (initialized to 0) that the caller is expected to set to 0/1 when evaluating
    alternating sums (e.g., epistatic quantities).

    Parameters
    ----------
    N : int
        Number of loci.
    fixed_idx : sequence of int
        Indices that will be toggled later (size k). These are *not* enumerated here.

    Returns
    -------
    X : (2^{N-k}, N) ndarray of int in {0,1}
        Matrix whose rows are the backgrounds. Entries at `fixed_idx` are zero
        placeholders; entries elsewhere enumerate all binary combinations.
    """
    if N <= 0:
        raise ValueError("N must be positive.")
    fixed = np.array(sorted(set(int(i) for i in fixed_idx)), dtype=int)
    if np.any((fixed < 0) | (fixed >= N)):
        raise ValueError("fixed_idx entries must be in [0, N-1].")
    k = fixed.shape[0]
    free = [i for i in range(N) if i not in set(fixed.tolist())]
    M = 1 << (N - k)
    X = np.zeros((M, N), dtype=int)
    # Fill the free positions in lexicographic order on the (N-k)-cube
    for m, bits in enumerate(product([0, 1], repeat=N - k)):
        X[m, free] = bits
    return X


#######################################################################
#                            SYMMBOLIC                                #
#######################################################################

def epistasis_symbolic(N: int, idx: Sequence[int]) -> str:
    """Return a human-readable symbolic expression of ε_{i1..ik}(x).
    
    
    The expression follows the biological convention (present − absent), i.e.,
    signs given by (-1)^{k - |a|} with a ∈ {0,1}^k, and lists terms in
    lexicographic order of a.
    
    
    Example (k=2, idx=(i,j)):
    "ε_{i,j}(x) = +F(x^{ij}_{11}) - F(x^{ij}_{10}) - F(x^{ij}_{01}) + F(x^{ij}_{00})"
    """
    
    idx = tuple(sorted(int(i) for i in idx))
    k = len(idx)
    if k == 0:
        raise ValueError("idx must contain at least one locus.")
    
    # Build header
    header = "ε_{" + ",".join(str(i) for i in idx) + "}(x) = "
    terms = []
    for a in product([0, 1], repeat=k):
        sign = "+" if ((k + sum(a)) % 2 == 0) else "-"
        sub = "".join(str(i) for i in idx)
        mask = "".join(str(ai) for ai in a)
        terms.append(f" {sign}F(x^{{{sub}}}_{{{mask}}})")
        
    # Order from 11..1 down to 00..0 is often visually clearer; reverse if desired
    # Here we keep standard lex order: 00..0, 00..1, ..., 11..1
    # To match the conventional display (11..1 first), reverse the list:
    terms = list(reversed(terms)) if k >= 1 else terms
    # Fix sign spacing for the very first term
    expr = header + terms[0].replace(" +", "+").replace(" -", "-")
    for t in terms[1:]:
        expr += t
    return expr

def _pack_bg_key(bg01: np.ndarray) -> np.ndarray:
    """
    Pack a (M, N-1) {0,1} array into fixed-width byte-string keys (hashable).
    Returns shape (M,) with dtype like '|S{k}'.
    """
    bg01 = np.asarray(bg01, dtype=np.uint8)
    packed = np.packbits(bg01, axis=1, bitorder="big")  # (M, nbytes)
    packed = np.ascontiguousarray(packed)
    nbytes = packed.shape[1]
    # View each row as a fixed-width bytes string => hashable scalar keys
    return packed.view(f"|S{nbytes}").ravel()



def _build_unique_map(keys: np.ndarray, abs_idx: np.ndarray, *, duplicate_policy: str) -> dict:
    """
    Build a map key -> absolute row index in the original states array.
    duplicate_policy:
        - "error": raise if a key appears more than once
        - "first": keep the first occurrence (smallest absolute index)
    """
    m = {}
    for k, idx in zip(keys, abs_idx):
        if k in m:
            if duplicate_policy == "error":
                raise ValueError(
                    "Duplicate background key detected while pairing FEEs. "
                    "This usually means duplicated genotypes in L.states (replication-in-rows "
                    "or repeated measurements). Collapse upstream or set duplicate_policy='first'."
                )
            elif duplicate_policy == "first":
                # keep deterministic choice: smallest abs index
                if idx < m[k]:
                    m[k] = idx
            else:
                raise ValueError("duplicate_policy must be 'error' or 'first'.")
        else:
            m[k] = int(idx)
    return m


def _pair_focal_backgrounds(
    states01: np.ndarray,
    i: int,
    *,
    missing_policy: MissingPolicy = "drop",
    duplicate_policy: str = "error",   # NEW
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Pair backgrounds for locus i: find rows with xi=0 and xi=1 for the same background.

    Returns:
      bg_bits  : (K, N-1)
      abs_idx0 : (K,) indices into original states01 with xi=0
      abs_idx1 : (K,) indices into original states01 with xi=1
      bg_key   : (K,) packed keys (np.void)
    """
    states01 = np.asarray(states01, dtype=np.uint8)
    if states01.ndim != 2:
        raise ValueError("states01 must be 2D (M,N).")
    M, N = states01.shape
    if not (0 <= i < N):
        raise ValueError(f"i must be in [0, {N-1}].")

    xi = states01[:, i]
    mask0 = (xi == 0)
    mask1 = (xi == 1)

    abs0_all = np.flatnonzero(mask0).astype(int)
    abs1_all = np.flatnonzero(mask1).astype(int)

    bg0 = np.delete(states01[mask0], i, axis=1)  # (M0, N-1)
    bg1 = np.delete(states01[mask1], i, axis=1)  # (M1, N-1)

    key0 = _pack_bg_key(bg0)
    key1 = _pack_bg_key(bg1)

    # Build explicit key -> absolute-row maps (robust)
    map0 = _build_unique_map(key0, abs0_all, duplicate_policy=duplicate_policy)
    map1 = _build_unique_map(key1, abs1_all, duplicate_policy=duplicate_policy)

    common_keys = sorted(set(map0.keys()) & set(map1.keys()))
    if len(common_keys) == 0:
        raise ValueError(f"No paired backgrounds found for locus i={i}.")

    if missing_policy == "error":
        if (len(common_keys) != len(map0)) or (len(common_keys) != len(map1)):
            raise ValueError(
                f"Missing paired backgrounds for locus i={i}. "
                f"Found {len(map0)} unique backgrounds with xi=0, {len(map1)} with xi=1, "
                f"but only {len(common_keys)} are paired."
            )
    elif missing_policy != "drop":
        raise ValueError("missing_policy must be 'drop' or 'error'.")

    abs_idx0 = np.array([map0[k] for k in common_keys], dtype=int)
    abs_idx1 = np.array([map1[k] for k in common_keys], dtype=int)

    # Recover bg bits from the paired xi=0 rows (guaranteed same bg as xi=1 by key)
    bg_bits = np.delete(states01[abs_idx0], i, axis=1).astype(np.uint8, copy=False)
    bg_key = np.array(common_keys, dtype=key0.dtype)

    return bg_bits, abs_idx0, abs_idx1, bg_key

################################################################
#                       FOCAL EFFECT (k=1)                     #
################################################################

def focal_effect(
    L: "Landscape",
    i: int,
    *,
    missing_policy: str = "error",
    nan_policy: str = "omit",
    B_uncertainty: int = 0,
    uncertainty_flavor: str = "iid",
    B_null: int = 0,
    preserve_residual_corr: bool = False,
    multipliers: str = "rademacher",
    ci_level: float = 0.95,
    rng: np.random.Generator | None = None,
    as_dataframe: bool = True,
) -> "Epistasis":

    # ----- Input validation -----
    if not hasattr(L, "states") or not hasattr(L, "values"):
        raise TypeError("L must provide .states and .values.")
    if not (0 <= i < L.N):
        raise ValueError(f"Invalid focal index: {i}")
    if missing_policy not in {"error", "drop"}:
        raise ValueError("missing_policy must be 'error' or 'drop'")
    if nan_policy not in {"omit", "propagate"}:
        raise ValueError("nan_policy must be 'omit' or 'propagate'")
    if rng is None:
        rng = np.random.default_rng()

    states = L.states.astype(int, copy=False)
    values = L.values
    M, N = states.shape
    R = values.shape[1]

    if R < 2 and (B_uncertainty > 0 or B_null > 0):
        raise ValueError("At least 2 replicates required for bootstrap.")

    # ----- Pair backgrounds (xi=1 vs xi=0) -----
    backgrounds, abs_idx0, abs_idx1, bg_key = _pair_focal_backgrounds(
        states, i, missing_policy=missing_policy
    )
    Mprime = backgrounds.shape[0]
    
    # ----- Extract replicate matrices -----
    F1 = values[abs_idx1]
    F0 = values[abs_idx0]


    # ----- Compute delta = F1 - F0 -----
    if nan_policy == "omit":
        F1_eff = np.where(np.isnan(F1), 0.0, F1)
        F0_eff = np.where(np.isnan(F0), 0.0, F0)
        delta = F1_eff - F0_eff
    else:
        F1_eff = np.where(np.isnan(F1), 0.0, F1)
        F0_eff = np.where(np.isnan(F0), 0.0, F0)
        delta = F1_eff - F0_eff
        bad1 = np.all(np.isnan(F1), axis=1)
        bad0 = np.all(np.isnan(F0), axis=1)
        bad = bad1 | bad0
        delta[bad, :] = np.nan

    mean = np.nanmean(delta, axis=1)
    std_exp = np.nanstd(delta, axis=1, ddof=1)
    std_bio = std_exp / np.sqrt(R)

    indices = (i,)
    indices_names = (L.feature_names[i],)
    bg_indices = tuple(j for j in range(L.N) if j != i)
    bg_names = tuple(L.feature_names[j] for j in bg_indices)

    # ----- Build Epistasis container -----
    E = Epistasis(
        values=delta,
        mean=mean,
        std_experimental=std_exp,
        std_biological=std_bio,
        order=1,
        indices=indices,
        backgrounds=backgrounds,
        indices_names=indices_names,
        background_indices=bg_indices,
        background_names=bg_names,
        meta={
        "focal_index": i,
        "missing_policy": missing_policy,
        "nan_policy": nan_policy,
        # Required by fees.fee_data() to reconstruct F0/F1 without re-pairing
        "abs_idx0": abs_idx0,
        "abs_idx1": abs_idx1,
        # Optional but useful for debugging/joins
        "bg_key": bg_key,},
    )

    # Early exit if no bootstrap
    if B_uncertainty <= 0 and B_null <= 0:
        return E.to_dataframe() if as_dataframe else E

    # ----------------------------------------------------------------------
    #                               BOOTSTRAPS
    # ----------------------------------------------------------------------

    F_df = pd.DataFrame(
        np.hstack([states, values]),
        columns=[f"s{j}" for j in range(N)] + [f"rep{r}" for r in range(R)],
    )
    alpha = 1.0 - ci_level
    lo_q = 100 * (alpha / 2)
    hi_q = 100 * (1 - alpha / 2)

    # ============================================================
    #                UNCERTAINTY BOOTSTRAP
    # ============================================================
    if B_uncertainty > 0:
        Fb_unc = bootstrap_uncertainty(
            F_df,
            B=B_uncertainty,
            multipliers=multipliers,
            flavor=uncertainty_flavor,
            rng=rng,
        )
        boots = Fb_unc.values[:, N:]  # (M, B)

        delta_b = boots[abs_idx1] - boots[abs_idx0]  # (M', B)

        E.ci_low = np.nanpercentile(delta_b, lo_q, axis=1)
        E.ci_high = np.nanpercentile(delta_b, hi_q, axis=1)
        E.sign_prob_pos = np.mean(delta_b > 0, axis=1)
        E.sign_prob_neg = np.mean(delta_b < 0, axis=1)

        if E.meta is None:
            E.meta = {}
        E.meta["uncertainty"] = {
            "B": B_uncertainty,
            "flavor": uncertainty_flavor,
            "multipliers": multipliers,
            "ci_level": ci_level,
            "sign_null_value": 0.0,
        }

    # ============================================================
    #                NULL BOOTSTRAP  (NO AJUSTE)
    # ============================================================
    if B_null > 0:
        Fb_null = bootstrap_null(
            F_df,
            B=B_null,
            multipliers=multipliers,
            preserve_residual_corr=preserve_residual_corr,
            rng=rng,
        )
        boots = Fb_null.values[:, N:]  # (M, B)

        delta_null_b = boots[abs_idx1] - boots[abs_idx0]

        # Global variance test
        var_obs = float(np.nanvar(mean, ddof=1))
        var_null_global = np.nanvar(delta_null_b, axis=0, ddof=1)
        var_null_global = var_null_global[np.isfinite(var_null_global)]

        if var_null_global.size:
            var_null_median = float(np.nanmedian(var_null_global))
            p_value_global = float(np.mean(var_null_global >= var_obs))
        else:
            var_null_median = np.nan
            p_value_global = np.nan

        E.var_obs = var_obs
        E.var_null_median = var_null_median
        E.p_value_var = p_value_global
        E.effect_size_var = var_obs - var_null_median

        # Per-background null CI
        E.null_ci_low = np.nanpercentile(delta_null_b, lo_q, axis=1)
        E.null_ci_high = np.nanpercentile(delta_null_b, hi_q, axis=1)

        # Per-background variance
        var_null_per_background = np.nanvar(delta_null_b, axis=1, ddof=1)
        E.var_null_per_background = var_null_per_background

        # Per-background SNR
        noise_std = np.sqrt(var_null_per_background)
        with np.errstate(divide="ignore", invalid="ignore"):
            snr = np.abs(mean) / noise_std
            snr[~np.isfinite(snr)] = np.nan
        E.snr_null = snr

        # Per-background *raw* p-values
        p_values = np.mean(np.abs(delta_null_b) >= np.abs(mean[:, None]), axis=1)
        E.p_values_per_background = p_values

        # 👇 NO adjusted p-values, no correction

        if E.meta is None:
            E.meta = {}
        E.meta["null"] = {
            "B": B_null,
            "preserve_residual_corr": preserve_residual_corr,
            "multipliers": multipliers,
            "ci_level": ci_level,
            "var_null_distribution": var_null_global.tolist() if var_null_global.size else [],
        }

    # ----------------------------------------------------------------------
    return E.to_dataframe() if as_dataframe else E


###########################################################################
#                            GENERAL EPISTASIS                            #  
###########################################################################

def epistasis_k(
    L: "Landscape",
    indices: Tuple[int, ...],
    *,
    # Robustness options
    missing_policy: str = "error",     # {"error","drop"}
    nan_policy: str = "omit",          # {"omit","propagate"}
    # Bootstrap options
    B_uncertainty: int = 0,
    uncertainty_flavor: str = "iid",   # {"iid","wildcluster"}
    B_null: int = 0,
    preserve_residual_corr: bool = False,   # null with/without biological correlations
    multipliers: str = "rademacher",   # {"rademacher","normal"}
    ci_level: float = 0.95,
    rng: np.random.Generator | None = None,
    # Output options
    as_dataframe: bool = True,
) -> "Epistasis":
    """
    Compute k-th order epistasis ε_I over all backgrounds (free loci), robust to:
      - missing states via `missing_policy`,
      - NaNs via `nan_policy`.

    It optionally attaches:
      - bootstrap uncertainty bands around F̂ (uncertainty bootstrap),
      - a null test via observed null landscapes (iid or wild-cluster), including:
          * per-background null CIs
          * per-background null variance
          * per-background p-values (raw)
          * global variance test across backgrounds.
    """
    # ---------- validation ----------
    if not hasattr(L, "states") or not hasattr(L, "values"):
        raise TypeError("L must provide .states (M,N) and .values (M,R).")
    if rng is None:
        rng = np.random.default_rng()

    states = L.states.astype(int, copy=False)
    values = L.values
    M, N = states.shape
    if values.shape[0] != M:
        raise ValueError("States and values must have the same number of rows.")
    R = values.shape[1]
    if R < 2 and (B_uncertainty > 0 or B_null > 0):
        raise ValueError("At least 2 replicates are required for bootstrap procedures.")

    I = tuple(sorted(indices))
    k = len(I)
    if k == 0:
        raise ValueError("indices must contain at least one locus.")
    if any(i < 0 or i >= N for i in I):
        raise ValueError(f"indices {I} out of bounds for N={N}.")
    if missing_policy not in {"error", "drop"}:
        raise ValueError("missing_policy must be 'error' or 'drop'.")
    if nan_policy not in {"omit", "propagate"}:
        raise ValueError("nan_policy must be 'omit' or 'propagate'.")

    # Free loci J = complement of I
    J = tuple(j for j in range(N) if j not in I)

    # ---------- all patterns over I ----------
    n_patterns = 1 << k
    patterns = np.array(
        [list(map(int, np.binary_repr(a, width=k))) for a in range(n_patterns)],
        dtype=int
    )  # (2^k, k)

    # Signs: s[a] = (-1)^(k + |a|)
    s = ((-1) ** (k + patterns.sum(axis=1))).astype(int)  # (2^k,)

    # ---------- collect background sets and absolute row ids per pattern ----------
    def rows_to_tuples(a: np.ndarray):
        if a.ndim == 1:
            return [tuple([int(a_i)]) for a_i in a.tolist()]
        return [tuple(row.tolist()) for row in a]

    bg_keys_by_pat = []
    abs_ids_by_pat = []
    for a_idx, a in enumerate(patterns):
        mask_a = np.all(states[:, I] == a, axis=1)
        rows_a = np.where(mask_a)[0]
        if rows_a.size == 0:
            if missing_policy == "error":
                raise ValueError(f"No rows found for pattern {a.tolist()} on loci {I}.")
            else:
                bg_keys_by_pat.append(set())
                abs_ids_by_pat.append({})
                continue
        bg_a = states[rows_a][:, J]
        keys_list = rows_to_tuples(bg_a)
        keys_a = set(keys_list)
        bg_keys_by_pat.append(keys_a)
        abs_ids_by_pat.append({key: rows_a[idx] for idx, key in enumerate(keys_list)})

    # ---------- decide the set of backgrounds to keep ----------
    if missing_policy == "error":
        first = None
        for keys in bg_keys_by_pat:
            if first is None:
                first = keys
            elif keys != first:
                raise ValueError("Missing states: not all patterns share the same background set.")
        kept_keys = sorted(first)
    else:  # "drop"
        if len(bg_keys_by_pat) == 0:
            kept_keys = []
        else:
            kept = set.intersection(*bg_keys_by_pat) if all(len(s_) > 0 for s_ in bg_keys_by_pat) else set()
            if len(kept) == 0:
                raise ValueError("No common backgrounds across patterns (after dropping missing).")
            kept_keys = sorted(kept)

    # backgrounds array in lex order over J
    if len(J) == 0:
        backgrounds = np.empty((len(kept_keys), 0), dtype=int)
    else:
        backgrounds = np.array(kept_keys, dtype=int)
        if backgrounds.ndim == 1:
            backgrounds = backgrounds.reshape(-1, 1)
        if backgrounds.size:
            order_bg = np.lexsort(backgrounds.T[::-1])
            backgrounds = backgrounds[order_bg]
    Mprime = backgrounds.shape[0]
    bg_keys = rows_to_tuples(backgrounds)

    # ---------- build absolute-row index matrix aligned to 'backgrounds' ----------
    idx_rows = np.empty((n_patterns, Mprime), dtype=int)
    for a_idx in range(n_patterns):
        mapping = abs_ids_by_pat[a_idx]
        try:
            idx_rows[a_idx] = np.array([mapping[k_] for k_ in bg_keys], dtype=int)
        except KeyError as e:
            if missing_policy == "error":
                raise ValueError(
                    f"Background {e.args[0]} missing for pattern {patterns[a_idx].tolist()}."
                )
            else:
                raise ValueError(
                    f"Inconsistent backgrounds after intersection for pattern {patterns[a_idx].tolist()}."
                )

    # ---------- build tensor Y: (M', 2^k, R) ----------
    Y = np.stack([values[idx_rows[a_idx]] for a_idx in range(n_patterns)], axis=1)

    # ---------- NaN handling before alternating sum ----------
    if nan_policy == "omit":
        Y_eff = np.where(np.isnan(Y), 0.0, Y)
        eps_values = (Y_eff * s[None, :, None]).sum(axis=1)  # (M', R)
    else:  # "propagate"
        Y_eff = np.where(np.isnan(Y), 0.0, Y)
        eps_values = (Y_eff * s[None, :, None]).sum(axis=1)  # (M', R)
        all_nan_term = np.all(np.isnan(Y), axis=2)           # (M', 2^k)
        bad_row = np.any(all_nan_term, axis=1)               # (M',)
        if np.any(bad_row):
            eps_values[bad_row, :] = np.nan

    # ---------- summary statistics ----------
    mean = np.nanmean(eps_values, axis=1)        # (M',)
    std_exp = np.nanstd(eps_values, axis=1, ddof=1)
    std_bio = std_exp / np.sqrt(R)

    # ---------- Epistasis container ----------
    indices_names = tuple(L.feature_names[j] for j in I)
    bg_indices = tuple(j for j in range(L.N) if j not in I)
    bg_names = tuple(L.feature_names[j] for j in bg_indices)

    E = Epistasis(
        values=eps_values,
        mean=mean,
        std_experimental=std_exp,
        std_biological=std_bio,
        order=k,
        indices=I,
        backgrounds=backgrounds,
        indices_names=indices_names,
        background_indices=bg_indices,
        background_names=bg_names,
        meta={
            "indices": I,
            "order": k,
            "missing_policy": missing_policy,
            "nan_policy": nan_policy,
        },
    )

    # Early exit if no bootstrap requested
    if B_uncertainty <= 0 and B_null <= 0:
        return E.to_dataframe() if as_dataframe else E

    # ---------- DataFrame for bootstrap engines ----------
    F_df = pd.DataFrame(
        np.hstack([states, values]),
        columns=[f"s{j}" for j in range(N)] + [f"rep{r}" for r in range(R)]
    )

    # ==================== UNCERTAINTY BOOTSTRAP ====================
    if B_uncertainty > 0:
        Fb_unc = bootstrap_uncertainty(
            F_df,
            B=B_uncertainty,
            multipliers=multipliers,
            flavor=uncertainty_flavor,
            rng=rng,
        )
        boots_mat = Fb_unc.values[:, N:]  # (M, B)

        eps_b = None
        for a_idx in range(n_patterns):
            part = boots_mat[idx_rows[a_idx]]  # (M', B)
            part = part * s[a_idx]
            eps_b = part if eps_b is None else (eps_b + part)

        alpha_unc = 1.0 - ci_level
        lo_q, hi_q = 100 * (alpha_unc / 2.0), 100 * (1.0 - alpha_unc / 2.0)
        ci_low = np.nanpercentile(eps_b, lo_q, axis=1)
        ci_high = np.nanpercentile(eps_b, hi_q, axis=1)

        sign_prob_pos = np.mean(eps_b > 0, axis=1)
        sign_prob_neg = np.mean(eps_b < 0, axis=1)

        if E.meta is None:
            E.meta = {}
        E.meta["uncertainty"] = {
            "B": B_uncertainty,
            "flavor": uncertainty_flavor,
            "multipliers": multipliers,
            "ci_level": ci_level,
            "sign_null_value": 0.0,
        }

        E.ci_low = ci_low
        E.ci_high = ci_high
        E.sign_prob_pos = sign_prob_pos
        E.sign_prob_neg = sign_prob_neg

    # ==================== NULL BOOTSTRAP ====================
    if B_null > 0:
        Fb_null = bootstrap_null(
            F_df,
            B=B_null,
            multipliers=multipliers,
            preserve_residual_corr=preserve_residual_corr,
            rng=rng,
        )
        boots_mat = Fb_null.values[:, N:]  # (M, B)

        eps_null_b = None
        for a_idx in range(n_patterns):
            part = boots_mat[idx_rows[a_idx]]  # (M', B)
            part = part * s[a_idx]
            eps_null_b = part if eps_null_b is None else (eps_null_b + part)

        # ---------- Global variance test across backgrounds ----------
        var_obs = float(np.nanvar(mean, ddof=1))                 # scalar
        var_null_global = np.nanvar(eps_null_b, axis=0, ddof=1)  # (B,)
        var_null_global = var_null_global[np.isfinite(var_null_global)]
        if var_null_global.size > 0:
            var_null_median = float(np.nanmedian(var_null_global))
            p_value = float(np.mean(var_null_global >= var_obs))
        else:
            var_null_median = np.nan
            p_value = np.nan
        effect_size = var_obs - var_null_median if np.isfinite(var_null_median) else np.nan

        # ---------- Per-background null CI ----------
        alpha_null = 1.0 - ci_level
        lo_q, hi_q = 100 * (alpha_null / 2.0), 100 * (1.0 - alpha_null / 2.0)
        null_ci_low = np.nanpercentile(eps_null_b, lo_q, axis=1)   # (M',)
        null_ci_high = np.nanpercentile(eps_null_b, hi_q, axis=1)  # (M',)

        # ---------- Per-background null variance ----------
        var_null_per_background = np.nanvar(eps_null_b, axis=1, ddof=1)  # (M',)

        # ---------- Per-background SNR ----------
        noise_std_null = np.sqrt(var_null_per_background)
        with np.errstate(divide="ignore", invalid="ignore"):
            snr_null = np.abs(mean) / noise_std_null
            snr_null[~np.isfinite(snr_null)] = np.nan

        # ---------- Per-background p-values (two-sided, raw) ----------
        p_values_per_background = np.mean(
            np.abs(eps_null_b) >= np.abs(mean)[:, None],
            axis=1
        )  # (M',)

        if E.meta is None:
            E.meta = {}
        E.meta["null"] = {
            "B": B_null,
            "preserve_residual_corr": preserve_residual_corr,
            "multipliers": multipliers,
            "var_null_distribution": var_null_global.tolist(),
            "ci_level": ci_level,
        }

        E.var_obs = var_obs
        E.var_null_median = var_null_median
        E.p_value_var = p_value
        E.effect_size_var = effect_size
        E.null_ci_low = null_ci_low
        E.null_ci_high = null_ci_high
        E.snr_null = snr_null
        E.var_null_per_background = var_null_per_background
        E.p_values_per_background = p_values_per_background

    # --- Final output selection ---
    return E.to_dataframe() if as_dataframe else E


def epistasis_k_replics(
    F: np.ndarray,
    N: int,
    idx: Sequence[int],
    *,
    nan_policy: str = "omit",   # {"omit", "propagate"}
) -> np.ndarray:
    """
    Compute ε_{i1..ik}(x) across all backgrounds for a full 2^N landscape.

    Fast, low-level helper assuming:
      - states are the full hypercube in MSB-first lexicographic order,
      - no bootstrap, no null model; just the raw epistasis tensor.

    Parameters
    ----------
    F : array_like of float
        Landscape values in lexicographic order (MSB-first).
        Accepts:
          - shape (2^N,)   → single landscape
          - shape (2^N, R) → multiple replicates in columns
    N : int
        Number of loci (so F.shape[0] must equal 2^N).
    idx : sequence of int
        Locus indices defining the k-wise term.
    nan_policy : {"omit", "propagate"}, default="omit"
        - "omit": treat NaNs as 0 in each term before the alternating sum.
        - "propagate": if any term (pattern over idx) is entirely NaN across
          replicates for a given background, the resulting epistasis for that
          background is set to NaN (otherwise NaNs are treated as 0 in the sum).

    Returns
    -------
    eps : ndarray of float
        Epistatic values across all backgrounds where these k loci are free.
        - shape (2^{N-k},)   if input was (2^N,)
        - shape (2^{N-k}, R) if input was (2^N, R)

    Notes
    -----
    - Biological sign convention "present minus absent":
        ε_I(x) = Σ_{a∈{0,1}^k} (-1)^{k - |a|} F(x_I = a, x_J fixed)
      where I = idx, J = complement.
    - This is a low-level routine assuming perfect coverage and ordering;
      for general / missing / irregular layouts, use `epistasis_k` over a
      `Landscape` objeto.
    """
    A = np.asarray(F, dtype=float)
    if A.ndim == 1:
        A = A[:, None]          # promote to (2^N, 1)
        squeeze_out = True
    elif A.ndim == 2:
        squeeze_out = False
    else:
        raise ValueError("F must be 1-D (2^N,) or 2-D (2^N, R).")

    if A.shape[0] != (1 << N):
        raise ValueError(f"First dimension of F must be 2^N, got {A.shape[0]} != {1<<N}.")

    if nan_policy not in {"omit", "propagate"}:
        raise ValueError("nan_policy must be 'omit' or 'propagate'.")

    # Validate loci
    idx = np.array(sorted(set(int(i) for i in idx)), dtype=int)
    if idx.size == 0:
        raise ValueError("idx must contain at least one locus.")
    if np.any((idx < 0) | (idx >= N)):
        raise ValueError("idx entries must be in [0, N-1].")
    k = idx.size

    # Enumerate all backgrounds with the k loci free (placeholders at idx are 0's)
    X = _states_for_backgrounds(N, idx)   # shape (2^{N-k}, N)
    M = X.shape[0]

    # Precompute integer indices in lex order.
    # index(x) = sum_m x[m] * 2^{N-1-m}. Since X[:, idx] == 0, base_idx ignores those bits.
    weights = (1 << np.arange(N-1, -1, -1))   # 2^{N-1-m}, m=0..N-1
    w_base = weights.copy()
    w_base[idx] = 0
    base_idx = X @ w_base                     # shape (M,)

    # Bit weights for the k free loci (their contribution when set to 1)
    bit_w = (1 << (N - 1 - idx))              # shape (k,)

    # Prepare output and (opcional) NaN tracking
    R = A.shape[1]
    out = np.zeros((M, R), dtype=float)

    if nan_policy == "propagate":
        # all_nan_term[m, a] = True si para el patrón a todos los reps son NaN en ese background
        # (M, 2^k)
        from itertools import product
        all_nan_term = np.zeros((M, 1 << k), dtype=bool)

        for a_idx, a_bits in enumerate(product([0, 1], repeat=k)):
            # (-1)^{k - |a|} with |a| = number of ones in a
            sign = 1.0 if ((k + sum(a_bits)) % 2 == 0) else -1.0
            offset = np.dot(a_bits, bit_w) if k > 0 else 0
            idx_vec = base_idx + offset           # shape (M,)

            vals = A[idx_vec, :]                  # (M, R)
            # track all-NaN patterns
            all_nan_term[:, a_idx] = np.all(np.isnan(vals), axis=1)

            # replace NaNs by 0 for the sum
            vals_eff = np.where(np.isnan(vals), 0.0, vals)
            out += sign * vals_eff

        # backgrounds where ANY pattern is all-NaN across reps → whole row NaN
        bad_row = np.any(all_nan_term, axis=1)
        if np.any(bad_row):
            out[bad_row, :] = np.nan

    else:  # nan_policy == "omit"
        from itertools import product
        for a_bits in product([0, 1], repeat=k):
            sign = 1.0 if ((k + sum(a_bits)) % 2 == 0) else -1.0
            offset = np.dot(a_bits, bit_w) if k > 0 else 0
            idx_vec = base_idx + offset           # shape (M,)

            vals = A[idx_vec, :]                  # (M, R)
            vals_eff = np.where(np.isnan(vals), 0.0, vals)
            out += sign * vals_eff

    return out[:, 0] if squeeze_out else out


###########################################################################
#                             FULL EPISTASIS                              #  
###########################################################################

def compute_full_epistasis(
    L: "Landscape",
    *,
    min_order: int = 1,
    max_order: Optional[int] = None,
    # Robustness options
    missing_policy: str = "error",
    nan_policy: str = "omit",
    # Bootstrap options
    B_uncertainty: int = 0,
    uncertainty_flavor: str = "iid",   # {"iid","wildcluster"}
    B_null: int = 0,
    preserve_residual_corr: bool = False,   # null with/without biological correlations
    multipliers: str = "rademacher",   # {"rademacher","normal"}
    ci_level: float = 0.95,
    rng: np.random.Generator | None = None,
    # Output format
    as_dataframe: bool = True,
) -> Union[pd.DataFrame, Dict[int, Dict[Tuple[int, ...], "Epistasis"]]]:
    
    """
    Compute all k-th order epistatic effects ε_I over all backgrounds
    for all subsets I of loci, with k ranging from min_order to max_order.

    Internally calls `epistasis_k` for each locus combination I.

    Parameters
    ----------
    L : Landscape
        Landscape object providing .states (M, N) and .values (M, R).
    min_order : int
        Minimum interaction order to compute (>=1).
    max_order : int or None
        Maximum interaction order.
        If None, sets max_order = N (the total number of loci).
    missing_policy, nan_policy, B_uncertainty, uncertainty_flavor,
    B_null, preserve_residual_corr, multipliers, ci_level, rng :
        Passed to epistasis_k without modification.
    as_dataframe : bool
        If True (default), returns a single tidy “pretty” DataFrame
        If False, returns a nested dict:
            { k : { I : Epistasis } }

    Returns
    -------
    pd.DataFrame or dict
        Full epistasis table or a nested Epistasis-object dictionary.
    """

    N = L.N
    if max_order is None:
        max_order = N

    if min_order < 1 or min_order > N:
        raise ValueError(f"min_order must be in [1, N], got {min_order}.")
    if max_order < 1 or max_order > N:
        raise ValueError(f"max_order must be in [1, N], got {max_order}.")
    if min_order > max_order:
        raise ValueError("min_order must be <= max_order.")

    # Container for Epistasis objects
    results: Dict[int, Dict[Tuple[int, ...], "Epistasis"]] = {}

    # Loop over interaction orders k = min_order .. max_order
    for k in range(min_order, max_order + 1):
        results[k] = {}
        for I in combinations(range(N), k):
            E_I = epistasis_k(
                L,
                indices=I,
                missing_policy=missing_policy,
                nan_policy=nan_policy,
                B_uncertainty=B_uncertainty,
                uncertainty_flavor=uncertainty_flavor,
                B_null=B_null,
                preserve_residual_corr=preserve_residual_corr,
                multipliers=multipliers,
                ci_level=ci_level,
                rng=rng,
                as_dataframe=False,  # internal use always Epistasis object
            )
            results[k][I] = E_I

    if not as_dataframe:
        # Return structured dict for programmatic use
        return results

    # ------- Build a single "pretty" DataFrame -------
    frames = []
    for k, dict_k in results.items():
        for I, E_I in dict_k.items():
            df_I = E_I.to_dataframe()

            # Number of backgrounds for this interaction
            n_rows = len(df_I)

            # High-level descriptors
            df_I.insert(0, "Interaction type", [f"order-{k}"] * n_rows)
            #df_I.insert(1, "Loci tuple", [I] * n_rows)

            frames.append(df_I)

    if not frames:
        return pd.DataFrame()

    full_df = pd.concat(frames, ignore_index=True)
    return full_df

########################################################################
#                          EPISTASIS DISTRIBUTION                      #
########################################################################

EpistasisDists = namedtuple("EpistasisDists", ["pooled", "per_replicate"])

def epistasis_distributions_by_order(
    F: np.ndarray,
    N: int,
    *,
    nan_policy: str = "omit",   # {"omit", "propagate"}
) -> EpistasisDists:
    """
    Compute epistasis ε_{i1..iS}(x) for ALL orders S=1..N and ALL subsets,
    assuming a full 2^N landscape in lexicographic order, and return
    pooled distributions by order.

    This is a fast, low-level helper:

      - No bootstraps, no null model.
      - Asume hipercubo completo (sin estados ausentes).
      - Usa la convención biológica "present minus absent".

    Parameters
    ----------
    F : ndarray
        - If shape is (2^N,), it is a single landscape in lexicographic order.
        - If shape is (2^N, R), the last axis indexes R replicates.
    N : int
        Number of loci.

    nan_policy : {"omit", "propagate"}, default="omit"
        Passed to `epistasis_k_replics`:
        - "omit": trata NaNs como 0 en cada término antes de la suma alternante.
        - "propagate": si algún término (patrón sobre los loci del conjunto)
          es todo NaN en todos los replicados para un background dado,
          la ε correspondiente se marca como NaN.

    Returns
    -------
    EpistasisDists
        A namedtuple with two dicts indexed by order S:
          - pooled[S]        : 1D array with all ε concatenated across subsets,
                               backgrounds, and replicates. Length =
                               comb(N,S) * 2^(N-S) * R.
          - per_replicate[S] : 2D array with shape
                               (comb(N,S) * 2^(N-S), R). Each column is a replicate.

    Notes
    -----
    - El orden interno de los subconjuntos (i1..iS) es el de `itertools.combinations`
      sobre range(N), es decir, lexicográfico en los índices de loci.
      Esto no afecta a histogramas ni estadísticas de orden.
    """

    A = np.asarray(F, dtype=float)
    if A.ndim == 1:
        # Promote to (2^N, 1) for unified handling
        A = A[:, None]
    elif A.ndim != 2:
        raise ValueError("F must be 1-D (2^N,) or 2-D (2^N, R).")

    M, R = A.shape
    if M != (1 << N):
        raise ValueError(f"F must have 2^N rows; got {M} != {1<<N}.")

    if nan_policy not in {"omit", "propagate"}:
        raise ValueError("nan_policy must be 'omit' or 'propagate'.")

    pooled: dict[int, np.ndarray] = {}
    per_rep: dict[int, np.ndarray] = {}

    # Loop over epistasis orders S = 1..N
    for S in range(1, N + 1):
        blocks = []

        # All subsets of size S
        for idx in combinations(range(N), S):
            # eps has shape (2^{N-S}, R)
            eps = epistasis_k_replics(
                A,
                N=N,
                idx=idx,
                nan_policy=nan_policy,
            )
            # Aseguramos 2D (para el caso R=1)
            eps = np.asarray(eps, dtype=float)
            if eps.ndim == 1:
                eps = eps[:, None]
            blocks.append(eps)

        if blocks:
            # apilamos todas las interacciones de orden S
            big_mat = np.vstack(blocks)  # shape = (comb(N,S)*2^{N-S}, R)
            per_rep[S] = big_mat
            pooled[S] = big_mat.reshape(-1)  # flatten all reps
        else:
            # debería ocurrir solo si N < 1, pero lo dejamos por robustez
            per_rep[S] = np.empty((0, R))
            pooled[S] = np.empty((0,))

    return EpistasisDists(pooled=pooled, per_replicate=per_rep)

########################################################################
#                         ANALYSIS DEPENDENCES                         #
########################################################################

def filter_significant_interactions(
    full_df: pd.DataFrame,
    *,
    min_snr_null: float = 2.0,
    max_p_value_var: Optional[float] = 0.05,
    min_prob_sign: float = 0.95,
    max_p_null: Optional[float] = 0.05,
    include_features: Optional[Iterable[str]] = None,
    min_feature_overlap: int = 1,
    feature_col: str = "Loci names",
    min_order: int = 1,
    max_order: Optional[int] = None,
    p_col_for_filter: str = "p-null",
) -> pd.DataFrame:
    """
    Filter significant epistatic interactions using a combination of criteria:
    SNR, global-variance p-value, local per-background p-value, sign consistency,
    interaction order, and optional feature overlap.

    Parameters
    ----------
    full_df : pd.DataFrame
        Input DataFrame, typically produced by Epistasis.to_dataframe().
    min_snr_null : float, default 2.0
        Minimum required signal-to-noise ratio under the null model.
    max_p_value_var : float or None, default 0.05
        Maximum allowed global variance p-value ("p-var"). If None, this filter is skipped.
    min_prob_sign : float, default 0.95
        Minimum of max{Prob(Effect > 0), Prob(Effect < 0)}.
    max_p_null : float or None, default 0.05
        Maximum allowed p-value in the column specified by `p_col_for_filter`.
        If None, the local p-value filter is skipped.
    include_features : Iterable[str] or None
        If provided, interactions must include at least `min_feature_overlap`
        of these features in their `feature_col` column.
    min_feature_overlap : int, default 1
        Minimum required overlap between the interaction's features and `include_features`.
    feature_col : str, default "Loci names"
        Column that encodes the loci/feature names (tuple, list, or string).
    min_order : int, default 1
        Minimum interaction order to retain.
    max_order : int or None, default None
        Maximum interaction order to retain. If None, no upper bound is imposed.
    p_col_for_filter : str, default "p-null"
        Name of the p-value column used for the local p-value filter (typically "p-null").

    Returns
    -------
    pd.DataFrame
        Filtered DataFrame (index reset).
    """
    df = full_df.copy()

    # Normalize global variance p-value column
    if "P-value (variance excess)" in df.columns:
        df["p-var"] = df["P-value (variance excess)"]
    elif "p-var" not in df.columns:
        df["p-var"] = np.nan

    # Normalize raw per-background p-values
    if "P-value (per background)" in df.columns:
        df["p-null"] = df["P-value (per background)"]
    elif "p-null" not in df.columns:
        df["p-null"] = np.nan

    # Ensure that the selected p-value column exists (for local filter)
    if max_p_null is not None and p_col_for_filter not in df.columns:
        raise ValueError(
            f"The selected p-value column '{p_col_for_filter}' does not exist. "
            f"Available columns: {list(df.columns)}"
        )

    # Ensure required columns exist
    for col in ["Signal-to-Null-Noise (SNR)", "Prob(Effect > 0)", "Prob(Effect < 0)"]:
        if col not in df.columns:
            df[col] = np.nan

    # Strongest sign probability
    df["Prob(max sign)"] = df[["Prob(Effect > 0)", "Prob(Effect < 0)"]].max(axis=1)

    # Begin composite mask
    mask = np.ones(len(df), dtype=bool)

    # SNR filter
    mask &= (df["Signal-to-Null-Noise (SNR)"] >= min_snr_null)

    # Global variance p-value filter
    if max_p_value_var is not None:
        mask &= (df["p-var"] <= max_p_value_var)

    # Local p-value filter (always raw/null p-values now)
    if max_p_null is not None:
        mask &= (df[p_col_for_filter] <= max_p_null)

    # Sign consistency
    mask &= (df["Prob(max sign)"] >= min_prob_sign)

    # Interaction order filters
    mask &= (df["Order"] >= min_order)
    if max_order is not None:
        mask &= (df["Order"] <= max_order)

    # Optional feature overlap filter
    if include_features is not None and feature_col in df.columns:
        target = set(include_features)

        def extract_list(x):
            if isinstance(x, str):
                return [x]
            elif isinstance(x, (tuple, list)):
                return list(x)
            return []

        df["_feat"] = df[feature_col].apply(extract_list)
        df["overlap"] = df["_feat"].apply(lambda L: len(set(L) & target))
        mask &= (df["overlap"] >= min_feature_overlap)

    return df[mask].reset_index(drop=True)

#########################################################################
#                           EPISTASIS NETWORKS                          #
#########################################################################

def epistasis_to_network(
    full_df: pd.DataFrame,
    *,
    order: int = 2,
    agg: str = "mean_abs",
    feature_names: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Build a pairwise epistatic network from the full epistasis DataFrame.

    For a given interaction order (default: 2), this function aggregates
    per-background epistatic values into a single edge weight per locus pair.

    Nodes = loci (indices or names).
    Edges = pairs (i, j) for which an order-2 interaction ε_{i,j} was computed.

    Parameters
    ----------
    full_df : pd.DataFrame
        Output from `compute_full_epistasis`, in "pretty" format.
        Must contain at least:
            - "Interaction type"  (e.g. "order-2")
            - "Loci tuple"        (e.g. (i, j))
            - "Epistasis (mean)"
        Optionally:
            - "Signal-to-Null-Noise (SNR)"

    order : int, default=2
        Interaction order to build the network from.
        Currently only order=2 is supported for pairwise networks.

    agg : {"mean_abs", "max_abs", "mean_snr", "max_snr"}, default="mean_abs"
        Aggregation rule across backgrounds for each locus pair:
            - "mean_abs" : mean(|Epistasis (mean)|)
            - "max_abs"  : max(|Epistasis (mean)|)
            - "mean_snr" : mean(Signal-to-Null-Noise (SNR))
            - "max_snr"  : max(Signal-to-Null-Noise (SNR))

    feature_names : list of str or None, optional
        Optional list of locus names, indexed by locus id (0..N-1).
        If provided, the output will include columns:
            - "Locus i name"
            - "Locus j name"

    Returns
    -------
    pd.DataFrame
        Edge list with columns:
            - "i", "j"          : locus indices
            - "weight"          : aggregated value according to `agg`
            - "metric"          : name of the aggregation metric
            - optionally "Locus i name", "Locus j name" if `feature_names` is not None.
    """
    if order != 2:
        raise NotImplementedError(
            "epistasis_to_network currently supports only order=2 "
            "interactions for pairwise networks."
        )

    # Safety: ensure required columns exist
    required_cols = ["Interaction type", "Loci involved", "Epistasis (mean)"]
    for col in required_cols:
        if col not in full_df.columns:
            raise ValueError(f"Column '{col}' is required in full_df but was not found.")

    # Filter to the desired order
    df_ord = full_df[full_df["Interaction type"] == f"order-{order}"].copy()
    if df_ord.empty:
        # No interactions of the requested order
        return pd.DataFrame(columns=["i", "j", "weight", "metric"])

    # Ensure SNR column exists if needed
    if agg in {"mean_snr", "max_snr"} and "Signal-to-Null-Noise (SNR)" not in df_ord.columns:
        raise ValueError(
            "Aggregation mode requires 'Signal-to-Null-Noise (SNR)' but the column is missing."
        )

    # Extract i, j from "Loci tuple" (which is e.g. (i, j))
    # This creates two new integer columns "i" and "j"
    df_ord = df_ord.copy()
    df_ord["i"] = df_ord["Loci involved"].apply(lambda t: int(t[0]))
    df_ord["j"] = df_ord["Loci involved"].apply(lambda t: int(t[1]))

    # Define the aggregation per pair (i, j)
    if agg == "mean_abs":
        grouped = (
            df_ord
            .groupby(["i", "j"], as_index=False)["Epistasis (mean)"]
            .agg(lambda x: float(np.nanmean(np.abs(x))))
        )
        grouped.rename(columns={"Epistasis (mean)": "weight"}, inplace=True)
        metric_name = "mean_abs_epistasis"

    elif agg == "max_abs":
        grouped = (
            df_ord
            .groupby(["i", "j"], as_index=False)["Epistasis (mean)"]
            .agg(lambda x: float(np.nanmax(np.abs(x))))
        )
        grouped.rename(columns={"Epistasis (mean)": "weight"}, inplace=True)
        metric_name = "max_abs_epistasis"

    elif agg == "mean_snr":
        grouped = (
            df_ord
            .groupby(["i", "j"], as_index=False)["Signal-to-Null-Noise (SNR)"]
            .agg(lambda x: float(np.nanmean(x)))
        )
        grouped.rename(columns={"Signal-to-Null-Noise (SNR)": "weight"}, inplace=True)
        metric_name = "mean_snr_null"

    elif agg == "max_snr":
        grouped = (
            df_ord
            .groupby(["i", "j"], as_index=False)["Signal-to-Null-Noise (SNR)"]
            .agg(lambda x: float(np.nanmax(x)))
        )
        grouped.rename(columns={"Signal-to-Null-Noise (SNR)": "weight"}, inplace=True)
        metric_name = "max_snr_null"

    else:
        raise ValueError(
            f"Unknown aggregation mode '{agg}'. "
            "Use one of: 'mean_abs', 'max_abs', 'mean_snr', 'max_snr'."
        )

    grouped["metric"] = metric_name

    # Optionally attach locus names
    if feature_names is not None:
        grouped["Locus i name"] = grouped["i"].apply(
            lambda idx: feature_names[idx] if 0 <= idx < len(feature_names) else f"loc_{idx}"
        )
        grouped["Locus j name"] = grouped["j"].apply(
            lambda idx: feature_names[idx] if 0 <= idx < len(feature_names) else f"loc_{idx}"
        )

    return grouped


import networkx as nx
import matplotlib.pyplot as plt

def plot_epistasis_network(edges_df, *, use_names=False, node_size=800, font_size=10):
    """
    Plot a pairwise epistasis network from the edges DataFrame returned by epistasis_to_network.
    
    Parameters
    ----------
    edges_df : pd.DataFrame
        Must contain columns "i", "j", "weight".
        If use_names=True, must also contain "Locus i name" and "Locus j name".
    
    use_names : bool
        If True, nodes use biological names instead of indices.
    """

    G = nx.Graph()

    # --- Add nodes ---
    if use_names:
        nodes = set(edges_df["Locus i name"]) | set(edges_df["Locus j name"])
        for name in nodes:
            G.add_node(name)
    else:
        nodes = set(edges_df["i"]) | set(edges_df["j"])
        for idx in nodes:
            G.add_node(idx)

    # --- Add weighted edges ---
    for _, row in edges_df.iterrows():
        if use_names:
            u = row["Locus i name"]
            v = row["Locus j name"]
        else:
            u = row["i"]
            v = row["j"]
        w = row["weight"]
        G.add_edge(u, v, weight=w)

    # --- Layout (spring layout looks good for epistasis) ---
    pos = nx.spring_layout(G, seed=42, k=0.5)

    # --- Edge widths scaled by weight ---
    weights = [G[u][v]["weight"] for u, v in G.edges()]
    max_w = max(weights) if len(weights) > 0 else 1.0
    widths = [2 + 4*(w/max_w) for w in weights]

    # --- Draw ---
    plt.figure(figsize=(8, 6))
    nx.draw_networkx_nodes(G, pos, node_size=node_size, node_color="#4FA0FF")
    nx.draw_networkx_labels(G, pos, font_size=font_size, font_color="black")

    nx.draw_networkx_edges(
        G,
        pos,
        width=widths,
        edge_color="#444444",
    )

    plt.title("Epistasis Network (order 2)")
    plt.axis("off")
    plt.show()

##############################################
#            VOLCANO PLOT FUNCTION           #
##############################################

def plot_epistasis_volcano(
    full_df: pd.DataFrame,
    *,
    orders: Optional[Sequence[int]] = None,
    effect_col: str = "Epistasis (mean)",
    p_col: str = "p-null",           # typically "p-null" or any p-value column
    clip: float = 1e-6,
    return_data: bool = False,
    mode: str = "by-order",          # {"single", "by-order"}
    alpha: float | None = 0.05,      # threshold on p_col; None -> no line / no % text
    colormap: str = "default",       # {"default","custom", other->custom}
    ax: Optional["plt.Axes"] = None, # or array of Axes if mode="by-order"
    fig_kwargs: Optional[dict] = None,
):
    """
    Volcano plot of epistatic coefficients for selected interaction orders.

    Parameters
    ----------
    full_df : pd.DataFrame
        Output from `compute_full_epistasis` (or a filtered version) containing
        at least the columns:
            - "Order"
            - effect_col (default: "Epistasis (mean)")
            - p_col      (default: "p-null")
    orders : sequence of int or None, optional
        Interaction orders to include. If None, use all orders present in `full_df`.
    effect_col : str, default "Epistasis (mean)"
        Column name with the epistatic effect size.
    p_col : str, default "p-null"
        Column name with p-values associated to each coefficient. This is used
        to compute -log10(p) on the y-axis and to define significance thresholds.
    clip : float, default 1e-6
        Minimum p-value used when computing -log10(p). Any p < clip
        is replaced by clip to avoid infinities / NaNs.
    return_data : bool, default False
        If True, return the DataFrame used for plotting, with an extra
        column "neg_log10_p". If False, return None.
    mode : {"single", "by-order"}, default "by-order"
        - "single"   : all orders in a single panel (color-coded).
        - "by-order" : one subplot per order; arranged in 1 or 2 rows:
                       * <= 4 orders: 1 row, n_orders columns
                       * > 4 orders: 2 rows, ceil(n_orders / 2) columns
    alpha : float or None, default 0.05
        Significance threshold applied to `p_col`. If not None, a horizontal
        line at -log10(alpha) is drawn. In "by-order" mode, the percentage of
        coefficients with p_col <= alpha is annotated in each panel.
    colormap : {"default","custom", any}, default "default"
        - "default" -> use matplotlib "tab10"
        - "custom"  -> use an internal orange/blue gradient
        - any other string -> fall back to the custom gradient
    ax : matplotlib.axes.Axes or array-like of Axes, optional
        Existing Axes to draw on.
        - If mode="single": a single Axes.
        - If mode="by-order": array-like of Axes with length >= n_orders.
        If None, a new figure + axes are created.
    fig_kwargs : dict or None, optional
        Extra keyword arguments passed to plt.subplots when creating
        a new figure (ignored if `ax` is provided).

    Returns
    -------
    pd.DataFrame or None
        If return_data=True, returns the DataFrame used for the plot.
        Otherwise returns None.
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    from matplotlib.transforms import blended_transform_factory

    if mode not in {"single", "by-order"}:
        raise ValueError("mode must be 'single' or 'by-order'.")

    # --- Basic checks ---
    if "Order" not in full_df.columns:
        raise ValueError("full_df must contain an 'Order' column.")
    if effect_col not in full_df.columns:
        raise ValueError(f"Column '{effect_col}' not found in full_df.")
    if p_col not in full_df.columns:
        raise ValueError(f"Column '{p_col}' not found in full_df.")

    fig_kwargs = {} if fig_kwargs is None else dict(fig_kwargs)
    df = full_df.copy()

    # --- Filter by orders if requested ---
    if orders is not None:
        orders = list(orders)
        df = df[df["Order"].isin(orders)]

    # Drop rows without effect or p-value
    df = df.dropna(subset=[effect_col, p_col]).copy()
    if df.empty:
        raise ValueError("No rows left after filtering by order and removing NaNs.")

    # --- Clip p-values and compute -log10(p_col) ---
    p_raw = df[p_col].astype(float)
    df[p_col] = p_raw  # ensure numeric
    p_clipped = p_raw.clip(lower=clip)
    df["neg_log10_p"] = -np.log10(p_clipped)

    # Label for y-axis and threshold text (generic p)
    y_label = r"$-\log_{10}(p)$"
    thr_label = r"$p \leq " + f"{alpha:g}" + r"$"

    # Unique orders actually present
    orders_unique = np.sort(df["Order"].unique())
    if len(orders_unique) == 0:
        raise ValueError("No interaction orders available after filtering.")
    n_orders = len(orders_unique)

    # --- Choose base colormap from flag ---
    EP_VOLCANO_BASE_COLORS = [
        "#ff5400",
        "#ff8500",
        "#ff9e00",
        "#00b4d8",
        "#0077b6",
        "#03045e",
    ]
    EP_VOLCANO_CMAP = LinearSegmentedColormap.from_list(
        "epistasis_volcano",
        EP_VOLCANO_BASE_COLORS,
        N=256,
    )

    if colormap == "parula":
        PARULA_COLORS =  [[0.2081, 0.1663, 0.5292], [0.2116238095, 0.1897809524, 0.5776761905], 
            [0.212252381, 0.2137714286, 0.6269714286], [0.2081, 0.2386, 0.6770857143], 
            [0.1959047619, 0.2644571429, 0.7279], [0.1707285714, 0.2919380952, 
            0.779247619], [0.1252714286, 0.3242428571, 0.8302714286], 
            [0.0591333333, 0.3598333333, 0.8683333333], [0.0116952381, 0.3875095238, 
            0.8819571429], [0.0059571429, 0.4086142857, 0.8828428571], 
            [0.0165142857, 0.4266, 0.8786333333], [0.032852381, 0.4430428571, 
            0.8719571429], [0.0498142857, 0.4585714286, 0.8640571429], 
            [0.0629333333, 0.4736904762, 0.8554380952], [0.0722666667, 0.4886666667, 
            0.8467], [0.0779428571, 0.5039857143, 0.8383714286], 
            [0.079347619, 0.5200238095, 0.8311809524], [0.0749428571, 0.5375428571, 
            0.8262714286], [0.0640571429, 0.5569857143, 0.8239571429], 
            [0.0487714286, 0.5772238095, 0.8228285714], [0.0343428571, 0.5965809524, 
            0.819852381], [0.0265, 0.6137, 0.8135], [0.0238904762, 0.6286619048, 
            0.8037619048], [0.0230904762, 0.6417857143, 0.7912666667], 
            [0.0227714286, 0.6534857143, 0.7767571429], [0.0266619048, 0.6641952381, 
            0.7607190476], [0.0383714286, 0.6742714286, 0.743552381], 
            [0.0589714286, 0.6837571429, 0.7253857143], 
            [0.0843, 0.6928333333, 0.7061666667], [0.1132952381, 0.7015, 0.6858571429], 
            [0.1452714286, 0.7097571429, 0.6646285714], [0.1801333333, 0.7176571429, 
            0.6424333333], [0.2178285714, 0.7250428571, 0.6192619048], 
            [0.2586428571, 0.7317142857, 0.5954285714], [0.3021714286, 0.7376047619, 
            0.5711857143], [0.3481666667, 0.7424333333, 0.5472666667], 
            [0.3952571429, 0.7459, 0.5244428571], [0.4420095238, 0.7480809524, 
            0.5033142857], [0.4871238095, 0.7490619048, 0.4839761905], 
            [0.5300285714, 0.7491142857, 0.4661142857], [0.5708571429, 0.7485190476, 
            0.4493904762], [0.609852381, 0.7473142857, 0.4336857143], 
            [0.6473, 0.7456, 0.4188], [0.6834190476, 0.7434761905, 0.4044333333], 
            [0.7184095238, 0.7411333333, 0.3904761905], 
            [0.7524857143, 0.7384, 0.3768142857], [0.7858428571, 0.7355666667, 
            0.3632714286], [0.8185047619, 0.7327333333, 0.3497904762], 
            [0.8506571429, 0.7299, 0.3360285714], [0.8824333333, 0.7274333333, 0.3217], 
            [0.9139333333, 0.7257857143, 0.3062761905], [0.9449571429, 0.7261142857, 
            0.2886428571], [0.9738952381, 0.7313952381, 0.266647619], 
            [0.9937714286, 0.7454571429, 0.240347619], [0.9990428571, 0.7653142857, 
            0.2164142857], [0.9955333333, 0.7860571429, 0.196652381], 
            [0.988, 0.8066, 0.1793666667], [0.9788571429, 0.8271428571, 0.1633142857], 
            [0.9697, 0.8481380952, 0.147452381], [0.9625857143, 0.8705142857, 0.1309], 
            [0.9588714286, 0.8949, 0.1132428571], [0.9598238095, 0.9218333333, 
            0.0948380952], [0.9661, 0.9514428571, 0.0755333333], 
            [0.9763, 0.9831, 0.0538]]

        # Build a continuous colormap from your sampled parula fragment
        parula_cm = LinearSegmentedColormap.from_list("parula_custom", PARULA_COLORS)

        # Sample N uniform colors across the *full* range (includes yellow)
        PARULA_12 = parula_cm(np.linspace(0, 1, 12))

        # Discrete colormap with exactly 12 levels
        BASE_CMAP = LinearSegmentedColormap.from_list("parula12", PARULA_12, N=10)

    elif colormap == "default":
        BASE_CMAP = plt.get_cmap("tab10")

    elif colormap == "custom":
        BASE_CMAP = EP_VOLCANO_CMAP

    else:
        BASE_CMAP = EP_VOLCANO_CMAP

    # Colors per order
    colors = [BASE_CMAP(t) for t in np.linspace(0, 1, n_orders)]

    # ====================================================
    # MODE 1: all orders in a single panel
    # ====================================================
    if mode == "single":
        if ax is None:
            fig, ax = plt.subplots(figsize=(6, 5), **fig_kwargs)

        for j, o in enumerate(orders_unique):
            sub = df[df["Order"] == o]

            # mask for significance
            if alpha is not None:
                sig_mask = sub[p_col] <= alpha
            else:
                sig_mask = np.ones(len(sub), dtype=bool)

            # --- NON SIGNIFICANT points (gray) ---
            ax.scatter(
                sub[effect_col][~sig_mask],
                sub["neg_log10_p"][~sig_mask],
                s=20,
                alpha=0.35,
                color="0.7",
            )

            # --- SIGNIFICANT points (colored by order) ---
            ax.scatter(
                sub[effect_col][sig_mask],
                sub["neg_log10_p"][sig_mask],
                s=22,
                alpha=0.75,
                color=colors[j],
                label=f"{o}",
            )

        # -----------------------------
        # SIGNIFICANCE THRESHOLD LINE
        # -----------------------------
        if alpha is not None:
            y_thr = -np.log10(alpha)
            ax.axhline(y_thr, linestyle="--", linewidth=1.5, alpha=0.7, color="black")

            # LABEL: p ≤ threshold
            trans_thr = blended_transform_factory(ax.transAxes, ax.transData)
            ax.text(
                0.99,
                y_thr,
                thr_label,                     # e.g. "p ≤ 0.05"
                ha="right",
                va="bottom",
                fontsize=12,
                color="black",
                alpha=0.85,
                transform=trans_thr,
                bbox=dict(
                    boxstyle="round,pad=0.2",
                    fc="white",
                    ec="none",
                    alpha=0.5,
                ),
            )

            # -----------------------------
            # CLIPPING LEVEL (dotted gray)
            # -----------------------------
            y_clip = -np.log10(clip)

            if clip < alpha:
                ax.axhline(
                    y_clip,
                    color="0.7",
                    linestyle=":",
                    linewidth=2,
                    alpha=0.8,
                )

                # text label "clip"
                trans_clip = blended_transform_factory(ax.transAxes, ax.transData)
                ax.text(
                    0.99,
                    y_clip,
                    "clip",
                    ha="right",
                    va="bottom",
                    fontsize=16,
                    color="0.5",
                    alpha=0.85,
                    transform=trans_clip,
                    bbox=dict(
                        boxstyle="round,pad=0.15",
                        fc="white",
                        ec="none",
                        alpha=0.5,
                    ),
                )

        # -----------------------------
        # AXIS LABELS AND LEGEND
        # -----------------------------
        ax.set_xlabel(r"$\hat{\epsilon}_{\mathbf{s}}(\mathbf{x})$")
        ax.set_ylabel(y_label)
        ax.legend(
            title="Interaction order",
            frameon=False,
            loc="center left",
            bbox_to_anchor=(1.02, 0.5),
            borderaxespad=0,
        )

    # ====================================================
    # MODE 2: one panel per order, dynamic layout + % significant
    # ====================================================
    else:  # mode == "by-order"

        if ax is None:
            if n_orders <= 4:
                nrows = 1
                ncols = n_orders
            else:
                nrows = 2
                ncols = int(np.ceil(n_orders / 2))

            _, axes = plt.subplots(
                nrows,
                ncols,
                sharey=True,
                figsize=(4 * ncols, 4 * nrows),
                **fig_kwargs,
            )
        else:
            axes = np.atleast_1d(ax)
            if axes.ndim > 1:
                axes = axes.ravel()
            if len(axes) < n_orders:
                raise ValueError(
                    f"Provided ax has {len(axes)} axes, "
                    f"but {n_orders} are needed for 'by-order' mode."
                )

        axes = np.atleast_1d(axes).ravel()

        for j, o in enumerate(orders_unique):
            sub = df[df["Order"] == o]
            ax_j = axes[j]

            # Significant mask
            sig_mask = sub[p_col] <= alpha if alpha is not None else np.ones(len(sub), bool)

            # Plot non-significant in gray
            ax_j.scatter(
                sub[effect_col][~sig_mask],
                sub["neg_log10_p"][~sig_mask],
                s=20,
                alpha=0.35,
                color="0.7",
            )

            # Plot significant in color
            ax_j.scatter(
                sub[effect_col][sig_mask],
                sub["neg_log10_p"][sig_mask],
                s=22,
                alpha=0.75,
                color=colors[j],
            )

            # -----------------------------
            # SIGNIFICANCE LINE
            # -----------------------------
            if alpha is not None:
                y_thr = -np.log10(alpha)
                ax_j.axhline(y_thr, linestyle="--", linewidth=2.2, alpha=0.7, color="black")

                trans_thr = blended_transform_factory(ax_j.transAxes, ax_j.transData)
                ax_j.text(
                    0.99,
                    y_thr,
                    thr_label,
                    va="bottom",
                    ha="right",
                    fontsize=12,
                    alpha=0.8,
                    color="black",
                    transform=trans_thr,
                    bbox=dict(
                        boxstyle="round,pad=0.15",
                        fc="white",
                        ec="none",
                        alpha=0.6,
                    ),
                )

                # -----------------------------
                # CLIP LINE AND LABEL
                # -----------------------------
                y_clip = -np.log10(clip)
                if clip < alpha:
                    ax_j.axhline(
                        y_clip,
                        color="0.7",
                        linestyle=":",
                        linewidth=2,
                        alpha=0.8,
                    )

                    trans_clip = blended_transform_factory(ax_j.transAxes, ax_j.transData)
                    ax_j.text(
                        0.99,
                        y_clip,
                        "clip",
                        va="bottom",
                        ha="right",
                        fontsize=16,
                        alpha=0.8,
                        color="0.5",
                        transform=trans_clip,
                        bbox=dict(
                            boxstyle="round,pad=0.15",
                            fc="white",
                            ec="none",
                            alpha=1,
                        ),
                    )

                # % significant
                n_total = len(sub)
                if n_total > 0:
                    n_sig = np.sum(sub[p_col] <= alpha)
                    frac_sig = 100.0 * n_sig / n_total
                    ax_j.text(
                        0.05,
                        0.93,
                        f"{frac_sig:.1f}% sig. ({thr_label})",
                        transform=ax_j.transAxes,
                        va="top",
                        ha="left",
                        fontsize=13,
                    )

            ax_j.set_title(f"Order {o}", fontsize=20)

            # X label
            if ax is None:
                row = j // ncols
                col = j % ncols
                if row == nrows - 1:
                    ax_j.set_xlabel(r"$\hat{\epsilon}_{\mathbf{s}}(\mathbf{x})$", fontsize=17)
                else:
                    ax_j.set_xlabel("")
            else:
                ax_j.set_xlabel(r"$\hat{\epsilon}_{\mathbf{s}}(\mathbf{x})$", fontsize=17)

            # Y label: only leftmost col
            if ax is None:
                col = j % ncols
                if col == 0:
                    ax_j.set_ylabel(y_label, fontsize=15)
                else:
                    ax_j.set_ylabel("")
            else:
                ax_j.set_ylabel(y_label if j == 0 else "")

        # Turn off unused axes
        for k in range(len(orders_unique), len(axes)):
            axes[k].axis("off")

        plt.tight_layout()

    if return_data:
        cols_keep = [
            c for c in df.columns
            if c in {
                "Interaction type",
                "Background",
                "Background loci",
                "Background loci names",
                "Background active loci",
                "Background active names",
                "Order",
                "Loci involved",
                "Loci names",
                effect_col,
                p_col,
                "neg_log10_p",
            }
        ]
        return df[cols_keep].copy()

    return None
