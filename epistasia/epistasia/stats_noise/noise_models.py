##########################################
#          REQUIRED LIBRARIES            #
##########################################
from __future__ import annotations

import os
import numpy as np
from math import comb
from typing import Tuple, Optional, Dict, Any, TYPE_CHECKING
from cmdstanpy import CmdStanModel
import pandas as pd

if TYPE_CHECKING:
    from .core import Landscape  # only for typing, no runtime import


######################################################
#               STATSITIC / NOISE MODULE             #
######################################################


######################################################
#              BATCH EFFECT CORRECTION               #
######################################################

# -------------------- Stan compile (simple cache) --------------------

_MODEL_CACHE: Dict[str, CmdStanModel] = {}

def _stan_model() -> CmdStanModel:
    """
    Compile once and cache the batch_effect Stan model placed under epistasia/sn/.
    """
    key = "batch_effect"
    if key in _MODEL_CACHE:
        return _MODEL_CACHE[key]
    here = os.path.dirname(__file__)
    stan_file = os.path.join(here, "batch_effect.stan")  # <-- nombre corregido
    if not os.path.exists(stan_file):
        raise FileNotFoundError(
            f"Stan file not found at: {stan_file}. "
            "Place 'batch_effect.stan' under 'epistasia/sn/'."
        )
    model = CmdStanModel(stan_file=stan_file)
    _MODEL_CACHE[key] = model
    return model



# -------------------- Public API --------------------

def correct_batch_effect(
    L: Landscape,
    lambda_a: float = 1.0,   # kept for backward compatibility (unused)
    lambda_b: float = 1.0,   # kept for backward compatibility (unused)
    return_posteriors: bool = False,
    seed: Optional[int] = None,
    chains: int = 4,
    parallel_chains: Optional[int] = None,
    iter_warmup: int = 1000,
    iter_sampling: int = 1000,
    link: str = "log",
    log_epsilon: float = 1e-3,
    adjust_zero_state: bool = False,
    **sample_kwargs: Any,
) -> Landscape | Tuple[Landscape, Dict[str, np.ndarray]]:
    """
    Remove batch effects using a hierarchical Bayesian model defined directly
    on the observed landscape F_obs[x, r] and return a corrected Landscape.

    Model (Stan side)
    -----------------
        F_obs[x, r] ~ Normal(F_bar[x] + a[r], sigma * b[r])

    where:
        - F_bar[x] is the true underlying landscape (biological signal),
        - a[r] are additive batch effects with sum_r a[r] = 0,
        - b[r] are multiplicative batch effects with geommean_r b[r] = 1,
        - sigma is a global noise scale shared across configurations.

    Parameters
    ----------
    L : Landscape
        Input landscape with shape (X, R) in L.values (X = 2^N configs, R replicates).
    lambda_a, lambda_b : float
        Deprecated / unused in the new model. Kept only for backward compatibility.
    return_posteriors : bool
        If True, also return a dict with posterior draws.
    seed : int, optional
        Random seed for sampling.
    chains : int
        Number of MCMC chains.
    parallel_chains : int, optional
        Parallel chains (defaults to min(chains, os.cpu_count())).
    iter_warmup : int
        Warmup iterations.
    iter_sampling : int
        Sampling iterations.
    link : {"identity", "log"}
        Link function used on F before fitting the batch model.
        If "log", values of F must be >= 0 and are transformed as
        log(F + log_epsilon).
    log_epsilon : float
        Positive offset used with link="log" so that zeros in F are allowed
        (they map to log(log_epsilon)). Must be > 0 when link="log".
    adjust_zero_state : bool, default True
        If False, the all-zero configuration (state x = (0,...,0)) is kept at its
        original value after batch correction, provided it was exactly zero in all
        replicates before correction. If True, the zero state is corrected like any
        other configuration (backward-compatible behaviour).

    **sample_kwargs :
        Extra args passed to CmdStanModel.sample (e.g., adapt_delta, max_treedepth).

    Returns
    -------
    Landscape
        Batch-corrected landscape with same states/N/R as input.
    (Landscape, dict) if return_posteriors=True
        The dict contains numpy arrays:
            post['a']         -> shape (draws, R)
            post['b']         -> shape (draws, R)
            post['F_bar']     -> shape (draws, X)
            post['log_sigma'] -> shape (draws,)
            post['sigma']     -> shape (draws,)
    """

    # Local import to avoid circular import during package initialization
    from ..core import Landscape

    # --- Extract raw matrix (X, R) ---
    F = np.asarray(L.values, dtype=float)
    if F.ndim != 2:
        raise ValueError("Landscape.values must be a 2D array of shape (X, R).")
    X, R = F.shape

    # --- Identify the all-zero state configuration x = (0,...,0) ---
    #    and store its original values, if we might want to keep it fixed later.
    states = np.asarray(L.states)
    if states.shape[0] != X:
        raise ValueError("L.states and L.values have inconsistent number of rows.")
    zero_state_mask = np.all(states == 0, axis=1)  # True only for x = 0...0
    # Only store original values if that state exists and is exactly zero in all reps
    if np.any(zero_state_mask) and np.all(F[zero_state_mask, :] == 0.0):
        F_zero_original = F[zero_state_mask, :].copy()  # shape (n_zero, R), usually 1×R
    else:
        F_zero_original = None

    # --- Optional link transform ---
    if link not in ("identity", "log"):
        raise ValueError("link must be 'identity' or 'log'.")

    if link == "log":
        if log_epsilon <= 0:
            raise ValueError(
                "log_epsilon must be > 0 when link='log', so that F + log_epsilon > 0 "
                "even if some entries of F are zero."
            )
        # Allow zeros but forbid negative values
        if np.any(F < 0):
            raise ValueError(
                "All entries of F must be >= 0 when link='log'. "
                "Use link='identity' for signed / real-valued landscapes."
            )
        # Log-transform with a small offset: zeros map to log(log_epsilon)
        F_model = np.log(F + log_epsilon)
    else:
        F_model = F

    # --- Prepare data for Stan (model defined on F_obs, not on Z) ---
    data = {
        "M": int(X),
        "R": int(R),
        "F_obs": F_model,
    }

    model = _stan_model()
    if parallel_chains is None:
        parallel_chains = min(chains, os.cpu_count() or 1)

    fit = model.sample(
        data=data,
        seed=seed,
        chains=chains,
        parallel_chains=parallel_chains,
        iter_warmup=iter_warmup,
        iter_sampling=iter_sampling,
        **sample_kwargs,
    )

    # --- Extract posterior draws ---
    # Stan parameters are assumed to be:
    #   vector[M] F_bar;
    #   vector[R] a;
    #   vector[R] b;
    #   real      log_sigma;
    a_draws = fit.stan_variable("a")          # shape: (draws, R)
    b_draws = fit.stan_variable("b")          # shape: (draws, R)
    F_bar_draws = fit.stan_variable("F_bar")  # shape: (draws, X)
    log_sigma_draws = fit.stan_variable("log_sigma")  # shape: (draws,)

    # Posterior means as point estimates
    a_bar = a_draws.mean(axis=0)            # (R,)
    b_bar = b_draws.mean(axis=0)            # (R,)
    F_bar_hat = F_bar_draws.mean(axis=0)    # (X,)
    sigma_draws = np.exp(log_sigma_draws)   # (draws,)

    # --- Batch-effect correction in model space ---
    # Model (in transformed space):
    #   F_model[x, r] = F_bar[x] + a[r] + b[r] * pi[x, r]
    # Estimate residual noise:
    #   pi_hat[x, r] = (F_model[x, r] - F_bar_hat[x] - a_bar[r]) / b_bar[r]
    # Then define corrected values with b = 1 (no multiplicative batch effect):
    #   F_model_adj[x, r] = F_bar_hat[x] + pi_hat[x, r]
    residual_pi_hat = (F_model - F_bar_hat[:, None] - a_bar[None, :]) / b_bar[None, :]
    F_model_adj = F_bar_hat[:, None] + residual_pi_hat

    # --- Inverse transform to original scale ---
    if link == "log":
        F_adj = np.exp(F_model_adj)
    else:
        F_adj = F_model_adj

    # --- Optionally keep the all-zero state fixed at its original value ---
    if (F_zero_original is not None) and (not adjust_zero_state):
        # Restore original values for the all-zero state configuration
        F_adj[zero_state_mask, :] = F_zero_original

    # --- Build corrected Landscape (preserve metadata) ---
    L_corr = Landscape(
        L.states,
        F_adj,
        N=getattr(L, "N", None),
        R=getattr(L, "R", R),
        order=getattr(L, "order", "lex"),
        feature_names=getattr(L, "feature_names", None),
    )

    if return_posteriors:
        post = {
            "a": a_draws,
            "b": b_draws,
            "F_bar": F_bar_draws,
            "log_sigma": log_sigma_draws,
            "sigma": sigma_draws,
        }
        return L_corr, post

    return L_corr



######################################################
#                     HELPERS                        # 
######################################################

def residuals_and_covariance(L: Landscape) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute replicate-level residuals P and the empirical noise covariance \hat C.

    P_{x,r} = \tilde F_r(x) - \hat F(x),    \hat F(x) = mean_r \tilde F_r(x)
    C_hat = (P @ P.T) / (R - 1)

    Requires
    --------
    L.values shape (M,R). NaNs are allowed; rows with any NaN are skipped
    for covariance to avoid bias.

    Returns
    -------
    P : np.ndarray, shape (M_clean, R)
        Residuals for rows without NaNs.
    C_hat : np.ndarray, shape (M_clean, M_clean)
        Empirical covariance across states (rows correspond to the kept states).
    """
    # For stable covariance, drop rows with any NaNs
    Lc = L.drop_rows_with_any_nan()
    V = Lc.values  # (M_clean, R)
    Fhat = np.mean(V, axis=1, keepdims=True)  # (M_clean,1)
    P = V - Fhat
    R = Lc.R
    if R < 2:
        raise ValueError("At least 2 replicates required to estimate covariance.")
    C_hat = (P @ P.T) / (R - 1)
    return P, C_hat

######################

def _phi_matrix(states: np.ndarray, N: int) -> np.ndarray:
    """
    Build the matrix Phi of size (M, 2^N) whose columns are Walsh-like
    basis functions φ_s(x) = ∏_{i: s_i=1} z_i(x),
    with shifted spins z_i in {-1, +1} defined from binary states {0,1} by:
        z_i = +1 if state_i == 1, and -1 otherwise.

    Parameters
    ----------
    states : (M, N) array of {0,1}
        Observed genotypes/configurations.
    N : int
        Number of loci (bits).

    Returns
    -------
    Phi : (M, 2^N) float array
        Column s corresponds to φ_s evaluated on all rows x in 'states'.
        Lexicographic order of s (0..2^N-1) is assumed.
    """
    # Convert to spins in {-1, +1}
    Z = np.where(states == 1, 1.0, -1.0)  # (M, N)
    M = states.shape[0]
    K = 1 << N

    # Pre-allocate Phi
    Phi = np.empty((M, K), dtype=float)

    # Column s = 0 corresponds to the constant function φ_∅(x) = 1
    Phi[:, 0] = 1.0

    # For efficiency, accumulate products by reusing previously computed columns:
    # each s can be built from s' removing its least significant 1-bit.
    for s in range(1, K):
        # isolate least significant 1-bit
        lsb = s & -s
        s_prev = s ^ lsb
        # position of that bit (0-based from the right). We want index in [0..N-1] from left.
        bit_pos_from_right = (lsb.bit_length() - 1)
        i = N - 1 - bit_pos_from_right
        Phi[:, s] = Phi[:, s_prev] * Z[:, i]

    return Phi

########################################################
#                   NOISE PROJECTIONS                  #
########################################################

###################################
# Core projection (pure function) #
###################################

def project_noise_var_WH(
    C_hat: np.ndarray,
    states: np.ndarray,
    N: int,
    orders: np.ndarray,
) -> Tuple[np.ndarray, Dict[int, float]]:
    """
    Project the replicate covariance C_hat onto the Walsh-Hadamard (WH) basis.

    It computes, for each WH mode s (s=0..2^N-1),
        Var[η_s] = 2^{-2N} * φ_s^T C_hat φ_s,
    where φ_s is the column of Phi corresponding to mode s,
    and returns order-averaged K_S = <Var[η_s]>_{|s|=S} for S>=1.

    Parameters
    ----------
    C_hat : (M, M) float array
        Empirical covariance of replicate residuals across the M observed states.
        Must be aligned with 'states' rows (i.e., computed after dropping NaNs).
    states : (M, N) int/bool array in {0,1}
        The observed states used to build C_hat.
    N : int
        Number of loci (bits).
    orders : (2^N,) int array
        Hamming weight |s| for each WH mode index s in [0..2^N-1].
        (Use the same ordering as your WH transform.)

    Returns
    -------
    var_eta_s : (2^N,) float array
        Per-mode noise variance Var[η_s].
    K_S : dict[int, float]
        Order-averaged noise variance per order S (S>=1).
        K_S[S] = mean_s Var[η_s] over all s with |s|=S.
    """
    # Build Phi (M x 2^N)
    Phi = _phi_matrix(states, N)  # columns are φ_s(x)

    # Quadratic form diag( Phi^T C_hat Phi ) without forming the full product:
    # q_s = φ_s^T C_hat φ_s
    # einsum "ms,mt,st->s" uses:
    #  - ms: Phi
    #  - mt: Phi
    #  - st: C_hat  (s,t here denote state indices; name clash with 's' (mode) is harmless)
    q = np.einsum("ms,mt,st->s", Phi, Phi, C_hat, optimize=True)

    # Scale by 2^{-2N}
    var_eta_s = q / float(1 << (2 * N))

    # Order-averaged K_S for S >= 1 (exclude S=0 / constant)
    K_S: Dict[int, float] = {}
    unique_orders = np.unique(orders)
    for S in unique_orders:
        if S == 0:
            continue
        mask = (orders == S)
        # numeric safety if mask is empty
        if np.any(mask):
            K_S[int(S)] = float(np.mean(var_eta_s[mask]))

    return var_eta_s, K_S

##################################
#    Correct variance spectrum   #
##################################

def correct_variance_spectrum(V_S_obs: dict[int, np.ndarray], K_S: dict[int, float], N: int):
    """
    Return V_bar(S) ≈ V_obs(S) - C(N,S) * K_S, clipped at 0.
    """
    V_corr = {}
    for S, v in V_S_obs.items():
        noise = comb(N, S) * K_S.get(S, 0.0)
        V_corr[S] = np.clip(v - noise, a_min=0.0, a_max=None)
    return V_corr

###################################################################
#                           BOOTSTRAPS                            #
###################################################################

def bootstrap_null(
    F_df,
    B=1000,
    multipliers="normal",     # {"rademacher","normal"}
    preserve_residual_corr=False,      # True  -> keep cross-state correlations (wild-cluster)
                                  # False -> break correlations across states (iid-by-state)
    rng=None,
):
    """
    Null bootstrap for full fitness landscape (mean-zero synthetic landscapes).

    Concept
    -------
    Builds mean-zero bootstrap landscapes F^(b) = beta^(b) from replicate-level residuals.
    - If preserve_residual_corr=True: PRESERVES empirical cross-state covariance (wild-cluster).
    - If preserve_residual_corr=False: DESTROYS cross-state correlations (i.i.d. across states) by
      reshuffling residuals row-wise (per replicate) before forming the wild perturbation.

    Parameters
    ----------
    F_df : pandas.DataFrame
        First N columns = binary states, remaining columns = replicate fitness values.
    B : int
        Number of bootstrap draws.
    multipliers : {"rademacher","normal"}
        Type of bootstrap multipliers. Rademacher is robust to asymmetries.
    preserve_residual_corr : bool
        Whether to preserve biological/experimental correlations across states.
    rng : np.random.Generator or None
        Random generator. If None, a new Generator is created.

    Returns
    -------
    F_boot : pandas.DataFrame
        First N columns identical to F_df.
        Then B columns = bootstrap replicates of the NULL landscape (mean zero).
    """
    if rng is None:
        rng = np.random.default_rng()

    vals = F_df.values
    M, C = vals.shape  # M = number of states (2**N), C = N + R

    # Infer N = number of binary columns (all entries 0/1 until first non-binary column)
    N_guess = 0
    for j in range(C):
        if np.isin(vals[:, j], [0, 1]).all():
            N_guess += 1
        else:
            break
    N = N_guess

    # Replicate matrix X: shape (M, R)
    X = vals[:, N:]
    R = X.shape[1]
    if R < 2:
        raise ValueError("At least 2 replicates required for wild bootstrap.")

    # Row means and residuals (row-centered)
    F_bar = X.mean(axis=1, keepdims=True)  # (M, 1)
    Xi = X - F_bar                         # (M, R), residuals per replicate

    # Output container: each column is one null bootstrap draw (mean-zero landscape)
    F_boot_mat = np.empty((M, B), dtype=float)

    for b in range(B):
        # Draw wild multipliers
        if multipliers == "normal":
            w = rng.standard_normal(R)
        elif multipliers == "rademacher":
            w = rng.choice([-1.0, 1.0], size=R)
        else:
            raise ValueError("multipliers must be 'normal' or 'rademacher'")

        # Possibly break cross-state correlations (iid null)
        if preserve_residual_corr:
            # Keep empirical covariance structure across states (wild-cluster)
            Xi_eff = Xi
        else:
            # Destroy cross-state correlations:
            # independently permute the rows of each replicate column
            # (equivalently, sample with replacement over states).
            Xi_eff = np.empty_like(Xi)
            for r in range(R):
                Xi_eff[:, r] = Xi[rng.permutation(M), r]

        # Wild perturbation beta^(b) (mean-zero up to sampling noise)
        beta = (Xi_eff @ w) / np.sqrt(R * (R - 1))   # shape (M,)

        # Enforce exact zero-mean across states (pure null, no global level)
        beta -= beta.mean()

        # Null landscape: F^(b) = beta^(b)
        F_boot_mat[:, b] = beta

    # Build output DataFrame
    state_cols = list(F_df.columns[:N])
    boot_cols = [f"boot_{b+1}" for b in range(B)]
    F_boot = pd.DataFrame(
        np.hstack([vals[:, :N], F_boot_mat]),
        columns=state_cols + boot_cols
    )
    return F_boot


def bootstrap_uncertainty(
    F_df,
    B=1000,
    multipliers="normal",   # {"rademacher","normal"}
    flavor="iid",               # "iid" -> break cross-state correlations; "wildcluster" -> keep empirical C
    rng=None,
):
    """
    Uncertainty bootstrap for full fitness landscape (around the observed mean).

    Concept
    -------
    Builds mean-zero perturbations beta^(b) from replicate-level residuals and
    returns F^(b) = Fhat + beta^(b), where Fhat is the row-wise mean across replicates.
      - flavor="wildcluster": PRESERVES empirical cross-state covariance (wild-cluster).
      - flavor="iid":        DESTROYS cross-state correlations (i.i.d. across states)
                             by reshuffling residuals row-wise per replicate before
                             forming the wild perturbation.

    Parameters
    ----------
    F_df : pandas.DataFrame
        First N columns = binary states, remaining columns = replicate fitness values.
    B : int
        Number of bootstrap draws.
    multipliers : {"rademacher","normal"}
        Type of bootstrap multipliers. Rademacher is robust to asymmetries.
    flavor : {"iid","wildcluster"}
        Whether to preserve ("wildcluster") or break ("iid") cross-state correlations.
    rng : np.random.Generator or None
        Random generator. If None, a new Generator is created.

    Returns
    -------
    F_boot : pandas.DataFrame
        First N columns identical to F_df.
        Then B columns = bootstrap replicates of the landscape (uncertainty around Fhat).

    Notes
    -----
    - Fhat is the row-wise mean (per state) across replicates.
    - Wild-cluster uses beta^(b) = [1/sqrt(R(R-1))] * Xi @ w, where Xi are residuals
      (row-centered replicates) and w are wild multipliers.
    - For "iid", residuals are reshuffled row-wise (per replicate) before the same formula.
    """
    
    if rng is None:
        rng = np.random.default_rng()

    vals = F_df.values
    M, C = vals.shape  # M = number of states (2**N), C = N + R

    # Infer N = number of binary columns (all entries 0/1 until first non-binary column)
    N_guess = 0
    for j in range(C):
        if np.isin(vals[:, j], [0, 1]).all():
            N_guess += 1
        else:
            break
    N = N_guess

    # Replicate matrix X: shape (M, R)
    X = vals[:, N:]
    R = X.shape[1]
    if R < 2:
        raise ValueError("At least 2 replicates required for wild bootstrap.")

    # Row means (Fhat) and residuals (row-centered)
    Fhat = X.mean(axis=1)              # shape (M,)
    Xi = X - Fhat[:, None]             # shape (M, R)

    # Output container: each column is one uncertainty bootstrap draw around Fhat
    F_boot_mat = np.empty((M, B), dtype=float)

    for b in range(B):
        # Draw wild multipliers
        if multipliers == "normal":
            w = rng.standard_normal(R)
        elif multipliers == "rademacher":
            w = rng.choice([-1.0, 1.0], size=R)
        else:
            raise ValueError("multipliers must be 'normal' or 'rademacher'")

        # Possibly break cross-state correlations (iid uncertainty)
        if flavor == "wildcluster":
            Xi_eff = Xi
        elif flavor == "iid":
            # Destroy cross-state correlations: independently permute rows per replicate
            Xi_eff = np.empty_like(Xi)
            for r in range(R):
                Xi_eff[:, r] = Xi[rng.permutation(M), r]
        else:
            raise ValueError("flavor must be 'iid' or 'wildcluster'")

        # Wild perturbation beta^(b) (mean ~ 0 across states up to sampling noise)
        beta = (Xi_eff @ w) / np.sqrt(R * (R - 1))   # shape (M,)

        # Build uncertainty sample around Fhat
        F_b = Fhat + beta

        # Optional: enforce exact global-mean preservation (keeps the same grand mean as Fhat)
        F_b -= (F_b.mean() - Fhat.mean())

        F_boot_mat[:, b] = F_b

    # Build output DataFrame
    state_cols = list(F_df.columns[:N])
    boot_cols = [f"boot_{b+1}" for b in range(B)]
    F_boot = pd.DataFrame(
        np.hstack([vals[:, :N], F_boot_mat]),
        columns=state_cols + boot_cols
    )
    return F_boot

