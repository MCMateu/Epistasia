# epistasia/testing.py

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from .synlands import GaussianRandomConfig
from .core import Landscape
from .walshhadamard import epistasis_amplitude



def test_gaussian_bootstrap_consistency(
    N: int,
    R: int,
    *,
    sigma: float = 1.0,
    B_wild: int = 500,
    B_param: int = 500,
    ci_level: float = 0.95,
    seed_landscape: int = 42,
    seed_wild: int = 125,
    seed_param: int = 777,
    show: bool = True,
):
    """
    Compare wild null bootstrap vs parametric null bootstrap for the
    epistasis amplitude ⟨E²⟩_S in a Gaussian iid synthetic landscape.

    Both methods estimate the *null* distribution:
      - wild null: B_null > 0 on a single synthetic landscape
      - parametric null: many synthetic landscapes resampled from N(0, sigma^2)

    Parameters
    ----------
    N : int
        Number of loci (2^N configurations).
    R : int
        Number of biological replicates.
    sigma : float, default=1.0
        Std dev of the iid Gaussian generative model.
    B_wild : int, default=500
        Number of null bootstrap samples in epistasis_amplitude.
    B_param : int, default=500
        Number of parametric bootstrap samples (synthetic landscapes).
    ci_level : float, default=0.95
        Confidence interval level.
    """

    alpha = 1.0 - ci_level
    low_q = 100.0 * (alpha / 2.0)
    high_q = 100.0 * (1.0 - alpha / 2.0)

    # --------------------------------------------------------
    # 1) One synthetic iid Gaussian landscape
    # --------------------------------------------------------
    cfg = GaussianRandomConfig(
        N=N,
        R=R,
        seed=seed_landscape,
        mean=0.0,
        sigma=sigma,
        batch=False,
    )
    L, meta = cfg.sample()

    # Optional: set WT to zero
    L.values[0, :] = 0.0

    # --------------------------------------------------------
    # 2) Wild *null* bootstrap via epistasis_amplitude
    # --------------------------------------------------------
    rng_wild = np.random.default_rng(seed_wild)

    df_wild = epistasis_amplitude(
        L,
        B_uncertainty=0,
        B_null=B_wild,
        ci_level=ci_level,
        multipliers="normal",
        uncertainty_flavor="iid",     # all iid for this test
        preserve_residual_corr=False,      # no biological correlations
        as_dataframe=True,
        rng=rng_wild,
    )

    S = df_wild["Order"].values
    A_obs = df_wild["Epistasis amplitude <E^2>_k"].values
    A_low_w = df_wild["Null CI low"].values
    A_high_w = df_wild["Null CI high"].values

    # --------------------------------------------------------
    # 3) Parametric *null* bootstrap: many iid Gaussian landscapes
    # --------------------------------------------------------
    rng_param = np.random.default_rng(seed_param)
    M = 1 << N  # number of configurations
    n_orders = len(S)

    amps_param = np.empty((B_param, n_orders), dtype=float)

    for b in range(B_param):
        F_b = rng_param.normal(loc=0.0, scale=sigma, size=(M, R))
        L_b = Landscape(states=L.states, values=F_b, N=N, R=R)
        L_b.values[0, :] = 0.0

        df_b = epistasis_amplitude(
            L_b,
            B_uncertainty=0,
            B_null=0,
            ci_level=ci_level,        # irrelevant here
            multipliers="normal",
            uncertainty_flavor="iid",
            preserve_residual_corr=False,
            as_dataframe=True,
        )
        amps_param[b, :] = df_b["Epistasis amplitude <E^2>_k"].values

    A_param_mean = amps_param.mean(axis=0)
    A_param_low = np.percentile(amps_param, low_q, axis=0)
    A_param_high = np.percentile(amps_param, high_q, axis=0)

    df_param = pd.DataFrame(
        {
            "Order": S,
            "Parametric mean": A_param_mean,
            "Parametric CI low": A_param_low,
            "Parametric CI high": A_param_high,
        }
    )

    # --------------------------------------------------------
    # 4) Single-panel comparison
    # --------------------------------------------------------
    if show:
        fig, ax = plt.subplots(figsize=(6, 5))

        # Wild null CI
        ax.fill_between(
            S,
            A_low_w,
            A_high_w,
            color="tab:blue",
            alpha=0.30,
            label="Wild null bootstrap (95% CI)",
        )

        # Parametric null CI (hatched)
        ax.fill_between(
            S,
            A_param_low,
            A_param_high,
            facecolor="none",
            edgecolor="gray",
            hatch="///",
            linewidth=0.0,
            label="Parametric null bootstrap (95% CI)",
        )

        # Observed amplitude from the initial landscape (optional)
        ax.plot(
            S,
            A_obs,
            "o-",
            color="tab:red",
            label="Observed amplitude (one realization)",
        )

        ax.set_yscale("log")
        ax.yaxis.set_minor_locator(plt.NullLocator()) 
        ax.set_xlabel("Interaction order "+r"$k$")
        ax.set_ylabel("Epistasis amplitude "+r"$\langle\mathcal{E}^2\rangle_k$")
        #ax.grid(alpha=0.3)
        ax.legend(frameon=False)
        ax.set_title(f"Gaussian iid null: wild vs parametric (σ={sigma}, R={R})")

        #if show:
            #fig.tight_layout()
            #plt.show()

    return df_wild, df_param
