"""
Epistasia: A toolkit for the analysis of binary genotype–phenotype maps.

Submodule aliases:
    ep.core  → core.py                  (CORE)
    ep.wh    → walshhadamard.py         (Walsh–Hadamard)
    ep.epi   → epistasis.py             (Epistasis)
    ep.sn    → stats_noise/             (Stats/Noise package)
    ep.syn   → synlands.py              (Synthetic landscapes)
"""

from importlib import import_module
from typing import Any

# --- Lazy submodule import -------------------------------------------------------
def __getattr__(name: str) -> Any:
    """
    Lazily import heavy submodules to avoid circular dependencies.
    """
    if name == "core":
        return import_module(".core", __name__)
    if name == "wh":
        return import_module(".walshhadamard", __name__)
    if name == "epi":
        return import_module(".epistasis", __name__)
    if name == "sn":
        return import_module(".stats_noise", __name__)
    if name == "syn":
        return import_module(".synlands", __name__)
    if name == "eglobal":
        return import_module(".eglobal", __name__)   
    if name == "fees":
        return import_module(".fees", __name__)
    if name == "io":
        return import_module(".io", __name__)

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    
def __dir__():
    """Expose dynamic submodules and public attributes for IDE autocompletion."""
    return sorted(list(globals().keys()) + [
        "core", "wh", "epi", "sn", "syn",
        "fees",
        # Core
        "Landscape",
        # IO
        "read_table", "landscape_from_file",
        # WH
        "wh_transform", "fwht_shifted",
        "variance_spectrum", "variance_decomposition",
        "_binary_domain", "mode_orders", "states_to_spins_shifted",
        "walsh_analysis", "epistasis_amplitude", "variance_by_order_statistics",
        "plot_variance_and_amplitude",
        "plot_walsh_volcano",
        # Noise / Statistics (top-level re-exports)
        "residuals_and_covariance",
        "correct_variance_spectrum", "project_noise_var_WH",
        "bootstrap_uncertainty", "bootstrap_null",
        "correct_batch_effect"
        # Epistasis (deterministic layer)
        "focal_effect", "epistasis_k", "epistasis_pairs",
        "epistasis_distributions_by_order", "epistasis_symbolic",
        "compute_full_epistasis",  "filter_significant_interactions",
        "epistasis_to_network", "plot_epistasis_network",
        "epistasis_distributions_by_order","epistasis_k_replics",
        "plot_epistasis_volcano",
        # Synthetic
        "BaseConfig","GaussianRandomConfig", "combine_landscapes",
        "FlatEpistasisConfig",
        #Mechanistic models
        "gaussian_interaction_matrix", "simulate_lv", "evaluate_exact_F",
        "growth_rates","is_lv_stable", "is_lv_feasible",
        #Testing
        "test_gaussian_bootstrap_consistency",
        #Global
        "eglobal",
    ])

# --- Core exports ----------------------------------------------------------------
from .core import Landscape

from .io import read_table, landscape_from_file

# --- Walsh–Hadamard transforms and spectra ---------------------------------------
from .walshhadamard import (
    wh_transform,
    fwht_shifted,
    variance_spectrum,
    variance_decomposition,
    _binary_domain,
    mode_orders,
    states_to_spins_shifted,
    walsh_analysis,
    epistasis_amplitude,
    variance_by_order_statistics,
    plot_variance_and_amplitude,
    plot_walsh_volcano,
)

# --- Statistics / Noise utilities -------------------------------------------------
# These live inside the package 'stats_noise'
from .stats_noise import (
    residuals_and_covariance,
    correct_variance_spectrum,
    project_noise_var_WH,
    correct_batch_effect,           # keep as top-level convenience
)

# Bootstraps live in stats_noise/noise_models.py
from .stats_noise.noise_models import (
    bootstrap_uncertainty,
    bootstrap_null,
)

# --- Epistasis deterministic layer ------------------------------------------------
from .epistasis import (
    _bin_to_int,
    _flip_indices,
    _states_for_backgrounds,
    focal_effect,
    epistasis_k,
    epistasis_symbolic,
    compute_full_epistasis,
    filter_significant_interactions,
    epistasis_to_network,
    plot_epistasis_network,
    epistasis_distributions_by_order,
    epistasis_k_replics,
    plot_epistasis_volcano,
)

# --- Synthetic landscapes ---------------------------------------------------------
from .synlands import (
    BaseConfig,GaussianRandomConfig,
    FlatEpistasisConfig,
    combine_landscapes,
)

# --- Mechanistic models -------------------------------------------------------------
from .mechanistic_models import (
    gaussian_interaction_matrix,
    simulate_lv,
    evaluate_exact_F,
    growth_rates,
    is_lv_stable, 
    is_lv_feasible,
)

# --- Testing utilities -----------------------------------------------------------
from .test import test_gaussian_bootstrap_consistency

# --- Public namespace -------------------------------------------------------------
__all__ = [
    # Submodules
    "core", "wh", "epi", "sn", "syn","io",
    # Core
    "Landscape",
    #IO
    "read_table", "landscape_from_file",
    # WH
    "wh_transform", "fwht_shifted",
    "variance_spectrum", "variance_decomposition",
    "_binary_domain", "mode_orders", "states_to_spins_shifted",
    "walsh_analysis", "epistasis_amplitude", "variance_by_order_statistics",
    "plot_variance_and_amplitude",
    "plot_walsh_volcano",
    # Noise / Statistics
    "residuals_and_covariance",
    "correct_variance_spectrum",
    "project_noise_var_WH",
    "bootstrap_uncertainty",
    "bootstrap_null",
    "correct_batch_effect",
    # Epistasis
    "_bin_to_int", "_flip_indices", "_states_for_backgrounds",
    "focal_effect", "epistasis_symbolic", "epistasis_k", 
    "compute_full_epistasis", "filter_significant_interactions",
    "epistasis_to_network", "plot_epistasis_network",
    "epistasis_distributions_by_order","epistasis_k_replics",
    "plot_epistasis_volcano",
    # Synthetic
    "BaseConfig", "GaussianRandomConfig", "combine_landscapes", "FlatEpistasisConfig",
    #Mechanistic models
    "gaussian_interaction_matrix", "simulate_lv", "evaluate_exact_F",
    "growth_rates","is_lv_stable", "is_lv_feasible",
    #Testing
    "test_gaussian_bootstrap_consistency",
    #Global
    "eglobal",
    #FEEs
    "fees"
]
