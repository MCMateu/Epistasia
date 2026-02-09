# epistasia/sn/__init__.py
"""
epistasia.sn

Statistical and noise–correction utilities.
"""

from .noise_models import (
    correct_batch_effect,
    residuals_and_covariance,
    project_noise_var_WH,
    correct_variance_spectrum,
)

__all__ = [
    "correct_batch_effect",
    "residuals_and_covariance",
    "project_noise_var_WH",
    "correct_variance_spectrum",
    "bootstrap_uncertainty",
    "bootstrap_null"
]
