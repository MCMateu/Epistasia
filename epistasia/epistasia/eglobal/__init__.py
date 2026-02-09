"""
eglobal: Global epistasis and alignment analysis.

Core concepts
-------------
- DF(z): microscopic response vectors
- u1   : dominant direction (global epistasis)
- Q    : nematic order parameter
"""

# ======================
# Algebra / landscapes
# ======================
from .algebra import (
    compute_M,
    build_gh_plan,
    gh_from_fs,
    M_second_order_from_gh,
    orient_u1_with_popcount,
    # DF utilities
    compute_DF,
    DFResult,
    sample_DF_second_order,
    project_states_on_pcs,
    explain_latent_regions_by_presence,
    strong_regionality_gate,
    explain_latent_regions_by_presence_k,
)

# ======================
# Alignment / geometry
# ======================
from .alignment import (
    compute_alignment,
    AlignmentResult,
    compute_Q_from_DF,
    compute_u1_from_M,
    U1Result,
    cos2_angles_pairwise,
    cos2_angles_to_u1,
    cos2_null_beta_params,
    plot_cos2_vs_null,
)

# ======================
# Second-order inference
# ======================
from .qinference import (
    compute_q_second_order,
    QSecondOrderResult,
    compute_q_second_order_given_u1,
    plot_strong_regionality_q,
    Qmanifold_second_order,
    QManifoldSecondOrderResult,
    compute_Qmanifold_restricted_to_gate_from_gh,
)

__all__ = [
    # --- algebra ---
    "compute_M",
    "build_gh_plan",
    "gh_from_fs",
    "M_second_order_from_gh",
    "orient_u1_with_popcount",
    "compute_DF",
    "DFResult",
    "sample_DF_second_order",
    "project_states_on_pcs",
    "explain_latent_regions_by_presence",
    "strong_regionality_gate",
    "explain_latent_regions_by_presence_k",

    # --- alignment ---
    "compute_alignment",
    "AlignmentResult",
    "compute_Q_from_DF",
    "compute_u1_from_M",
    "U1Result",
    "cos2_angles_pairwise",
    "cos2_angles_to_u1",
    "cos2_null_beta_params",
    "plot_cos2_vs_null",

    # --- inference ---
    "compute_q_second_order",
    "QSecondOrderResult",
    "compute_q_second_order_given_u1",
    "plot_strong_regionality_q",
    "Qmanifold_second_order",
    "QManifoldSecondOrderResult",
    "compute_Qmanifold_restricted_to_gate_from_gh",
]
