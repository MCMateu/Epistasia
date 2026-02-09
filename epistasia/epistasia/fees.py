"""
fees: Public FEE API.

This module provides a user-friendly namespace for Functional Effect Equations (FEEs).

Implementation lives in epistasia.eglobal.fees, but we re-export the public functions/classes
here to offer a clean top-level access as:

    import epistasia as ep
    ep.fees.plot_fees_grid_from_landscape(L)
"""

from .eglobal.fees import (
    # empirical
    FEEData,
    fee_data,

    # theory
    #FEETheorySecondOrder,
    #fee_theory_second_order,

    # plotting
    #plot_fees_grid,
    plot_fees_grid_from_landscape,
)

__all__ = [
    "FEEData",
    "fee_data",
    #"FEETheorySecondOrder",
    #"fee_theory_second_order",
    #"plot_fees_grid",
    "plot_fees_grid_from_landscape",
]
