"""
Mechanistic models for epistasis based on Lotka–Volterra dynamics.

This module provides:
- Random generators for interaction matrices.
- Deterministic LV simulations.
- Exact fitness evaluation at LV equilibria for binary genotypes.
"""

from __future__ import annotations

from typing import Optional, Sequence, Tuple

import numpy as np
from numpy.typing import ArrayLike
from scipy.integrate import solve_ivp

__all__ = [
    "gaussian_interaction_matrix",
    "simulate_lv",
    "evaluate_exact_F",
]


############################################
#       INTERACTION MATRIX FUNCTIONS       #
############################################

def gaussian_interaction_matrix(
    S: int,
    C: float,
    mu: float,
    sigma: float,
    diag_value: float = -1.0,
    symmetric: bool = False,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """
    Generate a random interaction matrix A for a Lotka–Volterra system.

    Off-diagonal entries are Gaussian N(mu, sigma^2), thinned by connectance C.
    Diagonal entries are fixed to diag_value (usually negative).

    Parameters
    ----------
    S : int
        Number of species.
    C : float
        Connectance (probability that an off-diagonal interaction is non-zero).
    mu : float
        Mean of the Gaussian distribution for off-diagonal entries.
    sigma : float
        Standard deviation of the Gaussian distribution for off-diagonal entries.
    diag_value : float, default -1.0
        Value set on the diagonal (self-interactions).
    symmetric : bool, default False
        If True, build a symmetric matrix (A_ij = A_ji).
        Note that the diagonal is still fixed to diag_value.
    rng : np.random.Generator, optional
        Random number generator. If None, use default_rng().

    Returns
    -------
    A : ndarray, shape (S, S)
        Interaction matrix.
    """
    if rng is None:
        rng = np.random.default_rng()

    if not (0.0 <= C <= 1.0):
        raise ValueError("Connectance C must be in [0, 1].")

    # Start with a zero matrix
    A = np.zeros((S, S), dtype=float)

    # Generate upper triangle (excluding diagonal) if symmetric
    if symmetric:
        # random mask for upper triangle
        mask = rng.random((S, S))
        for i in range(S):
            mask[i, i] = 1.0  # avoid using mask on diagonal
        upper = (mask < C).astype(float)
        # draw Gaussian values
        vals = rng.normal(mu, sigma, size=(S, S))
        upper *= vals
        # symmetrize
        A = upper + upper.T
    else:
        # Full random mask (off-diagonal entries)
        mask = rng.random((S, S))
        # draw Gaussian values
        vals = rng.normal(mu, sigma, size=(S, S))
        A = (mask < C).astype(float) * vals

    # Set diagonal to fixed value (independent of connectance)
    np.fill_diagonal(A, diag_value)

    return A


############################################
#                LV SOLVER                 #
############################################

def simulate_lv(
    alpha: ArrayLike,
    r: ArrayLike,
    t_max: float = 100.0,
    N0: Optional[ArrayLike] = None,
    t_eval: Optional[ArrayLike] = None,
    rtol: float = 1e-8,
    atol: float = 1e-10,
    extinction_threshold: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Simulate the Lotka–Volterra dynamics:

        dN_i/dt = N_i * ( r_i + sum_j alpha_ij N_j )

    Parameters
    ----------
    alpha : array_like, shape (S, S)
        Interaction matrix.
    r : array_like, shape (S,)
        Intrinsic growth rates.
    t_max : float, default 100.0
        Final time of integration.
    N0 : array_like, shape (S,), optional
        Initial abundances. If None, random positive values are used.
    t_eval : array_like, optional
        Time points at which to store the computed solution. If None,
        the solver chooses adaptively.
    rtol : float, default 1e-8
        Relative tolerance for the ODE solver.
    atol : float, default 1e-10
        Absolute tolerance for the ODE solver.
    extinction_threshold : float, default 0.0
        If > 0, abundances are floored at this value inside the RHS
        to avoid negative densities.

    Returns
    -------
    t : ndarray, shape (n_times,)
        Time points of the simulation.
    N : ndarray, shape (S, n_times)
        Abundance trajectories of each species over time.
    """
    alpha = np.asarray(alpha, dtype=float)
    r = np.asarray(r, dtype=float)

    if alpha.ndim != 2 or alpha.shape[0] != alpha.shape[1]:
        raise ValueError("alpha must be a square (S, S) array.")
    S = alpha.shape[0]
    if r.shape != (S,):
        raise ValueError("r must have shape (S,) compatible with alpha.")

    if N0 is None:
        # Small positive initial condition
        rng = np.random.default_rng()
        N0 = rng.normal(loc=0.1, scale=0.05, size=S)
        N0 = np.abs(N0)
    else:
        N0 = np.asarray(N0, dtype=float)
        if N0.shape != (S,):
            raise ValueError("N0 must have shape (S,) compatible with alpha.")

    def lv_rhs(_t: float, N: np.ndarray) -> np.ndarray:
        """Right-hand side of LV dynamics."""
        N = np.asarray(N, dtype=float).flatten()
        if extinction_threshold > 0.0:
            # Avoid negative densities for numerical stability
            N = np.maximum(N, extinction_threshold)
        return N * (r + alpha @ N)

    sol = solve_ivp(
        lv_rhs,
        (0.0, t_max),
        N0,
        method="RK45",
        t_eval=t_eval,
        rtol=rtol,
        atol=atol,
    )

    if not sol.success:
        raise RuntimeError(f"LV integration failed: {sol.message}")

    return sol.t, sol.y


############################################
#       EXACT FITNESS AT EQUILIBRIUM       #
############################################

def evaluate_exact_F(
    X: np.ndarray,
    r: ArrayLike,
    phi: ArrayLike,
    A: ArrayLike,
    K: Optional[ArrayLike] = None,
    clip_negative: bool = False,
) -> np.ndarray:
    """
    Evaluate F(x) = sum_i phi_i * y_i * x_i, where y is the LV equilibrium:

        A y = - (r ⊙ x)

    for each binary configuration x in X. Here ⊙ denotes element-wise product.

    Parameters
    ----------
    X : ndarray, shape (M, S)
        Matrix of binary configurations (M genotypes, S species/traits).
        Each row is a genotype x in {0,1}^S.
    r : array_like, shape (S,)
        Growth rates.
    phi : array_like, shape (S,)
        Weights used to compute the scalar fitness F(x) from the equilibrium y.
    A : array_like, shape (S, S)
        Interaction matrix (must be invertible).
    K : array_like, shape (S,), optional
        Optional carrying capacities or scaling of the equilibrium y.
        If provided, y is multiplied element-wise by K.
        (Currently a simple rescaling; can be adapted to a more
         mechanistic use if needed.)
    clip_negative : bool, default False
        If True, negative components of y are set to zero (interpreting
        them as extinctions).

    Returns
    -------
    F_exact : ndarray, shape (M,)
        Array of F(x) values, one per configuration.
    """
    X = np.asarray(X, dtype=float)
    A = np.asarray(A, dtype=float)
    r = np.asarray(r, dtype=float)
    phi = np.asarray(phi, dtype=float)

    if X.ndim != 2:
        raise ValueError("X must be a 2D array of shape (M, S).")

    M, S = X.shape
    if A.shape != (S, S):
        raise ValueError("A must have shape (S, S) compatible with X.")
    if r.shape != (S,):
        raise ValueError("r must have shape (S,).")
    if phi.shape != (S,):
        raise ValueError("phi must have shape (S,).")

    if K is not None:
        K = np.asarray(K, dtype=float)
        if K.shape != (S,):
            raise ValueError("K must have shape (S,) if provided.")

    # Right-hand side for all genotypes:
    # For each x: b(x) = - (r ⊙ x)
    # Arrange as matrix B of shape (S, M), each column is b(x)
    RX = (X * r[None, :])  # shape (M, S)
    B = -RX.T              # shape (S, M)

    # Solve A Y = B for Y (shape (S, M)), one column per genotype
    # This avoids computing the explicit inverse of A.
    try:
        Y = np.linalg.solve(A, B)  # shape (S, M)
    except np.linalg.LinAlgError as exc:
        raise np.linalg.LinAlgError(
            "Interaction matrix A is singular or ill-conditioned; "
            "cannot compute equilibrium y."
        ) from exc

    # Optional scaling by K
    if K is not None:
        Y = Y * K[:, None]

    if clip_negative:
        Y = np.maximum(Y, 0.0)

    # Fitness: F(x) = sum_i phi_i * y_i(x) * x_i
    # Y: (S, M), X.T: (S, M)
    F_exact = np.sum(phi[:, None] * Y * X.T, axis=0)  # shape (M,)

    return F_exact


###################################
#         AUXILIAR FUNCTION       #
###################################

def growth_rates(N: int,r) -> np.ndarray:
    """
    Return intrinsic growth rates r for a system of size N.

    For now we choose all ones, but this can be customized.
    """
    return r*np.ones(N, dtype=float)


def is_lv_stable(A: np.ndarray) -> bool:
    """
    Check dynamical stability of the LV system with interaction matrix A.

    The criterion used here is:
        max(Re(eig(A))) < 0
    """
    eigs = np.linalg.eigvals(A)
    return np.max(np.real(eigs)) < 0.0


def is_lv_feasible(A: np.ndarray, r: np.ndarray, X: np.ndarray, eps: float = 0.0) -> bool:
    """
    Check feasibility of all non-zero genotypes for the LV equilibrium.

    For each non-zero genotype x:
        A y = - (r ⊙ x)
    we require y_i > eps only for species with x_i = 1 (present).
    """
    try:
        A_inv = np.linalg.inv(A)
    except np.linalg.LinAlgError:
        return False

    for x in X[1:]:  # skip all-zero genotype
        rx = r * x
        y = -A_inv @ rx

        # Only check species that are present in x
        present = x.astype(bool)
        if np.any(y[present] <= eps):
            return False

    return True

