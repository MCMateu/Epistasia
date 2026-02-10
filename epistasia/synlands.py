from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Literal, Tuple, Dict, Any
import numpy as np
from math import comb

from .core import Landscape


Ordering = Literal["lex", "binary", "gray"]

###################################################
#               BASE CONFIG CLASS                 #
###################################################

@dataclass(frozen=True, slots=True)
class BaseConfig:
    """
    Base configuration for synthetic landscapes.

    - N: number of loci (bits)
    - R: number of replicates
    - ordering: "lex", "binary" or "gray"
    - seed: RNG seed (None -> random unless test_mode=True)
    - test_mode: if True, make RNG deterministic even when seed is None
    """
    N: int
    R: int = 1
    ordering: Ordering = "lex"
    seed: Optional[int] = None
    test_mode: bool = False

    # --- Shared utilities for all children ---

    def rng(self) -> np.random.Generator:
        """Return a NumPy Generator with the appropriate seeding policy."""
        if self.test_mode:
            # In test mode we want determinism even if seed is None
            base_seed = 0 if self.seed is None else self.seed
        else:
            base_seed = self.seed
        return np.random.default_rng(base_seed)

    def effective_ordering(self) -> Ordering:
        """Ordering actually used to enumerate states."""
        # If you ever want to override ordering in test_mode, do it here.
        return self.ordering

    def states(self) -> np.ndarray:
        """Enumerate all 2^N states as a (M, N) uint8 array."""
        return _enumerate_states(self.N, ordering=self.effective_ordering())


##################################################
#                    HELPERS                     #
##################################################

def _binary_to_gray(x: np.ndarray) -> np.ndarray:
    return x ^ (x >> 1)

def _ints_to_bits(x: np.ndarray, N: int) -> np.ndarray:
    # Convert vector of integers to an (len(x), N) bit-matrix (MSB first).
    return ((x[:, None] >> np.arange(N - 1, -1, -1)) & 1).astype(np.uint8)

def _enumerate_states(N: int, ordering: Ordering = "lex") -> np.ndarray:
    """
    Return states as a (2^N, N) uint8 array in {0,1}, according to the chosen ordering.
    'lex'/'binary': natural integer order 0..2^N-1 in binary.
    'gray': Gray code order (successive states differ by one bit).
    """
    M = 1 << N
    if ordering in ("lex", "binary"):
        ints = np.arange(M, dtype=np.uint32)
    elif ordering == "gray":
        ints = _binary_to_gray(np.arange(M, dtype=np.uint32))
    else:
        raise ValueError(f"Unknown ordering={ordering!r}")

    states = _ints_to_bits(ints, N)
    return states.astype(np.uint8)

##################################################
#           GAUSSIAN RANDOM LANDSCAPE            #
##################################################

@dataclass(frozen=True, slots=True)
class GaussianRandomConfig(BaseConfig):
    """
    Configuration for a single-sigma Gaussian random landscape with
    optional affine batch effects per replicate:

        F_r(x) = a_r + b_r * Xi_r(x),
        Xi_r(x) ~ N(mean(x), sigma^2) iid over states and replicates.
    """
    mean: float | np.ndarray = 0.0
    sigma: float = 1.0
    batch: bool = False
    tau_a: float = 1.0
    tau_b: float = 1.0

    def sample(
        self,
        *,
        return_meta: bool = True,
    ) -> Landscape | tuple[Landscape, dict]:
        rng = self.rng()
        states = self.states()
        M = states.shape[0]

        # --- Mean vector ---
        if np.isscalar(self.mean):
            mean_vec = np.full(M, float(self.mean))
        else:
            mean_vec = np.asarray(self.mean, dtype=float)
            if mean_vec.shape != (M,):
                raise ValueError(
                    f"`mean` must be scalar or length {M} vector; "
                    f"got {mean_vec.shape}, expected {(M,)}."
                )

        if self.sigma < 0:
            raise ValueError("`sigma` must be non-negative.")

        # --- Batch parameters per replicate ---
        if self.batch:
            a_r = (
                rng.normal(0.0, self.tau_a, size=self.R)
                if self.tau_a > 0.0 else np.zeros(self.R)
            )
            logb = (
                rng.normal(0.0, self.tau_b, size=self.R)
                if self.tau_b > 0.0 else np.zeros(self.R)
            )
            b_r = np.exp(logb)  # strictly positive
        else:
            a_r = np.zeros(self.R)
            b_r = np.ones(self.R)

        # --- Replicate-specific Gaussian fields (shape = (M, R)) ---
        Xi = mean_vec[:, None] + self.sigma * rng.standard_normal(size=(M, self.R))

        # --- Apply affine per-replicate batch transform ---
        values = a_r[None, :] + b_r[None, :] * Xi  # shape (M, R)

        # --- Package result ---
        L = Landscape(states=states, values=values, N=self.N, R=self.R)
        #L.values[0,:] = 0.0 

        if not return_meta:
            return L

        meta = {
            "kind": "gaussian_single_sigma_affine_batch",
            "N": self.N,
            "R": self.R,
            "ordering": self.effective_ordering(),
            "seed": (
                0 if (self.test_mode and self.seed is None) else self.seed
            ),
            "batch": self.batch,
            "tau_a": float(self.tau_a),
            "tau_b": float(self.tau_b),
            "a_r": a_r,
            "b_r": b_r,
            "sigma": float(self.sigma),
            "mean": mean_vec,
        }
        return L, meta


###################################################
#             FLAT EPISASIS CONFIG                #
###################################################

UDef = Literal["pdf", "mean_per_term", "block_energy"]
# pdf:          U(S) = (4^S / binom(N,S)) * sum f_s^2
# mean_per_term U(S) = (1 / binom(N,S))   * sum f_s^2
# block_energy  U(S) =                     sum f_s^2


import numpy as np
from math import comb

# Importa tu Landscape y utilidades WH
from .core import Landscape                      # :contentReference[oaicite:2]{index=2}
from .walshhadamard import binary_domain         # :contentReference[oaicite:3]{index=3}


def _ifwht_shifted(fs: np.ndarray, N: int) -> np.ndarray:
    """
    Inverse of fwht_shifted used in walshhadamard.py.

    If fwht_shifted computes:
        fs_out = sign_s * (1/2^N) * H * a
    then inverse is:
        a = H * (sign_s * fs_out)

    where H is the unnormalized Hadamard butterfly operator.
    """
    F = np.asarray(fs, dtype=float)
    if F.ndim == 1:
        F = F[:, None]
        squeeze = True
    elif F.ndim == 2:
        squeeze = False
    else:
        raise ValueError("fs must be 1D (2^N,) or 2D (2^N, R).")

    M, R = F.shape
    if M != (1 << N):
        raise ValueError("First dimension must be 2^N and match N.")

    # sign_s = (-1)^{|s|}
    idx = np.arange(M)[:, None]
    hw = ((idx >> np.arange(N)) & 1).sum(axis=1)
    sign = np.where(hw % 2 == 0, 1.0, -1.0)

    # g = sign_s * fs_out  (undo the shifted correction applied in fwht_shifted)
    A = F * sign[:, None]

    # Apply unnormalized FWHT butterfly: A <- H * A
    h = 1
    while h < M:
        step = h << 1
        for i in range(0, M, step):
            j = i + h
            x = A[i:j, :]
            y = A[j:i + step, :]
            A[i:j, :], A[j:i + step, :] = x + y, x - y
        h = step

    return A[:, 0] if squeeze else A

class FlatEpistasisConfig:
    """
    Generator for 'flat epistatic landscapes':
        Var(f_s) = K^2 * 4^{-|s|}  for |s|>=1
    so that Var(E_s) is constant across orders.

    You can specify either:
      - K directly, OR
      - Vtot_target, using  Vtot = K^2 * [ (5/4)^N - 1 ].
    """

    def __init__(self, rng: np.random.Generator | None = None):
        self.rng = np.random.default_rng() if rng is None else rng

    @staticmethod
    def K_from_Vtot(N: int, Vtot_target: float) -> float:
        denom = (5.0 / 4.0) ** N - 1.0
        if denom <= 0:
            raise ValueError("Invalid denom for K_from_Vtot; check N.")
        if Vtot_target <= 0:
            raise ValueError("Vtot_target must be positive.")
        return float(np.sqrt(Vtot_target / denom))

    def sample(
        self,
        *,
        N: int,
        R: int = 1,
        K: float | None = None,
        Vtot_target: float | None = None,
        mu0: float = 0.0,              # coefficient of order 0 (global mean)
        noise_sd: float = 0.0,         # optional additive noise on F(x) per replicate
        feature_names: list[str] | None = None,
        return_fs: bool = True,        # keep WH coeffs in meta (can be big)
    ) -> tuple[Landscape, dict]:
        if (K is None) == (Vtot_target is None):
            raise ValueError("Provide exactly one of: K or Vtot_target.")

        if N <= 0:
            raise ValueError("N must be >= 1.")
        if R <= 0:
            raise ValueError("R must be >= 1.")

        if K is None:
            K = self.K_from_Vtot(N, Vtot_target)

        # All modes s in lex order (aligned with your FWHT assumptions)
        s_bits = binary_domain(N)                 # shape (2^N, N) :contentReference[oaicite:4]{index=4}
        orders = s_bits.sum(axis=1)               # |s|
        M = 1 << N

        # Sample fs for each replicate
        fs = np.zeros((M, R), dtype=float)

        # order-0 coefficient sets the global mean of F(x)
        fs[0, :] = mu0

        # For |s|>=1: Var(fs) = K^2 * 4^{-|s|} -> sd = K * 2^{-|s|}
        for idx in range(1, M):
            k = int(orders[idx])
            sd = K * (2.0 ** (-k))
            fs[idx, :] = self.rng.normal(loc=0.0, scale=sd, size=R)

        # Reconstruct phenotype values on all 2^N genotypes: F(x) = sum_s f_s phi_s(x)
        F = _ifwht_shifted(fs, N)                 # shape (2^N, R)

        # Optional measurement noise on the phenotype, per replicate
        if noise_sd > 0:
            F = F + self.rng.normal(0.0, noise_sd, size=F.shape)

        # States in lex order {0,1}^N
        states = binary_domain(N).astype(np.int8)

        L = Landscape(
            states=states,
            values=F,
            N=N,
            R=R,
            order="lex",
            feature_names=feature_names,
        )                                         # :contentReference[oaicite:5]{index=5}

        # Meta info (útil para debug/figuras)
        Vtot_expected = (K**2) * ((5.0 / 4.0) ** N - 1.0)

        meta = {
            "generator": "FlatEpistaticLandscapeGenerator",
            "N": N,
            "R": R,
            "K": float(K),
            "mu0": float(mu0),
            "noise_sd": float(noise_sd),
            "Vtot_expected_excl_order0": float(Vtot_expected),
            "orders": orders,
        }
        if return_fs:
            meta["fs"] = fs
            meta["s_bits"] = s_bits

        return L, meta

    
def combine_landscapes(
    L_signal: Landscape,
    L_noise: Landscape,
    *,
    w_signal: float = 1.0,
    w_noise: float = 1.0,
) -> Landscape:
    """
    Linear combination of two landscapes, with automatic broadcasting
    if one of the landscapes has R=1 and the other has R>1.
    """

    if L_signal.N != L_noise.N:
        raise ValueError("Landscapes must have the same N.")
    if not np.array_equal(L_signal.states, L_noise.states):
        raise ValueError("Landscapes must be defined on the same state set.")

    Rs, Rn = L_signal.R, L_noise.R

    # --- broadcasting logic ---
    if Rs == Rn:
        F_signal = L_signal.values
        F_noise  = L_noise.values
        R = Rs

    elif Rs == 1 and Rn > 1:
        # broadcast signal to match noise replicates
        F_signal = np.tile(L_signal.values, (1, Rn))
        F_noise  = L_noise.values
        R = Rn

    elif Rn == 1 and Rs > 1:
        # broadcast noise to match signal replicates
        F_signal = L_signal.values
        F_noise  = np.tile(L_noise.values, (1, Rs))
        R = Rs

    else:
        raise ValueError(
            f"Incompatible replicates: signal has R={Rs}, noise has R={Rn}"
        )

    values = w_signal * F_signal + w_noise * F_noise

    return Landscape(
        states=L_signal.states.copy(),
        values=values,
        N=L_signal.N,
        R=R,
    )

