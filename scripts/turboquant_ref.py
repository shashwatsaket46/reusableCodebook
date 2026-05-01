"""Reference implementation of TurboQuant (paper: arXiv 2504.19874).

Two modes:
  - 'mse'           : Algorithm 1, MSE-optimal quantizer
  - 'inner_product' : Algorithm 2, unbiased IP estimator (MSE + QJL on residual)

Total bits/dim:
  mode='mse'           : b
  mode='inner_product' : b   (because b-1 MSE bits + 1 QJL bit = b total)

Inputs are assumed to be unit-norm. If not, normalize first and store norms separately.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import beta as beta_dist


# ============================================================================
# Lloyd-Max codebook for symmetric Beta on [-1, 1]
# ============================================================================

def beta_alpha_for_dim(d: int) -> float:
    """For a unit-norm vector uniform on the (d-1)-sphere, each coordinate
    after a Haar rotation has density f(c) = (1-c^2)^((d-3)/2) / B(1/2, (d-1)/2)
    on [-1, 1] — symmetric Beta with parameter alpha = (d-1)/2 (re-scaled to [-1,1]).
    """
    return (d - 1) / 2.0


def _sample_beta_pm1(alpha: float, n: int, rng: np.random.Generator) -> np.ndarray:
    return 2 * rng.beta(alpha, alpha, size=n) - 1


def lloyd_max_beta(
        alpha: float, n_levels: int, n_iter: int = 300,
        n_samples: int = 200_000, seed: int = 42,
) -> np.ndarray:
    """Optimal Lloyd-Max scalar quantizer levels for Beta on [-1, 1]."""
    rng = np.random.default_rng(seed)
    samples = _sample_beta_pm1(alpha, n_samples, rng)
    levels = np.linspace(-0.95, 0.95, n_levels)
    for _ in range(n_iter):
        boundaries = (levels[:-1] + levels[1:]) / 2.0
        idx = np.digitize(samples, boundaries)
        new_levels = np.empty_like(levels)
        for i in range(n_levels):
            mask = idx == i
            new_levels[i] = samples[mask].mean() if mask.any() else levels[i]
        new_levels.sort()
        if np.max(np.abs(new_levels - levels)) < 1e-9:
            return new_levels
        levels = new_levels
    return levels


# ============================================================================
# Haar-distributed random rotation
# ============================================================================

def haar_rotation(d: int, seed: int = 42) -> np.ndarray:
    """Random orthogonal matrix uniformly distributed on O(d) (Haar measure).

    QR decomposition of a Gaussian matrix gives a Haar-distributed Q after
    sign correction (so that R has positive diagonal).
    """
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((d, d))
    Q, R = np.linalg.qr(A)
    # Sign-correct so Q is uniform on O(d) (avoids subtle non-uniformity)
    signs = np.sign(np.diag(R))
    Q = Q * signs[np.newaxis, :]
    return Q.astype(np.float64)


# ============================================================================
# Core quantizers
# ============================================================================

def _quantize_to_levels(values: np.ndarray, levels: np.ndarray) -> np.ndarray:
    """Map each value to the index of the nearest level. Vectorized."""
    boundaries = (levels[:-1] + levels[1:]) / 2.0  # (n_levels-1,)
    return np.digitize(values, boundaries)


class TurboQuantMSE:
    """Algorithm 1: MSE-optimal vector quantizer.

    Pipeline (per vector x of unit norm in R^d):
      1. Rotate: r = R x
      2. Quantize each coord r_i to nearest Lloyd-Max level for Beta(α=(d-1)/2)
      3. Store the indices

    Reconstruct:
      r_hat = lookup(idx)
      x_hat = R^T r_hat
    """

    def __init__(self, d: int, bits: int, seed: int = 42):
        if bits < 1:
            raise ValueError("bits must be >= 1")
        self.d = d
        self.bits = bits
        self.alpha = beta_alpha_for_dim(d)
        self.levels = lloyd_max_beta(self.alpha, 2 ** bits, seed=seed)
        self.R = haar_rotation(d, seed=seed)

    def quantize(self, X: np.ndarray) -> np.ndarray:
        """X: (N, d) unit-norm. Returns idx: (N, d) integer codes in [0, 2^bits)."""
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        rotated = X @ self.R  # (N, d)
        idx = _quantize_to_levels(rotated, self.levels).astype(np.int32)
        return idx

    def dequantize(self, idx: np.ndarray) -> np.ndarray:
        """idx: (N, d) -> X_hat: (N, d). Reconstructs in the original (un-rotated) space."""
        rotated_hat = self.levels[idx]              # (N, d)
        return rotated_hat @ self.R.T                # un-rotate

    def quantize_for_search(self, X: np.ndarray):
        """Storage-optimized: returns (idx, x_hat_rotated) for fast search.
        Avoids re-doing R x for queries that come later.
        """
        idx = self.quantize(X)
        x_hat_rotated = self.levels[idx]            # (N, d) in rotated space
        return idx, x_hat_rotated


class TurboQuantProd:
    """Algorithm 2: unbiased inner product quantizer (MSE + QJL).

    bits/dim total = `bits` (split as bits-1 for MSE part, 1 for QJL).

    For inner-product mode the paper requires bits >= 2.
    """

    def __init__(self, d: int, bits: int, seed: int = 42, qjl_dim: int | None = None):
        if bits < 2:
            raise ValueError("Prod mode requires bits >= 2")
        self.d = d
        self.bits = bits
        self.bits_mse = bits - 1
        self.qjl_dim = qjl_dim if qjl_dim is not None else d  # default m = d
        self.alpha = beta_alpha_for_dim(d)
        self.levels = lloyd_max_beta(self.alpha, 2 ** self.bits_mse, seed=seed)
        self.R = haar_rotation(d, seed=seed)

        # QJL projection: random Gaussian matrix in R^(qjl_dim, d).
        # Standard QJL uses entries ~ N(0, 1/qjl_dim) so that the resulting
        # projection has E[||S x||^2] = ||x||^2 (in expectation).
        rng = np.random.default_rng(seed + 12345)
        self.S = rng.standard_normal((self.qjl_dim, d)) / np.sqrt(self.qjl_dim)
        self.S = self.S.astype(np.float64)

    def quantize(self, X: np.ndarray):
        """X: (N, d) unit-norm. Returns dict with: idx, sign_codes, residual_norm.

        idx          : (N, d)    — MSE quantization codes
        sign_codes   : (N, m)    — sign(S * normalized_residual_in_rotated_space)
        residual_norm: (N,)      — ||residual_in_rotated_space||
        """
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        rotated = X @ self.R                                  # (N, d)
        idx = _quantize_to_levels(rotated, self.levels).astype(np.int32)
        rotated_hat = self.levels[idx]                         # (N, d)
        residual = rotated - rotated_hat                       # (N, d), in rotated space
        residual_norm = np.linalg.norm(residual, axis=1)       # (N,)
        # Normalize residual; safe-divide for zero-norm rows
        safe = np.where(residual_norm[:, None] > 1e-12, residual_norm[:, None], 1.0)
        residual_unit = residual / safe                        # (N, d)
        # QJL: sign of Gaussian projection
        # Note: S is (m, d), residual_unit is (N, d) — output is (N, m)
        sign_codes = np.sign(residual_unit @ self.S.T).astype(np.int8)
        # Replace zero signs (rare) with +1 to avoid degeneracy
        sign_codes[sign_codes == 0] = 1
        return {
            "idx": idx,
            "sign_codes": sign_codes,
            "residual_norm": residual_norm,
        }

    def inner_product(self, queries: np.ndarray, compressed: dict) -> np.ndarray:
        """Estimated inner products between queries and stored compressed vectors.

        queries: (NQ, d) unit-norm.
        Returns: (NQ, N).
        """
        queries = np.asarray(queries, dtype=np.float64)
        if queries.ndim == 1:
            queries = queries.reshape(1, -1)
        idx = compressed["idx"]                # (N, d)
        sign_codes = compressed["sign_codes"]  # (N, m)
        residual_norm = compressed["residual_norm"]  # (N,)
        N = idx.shape[0]

        # Rotate queries (queries are in the original space; rotation aligns them)
        q_rot = queries @ self.R               # (NQ, d)

        # MSE part: <q_rot, levels[idx]>
        rotated_hat = self.levels[idx]         # (N, d)
        mse_scores = q_rot @ rotated_hat.T     # (NQ, N)

        # QJL correction: scale * residual_norm[i] * <S q_rot, sign_codes[i]>
        # S has shape (m, d); q_rot has shape (NQ, d)
        Sq = q_rot @ self.S.T                  # (NQ, m)
        # sign_codes is int8 — cast to float for matmul
        qjl_dot = Sq @ sign_codes.astype(np.float64).T   # (NQ, N)
        scale = np.sqrt(np.pi / 2) / self.qjl_dim
        qjl_scores = scale * residual_norm[np.newaxis, :] * qjl_dot   # (NQ, N)

        return mse_scores + qjl_scores