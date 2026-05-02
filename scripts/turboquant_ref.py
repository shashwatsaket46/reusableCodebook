"""Reference TurboQuant implementation (paper: arXiv 2504.19874).

Built from scratch since the unofficial implementations gave inconsistent
results. Uses Gaussian Lloyd-Max on standardized rotated coordinates.

Two modes:
  - TurboQuantMSE  : Algorithm 1, MSE-optimal scalar quantizer
  - TurboQuantProd : Algorithm 2, MSE + 1-bit QJL on residual (unbiased IP)

Total bits/dim: equals `bits` argument in both modes.
For Prod: split as (bits-1) MSE bits + 1 QJL bit.

Inputs are assumed to be unit-norm.
"""
from __future__ import annotations
import numpy as np


# ============================================================================
# Lloyd-Max for unit Gaussian
# ============================================================================

def lloyd_max_gaussian(
        n_levels: int, n_iter: int = 300, n_samples: int = 500_000, seed: int = 42
) -> np.ndarray:
    """Lloyd-Max scalar quantizer levels for N(0, 1)."""
    rng = np.random.default_rng(seed)
    samples = rng.standard_normal(n_samples)
    levels = np.linspace(-3, 3, n_levels)
    if n_levels == 1:
        return np.array([0.0])
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
    """Random orthogonal matrix uniformly distributed on O(d) (Haar measure)."""
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((d, d))
    Q, R = np.linalg.qr(A)
    signs = np.sign(np.diag(R))
    Q = Q * signs[np.newaxis, :]
    return Q.astype(np.float64)


# ============================================================================
# Quantize / dequantize helpers
# ============================================================================

def _quantize_to_levels(values: np.ndarray, levels: np.ndarray) -> np.ndarray:
    """Map each value to the index of the nearest level."""
    if len(levels) == 1:
        return np.zeros_like(values, dtype=np.int32)
    boundaries = (levels[:-1] + levels[1:]) / 2.0
    return np.digitize(values, boundaries)


# ============================================================================
# TurboQuant MSE — Algorithm 1
# ============================================================================

class TurboQuantMSE:
    """MSE-optimal vector quantizer.

    Pipeline (per unit-norm input x in R^d):
      1. Rotate: r = R x  (coords have std 1/sqrt(d))
      2. Standardize: r_std = sqrt(d) * r  (now ~ N(0, 1) per coord)
      3. Quantize each coord r_std[i] to nearest Lloyd-Max-Gaussian level
      4. Reconstruct: r_hat = levels[idx] / sqrt(d), x_hat = R^T r_hat
    """

    def __init__(self, d: int, bits: int, seed: int = 42):
        if bits < 1:
            raise ValueError("bits must be >= 1")
        self.d = d
        self.bits = bits
        self.scale = np.sqrt(d)
        self.levels = lloyd_max_gaussian(2 ** bits, seed=seed)
        self.R = haar_rotation(d, seed=seed)

    def quantize(self, X: np.ndarray) -> np.ndarray:
        """X: (N, d) unit-norm. Returns idx: (N, d) integer codes in [0, 2^bits)."""
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        rotated = X @ self.R                          # (N, d), std ~ 1/sqrt(d)
        rotated_std = rotated * self.scale             # standardize
        return _quantize_to_levels(rotated_std, self.levels).astype(np.int32)

    def dequantize(self, idx: np.ndarray) -> np.ndarray:
        """idx: (N, d) -> X_hat: (N, d), reconstructed in original space."""
        rotated_hat_std = self.levels[idx]
        rotated_hat = rotated_hat_std / self.scale
        return rotated_hat @ self.R.T


# ============================================================================
# TurboQuant Prod — Algorithm 2 (MSE + QJL on residual)
# ============================================================================

class TurboQuantProd:
    """Unbiased inner product quantizer.

    Uses (bits-1)-bit MSE quantizer + 1-bit QJL on residual.
    Total bits/dim = bits.
    """

    def __init__(self, d: int, bits: int, seed: int = 42, qjl_dim: int | None = None):
        if bits < 2:
            raise ValueError("Prod mode requires bits >= 2")
        self.d = d
        self.bits = bits
        self.bits_mse = bits - 1
        self.qjl_dim = qjl_dim if qjl_dim is not None else d
        self.scale = np.sqrt(d)
        self.levels = lloyd_max_gaussian(2 ** self.bits_mse, seed=seed)
        self.R = haar_rotation(d, seed=seed)

        # QJL projection: Gaussian matrix scaled to unit-variance projections
        rng = np.random.default_rng(seed + 12345)
        self.S = (rng.standard_normal((self.qjl_dim, d)) /
                  np.sqrt(self.qjl_dim)).astype(np.float64)

    def quantize(self, X: np.ndarray):
        """X: (N, d) unit-norm. Returns dict with idx, sign_codes, residual_norm_std."""
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        rotated = X @ self.R                                  # (N, d)
        rotated_std = rotated * self.scale                     # standardize for Lloyd-Max
        idx = _quantize_to_levels(rotated_std, self.levels).astype(np.int32)
        rotated_hat_std = self.levels[idx]
        residual_std = rotated_std - rotated_hat_std
        residual_norm_std = np.linalg.norm(residual_std, axis=1)
        safe = np.where(residual_norm_std[:, None] > 1e-12, residual_norm_std[:, None], 1.0)
        residual_unit = residual_std / safe
        sign_codes = np.sign(residual_unit @ self.S.T).astype(np.int8)
        sign_codes[sign_codes == 0] = 1
        return {
            "idx": idx,
            "sign_codes": sign_codes,
            "residual_norm_std": residual_norm_std,
        }

    def inner_product(
            self, queries: np.ndarray, compressed: dict, query_batch: int = 256
    ) -> np.ndarray:
        """Estimated inner products. Chunked over queries to bound memory."""
        queries = np.asarray(queries, dtype=np.float64)
        if queries.ndim == 1:
            queries = queries.reshape(1, -1)
        idx = compressed["idx"]
        sign_codes = compressed["sign_codes"]
        residual_norm_std = compressed["residual_norm_std"]
        NQ, N = queries.shape[0], idx.shape[0]
        out = np.empty((NQ, N), dtype=np.float32)

        # Reconstruct DB in rotated (un-standardized) space
        rotated_hat = self.levels[idx] / self.scale          # (N, d)
        residual_norm = residual_norm_std / self.scale       # (N,)
        sign_codes_f = sign_codes.astype(np.float64)
        scale_qjl = np.sqrt(np.pi / 2) / self.qjl_dim

        for start in range(0, NQ, query_batch):
            end = min(start + query_batch, NQ)
            q_rot = queries[start:end] @ self.R              # (B, d)

            # MSE part: <q, x_hat> = <q_rot, rotated_hat>
            mse = q_rot @ rotated_hat.T                       # (B, N)

            # QJL part: <q, residual> via Gaussian projection sign trick
            Sq = q_rot @ self.S.T                             # (B, m)
            qjl_dot = Sq @ sign_codes_f.T                     # (B, N)
            qjl = scale_qjl * residual_norm[np.newaxis, :] * qjl_dot

            out[start:end] = (mse + qjl).astype(np.float32)
        return out