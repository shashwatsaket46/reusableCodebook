"""Reference TurboQuant implementation (paper: arXiv 2504.19874)."""
from __future__ import annotations
import numpy as np


# ============================================================================
# Lloyd-Max for unit Gaussian
# ============================================================================

def lloyd_max_gaussian(
        n_levels: int, n_iter: int = 300, n_samples: int = 500_000, seed: int = 42
) -> np.ndarray:
    """Lloyd-Max scalar quantizer for N(0, 1).

    Initial spread covers ±3σ which always has positive density, so all cells
    populate and converge correctly.
    """
    rng = np.random.default_rng(seed)
    samples = rng.standard_normal(n_samples)
    # Spread initial levels within the bulk of the distribution (±3σ)
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
# Haar rotation
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
    """MSE-optimal quantizer.

    For unit-norm input x in R^d:
      1. Rotate: r = R x  (coords have std 1/sqrt(d))
      2. Standardize: r_std = r * sqrt(d)  (now ~ N(0, 1) per coord)
      3. Quantize each coord r_std[i] to nearest Lloyd-Max-Gaussian level
      4. Reconstruct: r_hat = levels[idx] / sqrt(d), then x_hat = R^T r_hat
    """

    def __init__(self, d: int, bits: int, seed: int = 42):
        if bits < 1:
            raise ValueError("bits must be >= 1")
        self.d = d
        self.bits = bits
        self.scale = np.sqrt(d)  # standardization factor
        self.levels = lloyd_max_gaussian(2 ** bits, seed=seed)
        self.R = haar_rotation(d, seed=seed)

    def quantize(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        rotated = X @ self.R                    # (N, d), std ~ 1/sqrt(d)
        rotated_std = rotated * self.scale      # standardize to ~N(0, 1)
        idx = _quantize_to_levels(rotated_std, self.levels).astype(np.int32)
        return idx

    def dequantize(self, idx: np.ndarray) -> np.ndarray:
        rotated_hat_std = self.levels[idx]      # (N, d), values in N(0,1) range
        rotated_hat = rotated_hat_std / self.scale   # un-standardize
        return rotated_hat @ self.R.T


# ============================================================================
# TurboQuant Prod — Algorithm 2 (MSE + QJL)
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
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        rotated = X @ self.R                      # (N, d)
        rotated_std = rotated * self.scale         # standardize
        idx = _quantize_to_levels(rotated_std, self.levels).astype(np.int32)
        rotated_hat_std = self.levels[idx]         # (N, d), in std space
        # Residual in standardized rotated space
        residual_std = rotated_std - rotated_hat_std
        residual_norm = np.linalg.norm(residual_std, axis=1)   # (N,)
        safe = np.where(residual_norm[:, None] > 1e-12, residual_norm[:, None], 1.0)
        residual_unit = residual_std / safe                    # (N, d)
        sign_codes = np.sign(residual_unit @ self.S.T).astype(np.int8)
        sign_codes[sign_codes == 0] = 1
        return {
            "idx": idx,
            "sign_codes": sign_codes,
            "residual_norm": residual_norm,
        }

    def inner_product(
            self, queries: np.ndarray, compressed: dict, query_batch: int = 256
    ) -> np.ndarray:
        """Estimated <query, x>. Chunked over queries to bound memory."""
        queries = np.asarray(queries, dtype=np.float64)
        if queries.ndim == 1:
            queries = queries.reshape(1, -1)

        idx = compressed["idx"]
        sign_codes = compressed["sign_codes"]
        residual_norm = compressed["residual_norm"]
        NQ, N = queries.shape[0], idx.shape[0]
        out = np.empty((NQ, N), dtype=np.float32)

        # DB side, precomputed
        rotated_hat_std = self.levels[idx]               # (N, d) in std space
        # x_hat in standardized rotated coords, but inner product needs original coords.
        # Since R is orthogonal, <q, x_hat> = <R q, R x_hat>.
        # And rotated_hat_std = scale * R x_hat → R x_hat = rotated_hat_std / scale.
        # So <q, x_hat> = <R q, rotated_hat_std> / scale.
        sign_codes_f = sign_codes.astype(np.float64)
        # QJL inverse: residual_unit ≈ scale_qjl * S^T sign_codes / qjl_dim
        # Actually: <Sq, sign(Sr)> ≈ sqrt(2/π) * <q, r> / |r|, so
        # <q, r> ≈ |r| * sqrt(π/2) / qjl_dim * <Sq, sign(Sr)>
        scale_qjl = np.sqrt(np.pi / 2) / self.qjl_dim

        for start in range(0, NQ, query_batch):
            end = min(start + query_batch, NQ)
            q_rot = queries[start:end] @ self.R          # (B, d), in rotated space
            q_rot_std = q_rot * self.scale                # in standardized rotated space

            # MSE part: <q_rot_std, rotated_hat_std> / scale
            # because <q, x_hat> = <q_rot, x_hat_rot> = <q_rot_std, rotated_hat_std> / scale
            mse = (q_rot_std @ rotated_hat_std.T) / self.scale  # (B, N)

            # QJL part on residual (in std space)
            Sq = q_rot_std @ self.S.T                     # (B, m)
            qjl_dot = Sq @ sign_codes_f.T                 # (B, N)
            # The QJL gives <q_rot_std, residual_std>; we need <q, residual> = <q_rot_std, residual_std> / scale
            qjl = (scale_qjl * residual_norm[np.newaxis, :] * qjl_dot) / self.scale

            out[start:end] = (mse + qjl).astype(np.float32)
        return out