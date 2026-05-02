"""Test the from-scratch TurboQuant reference implementation.

Validates against:
  1. Lloyd-Max codebook sanity (no stuck endpoints, correct symmetric levels).
  2. Round-trip MSE bound from paper Theorem 1.
  3. GloVe-200 R@1 expected from paper Figure 4(a):
        TurboQuant 2-bit ~ 0.55, 4-bit ~ 0.86.
"""
from pathlib import Path
import sys
import time
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from turboquant_ref import (
    TurboQuantMSE,
    TurboQuantProd,
    lloyd_max_gaussian,
)


def _recall_at_ks(scores, GT, ks):
    top_max = max(ks)
    top_idx = np.argpartition(-scores, top_max, axis=1)[:, :top_max]
    row_scores = np.take_along_axis(scores, top_idx, axis=1)
    sort_order = np.argsort(-row_scores, axis=1)
    top_sorted = np.take_along_axis(top_idx, sort_order, axis=1)
    out = {}
    for k in ks:
        topk = top_sorted[:, :k]
        gt1 = GT[:, 0:1]
        out[k] = float((topk == gt1).any(axis=1).mean())
    return out


def test_levels():
    print("=" * 70)
    print("Test 1: Lloyd-Max-Gaussian levels (no stuck endpoints)")
    print("=" * 70)
    for b in [1, 2, 3, 4, 5]:
        levels = lloyd_max_gaussian(2 ** b)
        print(f"  b={b}  levels = {np.array2string(levels, precision=3, suppress_small=True)}")
    print()


def test_roundtrip_synthetic():
    print("=" * 70)
    print("Test 2: round-trip MSE on synthetic d=200 unit-norm Gaussians")
    print("Paper bound (Theorem 1): MSE per vector <= sqrt(3*pi)/2 * 4^(-b)")
    print("=" * 70)
    rng = np.random.default_rng(0)
    X = rng.standard_normal((1000, 200))
    X /= np.linalg.norm(X, axis=1, keepdims=True)

    for b in [2, 3, 4, 5]:
        tq = TurboQuantMSE(d=200, bits=b, seed=42)
        idx = tq.quantize(X)
        X_hat = tq.dequantize(idx)
        mse = np.mean(np.sum((X - X_hat) ** 2, axis=1))
        bound = (np.sqrt(3 * np.pi) / 2) * 4 ** (-b)
        print(f"  b={b}  empirical MSE = {mse:.4f}  bound = {bound:.4f}  ratio = {mse/bound:.2f}")
    print()


def test_glove200():
    print("=" * 70)
    print("Test 3: GloVe-200 R@1 (paper Figure 4(a): ~0.55 at b=2, ~0.86 at b=4)")
    print("=" * 70)

    DATA = ROOT / "third_party" / "Extended-RaBitQ" / "data" / "glove200_100k"
    X = np.load(DATA / "X.npy").astype(np.float64)
    Xq = np.load(DATA / "Xq.npy").astype(np.float64)
    GT = np.load(DATA / "GT.npy").astype(np.int32)
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    Xq /= np.linalg.norm(Xq, axis=1, keepdims=True)
    Xq, GT = Xq[:1000], GT[:1000]

    print("\nTurboQuantMSE:")
    for b in [2, 3, 4]:
        tq = TurboQuantMSE(d=200, bits=b, seed=42)
        t0 = time.time()
        idx = tq.quantize(X)
        t_q = time.time() - t0
        t0 = time.time()
        X_hat = tq.dequantize(idx)
        scores = (Xq @ X_hat.T).astype(np.float32)
        t_s = time.time() - t0
        rec = _recall_at_ks(scores, GT, [1, 4, 32])
        print(f"  bits={b}  quant={t_q:.1f}s  search={t_s:.1f}s  "
              f"R@1={rec[1]:.4f}  R@4={rec[4]:.4f}  R@32={rec[32]:.4f}")

    print("\nTurboQuantProd (with QJL bias correction):")
    for b in [2, 3, 4]:
        tq = TurboQuantProd(d=200, bits=b, seed=42)
        t0 = time.time()
        compressed = tq.quantize(X)
        t_q = time.time() - t0
        t0 = time.time()
        scores = tq.inner_product(Xq, compressed, query_batch=256)
        t_s = time.time() - t0
        rec = _recall_at_ks(scores, GT, [1, 4, 32])
        print(f"  bits={b}  quant={t_q:.1f}s  search={t_s:.1f}s  "
              f"R@1={rec[1]:.4f}  R@4={rec[4]:.4f}  R@32={rec[32]:.4f}")


if __name__ == "__main__":
    test_levels()
    test_roundtrip_synthetic()
    test_glove200()