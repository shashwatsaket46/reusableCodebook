"""Test the reference TurboQuant implementation against:
  1. A self-consistency check (round-trip MSE quantize/dequantize)
  2. The paper's expected GloVe-200 numbers (R@1 ≈ 0.55 at 2 bits)
"""
from pathlib import Path
import sys
import time
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from turboquant_ref import TurboQuantMSE, TurboQuantProd


# ============================================================================
# Test 1: round-trip on synthetic Gaussian data
# ============================================================================

def test_roundtrip_synthetic():
    print("=" * 70)
    print("Test 1: round-trip MSE on synthetic d=200")
    print("=" * 70)
    rng = np.random.default_rng(0)
    X = rng.standard_normal((1000, 200))
    X /= np.linalg.norm(X, axis=1, keepdims=True)

    for b in [2, 3, 4, 5]:
        tq = TurboQuantMSE(d=200, bits=b, seed=42)
        idx = tq.quantize(X)
        X_hat = tq.dequantize(idx)
        mse = np.mean(np.sum((X - X_hat) ** 2, axis=1))  # per-vector MSE
        # Theoretical bound from the paper: E[||x - x_hat||^2] ≤ (sqrt(3π)/2) * 4^(-b)
        bound = (np.sqrt(3 * np.pi) / 2) * 4 ** (-b)
        print(f"  b={b}  empirical MSE per vector = {mse:.4f}  "
              f"bound = {bound:.4f}  ratio = {mse/bound:.2f}")


# ============================================================================
# Test 2: GloVe-200 R@1
# ============================================================================

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


def test_glove200():
    print()
    print("=" * 70)
    print("Test 2: GloVe-200 (10000 queries, full)")
    print("Expected from paper plot: R@1 ≈ 0.55 at 2 bits, ≈ 0.86 at 4 bits")
    print("=" * 70)

    DATA = ROOT / "third_party" / "Extended-RaBitQ" / "data" / "glove200_100k"
    X = np.load(DATA / "X.npy").astype(np.float64)
    Xq = np.load(DATA / "Xq.npy").astype(np.float64)
    GT = np.load(DATA / "GT.npy").astype(np.int32)
    # Normalize
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    Xq /= np.linalg.norm(Xq, axis=1, keepdims=True)

    for b in [2, 4]:
        tq = TurboQuantProd(d=200, bits=b, seed=42)
        t0 = time.time()
        compressed = tq.quantize(X)
        t_q = time.time() - t0
        t0 = time.time()
        scores = tq.inner_product(Xq, compressed)
        t_s = time.time() - t0
        recalls = _recall_at_ks(scores, GT, [1, 4, 32])
        print(f"  bits={b}  quant={t_q:.1f}s  search={t_s:.1f}s  "
              f"R@1={recalls[1]:.4f}  R@4={recalls[4]:.4f}  R@32={recalls[32]:.4f}")


if __name__ == "__main__":
    test_roundtrip_synthetic()
    test_glove200()