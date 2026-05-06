#!/usr/bin/env python
"""PQ + reranking baseline for two-tier retrieval comparison.

Tests three pipelines:
  1. PQ-2bit prefilter + PQ-4bit rerank  (independent PQ codebooks)
  2. PQ-2bit prefilter + true float rerank  (idealized upper bound)
  3. Single-tier PQ-4bit (accuracy ceiling)

Compares against the top-down nested TurboQuant approach (run_two_tier_recall.py).

Outputs:
  - results/pq_rerank_baseline.csv
"""
from __future__ import annotations

from pathlib import Path
import csv
import os
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

DATA_DIR = ROOT / "third_party" / "Extended-RaBitQ" / "data"
RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Lazy import: install faiss if not present
try:
    import faiss
except ImportError:
    print("faiss not installed. Run: pip install faiss-cpu")
    sys.exit(1)


# ============================================================================
# Config
# ============================================================================

DATASETS_DEFAULT = ["glove200_100k", "openai1536", "openai3072"]
DATASETS = os.environ.get("DATASETS", " ".join(DATASETS_DEFAULT)).split()

T_VALUES = [10, 50, 100, 500, 1000, 5000]
KS = [1, 2, 4, 8, 16, 32]
BITS_CHEAP = 2
BITS_EXPENSIVE = 4

# Per-dataset query counts
N_QUERIES_DEFAULTS = {
    "glove200_100k": 10000,
    "openai1536": 1000,
    "openai3072": 1000,
}
N_QUERIES_GLOVE = int(os.environ.get("N_QUERIES_GLOVE", N_QUERIES_DEFAULTS["glove200_100k"]))
N_QUERIES_OPENAI = int(os.environ.get("N_QUERIES_OPENAI", N_QUERIES_DEFAULTS["openai1536"]))


def get_n_queries(name: str) -> int:
    if name.startswith("glove"):
        return N_QUERIES_GLOVE
    return N_QUERIES_OPENAI


# ============================================================================
# Data loading
# ============================================================================

def load_dataset(name: str):
    path = DATA_DIR / name
    X = np.load(path / "X.npy").astype(np.float32)
    Xq = np.load(path / "Xq.npy").astype(np.float32)
    GT = np.load(path / "GT.npy").astype(np.int32)
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    Xq /= np.linalg.norm(Xq, axis=1, keepdims=True)
    return X, Xq, GT


# ============================================================================
# Recall computation
# ============================================================================

def recall_at_k_from_indices(top_indices, GT, k):
    topk = top_indices[:, :k]
    gt1 = GT[:, 0:1]
    return float((topk == gt1).any(axis=1).mean())


def recall_at_ks_from_scores(scores, GT, ks):
    top_max = max(ks)
    top_idx = np.argpartition(-scores, top_max, axis=1)[:, :top_max]
    row_scores = np.take_along_axis(scores, top_idx, axis=1)
    sort_order = np.argsort(-row_scores, axis=1)
    top_sorted = np.take_along_axis(top_idx, sort_order, axis=1)
    return {k: recall_at_k_from_indices(top_sorted, GT, k) for k in ks}


# ============================================================================
# PQ helpers
# ============================================================================

def fit_pq(X: np.ndarray, nbits: int, m: int = None) -> "faiss.ProductQuantizer":
    d = X.shape[1]
    if m is None:
        m = d
    pq = faiss.ProductQuantizer(d, m, nbits)
    pq.train(X)
    return pq


def pq_decode(pq, X):
    codes = pq.compute_codes(X)
    return pq.decode(codes)


def pq_scores(X_hat, Xq, query_batch=512):
    NQ = Xq.shape[0]
    out = np.empty((NQ, X_hat.shape[0]), dtype=np.float32)
    for s in range(0, NQ, query_batch):
        e = min(s + query_batch, NQ)
        out[s:e] = Xq[s:e] @ X_hat.T
    return out


# ============================================================================
# Two-tier rerank using a different codebook
# ============================================================================

def rerank_with_codebook(X_hat, Xq, candidates):
    NQ, T = candidates.shape
    scores = np.empty((NQ, T), dtype=np.float32)
    for i in range(NQ):
        scores[i] = X_hat[candidates[i]] @ Xq[i]
    return scores


def two_tier_pq_pq(X_hat_cheap, X_hat_expensive, Xq, GT, T, ks):
    cheap_scores = pq_scores(X_hat_cheap, Xq)
    top_T_idx = np.argpartition(-cheap_scores, T, axis=1)[:, :T]
    rerank_scores = rerank_with_codebook(X_hat_expensive, Xq, top_T_idx)
    sort_order = np.argsort(-rerank_scores, axis=1)
    top_sorted = np.take_along_axis(top_T_idx, sort_order, axis=1)
    return {k: recall_at_k_from_indices(top_sorted, GT, k) for k in ks}


def two_tier_pq_float(X_hat_cheap, X, Xq, GT, T, ks):
    cheap_scores = pq_scores(X_hat_cheap, Xq)
    top_T_idx = np.argpartition(-cheap_scores, T, axis=1)[:, :T]
    NQ = Xq.shape[0]
    rerank_scores = np.empty((NQ, T), dtype=np.float32)
    for i in range(NQ):
        rerank_scores[i] = X[top_T_idx[i]] @ Xq[i]
    sort_order = np.argsort(-rerank_scores, axis=1)
    top_sorted = np.take_along_axis(top_T_idx, sort_order, axis=1)
    return {k: recall_at_k_from_indices(top_sorted, GT, k) for k in ks}


# ============================================================================
# Main loop
# ============================================================================

def main():
    print(f"PQ + rerank baseline")
    print(f"Datasets: {DATASETS}")
    print(f"Bits: cheap={BITS_CHEAP}, expensive={BITS_EXPENSIVE}")
    print(f"T values: {T_VALUES}")
    print()

    csv_path = RESULTS_DIR / "pq_rerank_baseline.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        header = ["dataset", "d", "pipeline", "T"]
        for k in KS:
            header.append(f"R@{k}")
        header.extend(["t_train_cheap", "t_train_exp", "t_search"])
        w.writerow(header)

    for ds_name in DATASETS:
        print(f"\n{'=' * 80}")
        print(f"Dataset: {ds_name}")
        print(f"{'=' * 80}")
        try:
            X, Xq, GT = load_dataset(ds_name)
        except FileNotFoundError as e:
            print(f"  SKIP: {e}")
            continue
        d = X.shape[1]
        n_q = get_n_queries(ds_name)
        if n_q < Xq.shape[0]:
            Xq = Xq[:n_q]
            GT = GT[:n_q]
        print(f"  N={X.shape[0]}  d={d}  NQ={Xq.shape[0]}")

        # Train PQ codebooks
        print(f"\n  Training PQ-{BITS_CHEAP}bit ...")
        t0 = time.time()
        pq_cheap = fit_pq(X, nbits=BITS_CHEAP, m=d)
        t_train_cheap = time.time() - t0
        X_hat_cheap = pq_decode(pq_cheap, X)
        print(f"    {t_train_cheap:.1f}s")

        print(f"  Training PQ-{BITS_EXPENSIVE}bit ...")
        t0 = time.time()
        pq_exp = fit_pq(X, nbits=BITS_EXPENSIVE, m=d)
        t_train_exp = time.time() - t0
        X_hat_exp = pq_decode(pq_exp, X)
        print(f"    {t_train_exp:.1f}s")

        # Single-tier b=4 ceiling
        print(f"\n  --- Single-tier PQ-{BITS_EXPENSIVE}bit (full scan) ---")
        t0 = time.time()
        scores = pq_scores(X_hat_exp, Xq)
        t_search = time.time() - t0
        rec = recall_at_ks_from_scores(scores, GT, KS)
        print(f"    t={t_search:.1f}s  " + "  ".join(f"R@{k}={rec[k]:.4f}" for k in KS))
        with open(csv_path, "a", newline="") as f:
            w = csv.writer(f)
            w.writerow([ds_name, d, "single_tier_pq4bit", -1] + [rec[k] for k in KS] +
                       [t_train_cheap, t_train_exp, t_search])

        # Single-tier b=2
        print(f"\n  --- Single-tier PQ-{BITS_CHEAP}bit (full scan) ---")
        t0 = time.time()
        scores = pq_scores(X_hat_cheap, Xq)
        t_search = time.time() - t0
        rec = recall_at_ks_from_scores(scores, GT, KS)
        print(f"    t={t_search:.1f}s  " + "  ".join(f"R@{k}={rec[k]:.4f}" for k in KS))
        with open(csv_path, "a", newline="") as f:
            w = csv.writer(f)
            w.writerow([ds_name, d, "single_tier_pq2bit", -1] + [rec[k] for k in KS] +
                       [t_train_cheap, t_train_exp, t_search])

        # Two-tier PQ-2 + PQ-4
        print(f"\n  --- Two-tier: PQ-{BITS_CHEAP}bit prefilter + PQ-{BITS_EXPENSIVE}bit rerank ---")
        for T in T_VALUES:
            if T >= X.shape[0]:
                continue
            t0 = time.time()
            rec = two_tier_pq_pq(X_hat_cheap, X_hat_exp, Xq, GT, T, KS)
            t_search = time.time() - t0
            print(f"    T={T:>5d}  t={t_search:.1f}s  " +
                  "  ".join(f"R@{k}={rec[k]:.4f}" for k in KS))
            with open(csv_path, "a", newline="") as f:
                w = csv.writer(f)
                w.writerow([ds_name, d, "two_tier_pq2_pq4", T] + [rec[k] for k in KS] +
                           [t_train_cheap, t_train_exp, t_search])

        # Two-tier PQ-2 + float (idealized upper bound)
        print(f"\n  --- Two-tier: PQ-{BITS_CHEAP}bit prefilter + FLOAT rerank (idealized) ---")
        for T in T_VALUES:
            if T >= X.shape[0]:
                continue
            t0 = time.time()
            rec = two_tier_pq_float(X_hat_cheap, X, Xq, GT, T, KS)
            t_search = time.time() - t0
            print(f"    T={T:>5d}  t={t_search:.1f}s  " +
                  "  ".join(f"R@{k}={rec[k]:.4f}" for k in KS))
            with open(csv_path, "a", newline="") as f:
                w = csv.writer(f)
                w.writerow([ds_name, d, "two_tier_pq2_float", T] + [rec[k] for k in KS] +
                           [t_train_cheap, t_train_exp, t_search])

    print(f"\nDone. CSV: {csv_path}")


if __name__ == "__main__":
    main()