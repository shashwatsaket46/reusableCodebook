#!/usr/bin/env python
"""Recall@k experiment for hierarchical TurboQuant codebooks.

For each (dataset, bits, strategy) combination, measure R@k using
TurboQuant's MSE quantizer with the specified codebook.

Strategies:
  - native    : unconstrained Lloyd-Max for that bit-width (no nesting)
  - bottom_up : nested, b=2 unconstrained, higher b's pay penalty
  - top_down  : nested, b=5 unconstrained, lower b's pay penalty

Outputs:
  - results/hierarchical_recall.csv
  - results/figures/hierarchical_recall_<dataset>.png
"""
from __future__ import annotations

from pathlib import Path
import csv
import os
import sys
import time

import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from hierarchical_codebook import (
    lloyd_max_unconstrained,
    build_hierarchical_codebooks,
    build_top_down_codebooks,
)
from turboquant_ref import haar_rotation, _quantize_to_levels


# ============================================================================
# Config
# ============================================================================

DATA_DIR = ROOT / "third_party" / "Extended-RaBitQ" / "data"
RESULTS_DIR = ROOT / "results"
FIGS_DIR = RESULTS_DIR / "figures"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
FIGS_DIR.mkdir(parents=True, exist_ok=True)

# Datasets configurable via env
DATASETS_DEFAULT = ["glove200_100k", "openai1536", "openai3072"]
DATASETS = os.environ.get("DATASETS", " ".join(DATASETS_DEFAULT)).split()
BITS = [2, 3, 4, 5]
STRATEGIES = ["native", "bottom_up", "top_down"]
KS = [1, 2, 4, 8, 16, 32, 64]
QUERY_BATCH = 512  # for matmul chunking
N_QUERIES_DEFAULT = 1000  # subsample for tractability


# ============================================================================
# Data loading
# ============================================================================

def load_dataset(name: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load (X, Xq, GT). Vectors normalized to unit length. GT shape: (NQ, k)."""
    path = DATA_DIR / name
    X = np.load(path / "X.npy").astype(np.float64)
    Xq = np.load(path / "Xq.npy").astype(np.float64)
    GT = np.load(path / "GT.npy").astype(np.int32)
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    Xq /= np.linalg.norm(Xq, axis=1, keepdims=True)
    return X, Xq, GT


# ============================================================================
# TurboQuant MSE search with arbitrary codebook
# ============================================================================

def turboquant_mse_search(
        X: np.ndarray,
        Xq: np.ndarray,
        levels: np.ndarray,
        R: np.ndarray,
        query_batch: int = QUERY_BATCH,
) -> np.ndarray:
    """Run TurboQuant MSE quantize + search using a custom codebook.

    Args:
        X: (N, d) unit-norm base vectors.
        Xq: (NQ, d) unit-norm queries.
        levels: 1D codebook levels (in standardized coordinate space).
        R: (d, d) Haar rotation matrix.
        query_batch: chunk size for memory-bounded search.

    Returns:
        scores: (NQ, N) float32 inner-product estimates.
    """
    d = X.shape[1]
    scale = np.sqrt(d)

    # Quantize DB
    rotated = X @ R                                       # (N, d)
    rotated_std = rotated * scale                          # standardize
    idx = _quantize_to_levels(rotated_std, levels).astype(np.int32)
    rotated_hat = levels[idx] / scale                      # (N, d) reconstructed

    # Search
    NQ, N = Xq.shape[0], X.shape[0]
    out = np.empty((NQ, N), dtype=np.float32)
    for start in range(0, NQ, query_batch):
        end = min(start + query_batch, NQ)
        q_rot = Xq[start:end] @ R                          # (B, d)
        out[start:end] = (q_rot @ rotated_hat.T).astype(np.float32)
    return out


# ============================================================================
# Recall@k
# ============================================================================

def recall_at_ks(scores: np.ndarray, GT: np.ndarray, ks: list[int]) -> dict[int, float]:
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


# ============================================================================
# Codebook construction (cache across datasets — same Gaussian codebook)
# ============================================================================

def build_all_codebooks(bits_list: list[int]) -> dict[tuple[str, int], np.ndarray]:
    """Build the three strategies' codebooks for all bit-widths."""
    cbs = {}
    print("Building codebooks (one-time, dimension-independent for Gaussian)...")
    # Native
    for b in bits_list:
        cbs[("native", b)] = lloyd_max_unconstrained(2 ** b)
    # Bottom-up nested
    bu = build_hierarchical_codebooks(bits_list)
    for b in bits_list:
        cbs[("bottom_up", b)] = bu[b]
    # Top-down nested
    td = build_top_down_codebooks(bits_list)
    for b in bits_list:
        cbs[("top_down", b)] = td[b]
    print(f"  Built {len(cbs)} codebooks.")
    return cbs


# ============================================================================
# Main experiment loop
# ============================================================================

def main():
    n_queries_str = os.environ.get("N_QUERIES", str(N_QUERIES_DEFAULT))
    n_queries = int(n_queries_str) if n_queries_str.isdigit() else N_QUERIES_DEFAULT

    print(f"Datasets: {DATASETS}")
    print(f"Bit-widths: {BITS}")
    print(f"Strategies: {STRATEGIES}")
    print(f"N_queries per dataset: {n_queries}")
    print()

    codebooks = build_all_codebooks(BITS)

    # Pre-output CSV header
    csv_path = RESULTS_DIR / "hierarchical_recall.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["dataset", "d", "strategy", "bits", "quant_time", "search_time"]
                   + [f"R@{k}" for k in KS])

    # For plot data: results[dataset][strategy][bits] = {k: recall}
    results: dict = {ds: {s: {} for s in STRATEGIES} for ds in DATASETS}

    for ds_name in DATASETS:
        print(f"\n{'=' * 70}")
        print(f"Dataset: {ds_name}")
        print(f"{'=' * 70}")
        try:
            X, Xq, GT = load_dataset(ds_name)
        except FileNotFoundError as e:
            print(f"  SKIP: {e}")
            continue

        d = X.shape[1]
        print(f"  N={X.shape[0]}  d={d}  NQ_total={Xq.shape[0]}")

        # Subsample queries
        if n_queries < Xq.shape[0]:
            Xq = Xq[:n_queries]
            GT = GT[:n_queries]
            print(f"  Subsampled to NQ={n_queries}")

        # One Haar rotation, shared across all strategies (so they're directly comparable)
        R = haar_rotation(d, seed=42)

        for strat in STRATEGIES:
            print(f"\n  Strategy: {strat}")
            for b in BITS:
                levels = codebooks[(strat, b)]
                t0 = time.time()
                scores = turboquant_mse_search(X, Xq, levels, R)
                t_total = time.time() - t0
                recalls = recall_at_ks(scores, GT, KS)
                results[ds_name][strat][b] = recalls

                rec_str = "  ".join(f"R@{k}={recalls[k]:.4f}" for k in KS)
                print(f"    b={b}  ({t_total:5.1f}s)  {rec_str}")

                # Append to CSV
                with open(csv_path, "a", newline="") as f:
                    w = csv.writer(f)
                    w.writerow([ds_name, d, strat, b, "", t_total]
                               + [recalls[k] for k in KS])

    # =========================================================================
    # Plotting
    # =========================================================================
    print("\nGenerating figures...")
    for ds_name in DATASETS:
        if not any(results[ds_name][s] for s in STRATEGIES):
            continue
        _plot_dataset(ds_name, results[ds_name])

    print(f"\nDone. CSV: {csv_path}")


# ============================================================================
# Plotting
# ============================================================================

def _plot_dataset(ds_name: str, ds_results: dict):
    """Plot R@k vs k for native/bottom_up/top_down at b=2 and b=4."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), sharey=True)

    bit_groups = [(2, axes[0]), (4, axes[1])]
    color_map = {"native": "#404040", "bottom_up": "#1f77b4", "top_down": "#ff7f0e"}
    style_map = {"native": "-", "bottom_up": "--", "top_down": "-."}
    marker_map = {"native": "o", "bottom_up": "s", "top_down": "^"}

    for b, ax in bit_groups:
        for strat in STRATEGIES:
            if b not in ds_results.get(strat, {}):
                continue
            recalls = ds_results[strat][b]
            ys = [recalls[k] for k in KS]
            ax.plot(KS, ys,
                    color=color_map[strat], linestyle=style_map[strat],
                    marker=marker_map[strat], markersize=8, linewidth=2,
                    label=f"{strat} (b={b})")
        ax.set_xscale("log", base=2)
        ax.set_xticks(KS)
        ax.set_xticklabels([str(k) for k in KS])
        ax.set_xlabel("Top-k", fontsize=12)
        ax.set_title(f"b = {b} bits/dim", fontsize=13)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="lower right", fontsize=10)
    axes[0].set_ylabel("Recall@1@k", fontsize=12)
    fig.suptitle(f"Hierarchical TurboQuant on {ds_name}", fontsize=14)
    fig.tight_layout()
    out = FIGS_DIR / f"hierarchical_recall_{ds_name}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


if __name__ == "__main__":
    main()