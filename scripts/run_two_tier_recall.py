#!/usr/bin/env python
"""Two-tier prefilter+rerank experiment for hierarchical TurboQuant.

Compares three pipelines:
  1. Single-tier b=2  : scan all N at 2 bits (cheap, lossy)
  2. Single-tier b=4  : scan all N at 4 bits (slow, accurate)
  3. Two-tier (b=2 + b=4): scan all N at 2 bits → keep top-T → rerank at 4 bits

Codebook strategies:
  - native   : independent codebooks per bit-width (no sharing)
  - top_down : one shared b=5 codebook, decimated to b=2 and b=4
  - bottom_up: one shared b=5 codebook, b=2 unconstrained anchor

Outputs:
  - results/two_tier_recall.csv
  - results/figures/two_tier_<dataset>.png
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

DATASETS_DEFAULT = ["glove200_100k", "openai1536", "openai3072"]
DATASETS = os.environ.get("DATASETS", " ".join(DATASETS_DEFAULT)).split()

# Prefilter top-T values to sweep
T_VALUES = [10, 50, 100, 500, 1000, 5000]

# Final R@k metrics
KS = [1, 4, 32]

# Bit-widths used in the two-tier pipeline
BITS_CHEAP = 2  # prefilter
BITS_EXPENSIVE = 4  # rerank

# Strategies to test
STRATEGIES = ["native", "top_down", "bottom_up"]

QUERY_BATCH = 512
N_QUERIES_DEFAULT = 1000


# ============================================================================
# Data loading
# ============================================================================

def load_dataset(name: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    path = DATA_DIR / name
    X = np.load(path / "X.npy").astype(np.float64)
    Xq = np.load(path / "Xq.npy").astype(np.float64)
    GT = np.load(path / "GT.npy").astype(np.int32)
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    Xq /= np.linalg.norm(Xq, axis=1, keepdims=True)
    return X, Xq, GT


# ============================================================================
# Quantize+search core (returns scores)
# ============================================================================

def turboquant_scores(
        X: np.ndarray,
        Xq: np.ndarray,
        levels: np.ndarray,
        R: np.ndarray,
        query_batch: int = QUERY_BATCH,
) -> np.ndarray:
    """Returns (NQ, N) score matrix. Quantize DB at given levels, search."""
    d = X.shape[1]
    scale = np.sqrt(d)
    rotated = X @ R
    rotated_std = rotated * scale
    idx = _quantize_to_levels(rotated_std, levels).astype(np.int32)
    rotated_hat = levels[idx] / scale
    NQ, N = Xq.shape[0], X.shape[0]
    out = np.empty((NQ, N), dtype=np.float32)
    for start in range(0, NQ, query_batch):
        end = min(start + query_batch, NQ)
        q_rot = Xq[start:end] @ R
        out[start:end] = (q_rot @ rotated_hat.T).astype(np.float32)
    return out


# ============================================================================
# Per-query rerank: given prefilter candidate indices, score them at b_expensive
# ============================================================================

def rerank_candidates(
        X: np.ndarray,
        Xq: np.ndarray,
        candidates: np.ndarray,         # (NQ, T) — indices of top-T per query
        levels_expensive: np.ndarray,
        R: np.ndarray,
) -> np.ndarray:
    """For each query, rescore its T candidates using the expensive codebook.

    Returns (NQ, T) reranked scores (higher is better) — and *parallel*
    to `candidates` so the i-th score corresponds to the i-th candidate.
    """
    d = X.shape[1]
    scale = np.sqrt(d)
    NQ, T = candidates.shape

    # Quantize the *candidate* vectors at b_expensive
    # (Cleaner: precompute the entire b_expensive reconstruction, then index.
    # Faster but uses more memory. For 100k vectors at d=200 that's 80MB — fine.)
    rotated = X @ R
    rotated_std = rotated * scale
    idx = _quantize_to_levels(rotated_std, levels_expensive).astype(np.int32)
    rotated_hat_expensive = levels_expensive[idx] / scale  # (N, d)

    # Pre-rotate queries
    q_rot_all = Xq @ R                                      # (NQ, d)

    # Score each query against its T candidates
    scores = np.empty((NQ, T), dtype=np.float32)
    for i in range(NQ):
        # Get the d-dim reconstructed vectors for this query's T candidates
        cand_recon = rotated_hat_expensive[candidates[i]]   # (T, d)
        scores[i] = (cand_recon @ q_rot_all[i]).astype(np.float32)
    return scores


# ============================================================================
# Recall computation
# ============================================================================

def recall_at_k_from_indices(top_indices: np.ndarray, GT: np.ndarray, k: int) -> float:
    """top_indices: (NQ, k+) sorted-by-score predicted top indices. GT: (NQ, *)."""
    topk = top_indices[:, :k]
    gt1 = GT[:, 0:1]
    return float((topk == gt1).any(axis=1).mean())


def recall_at_ks_from_scores(scores: np.ndarray, GT: np.ndarray, ks: list[int]) -> dict[int, float]:
    top_max = max(ks)
    top_idx = np.argpartition(-scores, top_max, axis=1)[:, :top_max]
    row_scores = np.take_along_axis(scores, top_idx, axis=1)
    sort_order = np.argsort(-row_scores, axis=1)
    top_sorted = np.take_along_axis(top_idx, sort_order, axis=1)
    return {k: recall_at_k_from_indices(top_sorted, GT, k) for k in ks}


# ============================================================================
# Two-tier pipeline
# ============================================================================

def two_tier_recall(
        X: np.ndarray, Xq: np.ndarray, GT: np.ndarray,
        levels_cheap: np.ndarray, levels_expensive: np.ndarray,
        R: np.ndarray,
        T: int, ks: list[int],
) -> tuple[dict[int, float], float, float]:
    """
    Returns:
      recalls : dict {k: R@k from two-tier pipeline}
      t_prefilter : seconds spent on cheap scan
      t_rerank    : seconds spent on rerank
    """
    # Stage 1: cheap-tier prefilter — full scan at b=2
    t0 = time.time()
    cheap_scores = turboquant_scores(X, Xq, levels_cheap, R)
    # Top-T candidates per query (unsorted — fine for prefilter)
    top_T_idx = np.argpartition(-cheap_scores, T, axis=1)[:, :T]   # (NQ, T)
    t_prefilter = time.time() - t0

    # Stage 2: rerank using expensive codebook
    t0 = time.time()
    rerank_scores = rerank_candidates(X, Xq, top_T_idx, levels_expensive, R)  # (NQ, T)
    # Sort within candidates
    top_max = max(ks)
    sort_order = np.argsort(-rerank_scores, axis=1)
    top_sorted_local = np.take_along_axis(top_T_idx, sort_order, axis=1)      # (NQ, T)
    t_rerank = time.time() - t0

    recalls = {k: recall_at_k_from_indices(top_sorted_local, GT, k) for k in ks}
    return recalls, t_prefilter, t_rerank


# ============================================================================
# Build all required codebook combinations
# ============================================================================

def build_codebooks_for_pipeline(strategy: str, bits_list: list[int]) -> dict[int, np.ndarray]:
    """For a given strategy, return codebooks at each bit-width.

    Note: for native, each bit-width is unconstrained.
    For top_down/bottom_up, codebooks share structure (nested).
    """
    if strategy == "native":
        return {b: lloyd_max_unconstrained(2 ** b) for b in bits_list}
    if strategy == "top_down":
        return build_top_down_codebooks(bits_list)
    if strategy == "bottom_up":
        return build_hierarchical_codebooks(bits_list)
    raise ValueError(f"Unknown strategy: {strategy}")


# ============================================================================
# Main experiment loop
# ============================================================================

def main():
    n_queries_str = os.environ.get("N_QUERIES", str(N_QUERIES_DEFAULT))
    n_queries = int(n_queries_str) if n_queries_str.isdigit() else N_QUERIES_DEFAULT

    print(f"Datasets: {DATASETS}")
    print(f"Strategies: {STRATEGIES}")
    print(f"Cheap-tier bits: {BITS_CHEAP}, expensive-tier bits: {BITS_EXPENSIVE}")
    print(f"Prefilter T values: {T_VALUES}")
    print(f"N_queries per dataset: {n_queries}")
    print()

    csv_path = RESULTS_DIR / "two_tier_recall.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["dataset", "d", "strategy", "T",
                    "single_b2_R@1", "single_b4_R@1", "two_tier_R@1",
                    "single_b2_R@4", "single_b4_R@4", "two_tier_R@4",
                    "single_b2_R@32", "single_b4_R@32", "two_tier_R@32",
                    "t_prefilter", "t_rerank"])

    all_results = {}  # all_results[ds][strategy] -> dict of results

    for ds_name in DATASETS:
        print(f"\n{'=' * 75}")
        print(f"Dataset: {ds_name}")
        print(f"{'=' * 75}")
        try:
            X, Xq, GT = load_dataset(ds_name)
        except FileNotFoundError as e:
            print(f"  SKIP: {e}")
            continue

        d = X.shape[1]
        print(f"  N={X.shape[0]}  d={d}  NQ_total={Xq.shape[0]}")

        if n_queries < Xq.shape[0]:
            Xq = Xq[:n_queries]
            GT = GT[:n_queries]
            print(f"  Subsampled to NQ={n_queries}")

        R = haar_rotation(d, seed=42)

        all_results[ds_name] = {}

        for strat in STRATEGIES:
            print(f"\n  Strategy: {strat}")
            cbs = build_codebooks_for_pipeline(strat, [BITS_CHEAP, BITS_EXPENSIVE])
            levels_cheap = cbs[BITS_CHEAP]
            levels_expensive = cbs[BITS_EXPENSIVE]

            # Reference: single-tier b=2 (cheap full scan)
            t0 = time.time()
            scores_cheap = turboquant_scores(X, Xq, levels_cheap, R)
            t_b2 = time.time() - t0
            single_b2 = recall_at_ks_from_scores(scores_cheap, GT, KS)

            # Reference: single-tier b=4 (expensive full scan — accuracy target)
            t0 = time.time()
            scores_expensive = turboquant_scores(X, Xq, levels_expensive, R)
            t_b4 = time.time() - t0
            single_b4 = recall_at_ks_from_scores(scores_expensive, GT, KS)

            print(f"    single b={BITS_CHEAP}: t={t_b2:.1f}s  "
                  + " ".join(f"R@{k}={single_b2[k]:.4f}" for k in KS))
            print(f"    single b={BITS_EXPENSIVE}: t={t_b4:.1f}s  "
                  + " ".join(f"R@{k}={single_b4[k]:.4f}" for k in KS))

            two_tier_results = {}
            for T in T_VALUES:
                if T >= X.shape[0]:
                    continue
                tt_recalls, t_pre, t_re = two_tier_recall(
                    X, Xq, GT, levels_cheap, levels_expensive, R, T, KS
                )
                two_tier_results[T] = {
                    "recalls": tt_recalls,
                    "t_prefilter": t_pre,
                    "t_rerank": t_re,
                }
                print(f"    two-tier T={T:>5d}: "
                      f"t_pre={t_pre:.1f}s t_re={t_re:.1f}s  "
                      + " ".join(f"R@{k}={tt_recalls[k]:.4f}" for k in KS))

                with open(csv_path, "a", newline="") as f:
                    w = csv.writer(f)
                    w.writerow([
                        ds_name, d, strat, T,
                        single_b2[1], single_b4[1], tt_recalls[1],
                        single_b2[4], single_b4[4], tt_recalls[4],
                        single_b2[32], single_b4[32], tt_recalls[32],
                        t_pre, t_re,
                    ])

            all_results[ds_name][strat] = {
                "single_b2": single_b2,
                "single_b4": single_b4,
                "two_tier": two_tier_results,
                "t_b2": t_b2,
                "t_b4": t_b4,
            }

    # =========================================================================
    # Plotting
    # =========================================================================
    print("\nGenerating figures...")
    for ds_name in all_results:
        if not all_results[ds_name]:
            continue
        _plot_two_tier(ds_name, all_results[ds_name])

    print(f"\nDone. CSV: {csv_path}")


# ============================================================================
# Plotting
# ============================================================================

def _plot_two_tier(ds_name: str, ds_results: dict):
    """For each strategy, plot two-tier R@1 vs T, with single-tier reference lines."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True)

    color_main = "#1f77b4"
    color_b2 = "#ff7f0e"
    color_b4 = "#2ca02c"

    for ax, strat in zip(axes, STRATEGIES):
        if strat not in ds_results:
            continue
        res = ds_results[strat]
        ts = sorted(res["two_tier"].keys())
        tt_r1 = [res["two_tier"][t]["recalls"][1] for t in ts]
        ax.plot(ts, tt_r1, "o-", color=color_main, markersize=8, linewidth=2,
                label="two-tier (b=2 → b=4)")
        ax.axhline(res["single_b4"][1], color=color_b4, linestyle="--",
                   linewidth=2, label=f"single b=4 (target)")
        ax.axhline(res["single_b2"][1], color=color_b2, linestyle=":",
                   linewidth=2, label=f"single b=2 (prefilter alone)")
        ax.set_xscale("log")
        ax.set_xticks(ts)
        ax.set_xticklabels([str(t) for t in ts], rotation=45, ha="right")
        ax.set_xlabel("Prefilter top-T", fontsize=11)
        ax.set_title(f"Strategy: {strat}", fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="lower right", fontsize=9)
    axes[0].set_ylabel("R@1", fontsize=11)
    fig.suptitle(f"Two-tier prefilter+rerank on {ds_name}", fontsize=13)
    fig.tight_layout()
    out = FIGS_DIR / f"two_tier_{ds_name}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


if __name__ == "__main__":
    main()