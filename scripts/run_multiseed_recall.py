#!/usr/bin/env python
"""Multi-seed variance experiment for hierarchical TurboQuant.

Repeats the two-tier prefilter+rerank experiment with multiple seeds for:
  - The Haar rotation matrix R
  - The Lloyd-Max codebook construction

Reports mean ± std for each (dataset, strategy, T) cell.

Outputs:
  - results/multiseed_recall.csv
  - results/figures/multiseed_<dataset>.png
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

T_VALUES = [10, 50, 100, 500, 1000]
KS = [1, 4, 8,16,32]
BITS_CHEAP = 2
BITS_EXPENSIVE = 4
STRATEGIES = ["native", "top_down", "bottom_up", "middle_anchor"]

QUERY_BATCH = 512
N_QUERIES_DEFAULT = 1000
N_SEEDS_DEFAULT = 5


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
# Quantize+search core
# ============================================================================

def turboquant_scores(
        X: np.ndarray, Xq: np.ndarray, levels: np.ndarray, R: np.ndarray,
        query_batch: int = QUERY_BATCH,
) -> np.ndarray:
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


def rerank_candidates(
        X: np.ndarray, Xq: np.ndarray,
        candidates: np.ndarray, levels_expensive: np.ndarray, R: np.ndarray,
) -> np.ndarray:
    d = X.shape[1]
    scale = np.sqrt(d)
    NQ, T = candidates.shape
    rotated = X @ R
    rotated_std = rotated * scale
    idx = _quantize_to_levels(rotated_std, levels_expensive).astype(np.int32)
    rotated_hat_expensive = levels_expensive[idx] / scale
    q_rot_all = Xq @ R
    scores = np.empty((NQ, T), dtype=np.float32)
    for i in range(NQ):
        cand_recon = rotated_hat_expensive[candidates[i]]
        scores[i] = (cand_recon @ q_rot_all[i]).astype(np.float32)
    return scores


def recall_at_k(top_idx: np.ndarray, GT: np.ndarray, k: int) -> float:
    return float((top_idx[:, :k] == GT[:, 0:1]).any(axis=1).mean())


def two_tier_pipeline(
        X: np.ndarray, Xq: np.ndarray, GT: np.ndarray,
        levels_cheap: np.ndarray, levels_expensive: np.ndarray, R: np.ndarray,
        T: int, ks: list[int],
) -> dict[int, float]:
    cheap_scores = turboquant_scores(X, Xq, levels_cheap, R)
    top_T_idx = np.argpartition(-cheap_scores, T, axis=1)[:, :T]
    rerank_scores = rerank_candidates(X, Xq, top_T_idx, levels_expensive, R)
    sort_order = np.argsort(-rerank_scores, axis=1)
    top_sorted_local = np.take_along_axis(top_T_idx, sort_order, axis=1)
    return {k: recall_at_k(top_sorted_local, GT, k) for k in ks}


def single_tier_pipeline(
        X: np.ndarray, Xq: np.ndarray, GT: np.ndarray,
        levels: np.ndarray, R: np.ndarray, ks: list[int],
) -> dict[int, float]:
    scores = turboquant_scores(X, Xq, levels, R)
    top_max = max(ks)
    top_idx = np.argpartition(-scores, top_max, axis=1)[:, :top_max]
    row_scores = np.take_along_axis(scores, top_idx, axis=1)
    sort_order = np.argsort(-row_scores, axis=1)
    top_sorted = np.take_along_axis(top_idx, sort_order, axis=1)
    return {k: recall_at_k(top_sorted, GT, k) for k in ks}


# ============================================================================
# Codebook construction (per-seed)
# ============================================================================

def build_codebooks(strategy: str, bits_list: list[int], seed: int):
    if strategy == "native":
        return {b: lloyd_max_unconstrained(2 ** b, seed=seed) for b in bits_list}
    if strategy == "top_down":
        return build_top_down_codebooks(bits_list, seed=seed)
    if strategy == "bottom_up":
        return build_hierarchical_codebooks(bits_list, seed=seed)
    if strategy == "middle_anchor":
        from middle_anchor import build_middle_anchor_codebooks
        bits_full = sorted(set(bits_list) | {3})
        cbs = build_middle_anchor_codebooks(bits_full, anchor_bits=3, seed=seed)
        return {b: cbs[b] for b in bits_list}
    raise ValueError(strategy)


# ============================================================================
# Main loop
# ============================================================================

def main():
    n_queries = int(os.environ.get("N_QUERIES", N_QUERIES_DEFAULT))
    n_seeds = int(os.environ.get("N_SEEDS", N_SEEDS_DEFAULT))
    base_seed = int(os.environ.get("BASE_SEED", 42))

    print(f"Datasets: {DATASETS}")
    print(f"Strategies: {STRATEGIES}")
    print(f"T values: {T_VALUES}")
    print(f"N_queries: {n_queries}")
    print(f"N_seeds: {n_seeds}  (base seed = {base_seed})")
    print()

    csv_path = RESULTS_DIR / "multiseed_recall.csv"
    with open(csv_path, "w", newline="") as f:
        header = ["dataset", "d", "strategy", "T", "seed"]
        for k in [1, 2, 4, 8, 16, 32]:
            header.extend([f"single_b2_R@{k}", f"single_b4_R@{k}", f"two_tier_R@{k}"])
        csv.writer(f).writerow(header)

    seeds = [base_seed + i for i in range(n_seeds)]
    # results[ds][strat][T] = list of R@1 values across seeds
    results: dict = {ds: {s: {T: [] for T in T_VALUES} for s in STRATEGIES} for ds in DATASETS}
    targets: dict = {ds: {s: [] for s in STRATEGIES} for ds in DATASETS}

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
        if n_queries < Xq.shape[0]:
            Xq = Xq[:n_queries]
            GT = GT[:n_queries]
        print(f"  N={X.shape[0]}  d={d}  NQ={Xq.shape[0]}")

        for strat in STRATEGIES:
            print(f"\n  Strategy: {strat}")
            for seed in seeds:
                R = haar_rotation(d, seed=seed)
                cbs = build_codebooks(strat, [BITS_CHEAP, BITS_EXPENSIVE], seed=seed)
                lc = cbs[BITS_CHEAP]
                le = cbs[BITS_EXPENSIVE]

                # Single-tier references
                single_b2 = single_tier_pipeline(X, Xq, GT, lc, R, KS)
                single_b4 = single_tier_pipeline(X, Xq, GT, le, R, KS)
                targets[ds_name][strat].append(single_b4[1])

                row_prefix = f"    seed={seed}  b2={single_b2[1]:.4f}  b4={single_b4[1]:.4f}  "
                tt_results = {}
                for T in T_VALUES:
                    tt = two_tier_pipeline(X, Xq, GT, lc, le, R, T, KS)
                    tt_results[T] = tt
                    results[ds_name][strat][T].append(tt[1])

                    with open(csv_path, "a", newline="") as f:
                        csv.writer(f).writerow([
                            ds_name, d, strat, T, seed,
                            single_b2[1], single_b4[1], tt[1],
                            single_b2[4], single_b4[4], tt[4],
                            single_b2[32], single_b4[32], tt[32],
                        ])
                tt_str = "  ".join(f"T={T}:{tt_results[T][1]:.3f}" for T in T_VALUES)
                print(row_prefix + tt_str)

    # Summary table
    print("\n" + "=" * 90)
    print("Mean ± Std across seeds (R@1)")
    print("=" * 90)
    for ds_name in DATASETS:
        if not results[ds_name]:
            continue
        print(f"\n  {ds_name}")
        print(f"  {'strategy':<12s} {'target b=4':>15s} " +
              "  ".join(f"T={T:>4d}".rjust(13) for T in T_VALUES))
        for strat in STRATEGIES:
            if not targets[ds_name][strat]:
                continue
            tgt = np.array(targets[ds_name][strat])
            tgt_str = f"{tgt.mean():.3f}±{tgt.std():.3f}"
            tt_strs = []
            for T in T_VALUES:
                vals = np.array(results[ds_name][strat][T])
                if len(vals) == 0:
                    tt_strs.append("        n/a")
                else:
                    tt_strs.append(f"{vals.mean():.3f}±{vals.std():.3f}")
            print(f"  {strat:<12s} {tgt_str:>15s} " +
                  "  ".join(s.rjust(13) for s in tt_strs))

    # Per-dataset plot with error bars
    for ds_name in DATASETS:
        if not results[ds_name]:
            continue
        _plot_with_errorbars(ds_name, results[ds_name], targets[ds_name])

    print(f"\nDone. CSV: {csv_path}")


def _plot_with_errorbars(ds_name: str, ds_results: dict, ds_targets: dict):
    """R@1 vs T with error bars (mean ± std across seeds), per strategy."""
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = {"native": "#404040", "top_down": "#ff7f0e", "bottom_up": "#1f77b4"}
    markers = {"native": "o", "top_down": "s", "bottom_up": "^"}

    for strat in STRATEGIES:
        if not ds_targets.get(strat):
            continue
        means, stds = [], []
        for T in T_VALUES:
            vals = np.array(ds_results[strat][T])
            if len(vals) == 0:
                continue
            means.append(vals.mean())
            stds.append(vals.std())
        Ts = T_VALUES[:len(means)]
        ax.errorbar(Ts, means, yerr=stds,
                    color=colors[strat], marker=markers[strat], markersize=8, linewidth=2,
                    capsize=4, label=strat)
        # Target line (mean across seeds)
        target_mean = np.mean(ds_targets[strat])
        ax.axhline(target_mean, color=colors[strat], linestyle=":", alpha=0.4, linewidth=1)

    ax.set_xscale("log")
    ax.set_xlabel("Prefilter top-T", fontsize=12)
    ax.set_ylabel("Two-tier R@1 (mean ± std)", fontsize=12)
    ax.set_title(f"Multi-seed two-tier R@1 — {ds_name}\n(error bars = std across seeds)",
                 fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right", fontsize=10)
    fig.tight_layout()
    out = FIGS_DIR / f"multiseed_{ds_name}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


if __name__ == "__main__":
    main()