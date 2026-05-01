#!/usr/bin/env python
"""TurboQuant (Prod variant) baseline on the same datasets as PQ and ExRaBitQ.

TurboQuantProd uses b magnitude bits + 1 QJL bit = (b+1) total bits/dim.
Bit-budget mapping for fair comparison with PQ-Nbit / ExRaBitQ-Nbit:
  Target N bits/dim | TurboQuantProd b
  2                 | 1
  3                 | 2
  4                 | 3
"""
from __future__ import annotations

import os
import pickle
import time
from pathlib import Path

import numpy as np

from config_utils import enabled_datasets, ensure_int_list, load_config

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "third_party" / "Extended-RaBitQ" / "data"
PKLS = ROOT / "results" / "tq_pickles"
PKLS.mkdir(parents=True, exist_ok=True)


def _load_npy(name: str):
    out = DATA / name
    return (
        np.load(out / "X.npy"),
        np.load(out / "Xq.npy"),
        np.load(out / "GT.npy"),
    )


def _normalize(X: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms = np.where(norms < 1e-12, 1.0, norms)
    return X / norms


def _recall_at_ks(scores: np.ndarray, GT: np.ndarray, ks: list[int]) -> dict[int, float]:
    """Recall1@k: fraction of queries where top-1 GT lands in top-k of scores."""
    top_max = max(ks)
    top_idx = np.argpartition(-scores, top_max, axis=1)[:, :top_max]
    row_scores = np.take_along_axis(scores, top_idx, axis=1)
    sort_order = np.argsort(-row_scores, axis=1)
    top_sorted = np.take_along_axis(top_idx, sort_order, axis=1)

    out = {}
    for k in ks:
        topk = top_sorted[:, :k]
        gt1 = GT[:, 0:1]
        hits = (topk == gt1).any(axis=1)
        out[k] = float(hits.mean())
    return out


def run_turboquant_prod(X: np.ndarray, Xq: np.ndarray, b: int):
    """TurboQuantProd at (b+1) bits/dim total."""
    from turboquant.main.prod import TurboQuantProd

    d = X.shape[1]
    tq = TurboQuantProd(d=d, b=b)

    t0 = time.time()
    idx, qjl, gamma = tq.quantize(X)
    t_quant = time.time() - t0

    t0 = time.time()
    X_hat = tq.dequantize(idx, qjl, gamma)
    scores = Xq @ X_hat.T
    t_search = time.time() - t0

    return scores, t_quant, t_search


def run_for_dataset(name: str, ks: list[int], target_bits: list[int]):
    out_path = PKLS / f"tq_results_{name}.pkl"
    if out_path.exists():
        print(f"[{name}] tq_results already exist; loading & skipping recompute")
        return pickle.load(open(out_path, "rb"))

    print(f"[{name}] loading data ...")
    X, Xq, GT = _load_npy(name)
    X = _normalize(X.astype(np.float32))
    Xq = _normalize(Xq.astype(np.float32))

    res = {}
    for target in target_bits:
        if target < 2:
            print(f"  skipping target={target} (Prod requires target >= 2)")
            continue
        b_prod = target - 1
        print(f"  TurboQuantProd-{target}bit (b={b_prod}, total={target} bits/dim) ...",
              end=" ", flush=True)
        try:
            scores, tq_t, srch_t = run_turboquant_prod(X, Xq, b_prod)
            recalls = _recall_at_ks(scores, GT, ks)
            res[f"TurboQuantProd-{target}bit"] = recalls
            print(f"quant={tq_t:.1f}s search={srch_t:.1f}s "
                  f"R@1={recalls[1]:.4f}")
        except Exception as e:
            print(f"FAILED: {e}")

    pickle.dump(res, open(out_path, "wb"))
    print(f"  saved {out_path.name}")
    for n, r in res.items():
        print(f"    {n:<32s}", " ".join(f"R@{k}={r[k]:.4f}" for k in ks))
    return res


if __name__ == "__main__":
    cfg = load_config()
    enabled = enabled_datasets(cfg)
    ks = ensure_int_list(cfg.get("eval_ks", []), "eval_ks")

    env_tq = os.environ.get("TQ_BITS", "").strip()
    if env_tq:
        target_bits = [int(x) for x in env_tq.split()]
    else:
        target_bits = ensure_int_list(
            cfg.get("turboquant_bits", cfg.get("pq_bits", [2, 4])),
            "turboquant_bits",
        )

    for name in enabled:
        try:
            run_for_dataset(name, ks, target_bits)
        except Exception as e:
            print(f"[{name}] failed: {e}")