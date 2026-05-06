"""Run TurboQuantMSE baseline on each dataset.

Uses the from-scratch reference implementation (Algorithm 1 from Zandieh et al.).
The MSE variant is used because:
  (a) it matches the paper's reported Figure 4(a) recall numbers,
  (b) it outperforms TurboQuantProd for ranking-based retrieval at every bit-budget,
  (c) the hierarchical codebook construction in our paper extends this codebook directly.

Outputs:
  results/tq_pickles/tq_results_<dataset>.pkl
  format: {"TurboQuant-Nbit": {k: recall, ...}}
"""
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from turboquant_ref import TurboQuantMSE
from config_utils import enabled_datasets, ensure_int_list, load_config

DATA_DIR = ROOT / "third_party" / "Extended-RaBitQ" / "data"
PKL_DIR = ROOT / "results" / "tq_pickles"
PKL_DIR.mkdir(parents=True, exist_ok=True)


def load_dataset(name):
    path = DATA_DIR / name
    X = np.load(path / "X.npy").astype(np.float64)
    Xq = np.load(path / "Xq.npy").astype(np.float64)
    GT = np.load(path / "GT.npy").astype(np.int32)
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    Xq /= np.linalg.norm(Xq, axis=1, keepdims=True)
    return X, Xq, GT


def recall_at_ks(scores, GT, ks):
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


def main():
    cfg = load_config()
    enabled = enabled_datasets(cfg)
    ks = ensure_int_list(cfg.get("eval_ks", []), "eval_ks")
    bits_list = ensure_int_list(
        cfg.get("turboquant_bits", os.environ.get("TQ_BITS", "2 4").split()),
        "turboquant_bits",
    )

    print(f"Datasets: {enabled}")
    print(f"Bits:     {bits_list}")
    print(f"Eval ks:  {ks}")
    print()

    for name in enabled:
        print(f"=== {name} ===")
        try:
            X, Xq, GT = load_dataset(name)
        except FileNotFoundError as e:
            print(f"  SKIP: {e}")
            continue

        d = X.shape[1]
        n_query = int(cfg["datasets"][name].get("n_query", Xq.shape[0]))
        if n_query < Xq.shape[0]:
            Xq = Xq[:n_query]
            GT = GT[:n_query]
        print(f"  N={X.shape[0]}  d={d}  NQ={Xq.shape[0]}")

        results = {}
        for b in bits_list:
            t0 = time.time()
            tq = TurboQuantMSE(d=d, bits=b, seed=42)
            idx = tq.quantize(X)
            t_q = time.time() - t0

            t0 = time.time()
            X_hat = tq.dequantize(idx)
            scores = (Xq @ X_hat.T).astype(np.float32)
            t_s = time.time() - t0

            recalls = recall_at_ks(scores, GT, ks)
            results[f"TurboQuant-{b}bit"] = recalls
            print(f"  b={b}  quant={t_q:.1f}s search={t_s:.1f}s  " +
                  " ".join(f"R@{k}={recalls[k]:.4f}" for k in ks))

        out = PKL_DIR / f"tq_results_{name}.pkl"
        with open(out, "wb") as f:
            pickle.dump(results, f)
        print(f"  saved {out}")


if __name__ == "__main__":
    main()