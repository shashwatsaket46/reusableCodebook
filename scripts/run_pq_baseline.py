#!/usr/bin/env python
"""FAISS PQ baselines configured from config.yaml.

Saves results to results/pq_pickles/pq_results_<dataset>.pkl.
"""
import gc
import pickle
import time
from pathlib import Path

import numpy as np
import faiss

from config_utils import enabled_datasets, ensure_int_list, load_config

ROOT = Path(__file__).resolve().parent.parent
UPSTREAM = ROOT / "third_party" / "Extended-RaBitQ"
OUT = ROOT / "results" / "pq_pickles"
OUT.mkdir(parents=True, exist_ok=True)

def recall_at_k(I, GT, ks):
    return {k: float(np.mean([GT[i, 0] in I[i, :k]
                              for i in range(len(I))])) for k in ks}


def choose_pq_layout(dim, bits_per_coord):
    """Choose (m, nbits, group_size) for target bits-per-coordinate.

    We prefer exact bitrate matches, then the smallest nbits (lighter training),
    then the largest group size (fewer sub-quantizers).
    """
    if bits_per_coord <= 0:
        raise ValueError("pq_bits values must be positive")

    candidates = []
    for group_size in range(1, min(32, dim) + 1):
        if dim % group_size != 0:
            continue
        nbits = bits_per_coord * group_size
        if nbits < 1 or nbits > 16:
            continue
        m = dim // group_size
        # Effective bits-per-coordinate for this layout.
        eff_bpc = (m * nbits) / dim
        err = abs(eff_bpc - bits_per_coord)
        candidates.append((err, nbits, -group_size, m, group_size))

    if not candidates:
        raise ValueError(
            f"No valid PQ layout for dim={dim}, bits={bits_per_coord}; "
            "requires nbits in [1, 16] and divisible group sizing"
        )

    _, nbits, _, m, group_size = sorted(candidates)[0]
    return m, nbits, group_size


def run_pq(X, Xq, GT, bits_per_coord, metric, k_max, ks):
    dim = X.shape[1]
    m, nbits, group_size = choose_pq_layout(dim, bits_per_coord)
    print(f"  PQ-{bits_per_coord}bit: m={m}, nbits={nbits}, group={group_size} ...",
          end=" ", flush=True)
    t = time.time()
    idx = faiss.IndexPQ(dim, m, nbits, metric)
    idx.train(X); idx.add(X)
    _, I = idx.search(Xq, k_max)
    print(f"{time.time()-t:.0f}s")
    return recall_at_k(I, GT, ks)

if __name__ == "__main__":
    cfg = load_config()
    enabled = enabled_datasets(cfg)
    ks = ensure_int_list(cfg.get("eval_ks", []), "eval_ks")
    k_max = max(ks)
    bits_list = ensure_int_list(cfg.get("pq_bits", []), "pq_bits")
    metric_name = str(cfg.get("pq_metric", "inner_product")).strip().lower()
    metric = {
        "inner_product": faiss.METRIC_INNER_PRODUCT,
        "l2": faiss.METRIC_L2,
    }.get(metric_name)
    if metric is None:
        raise ValueError("pq_metric must be 'inner_product' or 'l2'")

    for name in enabled:
        out_path = OUT / f"pq_results_{name}.pkl"
        if out_path.exists():
            print(f"[{name}] PQ pickle exists; skipping")
            continue

        data_dir = UPSTREAM / "data" / name
        if not (data_dir / "X.npy").exists():
            print(f"[{name}] data not prepared; skipping")
            continue

        print(f"[{name}] running PQ baselines (metric={metric_name}) ...")
        X = np.load(data_dir / "X.npy")
        Xq = np.load(data_dir / "Xq.npy")
        GT = np.load(data_dir / "GT.npy")

        results = {
            f"PQ-{bits}bit": run_pq(X, Xq, GT, bits, metric, k_max, ks)
            for bits in bits_list
        }
        with open(out_path, "wb") as f:
            pickle.dump(results, f)
        print(f"  saved {out_path.name}")
        for k, v in results.items():
            print(f"    {k}:", " ".join(f"R@{kk}={v[kk]:.4f}" for kk in ks))

        del X, Xq, GT
        gc.collect()
