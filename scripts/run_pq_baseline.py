#!/usr/bin/env python
"""FAISS PQ-2bit / PQ-4bit baselines on each dataset.

Saves results to results/pq_pickles/pq_results_<dataset>.pkl.
"""
import os, pickle, time, gc
from pathlib import Path
import numpy as np
import faiss

ROOT = Path(__file__).resolve().parent.parent
UPSTREAM = ROOT / "third_party" / "Extended-RaBitQ"
OUT = ROOT / "results" / "pq_pickles"
OUT.mkdir(parents=True, exist_ok=True)

ENABLED = os.environ.get("DATASETS",
    "glove200_100k openai1536 openai3072").split()
KS, K_MAX = [1, 2, 4, 8, 16, 32], 32

def recall_at_k(I, GT):
    return {k: float(np.mean([GT[i, 0] in I[i, :k]
                              for i in range(len(I))])) for k in KS}

def run_pq(X, Xq, GT, bits_per_coord):
    dim = X.shape[1]
    group_size = 4 if bits_per_coord == 2 else 2
    m, nbits = dim // group_size, bits_per_coord * group_size
    print(f"  PQ-{bits_per_coord}bit: m={m}, nbits={nbits} ...",
          end=" ", flush=True)
    t = time.time()
    idx = faiss.IndexPQ(dim, m, nbits, faiss.METRIC_INNER_PRODUCT)
    idx.train(X); idx.add(X)
    _, I = idx.search(Xq, K_MAX)
    print(f"{time.time()-t:.0f}s")
    return recall_at_k(I, GT)

for name in ENABLED:
    out_path = OUT / f"pq_results_{name}.pkl"
    if out_path.exists():
        print(f"[{name}] PQ pickle exists; skipping"); continue

    data_dir = UPSTREAM / "data" / name
    if not (data_dir / "X.npy").exists():
        print(f"[{name}] data not prepared; skipping"); continue

    print(f"[{name}] running PQ baselines ...")
    X  = np.load(data_dir / "X.npy")
    Xq = np.load(data_dir / "Xq.npy")
    GT = np.load(data_dir / "GT.npy")

    results = {
        "PQ-2bit": run_pq(X, Xq, GT, 2),
        "PQ-4bit": run_pq(X, Xq, GT, 4),
    }
    pickle.dump(results, open(out_path, "wb"))
    print(f"  saved {out_path.name}")
    for k, v in results.items():
        print(f"    {k}:", " ".join(f"R@{kk}={v[kk]:.4f}" for kk in KS))

    del X, Xq, GT; gc.collect()