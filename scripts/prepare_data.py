#!/usr/bin/env python
"""Download and convert all configured datasets to fvecs/ivecs.

Idempotent: skips datasets whose base.fvecs already exists.
Outputs land in third_party/Extended-RaBitQ/data/<name>/.

GloVe-200: reads from Kaggle dataset shashwatsaket/glove-200
           (attach it to your notebook before running)
OpenAI:    streams from HuggingFace.
"""
import os, sys, struct, gc, shutil
from pathlib import Path
import numpy as np
import faiss

ROOT = Path(__file__).resolve().parent.parent
UPSTREAM = ROOT / "third_party" / "Extended-RaBitQ"
DATA_BASE = UPSTREAM / "data"

ENABLED = os.environ.get("DATASETS",
                         "glove200_100k openai1536 openai3072").split()

K_MAX = 32
SEED = 42

# --------------------------------------------------------------------- IO
def write_fvecs(path, arr):
    arr = np.ascontiguousarray(arr, dtype=np.float32)
    n, d = arr.shape
    with open(path, "wb") as f:
        for v in arr:
            f.write(struct.pack("<i", d))
            f.write(v.tobytes())

def write_ivecs(path, arr):
    arr = np.ascontiguousarray(arr, dtype=np.int32)
    n, d = arr.shape
    with open(path, "wb") as f:
        for v in arr:
            f.write(struct.pack("<i", d))
            f.write(v.tobytes())

def build_gt_and_save(name, X, Xq, dim):
    out = DATA_BASE / name
    out.mkdir(parents=True, exist_ok=True)

    faiss.normalize_L2(X)
    faiss.normalize_L2(Xq)
    X  = np.ascontiguousarray(X,  dtype=np.float32)
    Xq = np.ascontiguousarray(Xq, dtype=np.float32)

    print(f"  building GT (FlatIP) for {len(Xq)} queries x {len(X)} base ...")
    gt = faiss.IndexFlatIP(dim); gt.add(X)
    _, GT = gt.search(Xq, K_MAX)
    GT = GT.astype(np.int32)

    write_fvecs(out / f"{name}_base.fvecs",        X)
    write_fvecs(out / f"{name}_query.fvecs",       Xq)
    write_ivecs(out / f"{name}_groundtruth.ivecs", GT)
    np.save(out / "X.npy",  X)
    np.save(out / "Xq.npy", Xq)
    np.save(out / "GT.npy", GT)

    print(f"  done. base {X.shape} | query {Xq.shape} | GT {GT.shape}")

def already_done(name):
    p = DATA_BASE / name / f"{name}_base.fvecs"
    return p.exists() and p.stat().st_size > 0

# ----------------------------------------------------------------- GloVe
def prepare_glove200_100k():
    name, dim = "glove200_100k", 200
    if already_done(name):
        print(f"[{name}] already prepared; skipping"); return
    print(f"[{name}] preparing ...")
    import h5py
    cache = ROOT / "data" / "_cache"
    cache.mkdir(parents=True, exist_ok=True)
    h5_path = cache / "glove-200-angular.hdf5"
    if not h5_path.exists():
        # Search common Kaggle input locations for the file
        candidates = [
            Path("/kaggle/input/glove-200/glove-200-angular.hdf5"),
            Path("/kaggle/input/datasets/shashwatsaket/glove-200/glove-200-angular.hdf5"),
        ]
        # Also search dynamically under /kaggle/input/
        import glob
        found = glob.glob("/kaggle/input/**/glove-200-angular.hdf5", recursive=True)
        candidates += [Path(p) for p in found]

        kaggle_path = next((p for p in candidates if p.exists()), None)
        if kaggle_path:
            print(f"  copying from {kaggle_path} ...")
            shutil.copy(kaggle_path, h5_path)
        else:
            import urllib.request
            url = "http://ann-benchmarks.com/glove-200-angular.hdf5"
            print(f"  downloading {url} ...")
            urllib.request.urlretrieve(url, h5_path)

    with h5py.File(h5_path, "r") as f:
        X_full  = np.array(f["train"], dtype=np.float32)
        Xq_full = np.array(f["test"],  dtype=np.float32)
    rng = np.random.RandomState(SEED)
    X  = X_full[rng.choice(len(X_full), 100_000, replace=False)]
    Xq = Xq_full[rng.choice(len(Xq_full), 10_000, replace=False)]
    del X_full, Xq_full
    build_gt_and_save(name, X, Xq, dim)
    gc.collect()

# ----------------------------------------------------------------- OpenAI
def prepare_openai(name, dim, hf_repo, hf_col):
    if already_done(name):
        print(f"[{name}] already prepared; skipping"); return
    print(f"[{name}] preparing ... (streaming from HF)")
    from datasets import load_dataset
    N_BASE, N_QUERY = 100_000, 1_000
    N_TOTAL = N_BASE + N_QUERY

    ds = load_dataset(hf_repo, split="train", streaming=True)
    emb = np.empty((N_TOTAL, dim), dtype=np.float32)
    i = 0
    for row in ds:
        if i == 0:
            assert hf_col in row, f"col {hf_col!r} not in {list(row)[:6]}"
        emb[i] = np.asarray(row[hf_col], dtype=np.float32)
        i += 1
        if i % 10_000 == 0:
            print(f"  {i}/{N_TOTAL}", flush=True)
        if i >= N_TOTAL: break
    assert i == N_TOTAL, f"got {i} rows, expected {N_TOTAL}"

    rng  = np.random.RandomState(SEED)
    perm = rng.permutation(N_TOTAL)
    X    = np.ascontiguousarray(emb[perm[:N_BASE]],        dtype=np.float32)
    Xq   = np.ascontiguousarray(emb[perm[N_BASE:N_TOTAL]], dtype=np.float32)
    del emb; gc.collect()

    build_gt_and_save(name, X, Xq, dim)
    gc.collect()

# ---------------------------------------------------------------- main
DATASETS = {
    "glove200_100k": prepare_glove200_100k,
    "openai1536":    lambda: prepare_openai(
        "openai1536", 1536,
        "Qdrant/dbpedia-entities-openai3-text-embedding-3-large-1536-1M",
        "text-embedding-3-large-1536-embedding"),
    "openai3072":    lambda: prepare_openai(
        "openai3072", 3072,
        "Qdrant/dbpedia-entities-openai3-text-embedding-3-large-3072-1M",
        "text-embedding-3-large-3072-embedding"),
}

if __name__ == "__main__":
    for name in ENABLED:
        if name not in DATASETS:
            print(f"unknown dataset {name!r}; skipping"); continue
        DATASETS[name]()
    print("\nall datasets ready in", DATA_BASE)