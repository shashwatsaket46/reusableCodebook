#!/usr/bin/env python
"""Download and convert configured datasets to fvecs/ivecs.

All dataset shape/source knobs are read from config.yaml.
"""
import gc
import shutil
import struct
from pathlib import Path
import warnings

import faiss
import numpy as np

from config_utils import enabled_datasets, ensure_int_list, load_config

ROOT = Path(__file__).resolve().parent.parent
UPSTREAM = ROOT / "third_party" / "Extended-RaBitQ"
DATA_BASE = UPSTREAM / "data"

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

def build_gt_and_save(name, X, Xq, dim, k_max, metric):
    out = DATA_BASE / name
    out.mkdir(parents=True, exist_ok=True)

    X  = np.ascontiguousarray(X,  dtype=np.float32)
    Xq = np.ascontiguousarray(Xq, dtype=np.float32)

    if metric == "inner_product":
        faiss.normalize_L2(X)
        faiss.normalize_L2(Xq)
        print(f"  building GT (FlatIP) for {len(Xq)} queries x {len(X)} base ...")
        gt = faiss.IndexFlatIP(dim)
    elif metric == "l2":
        print(f"  building GT (FlatL2) for {len(Xq)} queries x {len(X)} base ...")
        gt = faiss.IndexFlatL2(dim)
    else:
        raise ValueError(f"Unsupported pq_metric={metric!r}; expected inner_product|l2")
    gt.add(X)
    _, GT = gt.search(Xq, k_max)
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
def prepare_hdf5_dataset(name, ds_cfg, seed, k_max, metric):
    dim = int(ds_cfg["dim"])
    n_base = int(ds_cfg["n_base"])
    n_query = int(ds_cfg["n_query"])
    source = ds_cfg["source"]
    if already_done(name):
        print(f"[{name}] already prepared; skipping"); return
    print(f"[{name}] preparing ...")
    import h5py
    cache = ROOT / "data" / "_cache"
    cache.mkdir(parents=True, exist_ok=True)
    h5_path = cache / f"{name}.hdf5"
    if not h5_path.exists():
        candidates = [Path(p) for p in source.get("kaggle_paths", [])]
        import glob
        found = glob.glob("/kaggle/input/**/*.hdf5", recursive=True)
        candidates += [Path(p) for p in found]

        kaggle_path = next((p for p in candidates if p.exists()), None)
        if kaggle_path:
            print(f"  copying from {kaggle_path} ...")
            shutil.copy(kaggle_path, h5_path)
        else:
            import urllib.request
            url = source.get("url", "")
            if not url:
                warnings.warn(f"[{name}] no local hdf5 found and no source.url configured; skipping")
                return
            print(f"  downloading {url} ...")
            try:
                urllib.request.urlretrieve(url, h5_path)
            except Exception as e:
                warnings.warn(f"[{name}] failed to download hdf5 from {url}: {e}; skipping")
                return

    try:
        with h5py.File(h5_path, "r") as f:
            X_full  = np.array(f["train"], dtype=np.float32)
            Xq_full = np.array(f["test"],  dtype=np.float32)
    except Exception as e:
        warnings.warn(f"[{name}] failed to open/read hdf5 at {h5_path}: {e}; skipping")
        return
    rng = np.random.RandomState(seed)
    X = X_full[rng.choice(len(X_full), n_base, replace=False)]
    Xq = Xq_full[rng.choice(len(Xq_full), n_query, replace=False)]
    del X_full, Xq_full
    build_gt_and_save(name, X, Xq, dim, k_max, metric)
    gc.collect()

# ----------------------------------------------------------------- OpenAI
def prepare_hf_dataset(name, ds_cfg, seed, k_max, metric):
    dim = int(ds_cfg["dim"])
    n_base = int(ds_cfg["n_base"])
    n_query = int(ds_cfg["n_query"])
    source = ds_cfg["source"]
    if already_done(name):
        print(f"[{name}] already prepared; skipping"); return
    print(f"[{name}] preparing ... (streaming from HF)")
    from datasets import load_dataset
    hf_repo = source["repo"]
    hf_col = source["column"]
    N_BASE, N_QUERY = n_base, n_query
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

    rng  = np.random.RandomState(seed)
    perm = rng.permutation(N_TOTAL)
    X    = np.ascontiguousarray(emb[perm[:N_BASE]],        dtype=np.float32)
    Xq   = np.ascontiguousarray(emb[perm[N_BASE:N_TOTAL]], dtype=np.float32)
    del emb; gc.collect()

    build_gt_and_save(name, X, Xq, dim, k_max, metric)
    gc.collect()

# ---------------------------------------------------------------- main
if __name__ == "__main__":
    cfg = load_config()
    enabled = enabled_datasets(cfg)
    seed = int(cfg.get("random_seed", 42))
    eval_ks = ensure_int_list(cfg.get("eval_ks", []), "eval_ks")
    topk = int(cfg.get("cpp", {}).get("topk", max(eval_ks)))
    k_max = max(max(eval_ks), topk)
    metric = str(cfg.get("pq_metric", "inner_product")).strip().lower()

    for name in enabled:
        try:
            ds_cfg = cfg["datasets"][name]
            src_type = ds_cfg.get("source", {}).get("type")
            if src_type == "hdf5":
                prepare_hdf5_dataset(name, ds_cfg, seed, k_max, metric)
            elif src_type == "huggingface":
                prepare_hf_dataset(name, ds_cfg, seed, k_max, metric)
            else:
                print(f"[{name}] unsupported source.type={src_type!r}; skipping")
        except Exception as e:
            warnings.warn(f"[{name}] dataset preparation failed: {e}; continuing")
    print("\nall datasets ready in", DATA_BASE)