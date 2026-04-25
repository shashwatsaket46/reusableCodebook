"""Drop-in replacement for upstream Extended-RaBitQ/python/ivf.py.

Uses argparse instead of hardcoded constants. Output filenames match what
the C++ create_index expects: <name>_centroid_<K>.fvecs,
<name>_cluster_id_<K>.ivecs, <name>_dist_to_centroid_<K>.fvecs.
"""
import argparse, os, sys
import numpy as np
import faiss

# Reuse upstream's IO helpers
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "utils"))
from utils.io import read_fvecs, write_fvecs, write_ivecs

ap = argparse.ArgumentParser()
ap.add_argument("--base",    required=True, help="base.fvecs path")
ap.add_argument("--k",       type=int, required=True, help="num clusters")
ap.add_argument("--out_dir", required=True, help="output directory")
ap.add_argument("--name",    required=True, help="dataset name prefix")
ap.add_argument("--niter",   type=int, default=25)
ap.add_argument("--seed",    type=int, default=42)
args = ap.parse_args()

print(f"Reading {args.base}")
X = read_fvecs(args.base).astype(np.float32)
N, D = X.shape
print(f"  N={N}  D={D}  K={args.k}")

print("Running k-means ...")
km = faiss.Kmeans(D, args.k, niter=args.niter, verbose=True, seed=args.seed)
km.train(X)
centroids = km.centroids.astype(np.float32)

dist, cluster_ids = km.index.search(X, 1)
cluster_ids = cluster_ids.astype(np.int32)
dist_sq     = dist.astype(np.float32)   # squared L2 distance to assigned centroid

os.makedirs(args.out_dir, exist_ok=True)
write_fvecs(os.path.join(args.out_dir, f"{args.name}_centroid_{args.k}.fvecs"),
            centroids)
write_ivecs(os.path.join(args.out_dir, f"{args.name}_cluster_id_{args.k}.ivecs"),
            cluster_ids)
write_fvecs(os.path.join(args.out_dir, f"{args.name}_dist_to_centroid_{args.k}.fvecs"),
            dist_sq)
print("Done.")