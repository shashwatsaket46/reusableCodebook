# reusableCodebooks

Reproducing per-k recall comparison between **Product Quantization (PQ)** and
**Extended RaBitQ** [SIGMOD'25] under exhaustive search, across three
embedding dimensionalities: GloVe-200 (200d), OpenAI text-embedding-3-large
at 1536d and 3072d.

The upstream [Extended-RaBitQ](https://github.com/VectorDB-NTU/Extended-RaBitQ)
`test_search` reports average distance ratio at a single k. This repo
patches it to emit **Recall1@k** for k ∈ {1, 2, 4, 8, 16, 32}, matching the
metric used in the Python PQ baseline so the curves are directly comparable.

This repo also extends ExRaBitQ to support **bits = 2** (1 bit/dim total),
filling the gap between binary RaBitQ (1 bit) and the paper's lowest
supported `bits = 3` (3 bits/dim). See [`cpp/overrides/`](cpp/overrides/)
for the implementation.

---

## TL;DR — Run on Kaggle

Open a new Kaggle notebook with **CPU** accelerator and paste this single cell:

```python
import os
os.chdir("/kaggle/working")
!rm -rf reusableCodebook
!git clone https://github.com/shashwatsaket46/reusableCodebook.git
%cd reusableCodebook
!bash run_all.sh
```

Runtime ≈ 50 minutes end-to-end. Outputs land in `results/figures/`.

Display the final figure inline:

```python
from IPython.display import Image
Image("/kaggle/working/reusableCodebook/results/figures/recall_all_datasets.png")
```

### Selective runs

Two knobs available without editing config files:

```python
# Run a single dataset
!DATASETS="glove200_100k" bash run_all.sh

# Run with custom ExRaBitQ bit-widths
!DATASETS="glove200_100k" BITS="2 4" bash run_all.sh

# Both
!DATASETS="glove200_100k openai1536" BITS="2 3 4" bash run_all.sh
```

`DATASETS` and `BITS` env vars override the values in `config.yaml`.

---

## Hardware requirements

| Resource | Required | Notes |
|---|---|---|
| CPU | x86_64 with **AVX-512** | Required by upstream Extended-RaBitQ |
| RAM | ≥ 32 GB | OpenAI-3072 indexing is memory-hungry |
| Disk | ≥ 40 GB free | Datasets + indexes total ~25 GB |
| OS | Linux (Ubuntu 22.04 tested) | macOS unsupported (no AVX-512) |

Verify before running:

```bash
grep -q avx512f /proc/cpuinfo && echo "OK" || echo "AVX-512 MISSING"
free -h | head -2
```

Kaggle CPU notebooks satisfy all of the above. TPU and most GPU
accelerator types **do not** have AVX-512 — switch to plain CPU.

---

## Configuration

All experiment knobs are centralized in `config.yaml`:

| Section | Knob | Purpose |
|---|---|---|
| `datasets.<name>` | `dim`, `n_base`, `n_query`, `nlist`, `source` | Per-dataset shape and origin |
| Top-level | `exrabitq_bits` | List of bit-widths for ExRaBitQ runs |
| Top-level | `pq_bits` | List of bit-widths for PQ baseline |
| Top-level | `pq_metric` | `inner_product` or `l2` |
| Top-level | `eval_ks` | Recall@k values to compute |
| Top-level | `search_rounds` | How many timing rounds (deterministic recall, but stable QPS) |
| Top-level | `random_seed` | Reproducibility |
| `cpp.*` | `topk`, `high_acc_fast_scan` | C++ binary settings (rebuild required to change) |
| `plot.*` | `y_lim`, `colors` | Plot styling |

Env vars `DATASETS` and `BITS` override the defaults at runtime without
editing `config.yaml`.

---

## Method

### Datasets

| Name | dim | base | queries | source |
|---|---:|---:|---:|---|
| `glove200_100k` | 200 | 100k | 10k | GloVe-200-angular ([ann-benchmarks](http://ann-benchmarks.com/glove-200-angular.hdf5)) |
| `openai1536` | 1536 | 100k | 1k | HF `Qdrant/dbpedia-entities-openai3-text-embedding-3-large-1536-1M` |
| `openai3072` | 3072 | 100k | 1k | HF `Qdrant/dbpedia-entities-openai3-text-embedding-3-large-3072-1M` |

All vectors are L2-normalized; ground truth is computed via FAISS
`IndexFlatIP` over normalized vectors (equivalent to L2 ranking under
unit norm).

The OpenAI datasets stream from HuggingFace automatically. The GloVe
download from ann-benchmarks.com is sometimes slow on Kaggle — see
[Optional: Pre-upload GloVe-200](#optional-pre-upload-glove-200-to-kaggle).

### Quantizers

* **PQ** (FAISS `IndexPQ`): bit-widths from `config.yaml:pq_bits`
  (default `[2, 4]`), search via `config.yaml:pq_metric`.
* **Extended RaBitQ** (patched C++): bit-widths from
  `config.yaml:exrabitq_bits` (default `[2, 4]`).
  * `bits=2` → 1 bit/dim total (1-bit RaBitQ extension implemented in this repo)
  * `bits=3` → 3 bits/dim total (paper's minimum)
  * `bits=4` → 4 bits/dim total (paper's intended low-budget)
  * `bits=5` → 5 bits/dim total

### Search

Exhaustive across the board:
* **PQ**: flat asymmetric distance computation over all 100k codes.
* **ExRaBitQ**: IVF with `nprobe = nlist` (every cluster probed). The
  result is mathematically equivalent to flat search, with the only
  difference being the IVF residual encoding (each vector encoded
  relative to its assigned centroid).

### nlist configuration

| Dataset | nlist | Reason |
|---|---:|---|
| GloVe-200 | 256 | default |
| OpenAI-1536 | 256 | default |
| OpenAI-3072 | 64 | RAM-constrained: `HIGH_ACC_FAST_SCAN` at d=3072 with nlist=256 OOMs a 32GB instance |

Both configurations are exhaustive (`nprobe = nlist`), so the recall
numbers are directly comparable across datasets; only the residual-encoding
granularity differs.

### Metric — Recall1@k

```
Recall1@k = (1/|Q|) Σ_q  𝟙[ NN_true(q) ∈ top-k retrieved candidates ]
```

i.e. fraction of queries for which the true nearest neighbor lands in the
top-k. Standard in ANN literature; matches the metric used by the Extended
RaBitQ paper.

---

## Pipeline overview

`run_all.sh` orchestrates 7 steps. Each step is **idempotent** —
re-running skips work already completed.

1. **`scripts/setup.sh`** — clone Extended-RaBitQ, vendor Eigen + hnswlib,
   apply C++ overrides from `cpp/overrides/`, build via cmake.
2. **`scripts/prepare_data.py`** — download (GloVe via ann-benchmarks
   or pre-uploaded Kaggle dataset; OpenAI via HuggingFace streaming),
   normalize, sample base/query splits, build ground truth, write
   fvecs/ivecs.
3. **`scripts/run_ivf.sh`** — k-means clustering per dataset with
   per-dataset nlist from `config.yaml`.
4. **`scripts/build_index.sh`** — `create_index` for each
   `exrabitq_bits` value.
5. **`scripts/run_search.sh`** — patched `test_search`. EVAL lines
   captured to `results/logs/`.
6. **`scripts/run_pq_baseline.py`** — FAISS PQ baselines, pickled to
   `results/pq_pickles/`.
7. **`scripts/plot.py`** — parses logs + pickles, emits per-dataset and
   combined figures to `results/figures/`.

---

## Optional: Pre-upload GloVe-200 to Kaggle

The ann-benchmarks server is slow and rate-limited from Kaggle. To
avoid a multi-minute download in the middle of your run, upload the
file once as a Kaggle dataset and reuse it across notebooks.

**One-time setup (~5 min faster per run):**

1. Download http://ann-benchmarks.com/glove-200-angular.hdf5 locally (~1.3 GB).
2. In Kaggle: **+ Create → New Dataset**, drag the file in, name
   it `glove-200`. Visibility: Private. Click Create.
3. In your notebook: **+ Add Input** (right sidebar) → search
   `glove-200` → Add.

`scripts/prepare_data.py` checks `/kaggle/input/glove-200/` and other
common paths first, falling back to the ann-benchmarks download if
not found.

If the HuggingFace stream gets rate-limited, set an `HF_TOKEN`
[Kaggle secret](https://huggingface.co/settings/tokens) (free, no
credit card) before re-running.

---

## C++ overrides — what we modified upstream

`cpp/overrides/` mirrors the upstream Extended-RaBitQ tree. Files here
replace the corresponding upstream files when `scripts/setup.sh` runs.

| File | Purpose |
|---|---|
| `cpp/overrides/src/test_search.cpp` | Per-k recall metric; force exhaustive search; emit `EVAL` lines for downstream parsing |
| `cpp/overrides/src/create_index.cpp` | Allow `B=2` argument |
| `cpp/overrides/inc/index/IVF.hpp` | Allow `EX_BITS=1` (1 bit/dim long code) |
| `cpp/overrides/inc/index/Quantizer.hpp` | Add 1-bit packing branch in `store_compacted_code()` |
| `cpp/overrides/inc/index/Searcher.hpp` | Dispatch `IP_FUNC = IP_fxu1` for `ex_bits=1` |
| `cpp/overrides/inc/index/HASearcher.hpp` | Same dispatch in HighAccuracy variant |
| `cpp/overrides/inc/utils/space.hpp` | Define scalar `IP_fxu1()` for 1-bit code dot product |

Critically, `HIGH_ACC_FAST_SCAN` must remain enabled (line 1 of
`test_search.cpp`). Disabling it triggers an upstream code path that
returns zero matches.

---

## Repository layout

```
reusableCodebooks/
├── README.md
├── LICENSE                                 # Apache-2.0
├── requirements.txt
├── config.yaml                             # all experiment knobs
├── run_all.sh                              # entry point
├── .gitignore
├── cpp/
│   └── overrides/                          # files that replace upstream sources
│       ├── src/{create_index,test_search}.cpp
│       └── inc/
│           ├── index/{IVF,Quantizer,Searcher,HASearcher}.hpp
│           └── utils/space.hpp
├── scripts/
│   ├── setup.sh                            # clone + vendor + apply overrides + build
│   ├── prepare_data.py                     # download + GT + fvecs/ivecs
│   ├── run_ivf.sh                          # k-means
│   ├── build_index.sh                      # create_index
│   ├── run_search.sh                       # test_search → EVAL logs
│   ├── run_pq_baseline.py                  # FAISS PQ
│   ├── plot.py                             # parse + plot
│   ├── ivf_argparse.py                     # drop-in for upstream's ivf.py
│   ├── config_utils.py                     # config.yaml loader
│   ├── config_cli.py                       # shell-script bridge to config
│   └── generate_cpp_config.py              # (optional) generate cpp config header
├── data/                                   # gitignored; generated
├── results/
│   ├── logs/                               # gitignored
│   ├── pq_pickles/                         # gitignored
│   └── figures/                            # committed
└── third_party/                            # gitignored; upstream lands here
```

---

## Expected results

| Dataset | dim | nlist | metric | PQ-2bit | ExRaBitQ-2bit | ExRaBitQ-3bit | PQ-4bit | ExRaBitQ-4bit |
|---|---:|---:|---|---:|---:|---:|---:|---:|
| GloVe-200 | 200 | 256 | R1@1 | 0.564 | TBD | 0.801 | 0.842 | TBD |
|  |  |  | R1@8 | 0.925 | TBD | 0.996 | 0.999 | TBD |
| OpenAI-1536 | 1536 | 256 | R1@1 | … | TBD | 0.937 | … | TBD |
| OpenAI-3072 | 3072 | 64 | R1@1 | … | TBD | 0.951 | … | TBD |

Numbers are deterministic to ~0.5% across reruns due to non-deterministic
multithreaded k-means in FAISS.

The story across dimensionality:
* **GloVe-200**: largest visible gap — per-coordinate quantization is
  hardest at low dim, ExRaBitQ's structured codebook wins clearly.
* **OpenAI-1536 / 3072**: gap shrinks; both methods near-saturate by k=4.
  Consistent with the Extended RaBitQ paper's claim that
  high-dimensional, semantically-structured embeddings are easier
  per-coordinate.

---

## Output figures

After a successful run, `results/figures/` contains:

* `recall_glove200_100k.png` — single panel, GloVe-200
* `recall_openai1536.png` — single panel, OpenAI-1536
* `recall_openai3072.png` — single panel, OpenAI-3072
* `recall_all_datasets.png` — three panels side-by-side, shared y-axis

The combined figure is the headline result for the writeup.

---

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `AVX-512 MISSING` at setup | Wrong accelerator type | Switch Kaggle to **CPU** (not TPU/GPU) |
| Build error: `IP64_fxu1 was not declared` | `space.hpp` override missing function | Re-pull repo; verify `cpp/overrides/inc/utils/space.hpp` defines `IP_fxu1` |
| All EVAL recalls = 0 | `HIGH_ACC_FAST_SCAN` disabled | Re-enable line 1 of `cpp/overrides/src/test_search.cpp` |
| `bad_alloc` during `create_index openai3072` | RAM exhausted at high dim | Already handled — uses nlist=64 for 3072d |
| `FileNotFoundError: glove-200-angular.hdf5` | ann-benchmarks down/blocked | Manual upload (see [Optional: Pre-upload GloVe-200](#optional-pre-upload-glove-200-to-kaggle)) |
| HF streaming hangs | Rate limit | Set `HF_TOKEN` Kaggle secret |
| `_centroid_*.fvecs not exists` at create_index | k-means step skipped | Run `bash scripts/run_ivf.sh` |
| Plot has empty panel | Wrong nprobe filter | Ensure `scripts/plot.py` uses max-nprobe parser |
| Search runs slow with many EVAL lines per dataset | Override not applied; running upstream's nprobe sweep | Confirm `setup.sh` copies `cpp/overrides/`; rebuild |

---

## Citation

If you use this repository, please cite the upstream papers:

```bibtex
@inproceedings{gao2025practical,
  title     = {Practical and Asymptotically Optimal Quantization of
               High-Dimensional Vectors in Euclidean Space for
               Approximate Nearest Neighbor Search},
  author    = {Gao, Jianyang and Long, Cheng},
  booktitle = {SIGMOD},
  year      = {2025}
}

@inproceedings{gao2024rabitq,
  title     = {RaBitQ: Quantizing High-Dimensional Vectors with a
               Theoretical Error Bound for Approximate Nearest
               Neighbor Search},
  author    = {Gao, Jianyang and Long, Cheng},
  booktitle = {SIGMOD},
  year      = {2024}
}
```

---

## License

Apache-2.0. The upstream Extended-RaBitQ project retains its original
license; modifications in `cpp/overrides/` are derivative works
covered by the same license.

---

## Acknowledgements

Course project for **CS-GY 9223**, NYU. Mentor: Majid Daliri (TurboQuant).

Built on:
* Upstream Extended-RaBitQ implementation by Jianyang Gao and Cheng Long
  (https://github.com/VectorDB-NTU/Extended-RaBitQ).
* FAISS for PQ baselines and k-means
  (https://github.com/facebookresearch/faiss).
* hnswlib for IVF initialization (https://github.com/nmslib/hnswlib).