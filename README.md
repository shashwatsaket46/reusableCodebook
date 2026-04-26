# reusableCodebooks

Reproducing per-k recall comparison between **Product Quantization (PQ)** and
**Extended RaBitQ** [SIGMOD'25] under exhaustive search, across three
embedding dimensionalities: GloVe-200 (200d), OpenAI text-embedding-3-large
at 1536d and 3072d.

The upstream [Extended-RaBitQ](https://github.com/VectorDB-NTU/Extended-RaBitQ)
`test_search` reports average distance ratio at a single k. This repo
patches it to emit **Recall1@k** for k ∈ {1, 2, 4, 8, 16, 32}, matching the
metric used in the Python PQ baseline so the curves are directly comparable.

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

---

## Hardware requirements

| Resource | Required | Notes |
|---|---|---|
| CPU | x86_64 with **AVX-512** | Required by upstream Extended-RaBitQ |
| RAM | ≥ 32 GB | OpenAI-3072 indexing is memory-hungry |
| Disk | ≥ 40 GB free | Datasets + indexes total ~25 GB |
| OS | Linux (Ubuntu 22.04 tested) | macOS unsupported (no AVX-512) |

**Kaggle CPU notebooks satisfy all of the above.** TPU and GPU instances
typically lack AVX-512.

Verify before running:

```bash
grep -q avx512f /proc/cpuinfo && echo "OK" || echo "AVX-512 MISSING"
free -h | head -2
```

---

## Method

* **Datasets.**
  * GloVe-200: 100k base vectors, 10k queries, 200d.
  * OpenAI-1536: 100k base, 1k queries, 1536d. Source: HuggingFace
    `Qdrant/dbpedia-entities-openai3-text-embedding-3-large-1536-1M`.
  * OpenAI-3072: 100k base, 1k queries, 3072d. Same HF source family.
  * All vectors L2-normalized; ground truth via FAISS `IndexFlatIP`.

* **Quantizers.**
  * **PQ** (FAISS `IndexPQ`, METRIC_INNER_PRODUCT): 2 bits/dim and 4 bits/dim.
  * **Extended RaBitQ** (official C++): bits=3 (≡ binary RaBitQ + sign,
    3 bits/dim) and bits=5 (≡ 4-bit RaBitQ + sign, 5 bits/dim).
    Note: the +1 bit is for the sign; the magnitude uses the listed
    bits-per-coord.

* **Search.** Exhaustive across the board:
  * PQ: flat asymmetric distance computation over all 100k codes.
  * ExRaBitQ: IVF with `nprobe = nlist` (every cluster probed).

* **nlist configuration.**
  * GloVe-200, OpenAI-1536: **nlist = 256**
  * OpenAI-3072: **nlist = 64** (RAM-constrained: HIGH_ACC_FAST_SCAN at
    d=3072 with nlist=256 OOMs a 32GB instance). Both are still
    exhaustive (nprobe = nlist), so curves remain directly comparable;
    only residual-encoding granularity differs.

* **Metric — Recall1@k.**
