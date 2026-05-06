#!/usr/bin/env bash
# Full pipeline for the hierarchical TurboQuant project.
#
# Runs everything end-to-end:
#   1.  Setup + data prep
#   2.  Extended-RaBitQ pipeline (original baselines)
#   3.  PQ baselines (single-tier scan)
#   4.  From-scratch TurboQuant baselines
#   5.  Hierarchical codebook construction (3 strategies)
#   6.  Two-tier recall (single-seed)
#   7.  PQ + reranking baseline (two-tier comparison)
#   8.  Multi-seed variance experiment
#   9.  All plotting
#
# Configurable via env vars:
#   DATASETS         - space-separated list (default: glove200_100k openai1536 openai3072)
#   N_QUERIES_GLOVE  - queries for glove200_100k (default: 10000)
#   N_QUERIES_OPENAI - queries for openai datasets (default: 1000)
#   N_SEEDS          - seeds for multi-seed run (default: 5)
#   SKIP_*           - set to 1 to skip a stage
#
# Examples:
#   bash run_all.sh
#   DATASETS="glove200_100k" N_SEEDS=3 bash run_all.sh
#   SKIP_RABITQ=1 SKIP_PQ=1 bash run_all.sh
#
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

mkdir -p results/logs results/pq_pickles results/tq_pickles results/codebooks results/figures

# ----------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------

export DATASETS="${DATASETS:-glove200_100k openai1536 openai3072}"
export N_QUERIES_GLOVE="${N_QUERIES_GLOVE:-10000}"
export N_QUERIES_OPENAI="${N_QUERIES_OPENAI:-1000}"
N_SEEDS="${N_SEEDS:-5}"

SKIP_SETUP="${SKIP_SETUP:-0}"
SKIP_DATA_PREP="${SKIP_DATA_PREP:-0}"
SKIP_RABITQ="${SKIP_RABITQ:-0}"
SKIP_PQ="${SKIP_PQ:-0}"
SKIP_TURBOQUANT="${SKIP_TURBOQUANT:-0}"
SKIP_HIERARCHICAL="${SKIP_HIERARCHICAL:-0}"
SKIP_TWO_TIER="${SKIP_TWO_TIER:-0}"
SKIP_PQ_RERANK="${SKIP_PQ_RERANK:-0}"
SKIP_MULTISEED="${SKIP_MULTISEED:-0}"
SKIP_PLOTS="${SKIP_PLOTS:-0}"

LOG="$ROOT/results/logs/run_all.log"
exec > >(tee -a "$LOG") 2>&1

# ----------------------------------------------------------------------------
# Header
# ----------------------------------------------------------------------------

echo "============================================================================"
echo "Hierarchical TurboQuant — Full Pipeline"
echo "============================================================================"
echo "Datasets:           $DATASETS"
echo "Queries (glove):    $N_QUERIES_GLOVE"
echo "Queries (openai):   $N_QUERIES_OPENAI"
echo "Seeds (multiseed):  $N_SEEDS"
echo "Logs:               $ROOT/results/logs/"
echo "Started:            $(date)"
echo "============================================================================"
echo

pick_n_queries() {
    case "$1" in
        glove*) echo "$N_QUERIES_GLOVE" ;;
        *) echo "$N_QUERIES_OPENAI" ;;
    esac
}

# ----------------------------------------------------------------------------
# Stage 1: setup
# ----------------------------------------------------------------------------

if [ "$SKIP_SETUP" != "1" ]; then
    echo "[1/10] Setup: clone Extended-RaBitQ, install deps"
    bash scripts/setup.sh
    pip install -q --no-deps faiss-cpu || true
    echo
else
    echo "[1/10] SKIPPED: setup"; echo
fi

# ----------------------------------------------------------------------------
# Stage 2: data prep
# ----------------------------------------------------------------------------

if [ "$SKIP_DATA_PREP" != "1" ]; then
    echo "[2/10] Preparing datasets"
    python scripts/prepare_data.py
    echo
else
    echo "[2/10] SKIPPED: data prep"; echo
fi

# ----------------------------------------------------------------------------
# Stage 3-5: Extended-RaBitQ pipeline
# ----------------------------------------------------------------------------

if [ "$SKIP_RABITQ" != "1" ]; then
    echo "[3/10] Extended-RaBitQ: IVF clustering"
    bash scripts/run_ivf.sh
    echo
    echo "[4/10] Extended-RaBitQ: building indexes"
    bash scripts/build_index.sh
    echo
    echo "[5/10] Extended-RaBitQ: running test_search"
    bash scripts/run_search.sh
    echo
else
    echo "[3-5/10] SKIPPED: Extended-RaBitQ pipeline"; echo
fi

# ----------------------------------------------------------------------------
# Stage 6: PQ single-tier baselines
# ----------------------------------------------------------------------------

if [ "$SKIP_PQ" != "1" ]; then
    echo "[6/10] PQ single-tier baselines"
    python scripts/run_pq_baseline.py
    echo
else
    echo "[6/10] SKIPPED: PQ baselines"; echo
fi

# ----------------------------------------------------------------------------
# Stage 7: TurboQuant baselines
# ----------------------------------------------------------------------------

if [ "$SKIP_TURBOQUANT" != "1" ]; then
    echo "[7/10] TurboQuant from-scratch baselines"
    python scripts/run_turboquant_baseline.py
    echo
    echo "[7b] Sanity-check TurboQuant ref implementation"
    python scripts/test_turboquant_ref.py
    echo
else
    echo "[7/10] SKIPPED: TurboQuant baselines"; echo
fi

# ----------------------------------------------------------------------------
# Stage 8: Hierarchical codebook construction
# ----------------------------------------------------------------------------

if [ "$SKIP_HIERARCHICAL" != "1" ]; then
    echo "[8/10] Hierarchical codebook construction"
    echo "  -- Bottom-up + top-down"
    python scripts/hierarchical_codebook.py
    echo
    echo "  -- Middle-anchor (3rd Pareto point)"
    python scripts/middle_anchor.py
    echo
else
    echo "[8/10] SKIPPED: hierarchical codebook construction"; echo
fi

# ----------------------------------------------------------------------------
# Stage 9a: Two-tier prefilter+rerank — TurboQuant nested codebooks
# ----------------------------------------------------------------------------

if [ "$SKIP_TWO_TIER" != "1" ]; then
    echo "[9a/10] Two-tier prefilter+rerank (single-seed, 4 strategies)"
    for ds in $DATASETS; do
        nq=$(pick_n_queries "$ds")
        echo "  -- Dataset: $ds (NQ=$nq)"
        N_QUERIES="$nq" DATASETS="$ds" python scripts/run_two_tier_recall.py
        echo
    done
else
    echo "[9a/10] SKIPPED: two-tier (single-seed)"; echo
fi

# ----------------------------------------------------------------------------
# Stage 9b: PQ + reranking baseline (two-tier comparison)
# ----------------------------------------------------------------------------

if [ "$SKIP_PQ_RERANK" != "1" ]; then
    echo "[9b/10] PQ + reranking baseline"
    python scripts/run_pq_rerank_baseline.py
    echo
else
    echo "[9b/10] SKIPPED: PQ + rerank baseline"; echo
fi

# ----------------------------------------------------------------------------
# Stage 9c: Multi-seed variance experiment
# ----------------------------------------------------------------------------

if [ "$SKIP_MULTISEED" != "1" ]; then
    echo "[9c/10] Multi-seed variance experiment (N_SEEDS=$N_SEEDS)"
    for ds in $DATASETS; do
        nq=$(pick_n_queries "$ds")
        echo "  -- Dataset: $ds (NQ=$nq, seeds=$N_SEEDS)"
        N_QUERIES="$nq" N_SEEDS="$N_SEEDS" DATASETS="$ds" \
            python scripts/run_multiseed_recall.py
        echo
    done
else
    echo "[9c/10] SKIPPED: multi-seed experiment"; echo
fi

# ----------------------------------------------------------------------------
# Stage 10: All plotting
# ----------------------------------------------------------------------------

if [ "$SKIP_PLOTS" != "1" ]; then
    echo "[10/10] Final plotting"
    python scripts/plot.py
    echo

    if [ -f scripts/plot_recall_summary.py ]; then
        echo "  -- Recall summary plot"
        python scripts/plot_recall_summary.py
    fi
    echo
else
    echo "[10/10] SKIPPED: plotting"; echo
fi

# ----------------------------------------------------------------------------
# Footer
# ----------------------------------------------------------------------------

echo "============================================================================"
echo "Pipeline complete."
echo "Finished:  $(date)"
echo "Log:       $LOG"
echo
echo "Outputs:"
echo "  results/figures/                — all generated plots"
echo "  results/codebooks/              — saved codebooks (.npz, .csv)"
echo "  results/pq_pickles/             — PQ baseline results"
echo "  results/tq_pickles/             — TurboQuant baseline results"
echo "  results/two_tier_recall.csv     — single-seed two-tier results"
echo "  results/pq_rerank_baseline.csv  — PQ + rerank baseline"
echo "  results/multiseed_recall.csv    — multi-seed validation"
echo "============================================================================"
echo
echo "Top-level figures:"
ls -lh "$ROOT/results/figures/" | head -40