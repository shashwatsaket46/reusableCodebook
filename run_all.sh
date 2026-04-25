#!/usr/bin/env bash
# Full pipeline. Set DATASETS env var to limit, e.g.:
#   DATASETS="glove200_100k openai1536" bash run_all.sh
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

mkdir -p results/logs results/pq_pickles results/figures

export DATASETS="${DATASETS:-glove200_100k openai1536 openai3072}"
echo "Datasets: $DATASETS"
echo "Logs:     $ROOT/results/logs"
echo

LOG="$ROOT/results/logs/run_all.log"
exec > >(tee -a "$LOG") 2>&1

echo "[1/7] Setting up upstream repo + dependencies"
bash scripts/setup.sh

echo "[2/7] Preparing datasets"
python scripts/prepare_data.py

echo "[3/7] IVF clustering"
bash scripts/run_ivf.sh

echo "[4/7] Building Extended-RaBitQ indexes"
bash scripts/build_index.sh

echo "[5/7] Running test_search"
bash scripts/run_search.sh

echo "[6/7] Running PQ baseline"
python scripts/run_pq_baseline.py

echo "[7/7] Plotting"
python scripts/plot.py

echo
echo "Done. Figures in: $ROOT/results/figures/"
ls -lh "$ROOT/results/figures/"