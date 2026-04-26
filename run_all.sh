#!/usr/bin/env bash
# Full pipeline. Set DATASETS env var to limit, e.g.:
#   DATASETS="glove200_100k openai1536" bash run_all.sh
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

mkdir -p results/logs results/pq_pickles results/figures

export DATASETS="${DATASETS:-glove200_100k openai1536 openai3072}"
echo "Datasets: $DATASETS"
echo "Logs:     $ROOT/results/logs/"
echo

LOG="$ROOT/results/logs/run_all.log"
exec > >(tee -a "$LOG") 2>&1

echo "[1/7] Setting up upstream repo + dependencies"
bash scripts/setup.sh

echo "[2/7] Preparing datasets"
python scripts/prepare_data.py

echo "[3/7] IVF clustering"
# nlist=256 for low/mid-dim, nlist=64 for 3072d (RAM-constrained)
UPSTREAM="$ROOT/third_party/Extended-RaBitQ"
for NAME in $DATASETS; do
#  if [ "$NAME" = "openai3072" ]; then NLIST=64; else NLIST=256; fi
  NLIST=256
  CENT="$UPSTREAM/data/$NAME/${NAME}_centroid_${NLIST}.fvecs"
  if [ -f "$CENT" ]; then
    echo "  [$NAME] centroids exist (nlist=$NLIST), skipping"
    continue
  fi
  echo "  [$NAME] running k-means with K=$NLIST ..."
  cd "$UPSTREAM"
  python python/ivf.py \
    --base "data/$NAME/${NAME}_base.fvecs" \
    --k "$NLIST" \
    --out_dir "data/$NAME" \
    --name "$NAME"
  cd "$ROOT"
done

echo "[4/7] Building Extended-RaBitQ indexes"
cd "$UPSTREAM/bin"
for NAME in $DATASETS; do
  if [ "$NAME" = "openai3072" ]; then NLIST=64; else NLIST=256; fi
  for B in 3 5; do
    INDEX="../data/$NAME/ivf_exhaf${B}.index"
    if [ -f "$INDEX" ]; then
      echo "  [$NAME B=$B] index exists, skipping"
      continue
    fi
    echo "=== create_index $NAME $NLIST $B ==="
    ./create_index "$NAME" "$NLIST" "$B"
  done
done
cd "$ROOT"

echo "[5/7] Running test_search"
cd "$UPSTREAM/bin"
mkdir -p "$UPSTREAM/results/exrabitq"
for NAME in $DATASETS; do
  for B in 3 5; do
    OUT="$ROOT/results/logs/exrabitq_${NAME}_b${B}.log"
    echo "=== test_search $NAME $B ==="
    ./test_search "$NAME" "$B" 2>&1 | tee "$OUT"
    echo "  EVAL lines: $(grep -c '^EVAL ' "$OUT")"
  done
done
cd "$ROOT"

echo "[6/7] Running PQ baselines"
python scripts/run_pq_baseline.py

echo "[7/7] Plotting"
python scripts/plot.py

echo
echo "Done. Figures in: $ROOT/results/figures/"
ls -lh "$ROOT/results/figures/"