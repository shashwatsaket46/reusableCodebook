#!/usr/bin/env bash
# k-means clustering for each dataset (NLIST=256).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
UPSTREAM="$ROOT/third_party/Extended-RaBitQ"
NLIST="${NLIST:-256}"

cd "$UPSTREAM"
for NAME in $DATASETS; do
  CENT="data/$NAME/${NAME}_centroid_${NLIST}.fvecs"
  if [ -f "$CENT" ]; then
    echo "  [$NAME] centroids already exist, skipping"
    continue
  fi
  echo "  [$NAME] running k-means with K=$NLIST ..."
  python python/ivf.py \
    --base "data/$NAME/${NAME}_base.fvecs" \
    --k "$NLIST" \
    --out_dir "data/$NAME" \
    --name "$NAME"
done