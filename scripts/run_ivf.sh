#!/usr/bin/env bash
# k-means clustering for each dataset using config.yaml nlist.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
UPSTREAM="$ROOT/third_party/Extended-RaBitQ"

if [ -z "${DATASETS:-}" ]; then
  DATASETS="$(python "$ROOT/scripts/config_cli.py" datasets)"
fi

cd "$UPSTREAM"
for NAME in $DATASETS; do
  NLIST="$(python "$ROOT/scripts/config_cli.py" dataset_nlist "$NAME")"
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