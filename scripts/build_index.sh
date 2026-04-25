#!/usr/bin/env bash
# Build Extended-RaBitQ indexes at bits=3 (RaBitQ-2bit+sign equivalent)
# and bits=5 (RaBitQ-4bit+sign equivalent).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
UPSTREAM="$ROOT/third_party/Extended-RaBitQ"
NLIST="${NLIST:-256}"
BITS="${BITS:-3 5}"

cd "$UPSTREAM/bin"
for NAME in $DATASETS; do
  for B in $BITS; do
    INDEX="../data/$NAME/ivf_exhaf${B}.index"
    if [ -f "$INDEX" ]; then
      echo "  [$NAME B=$B] index exists, skipping"
      continue
    fi
    echo "=== create_index $NAME $NLIST $B ==="
    ./create_index "$NAME" "$NLIST" "$B"
  done
done