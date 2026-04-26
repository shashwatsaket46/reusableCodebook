#!/usr/bin/env bash
# Build Extended-RaBitQ indexes using config.yaml settings.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
UPSTREAM="$ROOT/third_party/Extended-RaBitQ"

if [ -z "${DATASETS:-}" ]; then
  DATASETS="$(python "$ROOT/scripts/config_cli.py" datasets)"
fi
BITS="${BITS:-$(python "$ROOT/scripts/config_cli.py" exrabitq_bits)}"

cd "$UPSTREAM/bin"
for NAME in $DATASETS; do
  NLIST="${NLIST:-$(python "$ROOT/scripts/config_cli.py" dataset_nlist "$NAME")}"
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