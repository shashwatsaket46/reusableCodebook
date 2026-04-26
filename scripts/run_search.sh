#!/usr/bin/env bash
# Run patched test_search; output EVAL lines to results/logs/.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
UPSTREAM="$ROOT/third_party/Extended-RaBitQ"
BITS="${BITS:-3 4}"
mkdir -p "$UPSTREAM/results/exrabitq" "$ROOT/results/logs"

cd "$UPSTREAM/bin"
for NAME in $DATASETS; do
  for B in $BITS; do
    OUT="$ROOT/results/logs/exrabitq_${NAME}_b${B}.log"
    echo "=== test_search $NAME $B ==="
    ./test_search "$NAME" "$B" 2>&1 | tee "$OUT"
    echo
    echo "  EVAL lines: $(grep -c '^EVAL ' "$OUT")"
  done
done