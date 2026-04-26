#!/usr/bin/env bash
# Clone upstream Extended-RaBitQ, vendor Eigen + hnswlib, copy patched src, build.
# Idempotent: skips steps already done.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
UPSTREAM="$ROOT/third_party/Extended-RaBitQ"
UPSTREAM_PIN="main"   # change to a SHA when ready to freeze

# AVX-512 sanity check
if ! grep -q avx512f /proc/cpuinfo; then
  echo "ERROR: CPU lacks AVX-512. Extended RaBitQ requires it."
  exit 1
fi

# Python deps
pip install -q -r "$ROOT/requirements.txt"

mkdir -p "$ROOT/third_party"

# Clone upstream
if [ ! -d "$UPSTREAM" ]; then
  echo "  cloning Extended-RaBitQ..."
  git clone --depth 1 --branch "$UPSTREAM_PIN" \
    https://github.com/VectorDB-NTU/Extended-RaBitQ.git "$UPSTREAM"
fi

# Eigen 3.4.0
if [ ! -e "$UPSTREAM/inc/third/Eigen" ]; then
  echo "  vendoring Eigen 3.4.0..."
  cd "$UPSTREAM/inc/third"
  wget -q https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.tar.gz
  tar xzf eigen-3.4.0.tar.gz
  ln -sf eigen-3.4.0/Eigen Eigen
  rm -f eigen-3.4.0.tar.gz
fi

# hnswlib (header-only)
if [ ! -f "$UPSTREAM/inc/third/hnswlib/hnswlib.h" ]; then
  echo "  vendoring hnswlib..."
  cd "$UPSTREAM/inc/third"
  rm -rf hnswlib_src hnswlib
  git clone --depth 1 https://github.com/nmslib/hnswlib.git hnswlib_src
  cp -r hnswlib_src/hnswlib hnswlib
  rm -rf hnswlib_src
fi

# Copy patched test_search.cpp from overrides (original stable version for per-k recall)
python "$ROOT/scripts/generate_cpp_config.py"
echo "  copying stable test_search.cpp from overrides..."
cp "$ROOT/cpp/overrides/src/test_search.cpp" "$UPSTREAM/src/test_search.cpp"
cp "$ROOT/cpp/overrides/src/create_index.cpp" "$UPSTREAM/src/create_index.cpp" 2>/dev/null || true

# Copy header overrides if present
if [ -d "$ROOT/cpp/overrides/inc" ]; then
  echo "  copying header overrides..."
  cp -r "$ROOT/cpp/overrides/inc/"* "$UPSTREAM/inc/" 2>/dev/null || true
fi

# Replace ivf.py with argparse version
cp "$ROOT/scripts/ivf_argparse.py" "$UPSTREAM/python/ivf.py"
echo "  cleaning old build artifacts..."
rm -rf "$UPSTREAM/build" "$UPSTREAM/bin"

# Build
mkdir -p "$UPSTREAM/build" "$UPSTREAM/bin"
cd "$UPSTREAM/build"
cmake -DCMAKE_BUILD_TYPE=Release .. > /tmp/cmake.log 2>&1 || {
  echo "CMAKE FAILED:"; tail -50 /tmp/cmake.log; exit 1;
}
make -j"$(nproc)" 2>&1 | tail -10

echo "  binaries:"
ls -lh "$UPSTREAM/bin/"