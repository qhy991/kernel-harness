#!/usr/bin/env bash
# Decode throughput sweep — batch_size ∈ {128, 256}, TP=8.
# Measures decoder tok/s per batch after a short prefill (--decode-input-len).
#
# --overrides <file.py> swaps in candidate kernels before sglang boots.

set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

ROCM_TORCH_VENV="${ROCM_TORCH_VENV:-/root/venvs/rocm-torch}"
PYTHON="${ROCM_TORCH_PYTHON:-$ROCM_TORCH_VENV/bin/python}"

ulimit -c 0

exec "$PYTHON" "$HERE/bench_glm5_e2e.py" decode "$@"
