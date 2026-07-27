#!/usr/bin/env bash
set -euo pipefail

# CPU-only preparation. This intentionally installs beside, never into, the
# repo venv. GPU scripts validate and prepend this exact overlay.
ROOT=/home/qinhaiyan/glm52-goal-runs/07-moe_w2_decode_kernel/kernel-harness
SGLANG=/home/qinhaiyan/glm52-goal-runs/07-moe_w2_decode_kernel/sglang
PYTHON="$ROOT/.venv/bin/python"
OVERLAY="$SGLANG/build/deep-gemm-stock-0.1.4.post1"
EXPECTED_EXTENSION_SHA=cd8beab174071777c972c5948af7706ae2cfb5d2adcdbb7e6fbea253ce3a81bf

if [[ ! -f "$OVERLAY/deep_gemm/_C.so" ]]; then
    mkdir -p "$OVERLAY"
    "$PYTHON" -m pip install \
        --target "$OVERLAY" \
        --no-deps \
        --upgrade \
        sgl-deep-gemm==0.1.4.post1
fi

actual_sha="$(sha256sum "$OVERLAY/deep_gemm/_C.so" | awk '{print $1}')"
if [[ "$actual_sha" != "$EXPECTED_EXTENSION_SHA" ]]; then
    echo "DeepGEMM overlay SHA mismatch: $actual_sha" >&2
    exit 1
fi
if [[ "$(tr -d '[:space:]' < "$OVERLAY/deep_gemm/VERSION")" != "0.1.4.post1" ]]; then
    echo "DeepGEMM overlay version mismatch" >&2
    exit 1
fi
echo "$OVERLAY"
