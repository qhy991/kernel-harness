#!/usr/bin/env bash
# Prefill TTFT sweep — aligned with llm_flops's bench_glm5_prefill.py scenarios.
# Runs sglang.bench_one_batch with:
#   · TP=8 (single node × 8 MI300X; GLM-5.2-FP8 ~700GB doesn't fit on fewer)
#   · KV cache pool = 65536 tokens
#   · input_len ∈ {1024, 2048, 4096}, output_len = 1
#   · reports TTFT (prefill_latency) per shape
#
# Pass --overrides <file.py> to replace named operators before boot.
# Without --overrides you get the vanilla sglang production dispatch (aiter
# for FP8/MLA/AR, sglang triton fused MoE) — that's your baseline.

set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

ROCM_TORCH_VENV="${ROCM_TORCH_VENV:-/root/venvs/rocm-torch}"
PYTHON="${ROCM_TORCH_PYTHON:-$ROCM_TORCH_VENV/bin/python}"

# Do NOT skip hooks / GPU coredump protection.
ulimit -c 0

# Default full sweep = 1024, 2048, 4096. Override with --input-len 1024 etc.
exec "$PYTHON" "$HERE/bench_glm5_e2e.py" prefill "$@"
