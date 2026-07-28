#!/usr/bin/env bash
set -euo pipefail

# Run this whole script as one command under with_flexible_gpu.sh.  Keeping the
# five workloads, all three A/B series, and the in-runner profiler/graph-node
# captures in one lease binds the complete smoke matrix to one physical B200.

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="${1:?usage: run_v2_identity_smoke.sh OUTPUT_DIR}"
PY="${KERNEL_HARNESS_PYTHON:-${ROOT}/.venv/bin/python}"
CANDIDATE="${ROOT}/serving_native/candidates/reference.py"

mkdir -p "${OUT}"

{
  echo "captured_utc=$(date --iso-8601=seconds)"
  echo "cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-}"
  echo "python=${PY}"
  echo "kernel_harness_head=$(git -C "${ROOT}" rev-parse HEAD)"
  echo "sglang_root=${SGLANG_ROOT:-$(cd "${ROOT}/.." && pwd)/sglang}"
  echo "dg_jit_cache_dir=${DG_JIT_CACHE_DIR:-}"
  echo "sglang_dg_cache_dir=${SGLANG_DG_CACHE_DIR:-}"
  echo "triton_cache_dir=${TRITON_CACHE_DIR:-}"
  echo "torch_extensions_dir=${TORCH_EXTENSIONS_DIR:-}"
} >"${OUT}/campaign_environment.txt"
nvidia-smi \
  --query-gpu=index,uuid,name,driver_version,clocks.current.sm,clocks.current.memory,pstate \
  --format=csv,noheader,nounits >"${OUT}/gpu_before.csv"
"${PY}" "${ROOT}/testbench/bin/sync_glm52_tasks.py" --check \
  >"${OUT}/generated_task_projection.log" 2>&1

run_one() {
  local task="$1"
  local mode="$2"
  local stem="${task}_${mode}"
  "${ROOT}/serving_native/run.sh" "${task}" \
    --candidate "${CANDIDATE}" \
    --execution-mode "${mode}" \
    --series 3 \
    --warmup 3 \
    --repeat 10 \
    --output "${OUT}/${stem}.json" \
    >"${OUT}/${stem}.log" 2>&1
  "${PY}" "${ROOT}/serving_native/audit_result.py" \
    "${OUT}/${stem}.json" --json >"${OUT}/${stem}.audit.json"
}

run_one linear_attn_o_prefill_m4096 eager
run_one linear_attn_o_decode_m16 eager
run_one linear_attn_o_decode_m16 cuda_graph
run_one linear_attn_o_decode_m32 eager
run_one linear_attn_o_decode_m32 cuda_graph

nvidia-smi \
  --query-gpu=index,uuid,name,driver_version,clocks.current.sm,clocks.current.memory,pstate \
  --format=csv,noheader,nounits >"${OUT}/gpu_after.csv"
find "${OUT}" -maxdepth 1 -type f ! -name SHA256SUMS -print0 \
  | sort -z \
  | xargs -0 sha256sum >"${OUT}/SHA256SUMS"
