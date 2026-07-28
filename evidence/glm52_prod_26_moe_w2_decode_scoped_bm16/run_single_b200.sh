#!/usr/bin/env bash
# Run only through:
#   /home/qinhaiyan/glm52-goal-runs/with_flexible_gpu.sh -- <this-script>
set -euo pipefail

readonly ROOT="/home/qinhaiyan/glm52-v2-goal-runs/worktrees/26-moe-w2-decode-scoped-bm16/kernel-harness"
readonly SGLANG_ROOT="/home/qinhaiyan/glm52-v2-goal-runs/worktrees/26-moe-w2-decode-scoped-bm16/sglang"
readonly TASK_CACHE_ROOT="/home/qinhaiyan/glm52-v2-goal-runs/cache/26-moe_w2_decode_scoped_bm16"
readonly PYTHON="${ROOT}/.venv/bin/python"
readonly RUNNER="${ROOT}/serving_native/runner.py"
readonly AUDITOR="${ROOT}/serving_native/audit_result.py"
readonly CANDIDATE="${ROOT}/serving_native/candidates/moe_w2_bm16.py"
readonly STOCK_LAUNCHER="${SGLANG_ROOT}/third_party/deepgemm_w2_bm16/run_with_exact_post1_stock.sh"
readonly DEFAULT_RUN_ROOT="${ROOT}/runs/glm52_prod_26_moe_w2_decode_scoped_bm16"

if [[ -z "${CUDA_VISIBLE_DEVICES:-}" || "${CUDA_VISIBLE_DEVICES}" == *,* ]]; then
  echo "ERROR: this portfolio requires one scheduler-assigned B200" >&2
  exit 64
fi

export SGLANG_ROOT
export DG_JIT_CACHE_DIR="${TASK_CACHE_ROOT}/deepgemm"
export SGLANG_DG_CACHE_DIR="${TASK_CACHE_ROOT}/deepgemm"
export TRITON_CACHE_DIR="${TASK_CACHE_ROOT}/triton"
export TORCH_EXTENSIONS_DIR="${TASK_CACHE_ROOT}/torch_extensions"

run_stamp="$(date -u +%Y%m%dT%H%M%SZ)"
run_root="${TASK26_RUN_ROOT:-${DEFAULT_RUN_ROOT}/${run_stamp}}"
mkdir -p "${run_root}/results" "${run_root}/logs" "${run_root}/audits"

check_disk() {
  local available_kib
  available_kib="$(df -Pk "${ROOT}" | awk 'NR == 2 {print $4}')"
  if (( available_kib < 8 * 1024 * 1024 )); then
    echo "ERROR: fewer than 8 GiB remain; refusing new JIT/profile growth" >&2
    exit 1
  fi
}

record_environment() {
  {
    echo "started_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "cuda_visible_devices=${CUDA_VISIBLE_DEVICES}"
    echo "kernel_harness_head=$(git -C "${ROOT}" rev-parse HEAD)"
    echo "sglang_head=$(git -C "${SGLANG_ROOT}" rev-parse HEAD)"
    echo "inherited_contract_commit=c1c48c3d1e826c243727ed45d52ef8dbfeb3f701"
    echo "inherited_local_commit=$(git -C "${ROOT}" rev-parse 8fc047c)"
    echo "dg_jit_cache_dir=${DG_JIT_CACHE_DIR}"
    echo "sglang_dg_cache_dir=${SGLANG_DG_CACHE_DIR}"
    echo "triton_cache_dir=${TRITON_CACHE_DIR}"
    echo "torch_extensions_dir=${TORCH_EXTENSIONS_DIR}"
    df -h /
    nvidia-smi -i "${CUDA_VISIBLE_DEVICES}" \
      --query-gpu=index,uuid,name,driver_version,clocks.current.sm,clocks.current.memory \
      --format=csv,noheader,nounits
    git -C "${ROOT}" status --short
    git -C "${SGLANG_ROOT}" status --short
  } >"${run_root}/environment.txt"
}

run_one() {
  local task="$1"
  local mode="$2"
  local stem="${task}__${mode}"
  check_disk
  echo "RUN ${stem}" >&2
  "${STOCK_LAUNCHER}" "${PYTHON}" "${RUNNER}" \
    --task "${task}" \
    --candidate "${CANDIDATE}" \
    --execution-mode "${mode}" \
    --warmup 3 \
    --repeat 10 \
    --series 3 \
    --output "${run_root}/results/${stem}.json" \
    >"${run_root}/logs/${stem}.log" 2>&1
  "${PYTHON}" "${AUDITOR}" --json \
    "${run_root}/results/${stem}.json" \
    >"${run_root}/audits/${stem}.json"
}

check_disk
record_environment

run_one moe_w2_grouped_decode_m16 eager
run_one moe_w2_grouped_decode_m16 cuda_graph
run_one moe_w2_grouped_decode_m16_current_source_m5 eager
run_one moe_w2_grouped_decode_m16_current_source_m5 cuda_graph
run_one moe_w2_grouped_decode_m32 eager
run_one moe_w2_grouped_decode_m32 cuda_graph
run_one moe_w2_grouped_decode_m32_current_source_m9 eager
run_one moe_w2_grouped_decode_m32_current_source_m9 cuda_graph
run_one moe_w13_swiglu_w2_region_decode_m16_current_source_m5 eager
run_one moe_w13_swiglu_w2_region_decode_m32_current_source_m9 eager

{
  echo "finished_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  df -h /
} >"${run_root}/completion.txt"
find "${run_root}" -type f ! -name artifact_sha256.txt -print0 \
  | sort -z \
  | xargs -0 sha256sum \
  >"${run_root}/artifact_sha256.txt"
echo "PASS task26 single-B200 portfolio: ${run_root}" >&2
