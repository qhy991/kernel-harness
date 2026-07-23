#!/usr/bin/env bash
set -euo pipefail

case "${CUDA_VISIBLE_DEVICES:-}" in
  0|1|2|3) ;;
  *)
    echo "run through /home/qinhaiyan/glm52-goal-runs/with_flexible_gpu.sh -- <command>" >&2
    exit 64
    ;;
esac

KH=/home/qinhaiyan/glm52-goal-runs/22-dsa_flashmla_kv_production/kernel-harness
SG=/home/qinhaiyan/glm52-goal-runs/22-dsa_flashmla_kv_production/sglang
FLASHMLA="$SG/third_party/FlashMLA-goal22"
BASE_RUN="$KH/profile/dsa-flashmla-kv-stock-b200-20260722"
RUN="$BASE_RUN"
if [[ "$#" -gt 0 ]]; then
  if [[ "$#" -ne 2 || "$1" != "--output-root" ]]; then
    echo "usage: collect_ncu.sh [--output-root CAMPAIGN_ROOT]" >&2
    exit 64
  fi
  RUN="$(realpath "$2")"
  case "$RUN" in
    "$BASE_RUN"/campaigns/*) ;;
    *)
      echo "campaign output must be below $BASE_RUN/campaigns" >&2
      exit 64
      ;;
  esac
fi
mkdir -p "$RUN/analysis" "$RUN/reports"
DRIVER="$BASE_RUN/harness/profile_driver.py"
CANDIDATE="$KH/serving_native/candidates/flashmla_goal22_overlay.py"
PYTHON="$KH/.venv/bin/python"

STOCK_OVERLAY="$FLASHMLA/build-artifacts/stock-pybind-tensor/overlay"
STOCK_MANIFEST="$BASE_RUN/analysis/build_stock_pybind_tensor.json"
CANDIDATE_OVERLAY="$FLASHMLA/build-artifacts/combine32-m16-tensor/overlay"
CANDIDATE_MANIFEST="$BASE_RUN/analysis/build_combine32_m16_tensor.json"

run_profile() {
  local mode=$1
  local label=$2
  local task=$3
  local overlay=$4
  local manifest=$5
  local kernel_regex=$6
  local report="$RUN/reports/$label"
  local runtime="$RUN/analysis/${label}_runtime.json"

  if [[ -e "${report}.ncu-rep" || -e "$runtime" ]]; then
    echo "refusing to overwrite profiler evidence for $label" >&2
    exit 65
  fi

  local -a sections
  if [[ "$mode" == full ]]; then
    sections=(--set full --section PmSampling --section PmSampling_WarpStates)
  elif [[ "$mode" == source ]]; then
    # NCU 2026.1.1 has no named `source` set. SourceCounters plus the basic
    # launch/SOL set is the version-correct equivalent for per-PC attribution.
    sections=(--set basic --section SourceCounters)
  else
    echo "unknown profile mode: $mode" >&2
    exit 64
  fi

  echo "==> $label ($task, $mode)"
  env \
    SGLANG_GLM52_OPT=0 \
    SGLANG_ROOT="$SG" \
    GOAL22_FLASHMLA_OVERLAY="$overlay" \
    GOAL22_FLASHMLA_MANIFEST="$manifest" \
    ncu "${sections[@]}" \
      --import-source yes \
      -k "regex:${kernel_regex}" \
      -s 3 \
      -c 1 \
      -o "$report" \
      "$PYTHON" "$DRIVER" \
        --task "$task" \
        --candidate "$CANDIDATE" \
        --warmup 3 \
        --output "$runtime"
}

# Dominant sparse main kernel for both fixed production buckets.
run_profile full full_stock_main_m16 dsa_flashmla_kv_decode_m16 \
  "$STOCK_OVERLAY" "$STOCK_MANIFEST" flash_fwd_splitkv_mla_fp8_sparse_kernel
run_profile full full_stock_main_m32 dsa_flashmla_kv_decode_m32 \
  "$STOCK_OVERLAY" "$STOCK_MANIFEST" flash_fwd_splitkv_mla_fp8_sparse_kernel
run_profile source source_stock_main_m16 dsa_flashmla_kv_decode_m16 \
  "$STOCK_OVERLAY" "$STOCK_MANIFEST" flash_fwd_splitkv_mla_fp8_sparse_kernel

# A/B profile the only changed device kernel: M16's combine reduction.
run_profile full full_stock_combine_m16 dsa_flashmla_kv_decode_m16 \
  "$STOCK_OVERLAY" "$STOCK_MANIFEST" flash_fwd_mla_combine_kernel
run_profile full full_combine32_combine_m16 dsa_flashmla_kv_decode_m16 \
  "$CANDIDATE_OVERLAY" "$CANDIDATE_MANIFEST" flash_fwd_mla_combine_kernel
run_profile source source_stock_combine_m16 dsa_flashmla_kv_decode_m16 \
  "$STOCK_OVERLAY" "$STOCK_MANIFEST" flash_fwd_mla_combine_kernel
run_profile source source_combine32_combine_m16 dsa_flashmla_kv_decode_m16 \
  "$CANDIDATE_OVERLAY" "$CANDIDATE_MANIFEST" flash_fwd_mla_combine_kernel
