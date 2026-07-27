#!/usr/bin/env bash
# Corrected fixed-model NCU collection. Invoke only through with_flexible_gpu.sh.
set -euo pipefail

KH_ROOT=/home/qinhaiyan/glm52-goal-runs/05-indexer_k_weights_prefill/kernel-harness
SGLANG_ROOT=/home/qinhaiyan/glm52-goal-runs/05-indexer_k_weights_prefill/sglang
EVIDENCE="$KH_ROOT/evidence/glm52_prod_05_indexer_k_weights_prefill"
PROFILE="$KH_ROOT/profile/indexer-wk-weights-prefill-m4096-20260722"
PY="$KH_ROOT/.venv/bin/python"
PROFILE_SCRIPT="$PROFILE/harness/profile_indexer_region.py"
OUT="$EVIDENCE/exact_ncu"

export SGLANG_ROOT
export PYTHONPATH="$SGLANG_ROOT/python:$KH_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export SGLANG_GLM52_OPT=0
unset SGLANG_DISABLE_DSA_INDEXER_FUSION
unset SGLANG_NVFP4_CKPT_FP8_GEMM_IN_ATTN
mkdir -p "$OUT" "$PROFILE/reports" "$PROFILE/analysis"

{
  printf 'timestamp_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  printf 'cuda_visible_devices=%s\n' "${CUDA_VISIBLE_DEVICES:-unset}"
  printf 'kh_sha=%s\n' "$(git -C "$KH_ROOT" rev-parse HEAD)"
  printf 'sglang_sha=%s\n' "$(git -C "$SGLANG_ROOT" rev-parse HEAD)"
  printf 'fixed_model_revision=%s\n' \
    aec724e8c7b8ee9db3b48c01c320f63f9cdaf8aa
  nvidia-smi --query-gpu=index,uuid,name,pstate,clocks.current.sm,clocks.current.memory,power.draw \
    --format=csv
} > "$OUT/environment.txt" 2>&1
"$PY" "$KH_ROOT/testbench/bin/check_env.py" > "$OUT/check_env.txt" 2>&1
: > "$OUT/status.txt"

run_ncu() {
  local label=$1
  shift
  if "$@" > "$PROFILE/analysis/ncu-exact-$label-console.txt" 2>&1; then
    printf '%s\n' "PASS ncu-exact-$label" >> "$OUT/status.txt"
  else
    local status=$?
    printf '%s\n' "FAIL($status) ncu-exact-$label" >> "$OUT/status.txt"
    return "$status"
  fi
}

GEMM_REGEX='regex:.*nvjet_sm100_tst.*'
INDEXER_REGEX='regex:.*(fused_q_indexer_rope_hadamard_quant|fused_k_indexer_norm_rope_store).*'

run_ncu gemms-full ncu --set full --section PmSampling \
  --section PmSampling_WarpStates --target-processes all \
  -k "$GEMM_REGEX" -c 2 -f \
  -o "$PROFILE/reports/full-exact-bf16-wq-gemms-m4096" \
  "$PY" "$PROFILE_SCRIPT" --warmup 2 \
  --trace-output "$OUT/runtime_abi_trace_gemms_full.json"

run_ncu gemms-source ncu --set source --section SourceCounters \
  --target-processes all -k "$GEMM_REGEX" -c 2 -f \
  -o "$PROFILE/reports/source-exact-bf16-wq-gemms-m4096" \
  "$PY" "$PROFILE_SCRIPT" --warmup 2 \
  --trace-output "$OUT/runtime_abi_trace_gemms_source.json"

run_ncu indexer-full ncu --set full --section PmSampling \
  --section PmSampling_WarpStates --target-processes all \
  -k "$INDEXER_REGEX" -c 2 -f \
  -o "$PROFILE/reports/full-exact-bf16-wq-indexer-post-m4096" \
  "$PY" "$PROFILE_SCRIPT" --warmup 2 \
  --trace-output "$OUT/runtime_abi_trace_indexer_full.json"

run_ncu indexer-source ncu --set source --section SourceCounters \
  --target-processes all -k "$INDEXER_REGEX" -c 2 -f \
  -o "$PROFILE/reports/source-exact-bf16-wq-indexer-post-m4096" \
  "$PY" "$PROFILE_SCRIPT" --warmup 2 \
  --trace-output "$OUT/runtime_abi_trace_indexer_source.json"

printf '%s\n' COMPLETE >> "$OUT/status.txt"
