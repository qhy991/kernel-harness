#!/usr/bin/env bash
set -euo pipefail

KH_ROOT=/home/qinhaiyan/glm52-goal-runs/05-indexer_k_weights_prefill/kernel-harness
SGLANG_ROOT=/home/qinhaiyan/glm52-goal-runs/05-indexer_k_weights_prefill/sglang
EVIDENCE="$KH_ROOT/evidence/glm52_prod_05_indexer_k_weights_prefill"
PROFILE="$KH_ROOT/profile/indexer-wk-weights-prefill-m4096-20260722"
PY="$KH_ROOT/.venv/bin/python"
ISOLATED=indexer_wk_weights_prefill_m4096
REGION=indexer_fused_prepare_store_prefill_m4096_eager_dual_stream
TORCH_MM="$KH_ROOT/serving_native/candidates/indexer_wk_torch_mm.py"
K_FIRST="$EVIDENCE/archived_candidates/indexer_k_first_schedule.py"
PROFILE_SCRIPT="$PROFILE/harness/profile_indexer_region.py"

export SGLANG_ROOT
export PYTHONPATH="$SGLANG_ROOT/python:$KH_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export SGLANG_GLM52_OPT=0
unset SGLANG_DISABLE_DSA_INDEXER_FUSION
mkdir -p "$EVIDENCE/logs" "$PROFILE/reports" "$PROFILE/analysis"
: > "$EVIDENCE/schedule_campaign_status.txt"

run_candidate() {
  local task=$1
  local candidate=$2
  local output=$3
  "$KH_ROOT/serving_native/run.sh" "$task" --candidate "$candidate" \
    --warmup 10 --repeat 60 --output "$output" \
    > "${output%.json}.log" 2>&1
}

for run in 01 02 03; do
  run_candidate "$ISOLATED" "$TORCH_MM" "$EVIDENCE/isolated_torch_mm_${run}.json"
  run_candidate "$REGION" "$TORCH_MM" "$EVIDENCE/region_torch_mm_${run}.json"
  run_candidate "$REGION" "$K_FIRST" "$EVIDENCE/region_k_first_${run}.json"
done

# Capture only the final NVTX-bracketed call so DeepGEMM/CuTe import-time JIT
# activity cannot pollute the stream timeline.
nsys profile --force-overwrite=true --trace=cuda,nvtx,cublas --sample=none \
  -c cudaProfilerApi --capture-range-end=stop --kill=none \
  -o "$PROFILE/reports/nsys-schedule-stock" \
  "$PY" "$PROFILE_SCRIPT" --warmup 10 --cuda-profiler-api \
  --trace-output "$EVIDENCE/runtime_abi_trace_schedule_stock.json" \
  > "$PROFILE/analysis/nsys-schedule-stock-console.txt" 2>&1

nsys profile --force-overwrite=true --trace=cuda,nvtx,cublas --sample=none \
  -c cudaProfilerApi --capture-range-end=stop --kill=none \
  -o "$PROFILE/reports/nsys-k-first" \
  "$PY" "$PROFILE_SCRIPT" --candidate "$K_FIRST" --warmup 10 \
  --cuda-profiler-api \
  --trace-output "$EVIDENCE/runtime_abi_trace_k_first.json" \
  > "$PROFILE/analysis/nsys-k-first-console.txt" 2>&1

for tag in schedule-stock k-first; do
  nsys stats --force-export=true --report cuda_gpu_trace:nvtx-name:base,nvtx_gpu_proj_sum \
    --format csv --output "$PROFILE/analysis/nsys-$tag" \
    "$PROFILE/reports/nsys-$tag.nsys-rep" \
    > "$PROFILE/analysis/nsys-$tag-stats-console.txt" 2>&1
done

run_ncu() {
  local label=$1
  shift
  if "$@" > "$PROFILE/analysis/ncu-$label-console.txt" 2>&1; then
    printf '%s\n' "PASS ncu-$label" >> "$EVIDENCE/schedule_campaign_status.txt"
  else
    status=$?
    printf '%s\n' "FAIL($status) ncu-$label" >> "$EVIDENCE/schedule_campaign_status.txt"
  fi
}

run_ncu wk-full ncu --set full --section PmSampling \
  --section PmSampling_WarpStates --target-processes all \
  -k 'regex:.*nvjet_sm100_tst.*' -c 1 -f \
  -o "$PROFILE/reports/full-stock-wk-m4096" \
  "$KH_ROOT/serving_native/run.sh" "$ISOLATED" --warmup 0 --repeat 1

run_ncu wk-source ncu --set source --section SourceCounters \
  --target-processes all -k 'regex:.*nvjet_sm100_tst.*' -c 1 -f \
  -o "$PROFILE/reports/source-stock-wk-m4096" \
  "$KH_ROOT/serving_native/run.sh" "$ISOLATED" --warmup 0 --repeat 1

INDEXER_KERNEL_REGEX='regex:.*(fused_q_indexer_rope_hadamard_quant|fused_k_indexer_norm_rope_store).*'
run_ncu indexer-full ncu --set full --section PmSampling \
  --section PmSampling_WarpStates --target-processes all \
  -k "$INDEXER_KERNEL_REGEX" -c 2 -f \
  -o "$PROFILE/reports/full-stock-indexer-post-m4096" \
  "$PY" "$PROFILE_SCRIPT" --warmup 2 \
  --trace-output "$EVIDENCE/runtime_abi_trace_ncu_stock.json"

run_ncu indexer-source ncu --set source --section SourceCounters \
  --target-processes all -k "$INDEXER_KERNEL_REGEX" -c 2 -f \
  -o "$PROFILE/reports/source-stock-indexer-post-m4096" \
  "$PY" "$PROFILE_SCRIPT" --warmup 2 \
  --trace-output "$EVIDENCE/runtime_abi_trace_ncu_source_stock.json"

printf '%s\n' "SGLang trial commit: $(git -C "$SGLANG_ROOT" rev-parse HEAD)" \
  >> "$EVIDENCE/schedule_campaign_status.txt"
