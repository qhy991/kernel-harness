#!/usr/bin/env bash
# Matched dual-vs-single stream experiment. Invoke only via with_flexible_gpu.sh.
set -euo pipefail

KH_ROOT=/home/qinhaiyan/glm52-goal-runs/05-indexer_k_weights_prefill/kernel-harness
SGLANG_ROOT=/home/qinhaiyan/glm52-goal-runs/05-indexer_k_weights_prefill/sglang
EVIDENCE="$KH_ROOT/evidence/glm52_prod_05_indexer_k_weights_prefill"
PROFILE="$KH_ROOT/profile/indexer-wk-weights-prefill-m4096-20260722"
PY="$KH_ROOT/.venv/bin/python"
REGION=indexer_fused_prepare_store_prefill_m4096_eager_dual_stream
CANDIDATE="$KH_ROOT/serving_native/candidates/indexer_single_stream.py"
PROFILE_SCRIPT="$PROFILE/harness/profile_indexer_region.py"
OUT="$EVIDENCE/exact_single_stream"

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

for run in 01 02 03; do
  "$KH_ROOT/serving_native/run.sh" "$REGION" --candidate "$CANDIDATE" \
    --warmup 10 --repeat 60 --output "$OUT/region_single_stream_${run}.json" \
    > "$OUT/region_single_stream_${run}.log" 2>&1
done

for mode in stock single-stream; do
  PROFILE_ARGS=()
  if [[ "$mode" == single-stream ]]; then
    PROFILE_ARGS=(--candidate "$CANDIDATE")
  fi
  nsys profile --force-overwrite=true --trace=cuda,nvtx,cublas --sample=none \
    -c cudaProfilerApi --capture-range-end=stop --kill=none \
    -o "$PROFILE/reports/nsys-exact-single-stream-$mode" \
    "$PY" "$PROFILE_SCRIPT" "${PROFILE_ARGS[@]}" --warmup 10 \
    --cuda-profiler-api --trace-output "$OUT/runtime_abi_trace_${mode}.json" \
    > "$PROFILE/analysis/nsys-exact-single-stream-$mode-console.txt" 2>&1
  nsys stats --force-export=true \
    --report cuda_gpu_trace:nvtx-name:base,nvtx_gpu_proj_sum \
    --format csv --output "$PROFILE/analysis/nsys-exact-single-stream-$mode" \
    "$PROFILE/reports/nsys-exact-single-stream-$mode.nsys-rep" \
    > "$PROFILE/analysis/nsys-exact-single-stream-$mode-stats-console.txt" 2>&1
done

printf '%s\n' PASS > "$OUT/campaign_status.txt"
