#!/usr/bin/env bash
# Corrected fixed-model campaign. Invoke only through with_flexible_gpu.sh.
set -euo pipefail

KH_ROOT=/home/qinhaiyan/glm52-goal-runs/05-indexer_k_weights_prefill/kernel-harness
SGLANG_ROOT=/home/qinhaiyan/glm52-goal-runs/05-indexer_k_weights_prefill/sglang
EVIDENCE="$KH_ROOT/evidence/glm52_prod_05_indexer_k_weights_prefill"
PROFILE="$KH_ROOT/profile/indexer-wk-weights-prefill-m4096-20260722"
PY="$KH_ROOT/.venv/bin/python"
ISOLATED=indexer_wk_weights_prefill_m4096
REGION=indexer_fused_prepare_store_prefill_m4096_eager_dual_stream
IDENTITY="$KH_ROOT/serving_native/candidates/reference.py"
TORCH_MM="$KH_ROOT/serving_native/candidates/indexer_wk_torch_mm.py"
TGV="$KH_ROOT/serving_native/candidates/indexer_wk_cutedsl_tgv.py"
PROFILE_SCRIPT="$PROFILE/harness/profile_indexer_region.py"

export SGLANG_ROOT
export PYTHONPATH="$SGLANG_ROOT/python:$KH_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export SGLANG_GLM52_OPT=0
unset SGLANG_DISABLE_DSA_INDEXER_FUSION
unset SGLANG_NVFP4_CKPT_FP8_GEMM_IN_ATTN

mkdir -p "$EVIDENCE/exact_bf16_wq" "$PROFILE/reports" "$PROFILE/analysis"
OUT="$EVIDENCE/exact_bf16_wq"

{
  printf 'timestamp_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  printf 'cuda_visible_devices=%s\n' "${CUDA_VISIBLE_DEVICES:-unset}"
  printf 'kh_sha=%s\n' "$(git -C "$KH_ROOT" rev-parse HEAD)"
  printf 'sglang_sha=%s\n' "$(git -C "$SGLANG_ROOT" rev-parse HEAD)"
  printf 'dsa_indexer_sha256='
  sha256sum "$SGLANG_ROOT/python/sglang/srt/layers/attention/dsa/dsa_indexer.py"
  printf 'fixed_model_revision=%s\n' \
    aec724e8c7b8ee9db3b48c01c320f63f9cdaf8aa
  nvidia-smi --query-gpu=index,uuid,name,pstate,clocks.current.sm,clocks.current.memory,power.draw \
    --format=csv
} > "$OUT/environment.txt" 2>&1

"$PY" "$KH_ROOT/testbench/bin/check_env.py" > "$OUT/check_env.txt" 2>&1

run_reference() {
  local task=$1
  local output=$2
  "$KH_ROOT/serving_native/run.sh" "$task" --warmup 10 --repeat 30 \
    --output "$output" > "${output%.json}.log" 2>&1
}

run_candidate() {
  local task=$1
  local candidate=$2
  local output=$3
  local repeat=${4:-60}
  "$KH_ROOT/serving_native/run.sh" "$task" --candidate "$candidate" \
    --warmup 10 --repeat "$repeat" --output "$output" \
    > "${output%.json}.log" 2>&1
}

# Three uncontended stock baselines for both the optimization target and its
# fused prepare/store subregion, all on this one locked physical GPU.
for run in 01 02 03; do
  run_reference "$ISOLATED" "$OUT/isolated_stock_${run}.json"
  run_reference "$REGION" "$OUT/region_stock_${run}.json"
done

# Paired identity controls establish the local noise floor.
run_candidate "$ISOLATED" "$IDENTITY" "$OUT/isolated_identity.json" 60
run_candidate "$REGION" "$IDENTITY" "$OUT/region_identity.json" 60

# The direct ATen spelling was the only isolated backend near stock. The TGV
# candidate is the strongest library-kernel hypothesis from prior art; three
# subregion repetitions prevent a single favorable noisy run from promoting.
for run in 01 02 03; do
  run_candidate "$ISOLATED" "$TORCH_MM" "$OUT/isolated_torch_mm_${run}.json" 60
  run_candidate "$REGION" "$TORCH_MM" "$OUT/region_torch_mm_${run}.json" 60
  run_candidate "$REGION" "$TGV" "$OUT/region_tgv_${run}.json" 60
done

# Profile stock and the least-invasive candidate in the same locked session.
nsys profile --force-overwrite=true --trace=cuda,nvtx,cublas --sample=none \
  -c cudaProfilerApi --capture-range-end=stop --kill=none \
  -o "$PROFILE/reports/nsys-exact-bf16-wq-stock" \
  "$PY" "$PROFILE_SCRIPT" --warmup 10 --cuda-profiler-api \
  --trace-output "$OUT/runtime_abi_trace_stock.json" \
  > "$PROFILE/analysis/nsys-exact-bf16-wq-stock-console.txt" 2>&1

nsys profile --force-overwrite=true --trace=cuda,nvtx,cublas --sample=none \
  -c cudaProfilerApi --capture-range-end=stop --kill=none \
  -o "$PROFILE/reports/nsys-exact-bf16-wq-torch-mm" \
  "$PY" "$PROFILE_SCRIPT" --candidate "$TORCH_MM" --warmup 10 \
  --cuda-profiler-api --trace-output "$OUT/runtime_abi_trace_torch_mm.json" \
  > "$PROFILE/analysis/nsys-exact-bf16-wq-torch-mm-console.txt" 2>&1

for tag in stock torch-mm; do
  nsys stats --force-export=true \
    --report cuda_gpu_trace:nvtx-name:base,nvtx_gpu_proj_sum \
    --format csv --output "$PROFILE/analysis/nsys-exact-bf16-wq-$tag" \
    "$PROFILE/reports/nsys-exact-bf16-wq-$tag.nsys-rep" \
    > "$PROFILE/analysis/nsys-exact-bf16-wq-$tag-stats-console.txt" 2>&1
done

printf '%s\n' PASS > "$OUT/campaign_status.txt"
