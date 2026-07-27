#!/usr/bin/env bash
set -euo pipefail

KH_ROOT=/home/qinhaiyan/glm52-goal-runs/05-indexer_k_weights_prefill/kernel-harness
SGLANG_ROOT=/home/qinhaiyan/glm52-goal-runs/05-indexer_k_weights_prefill/sglang
EVIDENCE="$KH_ROOT/evidence/glm52_prod_05_indexer_k_weights_prefill"
PROFILE="$KH_ROOT/profile/indexer-wk-weights-prefill-m4096-20260722"
PY="$KH_ROOT/.venv/bin/python"
ISOLATED=indexer_wk_weights_prefill_m4096
REGION=indexer_fused_prepare_store_prefill_m4096_eager_dual_stream
CUTEDSL="$KH_ROOT/serving_native/candidates/indexer_wk_cutedsl_tgv.py"
FLASHINFER="$KH_ROOT/serving_native/candidates/indexer_wk_flashinfer.py"
REFERENCE="$KH_ROOT/serving_native/candidates/reference.py"

export SGLANG_ROOT
export PYTHONPATH="$SGLANG_ROOT/python:$KH_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export SGLANG_GLM52_OPT=0
unset SGLANG_DISABLE_DSA_INDEXER_FUSION
mkdir -p "$EVIDENCE/logs" "$PROFILE/reports" "$PROFILE/analysis"

"$PY" "$EVIDENCE/collect_environment.py" > "$EVIDENCE/environment.json"
"$PY" "$KH_ROOT/testbench/bin/check_env.py" > "$EVIDENCE/check_env.txt" 2>&1

run_stock() {
  local task=$1
  local output=$2
  "$KH_ROOT/serving_native/run.sh" "$task" --warmup 5 --repeat 30 --output "$output" \
    > "${output%.json}.log" 2>&1
}

run_candidate() {
  local task=$1
  local candidate=$2
  local output=$3
  shift 3
  env "$@" "$KH_ROOT/serving_native/run.sh" "$task" \
    --candidate "$candidate" --warmup 5 --repeat 30 --output "$output" \
    > "${output%.json}.log" 2>&1
}

try_candidate() {
  local label=$1
  local task=$2
  local candidate=$3
  local output=$4
  shift 4
  if run_candidate "$task" "$candidate" "$output" "$@"; then
    printf '%s\n' "PASS $label" >> "$EVIDENCE/sweep_status.txt"
  else
    status=$?
    printf '%s\n' "FAIL($status) $label" >> "$EVIDENCE/sweep_status.txt"
  fi
}

: > "$EVIDENCE/sweep_status.txt"
for run in 01 02 03; do
  run_stock "$ISOLATED" "$EVIDENCE/isolated_baseline_${run}.json"
  run_stock "$REGION" "$EVIDENCE/region_baseline_${run}.json"
done

# Paired identity calls establish the local A/B noise floor.
run_candidate "$ISOLATED" "$REFERENCE" "$EVIDENCE/isolated_reference_control.json"
run_candidate "$REGION" "$REFERENCE" "$EVIDENCE/region_reference_control.json"

try_candidate cutedsl "$ISOLATED" "$CUTEDSL" \
  "$EVIDENCE/isolated_cutedsl_sweep.json"
for backend in cutlass tgv cublaslt cudnn auto; do
  try_candidate "flashinfer-$backend" "$ISOLATED" "$FLASHINFER" \
    "$EVIDENCE/isolated_flashinfer_${backend}_sweep.json" \
    "INDEXER_WK_FLASHINFER_BACKEND=$backend"
done

"$PY" "$EVIDENCE/select_best.py" "$EVIDENCE" "$EVIDENCE/best_selection.json"
BEST_CANDIDATE=$("$PY" -c 'import json,sys; print(json.load(open(sys.argv[1]))["selected"]["candidate_path"])' "$EVIDENCE/best_selection.json")
BEST_BACKEND=$("$PY" -c 'import json,sys; print(json.load(open(sys.argv[1]))["selected"]["backend"])' "$EVIDENCE/best_selection.json")
BEST_ENV=()
if [[ "$BEST_BACKEND" == flashinfer_mm_bf16_* ]]; then
  BEST_ENV=("INDEXER_WK_FLASHINFER_BACKEND=${BEST_BACKEND#flashinfer_mm_bf16_}")
fi

# Confirm the strongest isolated choice and then measure the complete overlap
# region three times on the same locked physical GPU.
for run in 01 02 03; do
  run_candidate "$ISOLATED" "$BEST_CANDIDATE" \
    "$EVIDENCE/isolated_best_${run}.json" "${BEST_ENV[@]}"
  run_candidate "$REGION" "$BEST_CANDIDATE" \
    "$EVIDENCE/region_best_${run}.json" "${BEST_ENV[@]}"
done

PROFILE_SCRIPT="$PROFILE/harness/profile_indexer_region.py"
nsys profile --force-overwrite=true --trace=cuda,nvtx,osrt --sample=none \
  -o "$PROFILE/reports/nsys-stock" \
  "$PY" "$PROFILE_SCRIPT" --warmup 5 \
  --trace-output "$EVIDENCE/runtime_abi_trace_stock.json" \
  > "$PROFILE/analysis/nsys-stock-console.txt" 2>&1

env "${BEST_ENV[@]}" nsys profile --force-overwrite=true \
  --trace=cuda,nvtx,osrt --sample=none \
  -o "$PROFILE/reports/nsys-best" \
  "$PY" "$PROFILE_SCRIPT" --candidate "$BEST_CANDIDATE" --warmup 5 \
  --trace-output "$EVIDENCE/runtime_abi_trace_best.json" \
  > "$PROFILE/analysis/nsys-best-console.txt" 2>&1

for tag in stock best; do
  nsys stats --force-export=true --report cuda_gpu_kern_sum,cuda_gpu_trace,nvtx_gpu_proj_sum \
    --format csv --output "$PROFILE/analysis/nsys-$tag" \
    "$PROFILE/reports/nsys-$tag.nsys-rep" \
    > "$PROFILE/analysis/nsys-$tag-stats-console.txt" 2>&1
done

printf '%s\n' "$BEST_BACKEND" > "$EVIDENCE/best_backend.txt"
