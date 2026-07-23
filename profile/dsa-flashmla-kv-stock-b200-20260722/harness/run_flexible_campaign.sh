#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -ne 1 || ! "$1" =~ ^flex-[0-9]{8}T[0-9]{6}Z[a-z]?$ ]]; then
  echo "usage: run_flexible_campaign.sh flex-YYYYMMDDTHHMMSSZ" >&2
  exit 64
fi
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
BASE="$KH/profile/dsa-flashmla-kv-stock-b200-20260722"
HARNESS="$BASE/harness"
CAMPAIGN_ID=$1
OUT="$BASE/campaigns/$CAMPAIGN_ID"
PY="$KH/.venv/bin/python"
CANDIDATE="$KH/serving_native/candidates/flashmla_goal22_overlay.py"
STOCK_OVERLAY="$FLASHMLA/build-artifacts/stock-pybind-tensor/overlay"
STOCK_MANIFEST="$BASE/analysis/build_stock_pybind_tensor.json"
CANDIDATE_OVERLAY="$FLASHMLA/build-artifacts/combine32-m16-tensor/overlay"
CANDIDATE_MANIFEST="$BASE/analysis/build_combine32_m16_tensor.json"

mkdir -p "$OUT/analysis" "$OUT/logs" "$OUT/reports"
if [[ -e "$OUT/campaign.started" || -e "$OUT/campaign.complete" ]]; then
  echo "refusing to reuse campaign directory: $OUT" >&2
  exit 65
fi
date -u +%Y-%m-%dT%H:%M:%SZ > "$OUT/campaign.started"

export SGLANG_ROOT="$SG"
export SGLANG_GLM52_OPT=0
export KERNEL_HARNESS_PYTHON="$PY"
export PYTHONPATH="$SG/python:$KH:${PYTHONPATH:-}"
export GOAL22_CAMPAIGN_ID="$CAMPAIGN_ID"
export GOAL22_PHYSICAL_GPU="$CUDA_VISIBLE_DEVICES"
export GOAL22_GPU_UUID
GOAL22_GPU_UUID="$(nvidia-smi -i "$CUDA_VISIBLE_DEVICES" --query-gpu=uuid --format=csv,noheader | head -1 | tr -d '[:space:]')"
export GOAL22_STOCK_OVERLAY="$STOCK_OVERLAY"
export GOAL22_STOCK_MANIFEST="$STOCK_MANIFEST"
export GOAL22_CANDIDATE_OVERLAY="$CANDIDATE_OVERLAY"
export GOAL22_CANDIDATE_MANIFEST="$CANDIDATE_MANIFEST"

run_logged() {
  local label=$1
  shift
  echo "==> $label"
  "$@" > "$OUT/logs/$label.txt" 2>&1
}

record_device() {
  local stage=$1
  run_logged "device_$stage" \
    "$PY" "$HARNESS/record_gpu_environment.py" \
      --stage "$stage" \
      --output "$OUT/analysis/device_$stage.json"
}

run_paired() {
  local mode=$1
  local variant=$2
  local bucket=$3
  local session=$4
  local overlay manifest prefix task
  if [[ "$variant" == control ]]; then
    overlay=$STOCK_OVERLAY
    manifest=$STOCK_MANIFEST
  else
    overlay=$CANDIDATE_OVERLAY
    manifest=$CANDIDATE_MANIFEST
  fi
  task="dsa_flashmla_kv_decode_$bucket"
  if [[ "$mode" == eager ]]; then
    prefix=$([[ "$variant" == control ]] && echo paired_control || echo paired_combine32)
    run_logged "${prefix}_${bucket}_r${session}" \
      env GOAL22_FLASHMLA_OVERLAY="$overlay" GOAL22_FLASHMLA_MANIFEST="$manifest" \
      "$KH/serving_native/run.sh" "$task" \
        --candidate "$CANDIDATE" --warmup 100 --repeat 100 \
        --output "$OUT/analysis/${prefix}_${bucket}_r${session}.json"
  else
    prefix=$([[ "$variant" == control ]] && echo graph_control || echo graph_combine32)
    run_logged "${prefix}_${bucket}_r${session}" \
      env GOAL22_FLASHMLA_OVERLAY="$overlay" GOAL22_FLASHMLA_MANIFEST="$manifest" \
      "$PY" "$HARNESS/compare_cuda_graph.py" \
        --task "$task" --candidate "$CANDIDATE" \
        --warmup 10 --repeat 100 \
        --output "$OUT/analysis/${prefix}_${bucket}_r${session}.json"
  fi
}

run_nsys() {
  local label=$1
  local task=$2
  local overlay=$3
  local manifest=$4
  local candidate_args=()
  if [[ "$label" == combine32_m16 ]]; then
    candidate_args=(--candidate "$CANDIDATE")
  fi
  run_logged "nsys_$label" \
    env GOAL22_FLASHMLA_OVERLAY="$overlay" GOAL22_FLASHMLA_MANIFEST="$manifest" \
    nsys profile --trace=cuda,nvtx --sample=none --cpuctxsw=none \
      --force-overwrite=false --output "$OUT/reports/nsys_$label" \
      "$PY" "$HARNESS/profile_driver.py" \
        --task "$task" "${candidate_args[@]}" --warmup 3 \
        --output "$OUT/analysis/nsys_${label}_runtime.json"
}

record_device start

run_logged check_env \
  "$PY" "$KH/testbench/bin/check_env.py"
run_logged verify_harness \
  python3 "$KH/testbench/bin/verify_harness.py"

echo "==> exact SGLang production-ABI tests"
(
  cd "$SG"
  "$PY" test/registered/attention/unittests/dsa/test_dsa.py \
    TestDSAAttentionBackendCorrectness.test_production_flashmla_kv_decode_cases \
    TestDSAAttentionBackendCorrectness.test_production_flashmla_kv_cuda_graph_metadata_lifecycle_cases
) > "$OUT/logs/sglang_exact_tests.txt" 2>&1

for bucket in m16 m32; do
  run_logged "runtime_stock_$bucket" \
    "$PY" "$HARNESS/validate_runtime.py" \
      --task "dsa_flashmla_kv_decode_$bucket" \
      --output "$OUT/analysis/runtime_stock_$bucket.json"
done

for session in 1 2 3; do
  if [[ "$session" == 2 ]]; then
    variants=(candidate control)
    buckets=(m32 m16)
  else
    variants=(control candidate)
    buckets=(m16 m32)
  fi
  for variant in "${variants[@]}"; do
    for bucket in "${buckets[@]}"; do
      run_paired eager "$variant" "$bucket" "$session"
      run_paired cuda_graph "$variant" "$bucket" "$session"
    done
  done
done

for session in 1 2 3; do
  buckets=(m16 m32)
  [[ "$session" == 2 ]] && buckets=(m32 m16)
  for bucket in "${buckets[@]}"; do
    run_logged "baseline_stock_${bucket}_r${session}" \
      "$KH/serving_native/run.sh" "dsa_flashmla_kv_decode_$bucket" \
        --warmup 5 --repeat 50 \
        --output "$OUT/analysis/baseline_stock_${bucket}_r${session}.json"
  done
done
record_device after_paired

run_nsys stock_m16 dsa_flashmla_kv_decode_m16 "$STOCK_OVERLAY" "$STOCK_MANIFEST"
run_nsys stock_m32 dsa_flashmla_kv_decode_m32 "$STOCK_OVERLAY" "$STOCK_MANIFEST"
run_nsys combine32_m16 dsa_flashmla_kv_decode_m16 "$CANDIDATE_OVERLAY" "$CANDIDATE_MANIFEST"
record_device after_nsys

run_logged ncu_collection \
  "$HARNESS/collect_ncu.sh" --output-root "$OUT"
record_device end

date -u +%Y-%m-%dT%H:%M:%SZ > "$OUT/campaign.complete"
echo "campaign complete: $OUT"
