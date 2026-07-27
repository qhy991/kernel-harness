#!/usr/bin/env bash
set -euo pipefail

HARNESS_ROOT=/home/qinhaiyan/glm52-goal-runs/16-indexer_score_decode/kernel-harness
SGLANG_TREE=/home/qinhaiyan/glm52-goal-runs/16-indexer_score_decode/sglang
EVIDENCE_ROOT="$HARNESS_ROOT/evidence/glm52_prod_16_indexer_score_decode"
TASK_PY="$HARNESS_ROOT/.venv/bin/python"

if [[ -z "${CUDA_VISIBLE_DEVICES:-}" || "${CUDA_VISIBLE_DEVICES}" == *,* ]]; then
  echo "run_containing_region_campaign.sh requires one wrapper-selected GPU" >&2
  exit 64
fi

RUN_STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
RUN_DIR="$EVIDENCE_ROOT/region_runs/$RUN_STAMP"
mkdir -p "$RUN_DIR"/{logs,results,contracts,nsys}

export SGLANG_ROOT="$SGLANG_TREE"
export PYTHONPATH="$SGLANG_TREE/python:$HARNESS_ROOT:${PYTHONPATH:-}"

FAILURES=0

run_logged() {
  local log_path="$1"
  shift
  set +e
  "$@" >"$log_path" 2>&1
  local command_rc=$?
  set -e
  if [[ $command_rc -ne 0 ]]; then
    printf '%s rc=%s\n' "$log_path" "$command_rc" >>"$RUN_DIR/failures.txt"
    FAILURES=1
  fi
  return 0
}

{
  printf 'run_stamp=%s\n' "$RUN_STAMP"
  printf 'cuda_visible_devices=%s\n' "$CUDA_VISIBLE_DEVICES"
  nvidia-smi -i "$CUDA_VISIBLE_DEVICES" \
    --query-gpu=index,uuid,name,pstate,temperature.gpu,clocks.sm,clocks.mem,power.draw \
    --format=csv
} >"$RUN_DIR/gpu_identity_start.txt"

git -C "$HARNESS_ROOT" rev-parse HEAD >"$RUN_DIR/kernel_harness_head.txt"
git -C "$HARNESS_ROOT" status --porcelain=v1 >"$RUN_DIR/kernel_harness_status.txt"
git -C "$SGLANG_TREE" rev-parse HEAD >"$RUN_DIR/sglang_head.txt"
git -C "$SGLANG_TREE" status --porcelain=v1 >"$RUN_DIR/sglang_status.txt"
sha256sum \
  "$HARNESS_ROOT/serving_native/runner.py" \
  "$HARNESS_ROOT/serving_native/workloads.py" \
  "$HARNESS_ROOT/serving_native/indexer_region.py" \
  "$HARNESS_ROOT/serving_native/candidates/reference.py" \
  "$HARNESS_ROOT/serving_native/candidates/indexer_region_cutedsl.py" \
  "$EVIDENCE_ROOT/region_profile_driver.py" >"$RUN_DIR/source_manifest.sha256"

run_logged "$RUN_DIR/logs/check_env.log" \
  "$TASK_PY" "$HARNESS_ROOT/testbench/bin/check_env.py"
run_logged "$RUN_DIR/logs/pip_freeze.log" "$TASK_PY" -m pip freeze

tasks=(
  indexer_complete_decode_m16
  indexer_complete_decode_m32
  indexer_dsa_decode_m16
  indexer_dsa_decode_m32
)
for task in "${tasks[@]}"; do
  for candidate_kind in identity cutedsl; do
    candidate_path="$HARNESS_ROOT/serving_native/candidates/reference.py"
    if [[ "$candidate_kind" == cutedsl ]]; then
      candidate_path="$HARNESS_ROOT/serving_native/candidates/indexer_region_cutedsl.py"
    fi
    for repetition in 01 02 03; do
      tag="${task}_${candidate_kind}_${repetition}"
      run_logged "$RUN_DIR/logs/$tag.log" \
        "$TASK_PY" "$HARNESS_ROOT/serving_native/runner.py" \
        --task "$task" \
        --candidate "$candidate_path" \
        --execution-mode cuda_graph \
        --warmup 5 \
        --repeat 60 \
        --output "$RUN_DIR/results/$tag.json"
    done
  done
done

for task in "${tasks[@]}"; do
  for backend in deepgemm cutedsl; do
    tag="${task}_${backend}"
    run_logged "$RUN_DIR/logs/nsys_$tag.log" \
      nsys profile \
      --trace=cuda,nvtx,osrt \
      --sample=none \
      --cpuctxsw=none \
      --capture-range=cudaProfilerApi \
      --capture-range-end=stop \
      --force-overwrite=true \
      --output="$RUN_DIR/nsys/$tag" \
      "$TASK_PY" "$EVIDENCE_ROOT/region_profile_driver.py" \
      --task "$task" \
      --backend "$backend" \
      --contract-output "$RUN_DIR/contracts/$tag.json"
    run_logged "$RUN_DIR/logs/nsys_stats_$tag.log" \
      nsys stats \
      --report cuda_gpu_kern_sum,cuda_api_sum,nvtx_gpu_proj_sum \
      --format csv \
      --output "$RUN_DIR/nsys/stats_$tag" \
      "$RUN_DIR/nsys/$tag.nsys-rep"
  done
done

run_logged "$RUN_DIR/logs/paired_summary.log" \
  "$TASK_PY" "$EVIDENCE_ROOT/summarize_regions.py" --run-dir "$RUN_DIR"

{
  printf 'run_dir=%s\n' "$RUN_DIR"
  printf 'failures=%s\n' "$FAILURES"
} >"$RUN_DIR/status.txt"
printf '%s\n' "$RUN_DIR" >"$EVIDENCE_ROOT/latest_region_run.txt"

nvidia-smi -i "$CUDA_VISIBLE_DEVICES" \
  --query-gpu=index,uuid,name,pstate,temperature.gpu,clocks.sm,clocks.mem,power.draw \
  --format=csv >"$RUN_DIR/gpu_identity_end.txt"

find "$RUN_DIR" -type f ! -name artifact_manifest.sha256 -print0 |
  sort -z |
  xargs -0 sha256sum >"$RUN_DIR/artifact_manifest.sha256"

if [[ $FAILURES -ne 0 ]]; then
  exit 1
fi
