#!/usr/bin/env bash
set -euo pipefail

HARNESS_ROOT=/home/qinhaiyan/glm52-goal-runs/16-indexer_score_decode/kernel-harness
SGLANG_TREE=/home/qinhaiyan/glm52-goal-runs/16-indexer_score_decode/sglang
EVIDENCE_ROOT="$HARNESS_ROOT/evidence/glm52_prod_16_indexer_score_decode"
TASK_PY="$HARNESS_ROOT/.venv/bin/python"
NCU_HELPERS=/home/qinhaiyan/kernel-design-agents/skills/ncu-report-skill/helpers
NCU_PYTHON=/opt/nvidia/nsight-compute/2026.1.1/extras/python

if [[ -z "${CUDA_VISIBLE_DEVICES:-}" || "${CUDA_VISIBLE_DEVICES}" == *,* ]]; then
  echo "run_single_gpu_campaign.sh requires one wrapper-selected GPU" >&2
  exit 64
fi

RUN_STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
RUN_DIR="$EVIDENCE_ROOT/runs/$RUN_STAMP"
PROFILE_RUN="$HARNESS_ROOT/profile/indexer-score-decode-$RUN_STAMP"
mkdir -p "$RUN_DIR"/{logs,results,contracts,nsys} \
  "$PROFILE_RUN"/{harness,reports,analysis}

export SGLANG_ROOT="$SGLANG_TREE"
export PYTHONPATH="$SGLANG_TREE/python:$HARNESS_ROOT:${PYTHONPATH:-}"

cp "$EVIDENCE_ROOT/profile_driver.py" "$PROFILE_RUN/harness/profile_driver.py"
cp "$NCU_HELPERS/analyze_reports.py" "$PROFILE_RUN/analysis/analyze_reports.py"
cp "$NCU_HELPERS/extract_stall_hotspots.py" \
  "$PROFILE_RUN/analysis/extract_stall_hotspots.py"
cp "$NCU_HELPERS/plot_timeline.py" "$PROFILE_RUN/analysis/plot_timeline.py"
cp "$NCU_HELPERS/ncu_utils.py" "$PROFILE_RUN/analysis/ncu_utils.py"

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
  "$HARNESS_ROOT/serving_native/candidates/reference.py" \
  "$HARNESS_ROOT/serving_native/candidates/indexer_score_cutedsl.py" \
  "$PROFILE_RUN/harness/profile_driver.py" >"$RUN_DIR/source_manifest.sha256"

run_logged "$RUN_DIR/logs/check_env.log" \
  "$TASK_PY" "$HARNESS_ROOT/testbench/bin/check_env.py"
run_logged "$RUN_DIR/logs/pip_freeze.log" "$TASK_PY" -m pip freeze

for execution_mode in eager cuda_graph; do
  mode_tag=eager
  if [[ "$execution_mode" == cuda_graph ]]; then
    mode_tag=graph
  fi
  for batch in 16 32; do
    for candidate_kind in identity cutedsl; do
      candidate_path="$HARNESS_ROOT/serving_native/candidates/reference.py"
      if [[ "$candidate_kind" == cutedsl ]]; then
        candidate_path="$HARNESS_ROOT/serving_native/candidates/indexer_score_cutedsl.py"
      fi
      for repetition in 01 02 03; do
        tag="${mode_tag}_m${batch}_${candidate_kind}_${repetition}"
        run_logged "$RUN_DIR/logs/$tag.log" \
          "$TASK_PY" "$HARNESS_ROOT/serving_native/runner.py" \
          --task "indexer_score_decode_m${batch}" \
          --candidate "$candidate_path" \
          --execution-mode "$execution_mode" \
          --warmup 5 \
          --repeat 60 \
          --output "$RUN_DIR/results/$tag.json"
      done
    done
  done
done

for backend in deepgemm cutedsl; do
  for batch in 16 32; do
    tag="${backend}_m${batch}"
    range_name="indexer_score_topk_${backend}_m${batch}"
    run_logged "$RUN_DIR/logs/nsys_$tag.log" \
      nsys profile \
      --trace=cuda,nvtx,osrt \
      --sample=none \
      --cpuctxsw=none \
      --capture-range=cudaProfilerApi \
      --capture-range-end=stop \
      --force-overwrite=true \
      --output="$RUN_DIR/nsys/$tag" \
      "$TASK_PY" "$PROFILE_RUN/harness/profile_driver.py" \
      --batch "$batch" \
      --backend "$backend" \
      --contract-output "$RUN_DIR/contracts/$tag.json"

    run_logged "$RUN_DIR/logs/nsys_stats_$tag.log" \
      nsys stats \
      --report cuda_gpu_kern_sum,cuda_api_sum,nvtx_gpu_proj_sum \
      --format csv \
      --output "$RUN_DIR/nsys/stats_$tag" \
      "$RUN_DIR/nsys/$tag.nsys-rep"

    run_logged "$RUN_DIR/logs/ncu_full_$tag.log" \
      ncu \
      --set full \
      --section PmSampling \
      --section PmSampling_WarpStates \
      --nvtx \
      --nvtx-include "$range_name/" \
      --launch-count 1 \
      --force-overwrite \
      -o "$PROFILE_RUN/reports/full_$tag" \
      "$TASK_PY" "$PROFILE_RUN/harness/profile_driver.py" \
      --batch "$batch" \
      --backend "$backend"

    run_logged "$RUN_DIR/logs/ncu_source_$tag.log" \
      ncu \
      --set source \
      --section SourceCounters \
      --nvtx \
      --nvtx-include "$range_name/" \
      --launch-count 1 \
      --force-overwrite \
      -o "$PROFILE_RUN/reports/source_$tag" \
      "$TASK_PY" "$PROFILE_RUN/harness/profile_driver.py" \
      --batch "$batch" \
      --backend "$backend"
  done
done

for batch in 16 32; do
  tag="topk_m${batch}"
  range_name="indexer_score_topk_deepgemm_m${batch}"
  run_logged "$RUN_DIR/logs/ncu_full_$tag.log" \
    ncu \
    --set full \
    --section PmSampling \
    --section PmSampling_WarpStates \
    --nvtx \
    --nvtx-include "$range_name/" \
    --kernel-name 'regex:topk_(small_batch|main|persistent_cluster)_kernel' \
    --launch-count 1 \
    --force-overwrite \
    -o "$PROFILE_RUN/reports/full_$tag" \
    "$TASK_PY" "$PROFILE_RUN/harness/profile_driver.py" \
    --batch "$batch" \
    --backend deepgemm

  run_logged "$RUN_DIR/logs/ncu_source_$tag.log" \
    ncu \
    --set source \
    --section SourceCounters \
    --nvtx \
    --nvtx-include "$range_name/" \
    --kernel-name 'regex:topk_(small_batch|main|persistent_cluster)_kernel' \
    --launch-count 1 \
    --force-overwrite \
    -o "$PROFILE_RUN/reports/source_$tag" \
    "$TASK_PY" "$PROFILE_RUN/harness/profile_driver.py" \
    --batch "$batch" \
    --backend deepgemm
done

run_logged "$RUN_DIR/logs/paired_summary.log" \
  "$TASK_PY" "$EVIDENCE_ROOT/summarize_campaign.py" --run-dir "$RUN_DIR"

analysis_command=(
  "$TASK_PY" "$PROFILE_RUN/analysis/analyze_reports.py"
  --run-dir "$PROFILE_RUN"
)
timeline_command=(
  "$TASK_PY" "$PROFILE_RUN/analysis/plot_timeline.py"
  --run-dir "$PROFILE_RUN"
)
source_command=(
  "$TASK_PY" "$PROFILE_RUN/analysis/extract_stall_hotspots.py"
  --run-dir "$PROFILE_RUN"
)
for tag in deepgemm_m16 deepgemm_m32 cutedsl_m16 cutedsl_m32 topk_m16 topk_m32; do
  analysis_command+=(--report "$PROFILE_RUN/reports/full_$tag.ncu-rep" --tag "$tag")
  timeline_command+=(--report "$PROFILE_RUN/reports/full_$tag.ncu-rep" --tag "$tag")
  source_command+=(--report "$PROFILE_RUN/reports/source_$tag.ncu-rep" --tag "$tag")
  run_logged "$PROFILE_RUN/analysis/details_$tag.txt" \
    ncu --import "$PROFILE_RUN/reports/full_$tag.ncu-rep" --page details
done
run_logged "$RUN_DIR/logs/analyze_ncu.log" \
  env PYTHONPATH="$NCU_PYTHON:$PROFILE_RUN/analysis" "${analysis_command[@]}"
run_logged "$RUN_DIR/logs/timeline_ncu.log" \
  env PYTHONPATH="$NCU_PYTHON:$PROFILE_RUN/analysis" "${timeline_command[@]}"
run_logged "$RUN_DIR/logs/source_ncu.log" \
  env PYTHONPATH="$NCU_PYTHON:$PROFILE_RUN/analysis" "${source_command[@]}"

{
  printf 'run_dir=%s\n' "$RUN_DIR"
  printf 'profile_run=%s\n' "$PROFILE_RUN"
  printf 'failures=%s\n' "$FAILURES"
} >"$RUN_DIR/status.txt"
printf '%s\n' "$RUN_DIR" >"$EVIDENCE_ROOT/latest_single_gpu_run.txt"
printf '%s\n' "$PROFILE_RUN" >"$EVIDENCE_ROOT/latest_profile_run.txt"

{
  nvidia-smi -i "$CUDA_VISIBLE_DEVICES" \
    --query-gpu=index,uuid,name,pstate,temperature.gpu,clocks.sm,clocks.mem,power.draw \
    --format=csv
} >"$RUN_DIR/gpu_identity_end.txt"

find "$RUN_DIR" "$PROFILE_RUN" -type f \
  ! -name artifact_manifest.sha256 -print0 |
  sort -z |
  xargs -0 sha256sum >"$RUN_DIR/artifact_manifest.sha256"

if [[ $FAILURES -ne 0 ]]; then
  exit 1
fi
