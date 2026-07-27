#!/usr/bin/env bash
# Invoke this script exactly once through with_flexible_gpu.sh. It keeps all
# alternating series and profiler collections on one physical B200.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
EVIDENCE_ROOT="${ROOT}/evidence/glm52_prod_17_indexer_score_prefill"
SGLANG_ROOT="${SGLANG_ROOT:-/home/qinhaiyan/glm52-goal-runs/17-indexer_score_prefill/sglang}"
PY="${KERNEL_HARNESS_PYTHON:-${ROOT}/.venv/bin/python}"
RUN_ID="${1:-$(date -u +%Y%m%dT%H%M%SZ)}"
CAMPAIGN_DIR="${EVIDENCE_ROOT}/campaigns/${RUN_ID}"
PROFILE_DIR="${ROOT}/profile/indexer-score-prefill-${RUN_ID}"
CANDIDATE="${ROOT}/serving_native/candidates/indexer_score_balanced_chunks.py"
DRIVER="${EVIDENCE_ROOT}/profile_driver.py"
STATUS="${CAMPAIGN_DIR}/status.tsv"

if [[ -e "${CAMPAIGN_DIR}" || -e "${PROFILE_DIR}" ]]; then
  echo "refusing to overwrite campaign ${RUN_ID}" >&2
  exit 3
fi
mkdir -p "${CAMPAIGN_DIR}/logs" "${CAMPAIGN_DIR}/paired"
mkdir -p "${PROFILE_DIR}/harness" "${PROFILE_DIR}/reports" "${PROFILE_DIR}/analysis"
printf 'step\texit_code\tutc_finished\n' >"${STATUS}"
printf '%s\n' "${PROFILE_DIR}" >"${CAMPAIGN_DIR}/profile_dir.txt"

export SGLANG_ROOT
export PYTHONPATH="${SGLANG_ROOT}/python:${ROOT}:${PYTHONPATH:-}"
export SGLANG_GLM52_OPT=0

run_logged() {
  local step="$1"
  shift
  local log="${CAMPAIGN_DIR}/logs/${step}.log"
  mkdir -p "$(dirname "${log}")"
  echo "[$(date -u +%FT%TZ)] START ${step}"
  set +e
  "$@" > >(tee "${log}") 2>&1
  local code=$?
  set -e
  printf '%s\t%s\t%s\n' "${step}" "${code}" "$(date -u +%FT%TZ)" >>"${STATUS}"
  echo "[$(date -u +%FT%TZ)] END ${step} exit=${code}"
  return 0
}

{
  echo "run_id=${RUN_ID}"
  echo "started_utc=$(date -u +%FT%TZ)"
  echo "hostname=$(hostname)"
  echo "cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-unset}"
  echo "kernel_harness_root=${ROOT}"
  echo "sglang_root=${SGLANG_ROOT}"
  echo "python=${PY}"
  echo "kernel_harness_head=$(git -C "${ROOT}" rev-parse HEAD)"
  echo "sglang_head=$(git -C "${SGLANG_ROOT}" rev-parse HEAD)"
  echo "candidate_sha256=$(sha256sum "${CANDIDATE}" | awk '{print $1}')"
} >"${CAMPAIGN_DIR}/environment.txt"

git -C "${ROOT}" status --short >"${CAMPAIGN_DIR}/kernel_harness_status.txt"
git -C "${SGLANG_ROOT}" status --short >"${CAMPAIGN_DIR}/sglang_status.txt"
git -C "${ROOT}" diff --binary >"${CAMPAIGN_DIR}/kernel_harness_source.diff"
git -C "${SGLANG_ROOT}" diff --binary >"${CAMPAIGN_DIR}/sglang_source.diff"
cp "${DRIVER}" "${PROFILE_DIR}/harness/profile_driver.py"
cp \
  /home/qinhaiyan/kernel-design-agents/skills/ncu-report-skill/helpers/analyze_reports.py \
  /home/qinhaiyan/kernel-design-agents/skills/ncu-report-skill/helpers/ncu_utils.py \
  /home/qinhaiyan/kernel-design-agents/skills/ncu-report-skill/helpers/plot_timeline.py \
  /home/qinhaiyan/kernel-design-agents/skills/ncu-report-skill/helpers/extract_stall_hotspots.py \
  "${PROFILE_DIR}/analysis/"

run_logged env_check "${PY}" "${ROOT}/testbench/bin/check_env.py"
run_logged gpu_identity nvidia-smi \
  --query-gpu=index,name,uuid,pci.bus_id,pstate,clocks.current.sm,clocks.current.memory,power.draw,temperature.gpu,memory.total,memory.free \
  --format=csv
run_logged gpu_detail nvidia-smi -q
run_logged tool_versions bash -c \
  'nvcc --version; ncu --version; nsys --version'
run_logged package_paths "${PY}" -c '
import importlib, json, torch
mods = {}
for name in ("torch", "deep_gemm", "sgl_kernel", "flashinfer", "triton", "sglang"):
    mod = importlib.import_module(name)
    mods[name] = {
        "file": getattr(mod, "__file__", None),
        "version": getattr(mod, "__version__", None),
    }
print(json.dumps({
    "device": torch.cuda.get_device_name(0),
    "capability": torch.cuda.get_device_capability(0),
    "properties": str(torch.cuda.get_device_properties(0)),
    "modules": mods,
}, indent=2, sort_keys=True))
'
run_logged correctness_diagnostic \
  "${PY}" "${EVIDENCE_ROOT}/diagnose_correctness.py"

# Three independent paired series per exact scope. Each series internally
# alternates A/B order; raw per-sample latencies and ratios are persisted.
SCORE_TASKS=(
  indexer_score_prefill_m4096
  indexer_score_prefill_m4096_c256
  indexer_score_prefill_m4096_mixed
)
REGION_TASKS=(
  indexer_complete_prefill_m4096
  indexer_complete_prefill_m4096_c256
  indexer_graph_split_prefill_m4096
  indexer_graph_split_prefill_m4096_c256
  indexer_dsa_prefill_m4096
  indexer_dsa_prefill_m4096_c256
)

for task in "${SCORE_TASKS[@]}"; do
  for series in 1 2 3; do
    output="${CAMPAIGN_DIR}/paired/${task}.series${series}.json"
    run_logged "paired/${task}.series${series}" \
      "${ROOT}/serving_native/run.sh" "${task}" \
      --candidate "${CANDIDATE}" --warmup 5 --repeat 40 --output "${output}"
  done
done

for task in "${REGION_TASKS[@]}"; do
  for series in 1 2 3; do
    output="${CAMPAIGN_DIR}/paired/${task}.series${series}.json"
    run_logged "paired/${task}.series${series}" \
      "${ROOT}/serving_native/run.sh" "${task}" \
      --candidate "${CANDIDATE}" --warmup 5 --repeat 24 --output "${output}"
  done
done

run_logged paired_summary \
  "${PY}" "${EVIDENCE_ROOT}/summarize_campaign.py" "${CAMPAIGN_DIR}"

profile_nsys() {
  local tag="$1"
  local task="$2"
  local variant="$3"
  local base="${PROFILE_DIR}/reports/nsys_${tag}"
  run_logged "profile/nsys_${tag}" \
    nsys profile \
    --trace=cuda,nvtx,osrt \
    --sample=none \
    --cpuctxsw=none \
    --capture-range=cudaProfilerApi \
    --capture-range-end=stop \
    --force-overwrite=true \
    --output="${base}" \
    "${PY}" "${DRIVER}" --task "${task}" --variant "${variant}" --warmup 5
  if [[ -f "${base}.nsys-rep" ]]; then
    run_logged "profile/nsys_stats_${tag}" \
      nsys stats \
      --report cuda_gpu_kern_sum,cuda_gpu_kern_gb_sum,cuda_kern_exec_sum,nvtx_gpu_proj_sum \
      --format csv \
      "${base}.nsys-rep"
  fi
}

profile_nsys score_main_stock indexer_score_prefill_m4096 stock
profile_nsys score_c256_stock indexer_score_prefill_m4096_c256 stock
profile_nsys score_c256_balanced indexer_score_prefill_m4096_c256 balanced
profile_nsys score_mixed_stock indexer_score_prefill_m4096_mixed stock
profile_nsys score_mixed_balanced indexer_score_prefill_m4096_mixed balanced
profile_nsys complete_c256_stock indexer_complete_prefill_m4096_c256 stock
profile_nsys complete_c256_balanced indexer_complete_prefill_m4096_c256 balanced
profile_nsys graph_split_c256_stock indexer_graph_split_prefill_m4096_c256 stock
profile_nsys graph_split_c256_balanced indexer_graph_split_prefill_m4096_c256 balanced
profile_nsys dsa_c256_stock indexer_dsa_prefill_m4096_c256 stock
profile_nsys dsa_c256_balanced indexer_dsa_prefill_m4096_c256 balanced

profile_ncu() {
  local prefix="$1"
  local tag="$2"
  local task="$3"
  local variant="$4"
  local set_name="$5"
  local kernel_regex="$6"
  local launch_skip="$7"
  local report_base="${PROFILE_DIR}/reports/${prefix}_${tag}"
  run_logged "profile/${prefix}_${tag}" \
    ncu \
    --force-overwrite \
    --target-processes all \
    --profile-from-start off \
    --kernel-name-base function \
    --kernel-name "regex:${kernel_regex}" \
    --launch-skip "${launch_skip}" \
    --launch-count 1 \
    --set "${set_name}" \
    --import-source yes \
    --source-folders "${ROOT},${SGLANG_ROOT}" \
    --export "${report_base}" \
    "${PY}" "${DRIVER}" --task "${task}" --variant "${variant}" --warmup 5
}

# Full six-dimension reports for the dominant score kernel, the tail shape,
# the schedule candidate, K gather, and the top-k handoff.
profile_ncu full score_main_mqa indexer_score_prefill_m4096 stock full '.*mqa_logits.*' 0
profile_ncu full score_c256_mqa_head indexer_score_prefill_m4096_c256 stock full '.*mqa_logits.*' 0
profile_ncu full score_c256_mqa_tail indexer_score_prefill_m4096_c256 stock full '.*mqa_logits.*' 1
profile_ncu full score_c256_balanced_mqa indexer_score_prefill_m4096_c256 balanced full '.*mqa_logits.*' 0
profile_ncu full score_main_gather indexer_score_prefill_m4096 stock full '.*get_k_and_s.*' 0
profile_ncu full score_c256_topk_head indexer_score_prefill_m4096_c256 stock full '.*topk_transform_prefill_kernel.*' 0
profile_ncu full score_c256_balanced_topk indexer_score_prefill_m4096_c256 balanced full '.*topk_transform_prefill_kernel.*' 0

# Nsight Compute 2026.1 has no named "source" set. The detailed set includes
# SourceCounters; --import-source and imported cubins retain source/SASS views.
profile_ncu source score_c256_mqa_head indexer_score_prefill_m4096_c256 stock detailed '.*mqa_logits.*' 0
profile_ncu source score_c256_mqa_tail indexer_score_prefill_m4096_c256 stock detailed '.*mqa_logits.*' 1
profile_ncu source score_c256_balanced_mqa indexer_score_prefill_m4096_c256 balanced detailed '.*mqa_logits.*' 0

NCU_PYTHON=/opt/nvidia/nsight-compute/2026.1.1/extras/python
FULL_ARGS=()
SOURCE_ARGS=()
for report in "${PROFILE_DIR}"/reports/full_*.ncu-rep; do
  [[ -f "${report}" ]] || continue
  tag="$(basename "${report}" .ncu-rep)"
  tag="${tag#full_}"
  FULL_ARGS+=(--report "${report}" --tag "${tag}")
done
for report in "${PROFILE_DIR}"/reports/source_*.ncu-rep; do
  [[ -f "${report}" ]] || continue
  tag="$(basename "${report}" .ncu-rep)"
  tag="${tag#source_}"
  SOURCE_ARGS+=(--report "${report}" --tag "${tag}")
  run_logged "profile/source_dump_${tag}" \
    ncu --import "${report}" --page source --print-source sass
done

if ((${#FULL_ARGS[@]})); then
  run_logged profile/analyze_full \
    env "PYTHONPATH=${NCU_PYTHON}" "${PY}" \
    "${PROFILE_DIR}/analysis/analyze_reports.py" \
    --run-dir "${PROFILE_DIR}" "${FULL_ARGS[@]}"
  run_logged profile/plot_timeline \
    env "PYTHONPATH=${NCU_PYTHON}" "${PY}" \
    "${PROFILE_DIR}/analysis/plot_timeline.py" \
    --run-dir "${PROFILE_DIR}" "${FULL_ARGS[@]}"
fi
if ((${#SOURCE_ARGS[@]})); then
  run_logged profile/extract_stalls \
    env "PYTHONPATH=${NCU_PYTHON}" "${PY}" \
    "${PROFILE_DIR}/analysis/extract_stall_hotspots.py" \
    --run-dir "${PROFILE_DIR}" "${SOURCE_ARGS[@]}"
fi

find "${PROFILE_DIR}" -type f -print0 | sort -z | xargs -0 sha256sum \
  >"${CAMPAIGN_DIR}/profile_sha256.txt"
printf 'finished_utc=%s\n' "$(date -u +%FT%TZ)" >>"${CAMPAIGN_DIR}/environment.txt"
find "${CAMPAIGN_DIR}" -type f ! -name campaign_sha256.txt -print0 \
  | sort -z | xargs -0 sha256sum >"${CAMPAIGN_DIR}/campaign_sha256.txt"
echo "CAMPAIGN_DIR=${CAMPAIGN_DIR}"
echo "PROFILE_DIR=${PROFILE_DIR}"
