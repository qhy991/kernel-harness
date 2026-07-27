#!/usr/bin/env bash
# Invoke once through with_flexible_gpu.sh so paired mixed-bucket measurements
# and the matching profiler collection stay on one physical B200.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
EVIDENCE_ROOT="${ROOT}/evidence/glm52_prod_17_indexer_score_prefill"
SGLANG_ROOT="${SGLANG_ROOT:-/home/qinhaiyan/glm52-goal-runs/17-indexer_score_prefill/sglang}"
PY="${KERNEL_HARNESS_PYTHON:-${ROOT}/.venv/bin/python}"
RUN_ID="${1:-$(date -u +%Y%m%dT%H%M%SZ)}"
CAMPAIGN_DIR="${EVIDENCE_ROOT}/campaigns/${RUN_ID}-mixed-confirmation"
PROFILE_DIR="${ROOT}/profile/indexer-score-prefill-${RUN_ID}-mixed-confirmation"
CANDIDATE="${ROOT}/serving_native/candidates/indexer_score_balanced_mixed_bucket.py"
DRIVER="${EVIDENCE_ROOT}/profile_driver.py"
STATUS="${CAMPAIGN_DIR}/status.tsv"

if [[ -e "${CAMPAIGN_DIR}" || -e "${PROFILE_DIR}" ]]; then
  echo "refusing to overwrite campaign ${RUN_ID}" >&2
  exit 3
fi
mkdir -p "${CAMPAIGN_DIR}/logs" "${CAMPAIGN_DIR}/paired"
mkdir -p "${PROFILE_DIR}/harness" "${PROFILE_DIR}/reports"
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
  echo "campaign_kind=mixed_bucket_confirmation"
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

run_logged gpu_identity nvidia-smi \
  --query-gpu=index,name,uuid,pci.bus_id,pstate,clocks.current.sm,clocks.current.memory,power.draw,temperature.gpu,memory.total,memory.free \
  --format=csv

MIXED_TASKS=(
  indexer_score_prefill_m4096_mixed
  indexer_complete_prefill_m4096_mixed
  indexer_graph_split_prefill_m4096_mixed
  indexer_dsa_prefill_m4096_mixed
)
for task in "${MIXED_TASKS[@]}"; do
  repeat=24
  if [[ "${task}" == indexer_score_prefill_m4096_mixed ]]; then
    repeat=40
  fi
  for series in 1 2 3; do
    output="${CAMPAIGN_DIR}/paired/${task}.series${series}.json"
    run_logged "paired/${task}.series${series}" \
      "${ROOT}/serving_native/run.sh" "${task}" \
      --candidate "${CANDIDATE}" --warmup 5 --repeat "${repeat}" \
      --output "${output}"
  done
done

# These are fallback diagnostics, not candidate performance claims. The
# focused candidate must invoke the stock method for both non-mixed buckets.
for task in indexer_score_prefill_m4096 indexer_score_prefill_m4096_c256; do
  output="${CAMPAIGN_DIR}/paired/${task}.fallback.json"
  run_logged "fallback/${task}" \
    "${ROOT}/serving_native/run.sh" "${task}" \
    --candidate "${CANDIDATE}" --warmup 5 --repeat 16 --output "${output}"
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
      --report cuda_gpu_kern_sum,cuda_gpu_kern_gb_sum,nvtx_gpu_proj_sum \
      --format csv \
      "${base}.nsys-rep"
  fi
}

for scope in score complete graph_split dsa; do
  case "${scope}" in
    score) task=indexer_score_prefill_m4096_mixed ;;
    complete) task=indexer_complete_prefill_m4096_mixed ;;
    graph_split) task=indexer_graph_split_prefill_m4096_mixed ;;
    dsa) task=indexer_dsa_prefill_m4096_mixed ;;
  esac
  profile_nsys "mixed_${scope}_stock" "${task}" stock
  profile_nsys "mixed_${scope}_balanced" "${task}" balanced
done

find "${PROFILE_DIR}" -type f -print0 | sort -z | xargs -0 sha256sum \
  >"${CAMPAIGN_DIR}/profile_sha256.txt"
printf 'finished_utc=%s\n' "$(date -u +%FT%TZ)" \
  >>"${CAMPAIGN_DIR}/environment.txt"
find "${CAMPAIGN_DIR}" -type f ! -name campaign_sha256.txt -print0 \
  | sort -z | xargs -0 sha256sum >"${CAMPAIGN_DIR}/campaign_sha256.txt"
echo "CAMPAIGN_DIR=${CAMPAIGN_DIR}"
echo "PROFILE_DIR=${PROFILE_DIR}"
