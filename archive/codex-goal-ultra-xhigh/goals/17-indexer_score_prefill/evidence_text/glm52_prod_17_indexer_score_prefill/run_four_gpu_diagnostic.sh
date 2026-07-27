#!/usr/bin/env bash
# Invoke only through with_all_gpus_lock.sh. This is a TP4/DP4 diagnostic,
# never a substitute for the official eight-rank production acceptance gate.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
EVIDENCE_ROOT="${ROOT}/evidence/glm52_prod_17_indexer_score_prefill"
SGLANG_ROOT="${SGLANG_ROOT:-/home/qinhaiyan/glm52-goal-runs/17-indexer_score_prefill/sglang}"
PY="${KERNEL_HARNESS_PYTHON:-${ROOT}/.venv/bin/python}"
RUN_ID="${1:-$(date -u +%Y%m%dT%H%M%SZ)}"
OUT="${EVIDENCE_ROOT}/four_gpu/${RUN_ID}"
CANDIDATE="${ROOT}/serving_native/candidates/indexer_score_balanced_chunks.py"
TASK=tp4_indexer_score_prefill_m4096_c256_diagnostic

if [[ -e "${OUT}" ]]; then
  echo "refusing to overwrite diagnostic ${RUN_ID}" >&2
  exit 3
fi
mkdir -p "${OUT}/paired" "${OUT}/logs"
export SGLANG_ROOT
export PYTHONPATH="${SGLANG_ROOT}/python:${ROOT}:${PYTHONPATH:-}"
export SGLANG_GLM52_OPT=0

{
  echo "run_id=${RUN_ID}"
  echo "started_utc=$(date -u +%FT%TZ)"
  echo "cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-unset}"
  echo "kernel_harness_head=$(git -C "${ROOT}" rev-parse HEAD)"
  echo "sglang_head=$(git -C "${SGLANG_ROOT}" rev-parse HEAD)"
  echo "topology=TP4/DP4/EP4 diagnostic only"
  echo "official_tp8_acceptance=NOT_SATISFIED"
  sha256sum "${ROOT}/serving_native/runner.py" "${CANDIDATE}"
} >"${OUT}/environment.txt"
git -C "${ROOT}" status --short --untracked-files=all \
  >"${OUT}/kernel_harness_status.txt"
git -C "${ROOT}" diff --binary >"${OUT}/kernel_harness_source.diff"
git -C "${SGLANG_ROOT}" status --short --untracked-files=all \
  >"${OUT}/sglang_status.txt"
git -C "${SGLANG_ROOT}" diff --binary >"${OUT}/sglang_source.diff"
nvidia-smi \
  --query-gpu=index,name,uuid,pci.bus_id,pstate,clocks.current.sm,clocks.current.memory,power.draw,temperature.gpu,memory.total,memory.free \
  --format=csv >"${OUT}/gpu_identity.csv"

printf 'series\texit_code\tutc_finished\n' >"${OUT}/status.tsv"
for series in 1 2 3; do
  json="${OUT}/paired/${TASK}.series${series}.json"
  log="${OUT}/logs/${TASK}.series${series}.log"
  echo "[$(date -u +%FT%TZ)] four-GPU series ${series}"
  set +e
  "${ROOT}/serving_native/run.sh" "${TASK}" \
    --candidate "${CANDIDATE}" --warmup 5 --repeat 30 --output "${json}" \
    > >(tee "${log}") 2>&1
  code=$?
  set -e
  printf '%s\t%s\t%s\n' "${series}" "${code}" "$(date -u +%FT%TZ)" \
    >>"${OUT}/status.tsv"
done

"${PY}" "${EVIDENCE_ROOT}/summarize_campaign.py" "${OUT}" \
  >"${OUT}/summary.log" 2>&1 || true
printf 'finished_utc=%s\n' "$(date -u +%FT%TZ)" >>"${OUT}/environment.txt"
find "${OUT}" -type f ! -name sha256.txt -print0 | sort -z | xargs -0 sha256sum \
  >"${OUT}/sha256.txt"
echo "FOUR_GPU_DIAGNOSTIC_DIR=${OUT}"
