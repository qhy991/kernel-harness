#!/usr/bin/env bash
# Production entry (the lease sentinel must be injected inside the wrapper):
#   /home/qinhaiyan/glm52-goal-runs/with_flexible_gpu.sh -- \
#     env TASK26_FLEXIBLE_GPU_LEASE_SENTINEL=glm52-task26-em8-bm16-stage11-v4-flexible-gpu-lease-v1 \
#     /home/qinhaiyan/glm52-v2-goal-runs/worktrees/26-moe-w2-decode-scoped-bm16/kernel-harness/evidence/glm52_prod_26_moe_w2_decode_scoped_bm16_em8_bm16_stage11_v4/run_single_b200.sh
set -euo pipefail

readonly EXPECTED_LEASE_SENTINEL="glm52-task26-em8-bm16-stage11-v4-flexible-gpu-lease-v1"
readonly PRODUCTION_ROOT="/home/qinhaiyan/glm52-v2-goal-runs/worktrees/26-moe-w2-decode-scoped-bm16/kernel-harness"
readonly PRODUCTION_SGLANG_ROOT="/home/qinhaiyan/glm52-v2-goal-runs/worktrees/26-moe-w2-decode-scoped-bm16/sglang"
readonly PRODUCTION_GPU_LOCK_ROOT="/home/qinhaiyan/glm52-goal-runs/locks"
readonly TASK_SHARED_CACHE_ROOT="/home/qinhaiyan/glm52-v2-goal-runs/cache/26-moe_w2_decode_scoped_bm16"
readonly TASK_CACHE_ROOT="${TASK_SHARED_CACHE_ROOT}/em8_bm16_stage11_v4"
readonly VARIANT_NAME="em8_bm16_stage11"
readonly VARIANT_VERSION="4"
readonly MIN_FREE_KIB=$((8 * 1024 * 1024))
readonly INHERITED_CONTRACT_COMMIT="c1c48c3d1e826c243727ed45d52ef8dbfeb3f701"

die() {
  echo "ERROR: $*" >&2
  exit 1
}

test_mode="${TASK26_DRIVER_TEST_MODE:-0}"
case "${test_mode}" in
  0)
    for override in \
      TASK26_TEST_ROOT \
      TASK26_TEST_SGLANG_ROOT \
      TASK26_TEST_PYTHON \
      TASK26_TEST_STOCK_LAUNCHER \
      TASK26_TEST_NVIDIA_SMI \
      TASK26_TEST_DF \
      TASK26_TEST_GPU_LOCK_ROOT \
      TASK26_TEST_CAMPAIGN_LOCK \
      TASK26_TEST_READY_TOOL \
      TASK26_TEST_READY_RECORD \
      TASK26_STAGE11_TEST_ATTEMPT_SENTINEL \
      TASK26_RUN_ROOT
    do
      if [[ -n "${!override:-}" ]]; then
        die "${override} is accepted only in CPU-only driver test mode"
      fi
    done
    ROOT="${PRODUCTION_ROOT}"
    SGLANG_ROOT="${PRODUCTION_SGLANG_ROOT}"
    PYTHON="${ROOT}/.venv/bin/python"
    STOCK_LAUNCHER="${SGLANG_ROOT}/third_party/deepgemm_w2_em8_bm16_stage11_v4/run_with_exact_post1_stock.sh"
    NVIDIA_SMI="nvidia-smi"
    DF="df"
    GPU_LOCK_ROOT="${PRODUCTION_GPU_LOCK_ROOT}"
    CAMPAIGN_LOCK="${TASK_CACHE_ROOT}/run_single_b200_stage11_v4.lock"
    ATTEMPT_SENTINEL="${TASK_CACHE_ROOT}/ONE_ATTEMPT_CONSUMED"
    READY_TOOL="${SGLANG_ROOT}/third_party/deepgemm_w2_em8_bm16_stage11_v4/ready_bundle.py"
    READY_RECORD=""
    ;;
  1)
    ROOT="${TASK26_TEST_ROOT:?TASK26_TEST_ROOT is required in driver test mode}"
    SGLANG_ROOT="${TASK26_TEST_SGLANG_ROOT:?TASK26_TEST_SGLANG_ROOT is required in driver test mode}"
    PYTHON="${TASK26_TEST_PYTHON:?TASK26_TEST_PYTHON is required in driver test mode}"
    STOCK_LAUNCHER="${TASK26_TEST_STOCK_LAUNCHER:?TASK26_TEST_STOCK_LAUNCHER is required in driver test mode}"
    NVIDIA_SMI="${TASK26_TEST_NVIDIA_SMI:?TASK26_TEST_NVIDIA_SMI is required in driver test mode}"
    DF="${TASK26_TEST_DF:?TASK26_TEST_DF is required in driver test mode}"
    GPU_LOCK_ROOT="${TASK26_TEST_GPU_LOCK_ROOT:?TASK26_TEST_GPU_LOCK_ROOT is required in driver test mode}"
    CAMPAIGN_LOCK="${TASK26_TEST_CAMPAIGN_LOCK:?TASK26_TEST_CAMPAIGN_LOCK is required in driver test mode}"
    ATTEMPT_SENTINEL="${TASK26_STAGE11_TEST_ATTEMPT_SENTINEL:?TASK26_STAGE11_TEST_ATTEMPT_SENTINEL is required in driver test mode}"
    READY_TOOL="${TASK26_TEST_READY_TOOL:?TASK26_TEST_READY_TOOL is required in driver test mode}"
    READY_RECORD="${TASK26_TEST_READY_RECORD:?TASK26_TEST_READY_RECORD is required in driver test mode}"
    ;;
  *)
    die "TASK26_DRIVER_TEST_MODE must be 0 or 1"
    ;;
esac
readonly test_mode ROOT SGLANG_ROOT PYTHON STOCK_LAUNCHER NVIDIA_SMI DF
readonly GPU_LOCK_ROOT CAMPAIGN_LOCK ATTEMPT_SENTINEL READY_TOOL
readonly RUNNER="${ROOT}/serving_native/runner.py"
readonly AUDITOR="${ROOT}/serving_native/audit_result.py"
readonly AUDIT_GATE="${ROOT}/serving_native/validate_portfolio_audit.py"
readonly CANDIDATE="${ROOT}/serving_native/candidates/moe_w2_em8_bm16_stage11_v4.py"
readonly DEFAULT_RUN_ROOT="${ROOT}/runs/glm52_prod_26_moe_w2_decode_scoped_bm16_em8_bm16_stage11_v4"
readonly SCRIPT_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")"

export SGLANG_ROOT
export DG_JIT_CACHE_DIR="${TASK_CACHE_ROOT}/deepgemm"
export SGLANG_DG_CACHE_DIR="${TASK_CACHE_ROOT}/deepgemm"
export TRITON_CACHE_DIR="${TASK_CACHE_ROOT}/triton"
export TORCH_EXTENSIONS_DIR="${TASK_CACHE_ROOT}/torch_extensions"

readonly -a LANES=(
  "moe_w2_grouped_decode_m32_em8_bm16_stage11_v4|eager"
  "moe_w2_grouped_decode_m32_em8_bm16_stage11_v4|cuda_graph"
  "moe_w13_swiglu_w2_region_decode_m32_em8_bm16_stage11_v4|eager"
  "moe_w13_swiglu_w2_region_decode_m32_em8_bm16_stage11_v4|cuda_graph"
)

for required_file in \
  "${RUNNER}" \
  "${AUDITOR}" \
  "${AUDIT_GATE}" \
  "${CANDIDATE}" \
  "${READY_TOOL}"
do
  [[ -f "${required_file}" ]] || die "required source is missing: ${required_file}"
done
for required_executable in "${PYTHON}" "${STOCK_LAUNCHER}"; do
  [[ -x "${required_executable}" ]] || die "required executable is missing: ${required_executable}"
done

require_clean_repo() {
  local repo="$1"
  local label="$2"
  local status
  if ! status="$(git -C "${repo}" status --porcelain=v1 --untracked-files=normal)"; then
    die "cannot inspect ${label} repository: ${repo}"
  fi
  if [[ -n "${status}" ]]; then
    echo "${status}" >&2
    die "${label} repository must be clean before READY verification"
  fi
}

require_clean_repo "${ROOT}" "Kernel-Harness"
require_clean_repo "${SGLANG_ROOT}" "SGLang"
if ! KERNEL_HARNESS_HEAD="$(git -C "${ROOT}" rev-parse HEAD)"; then
  die "cannot resolve Kernel-Harness HEAD"
fi
if ! SGLANG_HEAD="$(git -C "${SGLANG_ROOT}" rev-parse HEAD)"; then
  die "cannot resolve SGLang HEAD"
fi
readonly KERNEL_HARNESS_HEAD SGLANG_HEAD
if [[ "${test_mode}" == "0" ]]; then
  if ! INHERITED_LOCAL_COMMIT="$(git -C "${ROOT}" rev-parse 8fc047c^{commit})"; then
    die "cannot resolve inherited local V2 contract commit"
  fi
else
  INHERITED_LOCAL_COMMIT="${KERNEL_HARNESS_HEAD}"
fi
readonly INHERITED_LOCAL_COMMIT

assert_source_identity() {
  require_clean_repo "${ROOT}" "Kernel-Harness"
  require_clean_repo "${SGLANG_ROOT}" "SGLang"
  [[ "$(git -C "${ROOT}" rev-parse HEAD)" == "${KERNEL_HARNESS_HEAD}" ]] \
    || die "Kernel-Harness HEAD changed during the portfolio"
  [[ "$(git -C "${SGLANG_ROOT}" rev-parse HEAD)" == "${SGLANG_HEAD}" ]] \
    || die "SGLang HEAD changed during the portfolio"
}

# Phase 1 is deliberately CPU-only.  The content-addressed bundle and its
# tracked provenance must be valid before the driver inspects the scheduler
# lease, creates a run root, consumes the one-attempt sentinel, or queries a
# GPU.  The verifier also replays package/source hashes and both clean HEADs.
if [[ "${test_mode}" == "0" ]]; then
  if ! READY_RECORD="$(
    "${PYTHON}" "${READY_TOOL}" locate \
      --sglang-root "${SGLANG_ROOT}" \
      --print ready
  )"; then
    die "no unique stage11-v4 READY bundle is available"
  fi
fi
[[ "${READY_RECORD}" == /* ]] || die "READY record must be an absolute path"
[[ -f "${READY_RECORD}" ]] || die "stage11-v4 READY record is missing: ${READY_RECORD}"
if ! READY_RECORD_CANONICAL="$(readlink -f -- "${READY_RECORD}")"; then
  die "cannot canonicalize stage11-v4 READY record"
fi
if ! READY_EVIDENCE_JSON="$(
  "${PYTHON}" "${READY_TOOL}" verify \
    --ready "${READY_RECORD_CANONICAL}" \
    --sglang-root "${SGLANG_ROOT}" \
    --kernel-harness-root "${ROOT}" \
    --check-env \
    --json
)"; then
  die "stage11-v4 READY verification failed before GPU inspection"
fi
if ! READY_EVIDENCE_PATH="$(
  "${PYTHON}" -c \
    'import json, pathlib, sys
record = json.loads(sys.argv[1])
required = {
    "ready_path", "ready_sha256", "contract_sha256", "bundle_digest",
    "manifest_path", "manifest_sha256", "source_replay_path",
    "source_replay_sha256", "build_provenance_path",
    "build_provenance_sha256", "stock_package_tree_sha256",
    "candidate_package_tree_sha256", "stock_site", "candidate_package",
}
if not isinstance(record, dict) or set(record) != required:
    raise SystemExit("READY verifier returned an unexpected evidence contract")
print(pathlib.Path(record["ready_path"]).resolve())' \
    "${READY_EVIDENCE_JSON}"
)"; then
  die "stage11-v4 READY verifier returned malformed evidence"
fi
[[ "${READY_EVIDENCE_PATH}" == "${READY_RECORD_CANONICAL}" ]] \
  || die "stage11-v4 READY verifier returned a different record"
READY_RECORD_SHA256="$(sha256sum "${READY_RECORD_CANONICAL}" | awk '{print $1}')"
readonly READY_RECORD_CANONICAL READY_RECORD_SHA256 READY_EVIDENCE_JSON
export SGLANG_GLM52_W2_EM8_BM16_STAGE11_V4_READY="${READY_RECORD_CANONICAL}"
export TASK26_V4_KERNEL_HARNESS_ROOT="${ROOT}"

if [[ "${TASK26_FLEXIBLE_GPU_LEASE_SENTINEL:-}" != "${EXPECTED_LEASE_SENTINEL}" ]]; then
  die "missing exact flexible-GPU lease sentinel; use the documented wrapper entry"
fi
if [[ ! "${CUDA_VISIBLE_DEVICES:-}" =~ ^[0-9]+$ ]]; then
  die "this portfolio requires exactly one scheduler-assigned physical GPU index"
fi
command -v "${NVIDIA_SMI}" >/dev/null \
  || die "nvidia-smi executable is missing: ${NVIDIA_SMI}"
command -v "${DF}" >/dev/null || die "df executable is missing: ${DF}"
command -v flock >/dev/null || die "flock executable is missing"

EXPECTED_GPU_LOCK="${GPU_LOCK_ROOT}/gpu${CUDA_VISIBLE_DEVICES}.lock"
if ! EXPECTED_GPU_LOCK_CANONICAL="$(readlink -f -- "${EXPECTED_GPU_LOCK}")"; then
  die "cannot canonicalize expected scheduler GPU lock: ${EXPECTED_GPU_LOCK}"
fi
readonly EXPECTED_GPU_LOCK EXPECTED_GPU_LOCK_CANONICAL

LEASE_FD=""
for fd_link in /proc/$$/fd/[0-9]*; do
  [[ -e "${fd_link}" ]] || continue
  if ! fd_target="$(readlink -f -- "${fd_link}" 2>/dev/null)"; then
    continue
  fi
  if [[ "${fd_target}" == "${EXPECTED_GPU_LOCK_CANONICAL}" ]]; then
    LEASE_FD="${fd_link##*/}"
    break
  fi
done
[[ -n "${LEASE_FD}" ]] \
  || die "no inherited wrapper FD resolves to ${EXPECTED_GPU_LOCK}"
[[ -w "/proc/$$/fd/${LEASE_FD}" ]] \
  || die "inherited wrapper GPU-lock FD is not writable"
readonly LEASE_FD

assert_wrapper_lease_fd() {
  local target
  [[ -e "/proc/$$/fd/${LEASE_FD}" ]] \
    || die "inherited wrapper GPU-lock FD was closed"
  target="$(readlink -f -- "/proc/$$/fd/${LEASE_FD}")" \
    || die "cannot resolve inherited wrapper GPU-lock FD"
  [[ "${target}" == "${EXPECTED_GPU_LOCK_CANONICAL}" ]] \
    || die "inherited wrapper GPU-lock FD changed target"
}

[[ -d "$(dirname "${CAMPAIGN_LOCK}")" ]] \
  || die "task campaign-lock directory is missing"
exec {CAMPAIGN_LOCK_FD}>"${CAMPAIGN_LOCK}"
if ! flock -n "${CAMPAIGN_LOCK_FD}"; then
  die "another Task 26 single-B200 bundle already holds ${CAMPAIGN_LOCK}"
fi
if ! CAMPAIGN_LOCK_CANONICAL="$(readlink -f -- "${CAMPAIGN_LOCK}")"; then
  die "cannot canonicalize Task 26 campaign lock"
fi
readonly CAMPAIGN_LOCK_FD CAMPAIGN_LOCK_CANONICAL

assert_campaign_lock_fd() {
  local target
  [[ -e "/proc/$$/fd/${CAMPAIGN_LOCK_FD}" ]] \
    || die "Task 26 campaign-lock FD was closed"
  target="$(readlink -f -- "/proc/$$/fd/${CAMPAIGN_LOCK_FD}")" \
    || die "cannot resolve Task 26 campaign-lock FD"
  [[ "${target}" == "${CAMPAIGN_LOCK_CANONICAL}" ]] \
    || die "Task 26 campaign-lock FD changed target"
}

check_disk() {
  local report available_kib
  if ! report="$("${DF}" -Pk "${ROOT}")"; then
    die "disk-space query failed"
  fi
  available_kib="$(awk 'NR == 2 {print $4}' <<<"${report}")"
  [[ "${available_kib}" =~ ^[0-9]+$ ]] \
    || die "disk-space query returned invalid available KiB: ${available_kib}"
  if (( available_kib < MIN_FREE_KIB )); then
    die "fewer than 8 GiB remain; refusing new JIT/profile growth"
  fi
}

run_stamp="$(date -u +%Y%m%dT%H%M%SZ)"
if [[ "${test_mode}" == "1" ]]; then
  run_root="${TASK26_RUN_ROOT:?TASK26_RUN_ROOT is required in driver test mode}"
else
  run_root="${DEFAULT_RUN_ROOT}/${run_stamp}"
fi
[[ "${run_root}" == /* ]] || die "TASK26_RUN_ROOT must be an absolute path"
run_leaf="${run_root##*/}"
[[ -n "${run_leaf}" && "${run_leaf}" != "." && "${run_leaf}" != ".." ]] \
  || die "TASK26_RUN_ROOT must name one direct child"
[[ "${run_root}" == "${DEFAULT_RUN_ROOT}/${run_leaf}" ]] \
  || die "TASK26_RUN_ROOT must be a direct child of ${DEFAULT_RUN_ROOT}"
default_run_root_canonical="$(realpath -m -- "${DEFAULT_RUN_ROOT}")"
run_root_canonical="$(realpath -m -- "${run_root}")"
[[ "$(dirname "${run_root_canonical}")" == "${default_run_root_canonical}" ]] \
  || die "TASK26_RUN_ROOT escapes the canonical default run root"
if [[ -e "${run_root}" ]]; then
  die "requested run root already exists and will not be reused: ${run_root}"
fi

check_disk
assert_wrapper_lease_fd
assert_campaign_lock_fd
assert_source_identity
[[ -d "$(dirname "${ATTEMPT_SENTINEL}")" ]] \
  || die "persistent one-attempt sentinel parent is missing"
if [[ -e "${ATTEMPT_SENTINEL}" ]]; then
  die "stage11 one-attempt sentinel already exists: ${ATTEMPT_SENTINEL}"
fi
if ! mkdir -- "${ATTEMPT_SENTINEL}"; then
  die "stage11 one-attempt sentinel was claimed concurrently"
fi
ATTEMPT_SENTINEL_CANONICAL="$(readlink -f -- "${ATTEMPT_SENTINEL}")" \
  || die "cannot canonicalize persistent one-attempt sentinel"
readonly ATTEMPT_SENTINEL_CANONICAL
{
  echo "status=CLAIMED"
  echo "claimed_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "variant=${VARIANT_NAME}"
  echo "variant_version=${VARIANT_VERSION}"
  echo "ready_record=${READY_RECORD_CANONICAL}"
  echo "ready_record_sha256=${READY_RECORD_SHA256}"
  echo "kernel_harness_head=${KERNEL_HARNESS_HEAD}"
  echo "sglang_head=${SGLANG_HEAD}"
} >"${ATTEMPT_SENTINEL}/CLAIMED"
ATTEMPT_CLAIM_SHA256="$(sha256sum "${ATTEMPT_SENTINEL}/CLAIMED" | awk '{print $1}')"
readonly ATTEMPT_CLAIM_SHA256

assert_attempt_sentinel() {
  [[ -d "${ATTEMPT_SENTINEL}" ]] \
    || die "persistent one-attempt sentinel disappeared"
  [[ "$(readlink -f -- "${ATTEMPT_SENTINEL}")" == "${ATTEMPT_SENTINEL_CANONICAL}" ]] \
    || die "persistent one-attempt sentinel changed canonical identity"
  [[ -f "${ATTEMPT_SENTINEL}/CLAIMED" ]] \
    || die "persistent one-attempt claim disappeared"
  [[ "$(sha256sum "${ATTEMPT_SENTINEL}/CLAIMED" | awk '{print $1}')" == "${ATTEMPT_CLAIM_SHA256}" ]] \
    || die "persistent one-attempt claim was mutated"
}

assert_ready_identity() {
  [[ -f "${READY_RECORD_CANONICAL}" ]] \
    || die "stage11-v4 READY record disappeared"
  [[ "$(sha256sum "${READY_RECORD_CANONICAL}" | awk '{print $1}')" == "${READY_RECORD_SHA256}" ]] \
    || die "stage11-v4 READY record was mutated"
}

mkdir -p "${DEFAULT_RUN_ROOT}"
mkdir "${run_root}"
mkdir "${run_root}/results" "${run_root}/logs" "${run_root}/audits"
printf 'started_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >"${run_root}/IN_PROGRESS"
printf '%s\n' "${READY_EVIDENCE_JSON}" >"${run_root}/ready_evidence.json"
if [[ "${test_mode}" == "1" ]]; then
  {
    echo "artifact_class=TEST_ONLY"
    echo "driver_test_mode=1"
  } >"${run_root}/TEST_ONLY"
fi

run_initialized=1
record_failure() {
  local rc="$?"
  trap - EXIT
  if (( rc != 0 && run_initialized == 1 )); then
    {
      echo "status=FAILED"
      echo "failed_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
      echo "exit_code=${rc}"
    } >"${run_root}/FAILED"
    {
      echo "status=FAILED"
      echo "failed_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
      echo "exit_code=${rc}"
      echo "run_root=${run_root}"
    } >"${ATTEMPT_SENTINEL}/FAILED"
  fi
  exit "${rc}"
}
trap record_failure EXIT

GPU_QUERY_INDEX=""
GPU_QUERY_UUID=""
GPU_QUERY_NAME=""
GPU_QUERY_DRIVER=""
GPU_QUERY_SM_CLOCK=""
GPU_QUERY_MEMORY_CLOCK=""

trim_variable() {
  local variable_name="$1"
  local value="${!variable_name}"
  value="${value#"${value%%[![:space:]]*}"}"
  value="${value%"${value##*[![:space:]]}"}"
  printf -v "${variable_name}" '%s' "${value}"
}

read_single_gpu_identity() {
  local output extra field_name
  local -a rows
  if ! output="$("${NVIDIA_SMI}" -i "${CUDA_VISIBLE_DEVICES}" \
    --query-gpu=index,uuid,name,driver_version,clocks.current.sm,clocks.current.memory \
    --format=csv,noheader,nounits)"
  then
    echo "GPU identity query failed" >&2
    return 1
  fi
  mapfile -t rows <<<"${output}"
  if (( ${#rows[@]} != 1 )) || [[ -z "${rows[0]}" ]]; then
    echo "GPU identity query did not resolve exactly one physical GPU" >&2
    return 1
  fi
  IFS=',' read -r \
    GPU_QUERY_INDEX \
    GPU_QUERY_UUID \
    GPU_QUERY_NAME \
    GPU_QUERY_DRIVER \
    GPU_QUERY_SM_CLOCK \
    GPU_QUERY_MEMORY_CLOCK \
    extra <<<"${rows[0]}"
  for field_name in \
    GPU_QUERY_INDEX \
    GPU_QUERY_UUID \
    GPU_QUERY_NAME \
    GPU_QUERY_DRIVER \
    GPU_QUERY_SM_CLOCK \
    GPU_QUERY_MEMORY_CLOCK
  do
    trim_variable "${field_name}"
  done
  if [[ -n "${extra:-}" ]] \
    || [[ ! "${GPU_QUERY_INDEX}" =~ ^[0-9]+$ ]] \
    || [[ "${GPU_QUERY_INDEX}" != "${CUDA_VISIBLE_DEVICES}" ]] \
    || [[ ! "${GPU_QUERY_UUID}" =~ ^GPU-[[:alnum:]-]+$ ]] \
    || [[ "${GPU_QUERY_NAME}" != "NVIDIA B200" ]] \
    || [[ -z "${GPU_QUERY_DRIVER}" ]] \
    || [[ ! "${GPU_QUERY_SM_CLOCK}" =~ ^[0-9]+([.][0-9]+)?$ ]] \
    || [[ ! "${GPU_QUERY_MEMORY_CLOCK}" =~ ^[0-9]+([.][0-9]+)?$ ]] \
    || [[ "${GPU_QUERY_NAME}" == *$'\t'* ]] \
    || [[ "${GPU_QUERY_DRIVER}" == *$'\t'* ]] \
    || [[ "${GPU_QUERY_SM_CLOCK}" == *$'\t'* ]] \
    || [[ "${GPU_QUERY_MEMORY_CLOCK}" == *$'\t'* ]]
  then
    echo "GPU identity query returned an invalid physical identity: ${rows[0]}" >&2
    return 1
  fi
}

if ! read_single_gpu_identity; then
  die "cannot establish the initial single-GPU lease identity"
fi
readonly INITIAL_GPU_INDEX="${GPU_QUERY_INDEX}"
readonly INITIAL_GPU_UUID="${GPU_QUERY_UUID}"
readonly INITIAL_GPU_NAME="${GPU_QUERY_NAME}"
readonly INITIAL_GPU_DRIVER="${GPU_QUERY_DRIVER}"
readonly INITIAL_GPU_SM_CLOCK="${GPU_QUERY_SM_CLOCK}"
readonly INITIAL_GPU_MEMORY_CLOCK="${GPU_QUERY_MEMORY_CLOCK}"

readonly GPU_SNAPSHOTS="${run_root}/gpu_snapshots.tsv"
printf 'timestamp_utc\tstage\tindex\tuuid\tname\tdriver_version\tsm_clock_mhz\tmemory_clock_mhz\n' \
  >"${GPU_SNAPSHOTS}"

record_gpu_snapshot() {
  local stage="$1"
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
    "${stage}" \
    "${GPU_QUERY_INDEX}" \
    "${GPU_QUERY_UUID}" \
    "${GPU_QUERY_NAME}" \
    "${GPU_QUERY_DRIVER}" \
    "${GPU_QUERY_SM_CLOCK}" \
    "${GPU_QUERY_MEMORY_CLOCK}" \
    >>"${GPU_SNAPSHOTS}"
}
record_gpu_snapshot "initial"

assert_same_gpu() {
  local stage="$1"
  assert_ready_identity
  assert_wrapper_lease_fd
  assert_campaign_lock_fd
  assert_attempt_sentinel
  if ! read_single_gpu_identity; then
    die "cannot revalidate the single-GPU lease identity"
  fi
  [[ "${GPU_QUERY_INDEX}" == "${INITIAL_GPU_INDEX}" ]] \
    || die "physical GPU index changed during the portfolio"
  [[ "${GPU_QUERY_UUID}" == "${INITIAL_GPU_UUID}" ]] \
    || die "physical GPU UUID changed during the portfolio"
  record_gpu_snapshot "${stage}"
}

record_environment() {
  {
    echo "started_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "artifact_class=$([[ "${test_mode}" == "0" ]] && echo production || echo TEST_ONLY)"
    echo "driver_test_mode=${test_mode}"
    echo "driver_sha256=$(sha256sum "${SCRIPT_PATH}" | awk '{print $1}')"
    echo "lease_sentinel=${TASK26_FLEXIBLE_GPU_LEASE_SENTINEL}"
    echo "wrapper_gpu_lock=${EXPECTED_GPU_LOCK}"
    echo "wrapper_gpu_lock_canonical=${EXPECTED_GPU_LOCK_CANONICAL}"
    echo "wrapper_gpu_lock_fd=${LEASE_FD}"
    echo "campaign_lock=${CAMPAIGN_LOCK}"
    echo "campaign_lock_canonical=${CAMPAIGN_LOCK_CANONICAL}"
    echo "campaign_lock_fd=${CAMPAIGN_LOCK_FD}"
    echo "variant=${VARIANT_NAME}"
    echo "variant_version=${VARIANT_VERSION}"
    echo "persistent_one_attempt_sentinel=${ATTEMPT_SENTINEL}"
    echo "persistent_one_attempt_sentinel_canonical=${ATTEMPT_SENTINEL_CANONICAL}"
    echo "persistent_one_attempt_claim_sha256=${ATTEMPT_CLAIM_SHA256}"
    echo "ready_record=${READY_RECORD_CANONICAL}"
    echo "ready_record_sha256=${READY_RECORD_SHA256}"
    echo "ready_evidence_sha256=$(sha256sum "${run_root}/ready_evidence.json" | awk '{print $1}')"
    echo "predeclared_fallback=em8_bm16_stage10"
    echo "fallback_eligible=0"
    echo "cuda_visible_devices=${CUDA_VISIBLE_DEVICES}"
    echo "gpu_index=${INITIAL_GPU_INDEX}"
    echo "gpu_uuid=${INITIAL_GPU_UUID}"
    echo "gpu_name=${INITIAL_GPU_NAME}"
    echo "gpu_driver_version=${INITIAL_GPU_DRIVER}"
    echo "gpu_initial_sm_clock_mhz=${INITIAL_GPU_SM_CLOCK}"
    echo "gpu_initial_memory_clock_mhz=${INITIAL_GPU_MEMORY_CLOCK}"
    echo "kernel_harness_head=${KERNEL_HARNESS_HEAD}"
    echo "sglang_head=${SGLANG_HEAD}"
    echo "inherited_contract_commit=${INHERITED_CONTRACT_COMMIT}"
    echo "inherited_local_commit=${INHERITED_LOCAL_COMMIT}"
    echo "stock_launcher=${STOCK_LAUNCHER}"
    echo "runner=${RUNNER}"
    echo "candidate=${CANDIDATE}"
    echo "auditor=${AUDITOR}"
    echo "audit_gate=${AUDIT_GATE}"
    echo "warmup=3"
    echo "repeat=10"
    echo "series=3"
    echo "dg_jit_cache_dir=${DG_JIT_CACHE_DIR}"
    echo "sglang_dg_cache_dir=${SGLANG_DG_CACHE_DIR}"
    echo "triton_cache_dir=${TRITON_CACHE_DIR}"
    echo "torch_extensions_dir=${TORCH_EXTENSIONS_DIR}"
    "${DF}" -h /
  } >"${run_root}/environment.txt"
}

run_one() {
  local task="$1"
  local mode="$2"
  local stem="${task}__${mode}"
  check_disk
  assert_ready_identity
  assert_source_identity
  assert_same_gpu "before:${stem}"
  echo "RUN ${stem}" >&2
  "${STOCK_LAUNCHER}" "${PYTHON}" "${RUNNER}" \
    --task "${task}" \
    --candidate "${CANDIDATE}" \
    --execution-mode "${mode}" \
    --warmup 3 \
    --repeat 10 \
    --series 3 \
    --output "${run_root}/results/${stem}.json" \
    >"${run_root}/logs/${stem}.log" 2>&1
  [[ -s "${run_root}/results/${stem}.json" ]] \
    || die "runner did not produce a result for ${stem}"
  "${PYTHON}" "${AUDITOR}" --json \
    "${run_root}/results/${stem}.json" \
    >"${run_root}/audits/${stem}.json"
  [[ -s "${run_root}/audits/${stem}.json" ]] \
    || die "auditor did not produce a report for ${stem}"
  "${PYTHON}" "${AUDIT_GATE}" "${run_root}/audits/${stem}.json"
  assert_source_identity
  assert_same_gpu "after:${stem}"
}

record_environment
for lane in "${LANES[@]}"; do
  IFS='|' read -r task mode <<<"${lane}"
  run_one "${task}" "${mode}"
done

assert_source_identity
assert_ready_identity
assert_same_gpu "final"
{
  echo "finished_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "artifact_class=$([[ "${test_mode}" == "0" ]] && echo production || echo TEST_ONLY)"
  echo "driver_test_mode=${test_mode}"
  echo "gpu_index=${GPU_QUERY_INDEX}"
  echo "gpu_uuid=${GPU_QUERY_UUID}"
  echo "gpu_name=${GPU_QUERY_NAME}"
  echo "gpu_driver_version=${GPU_QUERY_DRIVER}"
  echo "gpu_final_sm_clock_mhz=${GPU_QUERY_SM_CLOCK}"
  echo "gpu_final_memory_clock_mhz=${GPU_QUERY_MEMORY_CLOCK}"
  "${DF}" -h /
} >"${run_root}/completion.txt"
find "${run_root}" -type f \
  ! -name artifact_sha256.txt \
  ! -name IN_PROGRESS \
  ! -name COMPLETE \
  ! -name TEST_COMPLETE \
  -print0 \
  | sort -z \
  | xargs -0 sha256sum \
  >"${run_root}/artifact_sha256.txt"

if [[ "${test_mode}" == "0" ]]; then
  grep -Fxq "driver_test_mode=0" "${run_root}/environment.txt" \
    || die "production environment lost its driver_test_mode=0 marker"
  grep -Fxq "artifact_class=production" "${run_root}/completion.txt" \
    || die "production completion lost its artifact class"
  [[ ! -e "${run_root}/TEST_ONLY" ]] \
    || die "production evidence contains a TEST_ONLY marker"
  mv "${run_root}/IN_PROGRESS" "${run_root}/COMPLETE"
  {
    echo "status=COMPLETE"
    echo "finished_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "run_root=${run_root}"
  } >"${ATTEMPT_SENTINEL}/COMPLETE"
  run_initialized=0
  echo "PASS task26 em8/BM16/stage11 single-B200 portfolio: ${run_root}" >&2
else
  grep -Fxq "driver_test_mode=1" "${run_root}/environment.txt" \
    || die "CPU simulation lost its driver_test_mode=1 marker"
  [[ -f "${run_root}/TEST_ONLY" ]] \
    || die "CPU simulation lost its TEST_ONLY marker"
  mv "${run_root}/IN_PROGRESS" "${run_root}/TEST_COMPLETE"
  {
    echo "status=TEST_COMPLETE"
    echo "finished_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "run_root=${run_root}"
  } >"${ATTEMPT_SENTINEL}/TEST_COMPLETE"
  run_initialized=0
  echo "TEST_ONLY task26 stage11 CPU driver simulation: ${run_root}" >&2
fi
