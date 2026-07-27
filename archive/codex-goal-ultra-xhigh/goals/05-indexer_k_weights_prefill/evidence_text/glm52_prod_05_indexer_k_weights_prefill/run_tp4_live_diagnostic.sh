#!/usr/bin/env bash
# Run only through:
#   /home/qinhaiyan/glm52-goal-runs/with_all_gpus_lock.sh \
#     evidence/glm52_prod_05_indexer_k_weights_prefill/run_tp4_live_diagnostic.sh
#
# This is a four-GPU TP4/DP4/EP4 dummy-weight diagnostic. It is not TP8/DP8/EP8
# production acceptance gate and must never be reported as one.
set -Eeuo pipefail

KH_ROOT=/home/qinhaiyan/glm52-goal-runs/05-indexer_k_weights_prefill/kernel-harness
SGLANG_ROOT=/home/qinhaiyan/glm52-goal-runs/05-indexer_k_weights_prefill/sglang
EVIDENCE_ROOT="$KH_ROOT/evidence/glm52_prod_05_indexer_k_weights_prefill"
PROFILE_ROOT="$KH_ROOT/profile/indexer-wk-weights-prefill-m4096-20260722"
PY="$KH_ROOT/.venv/bin/python"
HELPER="$EVIDENCE_ROOT/tp4_live_request.py"
HOST=127.0.0.1
MODEL_PATH=nvidia/GLM-5.2-NVFP4
MODEL_REVISION=aec724e8c7b8ee9db3b48c01c320f63f9cdaf8aa
STARTUP_TIMEOUT_S=${TP4_STARTUP_TIMEOUT_S:-1800}
GENERATE_TIMEOUT_S=${TP4_GENERATE_TIMEOUT_S:-1800}
RUN_ID=${TP4_RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)-$$}

if [[ ! "$RUN_ID" =~ ^[A-Za-z0-9_.-]+$ ]]; then
  printf 'TP4_RUN_ID contains unsafe path characters: %s\n' "$RUN_ID" >&2
  exit 64
fi
if [[ -n "${TP4_MODEL_PATH:-}" && "$TP4_MODEL_PATH" != "$MODEL_PATH" ]]; then
  printf 'refusing noncanonical TP4_MODEL_PATH: %s\n' "$TP4_MODEL_PATH" >&2
  exit 64
fi
if [[ -n "${TP4_MODEL_REVISION:-}" && \
      "$TP4_MODEL_REVISION" != "$MODEL_REVISION" ]]; then
  printf 'refusing noncanonical TP4_MODEL_REVISION: %s\n' \
    "$TP4_MODEL_REVISION" >&2
  exit 64
fi

RUN_DIR="$EVIDENCE_ROOT/tp4_live/$RUN_ID"
NSYS_BASE="$PROFILE_ROOT/reports/nsys-tp4-live-$RUN_ID"
STATUS_FILE="$RUN_DIR/status.txt"
SERVER_PID=""
SERVER_PGID=""
BASE_URL=""
PROFILE_MAY_BE_ACTIVE=0

write_artifact_manifest() {
  local manifest_tmp="$RUN_DIR/artifact_manifest.sha256.tmp"
  local manifest="$RUN_DIR/artifact_manifest.sha256"

  : > "$manifest_tmp"
  if ! find "$RUN_DIR" -type f \
      ! -name 'artifact_manifest.sha256' \
      ! -name 'artifact_manifest.sha256.tmp' \
      -print0 \
      | sort -z \
      | xargs -0 -r sha256sum >> "$manifest_tmp"; then
    return 1
  fi
  if [[ -s "$NSYS_BASE.nsys-rep" ]]; then
    if ! sha256sum "$NSYS_BASE.nsys-rep" >> "$manifest_tmp"; then
      return 1
    fi
  fi
  if ! mv "$manifest_tmp" "$manifest"; then
    return 1
  fi
  [[ -s "$manifest" ]]
}

if [[ -e "$RUN_DIR" || -e "$NSYS_BASE.nsys-rep" ]]; then
  printf 'refusing to overwrite TP4 diagnostic artifacts for run ID %s\n' \
    "$RUN_ID" >&2
  exit 73
fi
mkdir -p "$RUN_DIR" "$PROFILE_ROOT/reports" "$PROFILE_ROOT/analysis"
: > "$STATUS_FILE"
GLM52_EMPTY_ENV="$RUN_DIR/glm52_opt.empty.env"
: > "$GLM52_EMPTY_ENV"

record_status() {
  printf '%s\t%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" \
    >> "$STATUS_FILE"
}

process_group_has_non_zombie() {
  local target_pgid=$1
  ps -eo pgid=,stat= | awk -v target="$target_pgid" \
    '$1 == target && $2 !~ /^Z/ { found = 1 } END { exit(found ? 0 : 1) }'
}

stop_server_group() {
  if [[ -z "$SERVER_PGID" || ! "$SERVER_PGID" =~ ^[0-9]+$ ]]; then
    return
  fi
  if (( SERVER_PGID <= 1 )); then
    record_status "REFUSE cleanup for unsafe process group $SERVER_PGID"
    return
  fi

  if process_group_has_non_zombie "$SERVER_PGID"; then
    record_status "TERM tracked process group $SERVER_PGID"
    kill -TERM -- "-$SERVER_PGID" 2>/dev/null || true
    for _ in $(seq 1 30); do
      if ! process_group_has_non_zombie "$SERVER_PGID"; then
        break
      fi
      sleep 1
    done
  fi
  if process_group_has_non_zombie "$SERVER_PGID"; then
    record_status "KILL tracked process group $SERVER_PGID after TERM timeout"
    kill -KILL -- "-$SERVER_PGID" 2>/dev/null || true
  fi
  if [[ -n "$SERVER_PID" ]]; then
    wait "$SERVER_PID" 2>/dev/null || true
  fi
  SERVER_PID=""
  SERVER_PGID=""
}

on_exit() {
  local exit_code=$?
  trap - EXIT INT TERM HUP
  set +e

  if (( PROFILE_MAY_BE_ACTIVE )) && [[ -n "$BASE_URL" ]]; then
    record_status "best-effort stop_profile from EXIT cleanup"
    timeout 30s "$PY" "$HELPER" stop-profile \
      --base-url "$BASE_URL" \
      --output "$RUN_DIR/cleanup_stop_profile_response.json" \
      --timeout-s 15 || true
  fi
  stop_server_group

  if (( exit_code == 0 )) && [[ ! -s "$NSYS_BASE.nsys-rep" ]]; then
    record_status "FAIL expected nsys report is missing or empty: $NSYS_BASE.nsys-rep"
    exit_code=1
  fi
  if (( exit_code == 0 )); then
    record_status "PASS TP4 diagnostic; this is not TP8 production acceptance"
  else
    record_status "FAIL TP4 diagnostic exit_code=$exit_code"
  fi
  if ! write_artifact_manifest; then
    record_status "FAIL could not finalize artifact SHA-256 manifest"
    exit_code=1
    write_artifact_manifest || true
  fi
  exit "$exit_code"
}

trap on_exit EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
trap 'exit 129' HUP

for required in "$PY" "$HELPER" "$SGLANG_ROOT/python/sglang/launch_server.py"; do
  if [[ ! -f "$required" ]]; then
    record_status "FAIL missing required file: $required"
    exit 64
  fi
done
for required_command in \
  find git grep mv nsys nvidia-smi readlink setsid sha256sum sort timeout xargs; do
  if ! command -v "$required_command" >/dev/null 2>&1; then
    record_status "FAIL missing required command: $required_command"
    exit 64
  fi
done

if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  record_status "FAIL CUDA_VISIBLE_DEVICES is unset; use with_all_gpus_lock.sh"
  exit 64
fi
IFS=',' read -r -a VISIBLE_DEVICE_TOKENS <<< "$CUDA_VISIBLE_DEVICES"
if (( ${#VISIBLE_DEVICE_TOKENS[@]} != 4 )); then
  record_status "FAIL expected four wrapper-selected logical devices; observed ${#VISIBLE_DEVICE_TOKENS[@]}"
  exit 64
fi
declare -A SEEN_VISIBLE_DEVICES=()
WRAPPER_LOCK_RECORDS=()
EXPECTED_LOCK_PATHS=(/home/qinhaiyan/glm52-goal-runs/locks/all4.intent.lock)
for token in "${VISIBLE_DEVICE_TOKENS[@]}"; do
  if [[ ! "$token" =~ ^[0-9]+$ || -n "${SEEN_VISIBLE_DEVICES[$token]:-}" ]]; then
    record_status "FAIL invalid or duplicate wrapper-selected device token: $token"
    exit 64
  fi
  SEEN_VISIBLE_DEVICES[$token]=1
  EXPECTED_LOCK_PATHS+=("/home/qinhaiyan/glm52-goal-runs/locks/gpu${token}.lock")
done
for lock_path in "${EXPECTED_LOCK_PATHS[@]}"; do
  if ! lock_canonical=$(readlink -f "$lock_path"); then
    record_status "FAIL cannot resolve required wrapper lock: $lock_path"
    exit 64
  fi
  inherited_fd=""
  for fd_path in /proc/$$/fd/*; do
    if [[ "$(readlink -f "$fd_path" 2>/dev/null || true)" == "$lock_canonical" ]]; then
      inherited_fd=${fd_path##*/}
      break
    fi
  done
  if [[ -z "$inherited_fd" ]]; then
    record_status "FAIL required all-GPU wrapper lock is not inherited: $lock_path"
    exit 64
  fi
  if ! grep -Eq \
    '^lock:[[:space:]]+[0-9]+:[[:space:]]+FLOCK[[:space:]]+ADVISORY[[:space:]]+WRITE([[:space:]]|$)' \
    "/proc/$$/fdinfo/$inherited_fd"; then
    record_status "FAIL inherited wrapper lock FD is not flock-held for write: $lock_path fd=$inherited_fd"
    exit 64
  fi
  WRAPPER_LOCK_RECORDS+=("$lock_path canonical=$lock_canonical fd=$inherited_fd")
done

if [[ -n "${TP4_DIAG_PORT:-}" ]]; then
  PORT=$TP4_DIAG_PORT
  if [[ ! "$PORT" =~ ^[0-9]+$ ]] || (( PORT < 1024 || PORT > 65535 )); then
    record_status "FAIL invalid TP4_DIAG_PORT=$PORT"
    exit 64
  fi
  if ! "$PY" "$HELPER" check-port --host "$HOST" --port "$PORT"; then
    record_status "FAIL requested port is already in use: $PORT"
    exit 69
  fi
else
  PORT=$("$PY" "$HELPER" find-free-port --host "$HOST")
fi
BASE_URL="http://$HOST:$PORT"

export SGLANG_ROOT
export PYTHONPATH="$SGLANG_ROOT/python:$KH_ROOT"
export PYTHONNOUSERSITE=1
export PYTHONSAFEPATH=1
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export SGLANG_GLM52_OPT=0
export SGLANG_GLM52_ENV_FILE="$GLM52_EMPTY_ENV"
export SGLANG_GLM52_NSYS_GATE=0
export SGLANG_PROFILE_V2=0
export SGLANG_SIMULATE_ROUND_ROBIN_EXPERTS=1
unset PYTHONSTARTUP
unset CUDA_LAUNCH_BLOCKING
unset CUDA_DEVICE_MAX_CONNECTIONS
unset CUDA_CACHE_DISABLE
unset CUDA_FORCE_PTX_JIT
unset CUBLAS_WORKSPACE_CONFIG
unset NVIDIA_TF32_OVERRIDE
unset TORCH_ALLOW_TF32_CUBLAS_OVERRIDE
unset SGLANG_DISABLE_DSA_INDEXER_FUSION
unset SGLANG_NVFP4_CKPT_FP8_GEMM_IN_ATTN
unset SGLANG_GLM52_OPT_PROFILE
unset SGLANG_GLM52_OPT_OPS
unset SGLANG_GLM52_OPT_M_BUCKETS
unset SGLANG_GLM52_ALLOW_ABI_ADAPTER
unset SGLANG_GLM52_MANIFEST
unset SGLANG_GLM52_DEEPGEMM_VARIANT
unset SGLANG_GLM52_ARCHIVE
unset SGLANG_GLM52_DEEPGEMM_OVERLAY
unset SGLANG_GLM52_DEEPGEMM_MANIFEST
unset SGLANG_GLM52_NSYS_TRIGGER
unset SGLANG_GLM52_NSYS_SECONDS

if ! git -C "$KH_ROOT" diff --quiet -- \
    ':!evidence/glm52_prod_05_indexer_k_weights_prefill/tp4_live' \
    ':!profile/indexer-wk-weights-prefill-m4096-20260722/reports' \
    || ! git -C "$KH_ROOT" diff --cached --quiet; then
  record_status "FAIL Kernel-Harness tracked source differs from committed HEAD"
  exit 65
fi
if ! git -C "$SGLANG_ROOT" diff --quiet \
    || ! git -C "$SGLANG_ROOT" diff --cached --quiet; then
  record_status "FAIL SGLang worktree differs from committed HEAD"
  exit 65
fi

SOURCE_FILES=(
  "$EVIDENCE_ROOT/run_tp4_live_diagnostic.sh"
  "$HELPER"
  "$EVIDENCE_ROOT/test_tp4_live_request.py"
  "$SGLANG_ROOT/python/sglang/launch_server.py"
  "$SGLANG_ROOT/python/sglang/srt/server_args.py"
  "$SGLANG_ROOT/python/sglang/srt/layers/attention/dsa/dsa_indexer.py"
  "$SGLANG_ROOT/python/sglang/srt/layers/attention/dsa_backend.py"
  "$SGLANG_ROOT/python/sglang/srt/layers/linear.py"
  "$SGLANG_ROOT/python/sglang/srt/layers/quantization/unquant.py"
  "$SGLANG_ROOT/python/sglang/srt/models/deepseek_common/deepseek_weight_loader.py"
  "$SGLANG_ROOT/python/sglang/srt/models/glm4_moe.py"
  "$SGLANG_ROOT/python/sglang/jit_kernel/dsv4/elementwise.py"
  "$SGLANG_ROOT/python/sglang/jit_kernel/dsv32/elementwise.py"
)
for source_file in "${SOURCE_FILES[@]}"; do
  if [[ ! -f "$source_file" ]]; then
    record_status "FAIL missing source-manifest input: $source_file"
    exit 65
  fi
done
sha256sum "${SOURCE_FILES[@]}" > "$RUN_DIR/source_manifest.sha256"

"$PY" - "$SGLANG_ROOT" "$KH_ROOT" "$RUN_DIR/package_origins.json" <<'PY'
import hashlib
import importlib.metadata
import importlib.util
import json
import sys
from pathlib import Path

sglang_root = Path(sys.argv[1]).resolve()
kh_root = Path(sys.argv[2]).resolve()
output = Path(sys.argv[3])
logical_venv = kh_root / ".venv"
if Path(sys.prefix) != logical_venv:
    raise SystemExit(
        f"diagnostic was not launched through the repo-local venv: "
        f"sys.prefix={sys.prefix}, expected={logical_venv}"
    )
runtime_prefix = logical_venv.resolve()
module_names = ("sglang", "deep_gemm", "sgl_kernel", "flashinfer", "torch", "tvm_ffi")
package_distributions = importlib.metadata.packages_distributions()
records = {}
for module_name in module_names:
    spec = importlib.util.find_spec(module_name)
    if spec is None or spec.origin is None:
        raise SystemExit(f"cannot resolve required module {module_name!r}")
    origin = Path(spec.origin).resolve()
    if not origin.is_file():
        raise SystemExit(f"module origin is not a file: {module_name} -> {origin}")
    if module_name == "sglang":
        expected_root = sglang_root / "python"
    else:
        # The repo-local .venv is a symlinked virtual environment on this host.
        # Compare canonical module origins with its canonical runtime prefix,
        # while the sys.prefix check above proves the logical launcher was the
        # required repo-local .venv rather than a directly invoked global Python.
        expected_root = runtime_prefix
    if not origin.is_relative_to(expected_root):
        raise SystemExit(
            f"module resolved outside pinned root: {module_name} -> {origin}, "
            f"expected under {expected_root}"
        )
    distributions = []
    for distribution_name in package_distributions.get(module_name, []):
        try:
            version = importlib.metadata.version(distribution_name)
        except importlib.metadata.PackageNotFoundError:
            version = None
        distributions.append({"name": distribution_name, "version": version})
    records[module_name] = {
        "distributions": distributions,
        "expected_canonical_root": str(expected_root),
        "origin": str(origin),
        "origin_sha256": hashlib.sha256(origin.read_bytes()).hexdigest(),
        "search_locations": [
            str(Path(location).resolve())
            for location in (spec.submodule_search_locations or [])
        ],
    }
output.write_text(json.dumps(records, indent=2, sort_keys=True) + "\n")
PY

{
  printf 'timestamp_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  printf 'run_id=%s\n' "$RUN_ID"
  printf 'base_url=%s\n' "$BASE_URL"
  printf 'model_path=%s\n' "$MODEL_PATH"
  printf 'model_revision=%s\n' "$MODEL_REVISION"
  printf 'cuda_visible_devices=%s\n' "$CUDA_VISIBLE_DEVICES"
  printf 'wrapper_locks=%s\n' "${WRAPPER_LOCK_RECORDS[*]}"
  printf 'PYTHONNOUSERSITE=%s\n' "$PYTHONNOUSERSITE"
  printf 'PYTHONSAFEPATH=%s\n' "$PYTHONSAFEPATH"
  printf 'CUDA_DEVICE_ORDER=%s\n' "$CUDA_DEVICE_ORDER"
  printf 'CUDA_LAUNCH_BLOCKING=%s\n' "${CUDA_LAUNCH_BLOCKING-unset}"
  printf 'CUDA_DEVICE_MAX_CONNECTIONS=%s\n' \
    "${CUDA_DEVICE_MAX_CONNECTIONS-unset}"
  printf 'SGLANG_GLM52_OPT=%s\n' "$SGLANG_GLM52_OPT"
  printf 'SGLANG_GLM52_ENV_FILE=%s\n' "$SGLANG_GLM52_ENV_FILE"
  printf 'SGLANG_GLM52_NSYS_GATE=%s\n' "$SGLANG_GLM52_NSYS_GATE"
  printf 'SGLANG_PROFILE_V2=%s\n' "$SGLANG_PROFILE_V2"
  printf 'SGLANG_SIMULATE_ROUND_ROBIN_EXPERTS=%s\n' \
    "$SGLANG_SIMULATE_ROUND_ROBIN_EXPERTS"
  printf 'kh_sha=%s\n' "$(git -C "$KH_ROOT" rev-parse HEAD)"
  printf 'sglang_sha=%s\n' "$(git -C "$SGLANG_ROOT" rev-parse HEAD)"
  printf 'python=%s\n' "$($PY --version 2>&1)"
  printf 'nsys=%s\n' "$(nsys --version 2>&1 | head -n 1)"
  printf '\nGPU inventory at launch:\n'
  nvidia-smi --query-gpu=index,uuid,name,pstate,clocks.current.sm,clocks.current.memory,power.draw,memory.used,memory.free \
    --format=csv
  printf '\nGPU topology at launch:\n'
  nvidia-smi topo -m
  printf '\nKernel-Harness status:\n'
  git -C "$KH_ROOT" status --short
  printf '\nSGLang status:\n'
  git -C "$SGLANG_ROOT" status --short
} > "$RUN_DIR/environment.txt" 2>&1

NSYS_COMMAND=(
  nsys profile
  --force-overwrite=true
  --trace=cuda,nvtx,cublas,nccl
  --sample=none
  -c cudaProfilerApi
  --capture-range-end=stop
  --kill=none
  -o "$NSYS_BASE"
  "$PY" -m sglang.launch_server
  --model-path "$MODEL_PATH"
  --revision "$MODEL_REVISION"
  --tp-size 4
  --dp-size 4
  --ep-size 4
  --enable-dp-attention
  --moe-a2a-backend deepep
  --load-format dummy
  --quantization modelopt_fp4
  --kv-cache-dtype fp8_e4m3
  --attention-backend dsa
  --dsa-prefill-backend trtllm
  --dsa-topk-backend sgl-kernel
  --trust-remote-code
  --skip-tokenizer-init
  --disable-flashinfer-autotune
  --cuda-graph-backend-prefill disabled
  --cuda-graph-backend-decode disabled
  --disable-radix-cache
  --chunked-prefill-size 16384
  --max-prefill-tokens 4096
  --max-total-tokens 8192
  --context-length 8192
  --max-running-requests 4
  --prefill-max-requests 1
  --page-size 64
  --mem-fraction-static 0.80
  --enable-layerwise-nvtx-marker
  --host "$HOST"
  --port "$PORT"
)

{
  printf '# Run only inside with_all_gpus_lock.sh.\n'
  printf '%q ' "${NSYS_COMMAND[@]}"
  printf '\n'
} > "$RUN_DIR/launch_command.sh"

record_status "START TP4 dummy-weight eager diagnostic base_url=$BASE_URL"
setsid "${NSYS_COMMAND[@]}" > "$RUN_DIR/server.log" 2>&1 &
SERVER_PID=$!
SERVER_PGID=$SERVER_PID
printf '%s\n' "$SERVER_PID" > "$RUN_DIR/nsys_process_group.txt"

# Verify that setsid established an isolated group before recording it as a
# cleanup target.  Never signal the caller's group if this invariant fails.
sleep 1
ACTUAL_PGID=$(ps -o pgid= -p "$SERVER_PID" 2>/dev/null | tr -d '[:space:]')
if [[ -z "$ACTUAL_PGID" ]]; then
  record_status "FAIL nsys leader exited before process-group verification"
  # A leader may exit while children remain in its established session. The
  # negative PGID comes from our freshly-created child PID, never the caller's
  # pre-existing group, so retain it for scoped cleanup.
  stop_server_group
  exit 1
elif [[ "$ACTUAL_PGID" != "$SERVER_PGID" ]]; then
  record_status "FAIL setsid invariant: pid=$SERVER_PID pgid=${ACTUAL_PGID:-missing}"
  kill -TERM "$SERVER_PID" 2>/dev/null || true
  wait "$SERVER_PID" 2>/dev/null || true
  SERVER_PID=""
  SERVER_PGID=""
  exit 1
fi
record_status "TRACK nsys process group $SERVER_PGID"

"$PY" "$HELPER" wait-health \
  --base-url "$BASE_URL" \
  --process-pid "$SERVER_PID" \
  --timeout-s "$STARTUP_TIMEOUT_S" \
  --output "$RUN_DIR/health_response.json"
record_status "HEALTH health_generate ready"

"$PY" "$HELPER" capture-server-info \
  --base-url "$BASE_URL" \
  --expected-model "$MODEL_PATH" \
  --expected-revision "$MODEL_REVISION" \
  --output "$RUN_DIR/server_info_response.json"
record_status "CONFIG resolved TP4/DP4/EP4 dummy eager settings validated"

PROFILE_MAY_BE_ACTIVE=1
"$PY" "$HELPER" run-profiled-request \
  --base-url "$BASE_URL" \
  --output-dir "$RUN_DIR" \
  --generate-timeout-s "$GENERATE_TIMEOUT_S"
PROFILE_MAY_BE_ACTIVE=0
record_status "CAPTURE four concurrent deterministic local-M4096 requests completed and profiler stopped"

stop_server_group
sleep 2
if [[ ! -s "$NSYS_BASE.nsys-rep" ]]; then
  record_status "FAIL nsys did not finalize $NSYS_BASE.nsys-rep"
  exit 1
fi
record_status "ARTIFACT nsys report $NSYS_BASE.nsys-rep"

nsys stats --force-export=true \
  --report cuda_gpu_trace:nvtx-name:base \
  --format csv --output "$RUN_DIR/nsys-trace" \
  "$NSYS_BASE.nsys-rep" > "$RUN_DIR/nsys_trace_console.txt" 2>&1
TRACE_CSV="$RUN_DIR/nsys-trace_cuda_gpu_trace_nvtx-name_base.csv"
if [[ ! -s "$TRACE_CSV" ]]; then
  record_status "FAIL mandatory CUDA GPU trace CSV is missing"
  exit 1
fi
"$PY" "$HELPER" analyze-trace \
  --csv "$TRACE_CSV" \
  --output "$RUN_DIR/trace_reachability.json" \
  --expected-devices 4
record_status "REACH expected M4096 fused Q/K kernels observed on distinct streams across four devices"

if nsys stats --force-export=true \
  --report nvtx_gpu_proj_sum \
  --format csv --output "$RUN_DIR/nsys-nvtx" \
  "$NSYS_BASE.nsys-rep" > "$RUN_DIR/nsys_nvtx_stats_console.txt" 2>&1; then
  record_status "ARTIFACT nsys NVTX GPU projection summary exported"
else
  record_status "WARN CUDA trace passed but NVTX projection summary export failed"
fi
record_status "SCOPE TP4 DP4 EP4 dummy NVFP4 eager M4096 reachability only; not TP8/DP8/EP8 acceptance"
