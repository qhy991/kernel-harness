#!/usr/bin/env bash
# Invoke only through /home/qinhaiyan/glm52-goal-runs/with_flexible_gpu.sh -- ...
# All A/B series and their matched profiler captures stay inside that one lock.
set -Eeuo pipefail

KH_ROOT=/home/qinhaiyan/glm52-goal-runs/05-indexer_k_weights_prefill/kernel-harness
SGLANG_ROOT=/home/qinhaiyan/glm52-goal-runs/05-indexer_k_weights_prefill/sglang
EVIDENCE_ROOT="$KH_ROOT/evidence/glm52_prod_05_indexer_k_weights_prefill"
PROFILE_HELPER="$KH_ROOT/profile/indexer-wk-weights-prefill-m4096-20260722/harness/profile_indexer_region.py"
VALIDATOR="$EVIDENCE_ROOT/validate_hardened_campaign.py"
JIT_KERNEL_ROOT="$SGLANG_ROOT/python/sglang/jit_kernel"
PY="$KH_ROOT/.venv/bin/python"
SN="$KH_ROOT/serving_native/run.sh"
ISOLATED=indexer_wk_weights_prefill_m4096
REGION=indexer_fused_prepare_store_prefill_m4096_eager_dual_stream
MODEL_REVISION=aec724e8c7b8ee9db3b48c01c320f63f9cdaf8aa
RUN_ID=${HARDENED_RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)-$$}

# Resolve imports under the exact campaign path from the first Python process;
# never inherit a caller-provided module search path.
export PYTHONPATH="$SGLANG_ROOT/python:$KH_ROOT"
export PYTHONNOUSERSITE=1
export PYTHONSAFEPATH=1
export CUDA_DEVICE_ORDER=PCI_BUS_ID
unset PYTHONSTARTUP
unset CUDA_LAUNCH_BLOCKING
unset CUDA_DEVICE_MAX_CONNECTIONS
unset CUDA_CACHE_DISABLE
unset CUDA_FORCE_PTX_JIT
unset CUBLAS_WORKSPACE_CONFIG
unset NVIDIA_TF32_OVERRIDE
unset TORCH_ALLOW_TF32_CUBLAS_OVERRIDE

if [[ ! "$RUN_ID" =~ ^[A-Za-z0-9_.-]+$ ]]; then
  printf 'HARDENED_RUN_ID contains unsafe path characters: %s\n' "$RUN_ID" >&2
  exit 64
fi
if [[ ! "${CUDA_VISIBLE_DEVICES:-}" =~ ^[0-9]+$ ]]; then
  printf 'expected one wrapper-selected physical GPU in CUDA_VISIBLE_DEVICES\n' >&2
  exit 64
fi
EXPECTED_LOCK=/home/qinhaiyan/glm52-goal-runs/locks/gpu${CUDA_VISIBLE_DEVICES}.lock
EXPECTED_LOCK_CANONICAL=$(readlink -f "$EXPECTED_LOCK")
WRAPPER_LOCK_FD=""
for fd_path in /proc/$$/fd/*; do
  if [[ "$(readlink -f "$fd_path" 2>/dev/null || true)" == \
        "$EXPECTED_LOCK_CANONICAL" ]]; then
    WRAPPER_LOCK_FD=${fd_path##*/}
    break
  fi
done
if [[ -z "$WRAPPER_LOCK_FD" ]]; then
  printf 'required flexible-GPU wrapper lock is not inherited: %s\n' \
    "$EXPECTED_LOCK" >&2
  exit 64
fi
for command in cmp cut find nsys nvidia-smi sha256sum sort xargs git; do
  command -v "$command" >/dev/null 2>&1 || {
    printf 'missing required command: %s\n' "$command" >&2
    exit 64
  }
done
[[ -x "$PY" ]] || {
  printf 'missing repo-local Python: %s\n' "$PY" >&2
  exit 64
}
PY_PREFIX=$(
  "$PY" -c 'from pathlib import Path; import sys; print(Path(sys.prefix).resolve())'
)
SG_KERNEL_ROOT=$(
  "$PY" -c \
    'import importlib.util; from pathlib import Path; s=importlib.util.find_spec("sgl_kernel"); assert s and s.submodule_search_locations; print(Path(next(iter(s.submodule_search_locations))).resolve())'
)
case "$SG_KERNEL_ROOT" in
  "$PY_PREFIX"/*) ;;
  *)
    printf 'sgl_kernel did not resolve under the campaign Python prefix: %s\n' \
      "$SG_KERNEL_ROOT" >&2
    exit 65
    ;;
esac
SG_KERNEL_INIT="$SG_KERNEL_ROOT/__init__.py"
SG_KERNEL_ELEMENTWISE="$SG_KERNEL_ROOT/elementwise.py"
SG_KERNEL_SM100_COMMON_OPS="$SG_KERNEL_ROOT/sm100/common_ops.abi3.so"
for resolved_file in \
  "$SG_KERNEL_INIT" "$SG_KERNEL_ELEMENTWISE" "$SG_KERNEL_SM100_COMMON_OPS"; do
  [[ -f "$resolved_file" ]] || {
    printf 'missing resolved sgl_kernel file: %s\n' "$resolved_file" >&2
    exit 65
  }
done

KH_TRACKED_FILES=(
  "$KH_ROOT/serving_native/runner.py"
  "$KH_ROOT/serving_native/workloads.py"
  "$KH_ROOT/serving_native/launch.py"
  "$KH_ROOT/serving_native/run.sh"
  "$KH_ROOT/serving_native/selftest.py"
  "$KH_ROOT/serving_native/candidates/reference.py"
  "$KH_ROOT/serving_native/candidates/indexer_wk_torch_mm.py"
  "$KH_ROOT/serving_native/candidates/indexer_wk_cutedsl_tgv.py"
  "$KH_ROOT/serving_native/candidates/indexer_single_stream.py"
  "$PROFILE_HELPER"
  "$VALIDATOR"
  "$EVIDENCE_ROOT/run_hardened_validation_campaign.sh"
)
for source_file in "${KH_TRACKED_FILES[@]}"; do
  git -C "$KH_ROOT" ls-files --error-unmatch -- "${source_file#"$KH_ROOT"/}" \
    >/dev/null || {
      printf 'hardened source is not tracked: %s\n' "$source_file" >&2
      exit 65
    }
done
for required in "$PY" "$SN" "$PROFILE_HELPER" "$VALIDATOR"; do
  [[ -f "$required" ]] || {
    printf 'missing required file: %s\n' "$required" >&2
    exit 64
  }
done

# The complete importable worktree must be committed and immutable. Runtime
# artifacts become the only allowlisted untracked paths after this point.
if ! KH_PREFLIGHT_STATUS=$(git -C "$KH_ROOT" status --porcelain=v1 --untracked-files=all); then
  printf 'cannot audit Kernel-Harness worktree status\n' >&2
  exit 65
fi
if [[ -n "$KH_PREFLIGHT_STATUS" ]]; then
  printf 'refusing hardened campaign with dirty Kernel-Harness worktree\n' >&2
  exit 65
fi
if ! SGLANG_PREFLIGHT_STATUS=$(git -C "$SGLANG_ROOT" status --porcelain=v1 --untracked-files=all); then
  printf 'cannot audit SGLang worktree status\n' >&2
  exit 65
fi
if [[ -n "$SGLANG_PREFLIGHT_STATUS" ]]; then
  printf 'refusing hardened campaign with dirty SGLang worktree\n' >&2
  exit 65
fi

KH_STATUS_BEFORE=$(git -C "$KH_ROOT" status --porcelain=v1 --untracked-files=all)
SGLANG_STATUS_BEFORE=$(git -C "$SGLANG_ROOT" status --porcelain=v1 --untracked-files=all)
KH_HEAD_BEFORE=$(git -C "$KH_ROOT" rev-parse HEAD)
SGLANG_HEAD_BEFORE=$(git -C "$SGLANG_ROOT" rev-parse HEAD)
RUN_PARENT="$EVIDENCE_ROOT/hardened_runs"
RUN_DIR="$RUN_PARENT/$RUN_ID"
RUN_REL=${RUN_DIR#"$KH_ROOT"/}
mkdir -p "$RUN_PARENT"
if ! mkdir "$RUN_DIR"; then
  printf 'refusing to overwrite hardened run: %s\n' "$RUN_DIR" >&2
  exit 73
fi
mkdir "$RUN_DIR/logs" "$RUN_DIR/profiles" "$RUN_DIR/results" "$RUN_DIR/source"
mkdir "$RUN_DIR/tvm_ffi_cache"

STATUS_FILE="$RUN_DIR/status.txt"
SOURCE_CHECK="$RUN_DIR/source_manifest_check.txt"
FINALIZED=0

finalize_artifacts() {
  local disposition=$1
  local exit_code=$2
  local failed=0
  printf '%s\tdisposition=%s exit_code=%s\n' \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$disposition" "$exit_code" \
    >> "$STATUS_FILE" || failed=1
  local kh_final_head=""
  local sglang_final_head=""
  kh_final_head=$(git -C "$KH_ROOT" rev-parse HEAD) || failed=1
  sglang_final_head=$(git -C "$SGLANG_ROOT" rev-parse HEAD) || failed=1
  if ! {
    printf 'kh_head=%s\n' "$kh_final_head"
    printf 'sglang_head=%s\n' "$sglang_final_head"
    printf 'kh_head_unchanged=%s\n' \
      "$([[ "$kh_final_head" == "$KH_HEAD_BEFORE" ]] && echo true || echo false)"
    printf 'sglang_head_unchanged=%s\n' \
      "$([[ "$sglang_final_head" == "$SGLANG_HEAD_BEFORE" ]] && echo true || echo false)"
    if [[ "$kh_final_head" != "$KH_HEAD_BEFORE" || \
          "$sglang_final_head" != "$SGLANG_HEAD_BEFORE" ]]; then
      failed=1
    fi
    printf 'kh_tracked_diff_clean='
    if git -C "$KH_ROOT" diff --quiet && git -C "$KH_ROOT" diff --cached --quiet; then
      printf 'true\n'
    else
      printf 'false\n'
      failed=1
    fi
    printf 'sglang_status_porcelain='
    local sglang_final_status
    if ! sglang_final_status=$(git -C "$SGLANG_ROOT" status --porcelain=v1 --untracked-files=all); then
      printf 'status-command-failed\n'
      failed=1
      sglang_final_status=status-command-failed
    fi
    if [[ -z "$sglang_final_status" ]]; then
      printf 'clean\n'
    else
      printf 'dirty\n%s\n' "$sglang_final_status"
      failed=1
    fi
    printf 'kh_untracked_allowlist='
    local kh_final_status
    if ! kh_final_status=$(git -C "$KH_ROOT" status --porcelain=v1 --untracked-files=all); then
      printf 'status-command-failed\n'
      failed=1
    else
      local status_line
      local unexpected_status=0
      while IFS= read -r status_line; do
        [[ -z "$status_line" ]] && continue
        if [[ "$status_line" == "?? $RUN_REL/"* ]]; then
          continue
        fi
        printf '\nunexpected: %s' "$status_line"
        unexpected_status=1
      done <<< "$kh_final_status"
      if (( unexpected_status )); then
        printf '\ninvalid\n'
        failed=1
      else
        printf 'pass\n'
      fi
    fi
  } > "$RUN_DIR/final_repository_state.txt" 2>&1; then
    failed=1
  fi
  if [[ -s "$RUN_DIR/source_manifest.sha256" ]]; then
    sha256sum -c "$RUN_DIR/source_manifest.sha256" > "$SOURCE_CHECK" 2>&1 \
      || failed=1
  else
    failed=1
  fi
  find "$RUN_DIR" -type f ! -name artifact_manifest.sha256 -print0 \
    | sort -z | xargs -0 sha256sum > "$RUN_DIR/artifact_manifest.sha256" \
      || failed=1
  return "$failed"
}

on_exit() {
  local exit_code=$?
  trap - EXIT INT TERM HUP
  if (( ! FINALIZED )); then
    set +e
    finalize_artifacts FAIL "$exit_code" || true
  fi
  exit "$exit_code"
}
trap on_exit EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
trap 'exit 129' HUP

export SGLANG_ROOT
export SGLANG_DIR="$SGLANG_ROOT"
export KERNEL_HARNESS_PYTHON="$PY"
export SGLANG_GLM52_OPT=0
GLM52_EMPTY_ENV="$RUN_DIR/glm52_opt.empty.env"
: > "$GLM52_EMPTY_ENV"
export SGLANG_GLM52_ENV_FILE="$GLM52_EMPTY_ENV"
export SGLANG_GLM52_NSYS_GATE=0
export TVM_FFI_CACHE_DIR="$RUN_DIR/tvm_ffi_cache"
export RANK=0
export LOCAL_RANK=0
export WORLD_SIZE=1
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

cp -p "$KH_ROOT/serving_native/candidates/reference.py" "$RUN_DIR/source/reference.py"
cp -p "$KH_ROOT/serving_native/candidates/indexer_wk_torch_mm.py" \
  "$RUN_DIR/source/indexer_wk_torch_mm.py"
cp -p "$KH_ROOT/serving_native/candidates/indexer_wk_cutedsl_tgv.py" \
  "$RUN_DIR/source/indexer_wk_cutedsl_tgv.py"
cp -p "$KH_ROOT/serving_native/candidates/indexer_single_stream.py" \
  "$RUN_DIR/source/indexer_single_stream.py"
chmod a-w "$RUN_DIR/source"/*.py

cmp -s "$KH_ROOT/serving_native/candidates/reference.py" \
  "$RUN_DIR/source/reference.py"
cmp -s "$KH_ROOT/serving_native/candidates/indexer_wk_torch_mm.py" \
  "$RUN_DIR/source/indexer_wk_torch_mm.py"
cmp -s "$KH_ROOT/serving_native/candidates/indexer_wk_cutedsl_tgv.py" \
  "$RUN_DIR/source/indexer_wk_cutedsl_tgv.py"
cmp -s "$KH_ROOT/serving_native/candidates/indexer_single_stream.py" \
  "$RUN_DIR/source/indexer_single_stream.py"

IDENTITY="$RUN_DIR/source/reference.py"
TORCH_MM="$RUN_DIR/source/indexer_wk_torch_mm.py"
TGV="$RUN_DIR/source/indexer_wk_cutedsl_tgv.py"
SINGLE_STREAM="$RUN_DIR/source/indexer_single_stream.py"

SOURCE_FILES=(
  "${KH_TRACKED_FILES[@]}"
  "$IDENTITY"
  "$TORCH_MM"
  "$TGV"
  "$SINGLE_STREAM"
  "$GLM52_EMPTY_ENV"
  "$SGLANG_ROOT/python/sglang/srt/layers/attention/dsa/dsa_indexer.py"
  "$SGLANG_ROOT/python/sglang/srt/layers/attention/dsa/index_buf_accessor.py"
  "$SGLANG_ROOT/python/sglang/srt/layers/attention/dsa/triton_kernel.py"
  "$SGLANG_ROOT/python/sglang/srt/layers/glm52_opt/config.py"
  "$SGLANG_ROOT/python/sglang/jit_kernel/cutedsl_bf16_gemm.py"
  "$JIT_KERNEL_ROOT/utils.py"
  "$JIT_KERNEL_ROOT/dsv4/__init__.py"
  "$JIT_KERNEL_ROOT/dsv4/elementwise.py"
  "$JIT_KERNEL_ROOT/dsv4/utils.py"
  "$JIT_KERNEL_ROOT/csrc/deepseek_v4/main_norm_rope.cuh"
  "$JIT_KERNEL_ROOT/dsv32/__init__.py"
  "$SGLANG_ROOT/python/sglang/jit_kernel/dsv32/elementwise.py"
  "$SGLANG_ROOT/python/sglang/jit_kernel/csrc/deepseek_v32/indexer_k.cuh"
  "$SGLANG_ROOT/sgl-kernel/csrc/elementwise/dsv4_norm_rope.cu"
  "$SGLANG_ROOT/sgl-kernel/python/sgl_kernel/elementwise.py"
  "$SG_KERNEL_INIT"
  "$SG_KERNEL_ELEMENTWISE"
  "$SG_KERNEL_SM100_COMMON_OPS"
  "$SGLANG_ROOT/python/sglang/srt/layers/linear.py"
  "$SGLANG_ROOT/python/sglang/srt/layers/quantization/unquant.py"
  "$SGLANG_ROOT/python/sglang/srt/layers/quantization/modelopt_quant.py"
  "$SGLANG_ROOT/python/sglang/srt/layers/layernorm.py"
  "$SGLANG_ROOT/python/sglang/srt/layers/rotary_embedding/base.py"
  "$SGLANG_ROOT/python/sglang/srt/layers/rotary_embedding/factory.py"
  "$SGLANG_ROOT/python/sglang/srt/model_executor/forward_context.py"
)
# _local_jit_source_hash recursively consumes sgl_kernel/* includes. Hash the
# complete small local include tree so no reachable header can drift between
# the source manifest and the produced TVM-FFI modules.
JIT_HEADER_LIST="$RUN_DIR/jit_header_files.list0"
find "$JIT_KERNEL_ROOT/include/sgl_kernel" -type f -print0 \
  | sort -z > "$JIT_HEADER_LIST"
JIT_HEADER_COUNT=0
while IFS= read -r -d '' jit_header; do
  SOURCE_FILES+=("$jit_header")
  (( JIT_HEADER_COUNT += 1 ))
done < "$JIT_HEADER_LIST"
(( JIT_HEADER_COUNT > 0 )) || {
  printf 'no local JIT include headers found\n' >&2
  exit 65
}
for source_file in "${SOURCE_FILES[@]}"; do
  [[ -f "$source_file" ]] || {
    printf 'missing source file: %s\n' "$source_file" >&2
    exit 64
  }
done
cmp -s "$SGLANG_ROOT/sgl-kernel/python/sgl_kernel/elementwise.py" \
  "$SG_KERNEL_ELEMENTWISE" || {
  printf 'installed sgl_kernel elementwise.py differs from the pinned source tree\n' >&2
  exit 65
}
sha256sum "${SOURCE_FILES[@]}" > "$RUN_DIR/source_manifest.sha256"
for pair in \
  "$KH_ROOT/serving_native/candidates/reference.py:$IDENTITY" \
  "$KH_ROOT/serving_native/candidates/indexer_wk_torch_mm.py:$TORCH_MM" \
  "$KH_ROOT/serving_native/candidates/indexer_wk_cutedsl_tgv.py:$TGV" \
  "$KH_ROOT/serving_native/candidates/indexer_single_stream.py:$SINGLE_STREAM"; do
  original=${pair%%:*}
  snapshot=${pair#*:}
  [[ "$(sha256sum "$original" | cut -d' ' -f1)" == \
     "$(sha256sum "$snapshot" | cut -d' ' -f1)" ]] || {
    printf 'candidate snapshot hash mismatch: %s\n' "$original" >&2
    exit 65
  }
done

{
  printf 'timestamp_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  printf 'run_id=%s\n' "$RUN_ID"
  printf 'fixed_model_revision=%s\n' "$MODEL_REVISION"
  printf 'selected_physical_gpu=%s\n' "$CUDA_VISIBLE_DEVICES"
  printf 'wrapper_lock_path=%s\n' "$EXPECTED_LOCK"
  printf 'wrapper_lock_canonical_path=%s\n' "$EXPECTED_LOCK_CANONICAL"
  printf 'wrapper_lock_fd=%s\n' "$WRAPPER_LOCK_FD"
  printf 'kh_head=%s\n' "$KH_HEAD_BEFORE"
  printf 'kh_branch=%s\n' "$(git -C "$KH_ROOT" branch --show-current)"
  printf 'sglang_head=%s\n' "$SGLANG_HEAD_BEFORE"
  printf 'sglang_branch=%s\n' "$(git -C "$SGLANG_ROOT" branch --show-current)"
  printf 'python_realpath=%s\n' "$(readlink -f "$PY")"
  printf 'python_prefix=%s\n' "$PY_PREFIX"
  printf 'python_version=%s\n' "$($PY --version 2>&1)"
  printf 'PYTHONNOUSERSITE=%s\n' "$PYTHONNOUSERSITE"
  printf 'PYTHONSAFEPATH=%s\n' "$PYTHONSAFEPATH"
  printf 'CUDA_DEVICE_ORDER=%s\n' "$CUDA_DEVICE_ORDER"
  printf 'CUDA_LAUNCH_BLOCKING=%s\n' "${CUDA_LAUNCH_BLOCKING-unset}"
  printf 'CUDA_DEVICE_MAX_CONNECTIONS=%s\n' \
    "${CUDA_DEVICE_MAX_CONNECTIONS-unset}"
  printf 'CUDA_CACHE_DISABLE=%s\n' "${CUDA_CACHE_DISABLE-unset}"
  printf 'CUDA_FORCE_PTX_JIT=%s\n' "${CUDA_FORCE_PTX_JIT-unset}"
  printf 'CUBLAS_WORKSPACE_CONFIG=%s\n' "${CUBLAS_WORKSPACE_CONFIG-unset}"
  printf 'NVIDIA_TF32_OVERRIDE=%s\n' "${NVIDIA_TF32_OVERRIDE-unset}"
  printf 'TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=%s\n' \
    "${TORCH_ALLOW_TF32_CUBLAS_OVERRIDE-unset}"
  printf 'sgl_kernel_root=%s\n' "$SG_KERNEL_ROOT"
  printf 'sgl_kernel_init=%s\n' "$SG_KERNEL_INIT"
  printf 'sgl_kernel_elementwise=%s\n' "$SG_KERNEL_ELEMENTWISE"
  printf 'sgl_kernel_sm100_common_ops=%s\n' "$SG_KERNEL_SM100_COMMON_OPS"
  printf 'sgl_kernel_elementwise_matches_source=true\n'
  printf 'nsys_version=%s\n' "$(nsys --version 2>&1 | head -n 1)"
  printf 'SGLANG_ROOT=%s\n' "$SGLANG_ROOT"
  printf 'SGLANG_GLM52_OPT=%s\n' "$SGLANG_GLM52_OPT"
  printf 'SGLANG_GLM52_ENV_FILE=%s\n' "$SGLANG_GLM52_ENV_FILE"
  printf 'SGLANG_GLM52_NSYS_GATE=%s\n' "$SGLANG_GLM52_NSYS_GATE"
  printf 'TVM_FFI_CACHE_DIR=%s\n' "$TVM_FFI_CACHE_DIR"
  printf 'RANK=%s LOCAL_RANK=%s WORLD_SIZE=%s\n' "$RANK" "$LOCAL_RANK" "$WORLD_SIZE"
  printf 'SGLANG_DISABLE_DSA_INDEXER_FUSION=%s\n' \
    "${SGLANG_DISABLE_DSA_INDEXER_FUSION-unset}"
  printf 'SGLANG_NVFP4_CKPT_FP8_GEMM_IN_ATTN=%s\n' \
    "${SGLANG_NVFP4_CKPT_FP8_GEMM_IN_ATTN-unset}"
  printf '\nKernel-Harness status before artifact creation:\n%s\n' "$KH_STATUS_BEFORE"
  printf '\nSGLang status before artifact creation:\n%s\n' "$SGLANG_STATUS_BEFORE"
  printf '\nSelected GPU identity and clocks:\n'
  nvidia-smi -i "$CUDA_VISIBLE_DEVICES" \
    --query-gpu=index,uuid,name,pstate,clocks.current.sm,clocks.current.memory,power.draw \
    --format=csv
} > "$RUN_DIR/environment.txt" 2>&1
"$PY" - "$SGLANG_ROOT" "$PY_PREFIX" > "$RUN_DIR/module_origins.json" <<'PY'
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

sglang_root = Path(sys.argv[1]).resolve()
python_prefix = Path(sys.argv[2]).resolve()
records = {}
for name in ("cutlass", "sgl_kernel", "sglang", "torch", "tvm_ffi"):
    spec = importlib.util.find_spec(name)
    if spec is None or spec.origin is None:
        raise RuntimeError(f"cannot resolve required module: {name}")
    origin = Path(spec.origin).resolve()
    if not origin.is_file():
        raise RuntimeError(f"module origin is not a file: {name}: {origin}")
    expected_root = sglang_root / "python" if name == "sglang" else python_prefix
    if not origin.is_relative_to(expected_root):
        raise RuntimeError(
            f"module resolved outside expected root: {name}: {origin}: {expected_root}"
        )
    records[name] = {
        "origin": str(origin),
        "sha256": hashlib.sha256(origin.read_bytes()).hexdigest(),
    }
print(json.dumps(records, indent=2, sort_keys=True))
PY
git -C "$KH_ROOT" diff --binary > "$RUN_DIR/kernel_harness_unstaged.patch"
git -C "$KH_ROOT" diff --cached --binary > "$RUN_DIR/kernel_harness_staged.patch"
git -C "$SGLANG_ROOT" diff --binary > "$RUN_DIR/sglang_unstaged.patch"
"$PY" -m pip freeze > "$RUN_DIR/pip_freeze.txt"
"$PY" "$KH_ROOT/testbench/bin/check_env.py" > "$RUN_DIR/check_env.txt" 2>&1
printf '%s\tSTART immutable same-GPU validation\n' \
  "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$STATUS_FILE"

run_reference() {
  local workload=$1
  local stem=$2
  "$SN" "$workload" --warmup 10 --repeat 30 \
    --output "$RUN_DIR/results/$stem.json" > "$RUN_DIR/logs/$stem.log" 2>&1
}

run_candidate() {
  local workload=$1
  local candidate=$2
  local stem=$3
  "$SN" "$workload" --candidate "$candidate" --warmup 10 --repeat 60 \
    --output "$RUN_DIR/results/$stem.json" > "$RUN_DIR/logs/$stem.log" 2>&1
}

for run in 01 02 03; do
  run_reference "$ISOLATED" "isolated_stock_$run"
  run_reference "$REGION" "region_stock_$run"
done
run_candidate "$ISOLATED" "$IDENTITY" isolated_identity
run_candidate "$REGION" "$IDENTITY" region_identity

for run in 01 02 03; do
  run_candidate "$ISOLATED" "$TORCH_MM" "isolated_torch_mm_$run"
  run_candidate "$REGION" "$TORCH_MM" "region_torch_mm_$run"
  run_candidate "$ISOLATED" "$TGV" "isolated_tgv_$run"
  run_candidate "$REGION" "$TGV" "region_tgv_$run"
  run_candidate "$REGION" "$SINGLE_STREAM" "region_single_stream_$run"
done

profile_one() {
  local tag=$1
  local candidate=${2:-}
  local candidate_args=()
  if [[ -n "$candidate" ]]; then
    candidate_args=(--candidate "$candidate")
  fi
  nsys profile --force-overwrite=false --trace=cuda,nvtx,cublas --sample=none \
    -c cudaProfilerApi --capture-range-end=stop --kill=none \
    -o "$RUN_DIR/profiles/nsys-$tag" \
    "$PY" "$PROFILE_HELPER" "${candidate_args[@]}" --warmup 10 \
    --iterations 1 --cuda-profiler-api \
    --trace-output "$RUN_DIR/profiles/abi-$tag.json" \
    > "$RUN_DIR/logs/profile-$tag.log" 2>&1
  nsys stats --force-export=true \
    --report cuda_gpu_trace:nvtx-name:base,nvtx_gpu_proj_sum \
    --format csv --output "$RUN_DIR/profiles/nsys-$tag" \
    "$RUN_DIR/profiles/nsys-$tag.nsys-rep" \
    > "$RUN_DIR/logs/profile-$tag-stats.log" 2>&1
}

profile_one stock
profile_one torch-mm "$TORCH_MM"
profile_one single-stream "$SINGLE_STREAM"

JIT_ARTIFACT_LIST="$RUN_DIR/jit_artifact_files.list0"
find "$TVM_FFI_CACHE_DIR" -type f -name '*.so' -print0 \
  | sort -z > "$JIT_ARTIFACT_LIST"
mapfile -d '' -t JIT_ARTIFACTS < "$JIT_ARTIFACT_LIST"
(( ${#JIT_ARTIFACTS[@]} > 0 )) || {
  printf 'no run-local TVM-FFI JIT artifacts were produced\n' >&2
  exit 66
}
JIT_ARTIFACT_PATHS=$(printf '%s\n' "${JIT_ARTIFACTS[@]}")
[[ "$JIT_ARTIFACT_PATHS" == *dpsk_v4_main_q_indexer_rope_first_quant* ]] || {
    printf 'missing fused-Q TVM-FFI artifact\n' >&2
    exit 66
  }
[[ "$JIT_ARTIFACT_PATHS" == *dpsk_v32_k_indexer_norm_rope_store_p64* ]] || {
    printf 'missing fused-K/store TVM-FFI artifact\n' >&2
    exit 66
  }
sha256sum "${JIT_ARTIFACTS[@]}" > "$RUN_DIR/jit_artifact_manifest.sha256"
sha256sum -c "$RUN_DIR/jit_artifact_manifest.sha256" \
  > "$RUN_DIR/jit_artifact_manifest_check.txt" 2>&1

{
  printf 'timestamp_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  nvidia-smi -i "$CUDA_VISIBLE_DEVICES" \
    --query-gpu=index,uuid,name,pstate,clocks.current.sm,clocks.current.memory,power.draw \
    --format=csv
} > "$RUN_DIR/environment_after.txt" 2>&1

sha256sum -c "$RUN_DIR/source_manifest.sha256" > "$SOURCE_CHECK" 2>&1
"$PY" "$VALIDATOR" \
  --run-dir "$RUN_DIR" \
  --repo "$KH_ROOT" \
  --output-json "$RUN_DIR/validation.json" \
  --output-md "$RUN_DIR/summary.md" \
  > "$RUN_DIR/logs/validator.log" 2>&1

PROMOTION_REQUIRED=$(
  "$PY" -c \
    'import json,sys; print("1" if json.load(open(sys.argv[1]))["stable_region_candidates_at_1_03x"] else "0")' \
    "$RUN_DIR/validation.json"
)
if [[ "$PROMOTION_REQUIRED" == 1 ]]; then
  DISPOSITION=PROMOTION_REQUIRED
  EXIT_CODE=42
else
  DISPOSITION=PASS_NO_REPLACEMENT_INNER_GATE
  EXIT_CODE=0
fi
if ! finalize_artifacts "$DISPOSITION" "$EXIT_CODE"; then
  printf 'final source/artifact integrity check failed\n' >&2
  exit 74
fi
FINALIZED=1
trap - EXIT INT TERM HUP
printf '%s\n' "$RUN_DIR"
exit "$EXIT_CODE"
