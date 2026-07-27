#!/usr/bin/env bash
set -uo pipefail

# The caller must hold with_all_gpus_lock.sh for this entire script. Every
# result is TP4/DP4/EP4 diagnostic evidence, never TP8/DP8/EP8 acceptance.
ROOT=/home/qinhaiyan/glm52-goal-runs/07-moe_w2_decode_kernel/kernel-harness
SGLANG=/home/qinhaiyan/glm52-goal-runs/07-moe_w2_decode_kernel/sglang
PY="$ROOT/.venv/bin/python"
OVERLAY="$SGLANG/build/deepep-overlay"
DEEP_GEMM_OVERLAY="$SGLANG/build/deep-gemm-stock-0.1.4.post1"
OUT_BASE="$ROOT/evidence/glm52_prod_07_moe_w2_decode_kernel/tp4_diagnostic"
ATTEMPT="${MOE_W2_TP4_ATTEMPT:-tp4_$(date -u +%Y%m%dT%H%M%SZ)_$$_${RANDOM}}"

if [[ "${CUDA_VISIBLE_DEVICES:-}" != "0,1,2,3" ]]; then
    echo "refusing TP4 campaign without the four-GPU wrapper" >&2
    exit 64
fi
if [[ ! "$ATTEMPT" =~ ^tp4_[A-Za-z0-9_.-]+$ ]]; then
    echo "invalid MOE_W2_TP4_ATTEMPT=$ATTEMPT" >&2
    exit 64
fi
if [[ ! -f "$OVERLAY/deep_ep/__init__.py" ]]; then
    echo "DeepEP overlay missing: $OVERLAY" >&2
    exit 64
fi

mkdir -p "$OUT_BASE"
OUT="$OUT_BASE/$ATTEMPT"
if ! mkdir "$OUT"; then
    echo "refusing to overwrite TP4 attempt: $OUT" >&2
    exit 64
fi
mkdir "$OUT/logs" "$OUT/results" "$OUT/profiles"
MANIFEST="$OUT/manifest.tsv"
printf 'label\texit_code\tlog\n' > "$MANIFEST"
export SGLANG_ROOT="$SGLANG"
export DEEP_EP_ROOT="$OVERLAY"
export KERNEL_HARNESS_PYTHON="$PY"
export TP4_ATTEMPT_DIR="$OUT"
export DEEP_GEMM_ROOT="$DEEP_GEMM_OVERLAY"
export EXPECTED_DEEP_GEMM_VERSION=0.1.4.post1
export EXPECTED_DEEP_GEMM_EXTENSION_SHA256=cd8beab174071777c972c5948af7706ae2cfb5d2adcdbb7e6fbea253ce3a81bf
export EXPECTED_DEEP_GEMM_DEVICE_SOURCE_SHA256=9c1e70677ede6ba09ab98e629482da7874182f8227907382efe0a81658da5a37
export PYTHONPATH="$OVERLAY:$DEEP_GEMM_OVERLAY:$SGLANG/python:$ROOT"
export SGLANG_GLM52_OPT=0
export SGLANG_DEEPGEMM_PDL=true
export SGLANG_JIT_DEEPGEMM_PRECOMPILE=0
export SGLANG_JIT_DEEPGEMM_FAST_WARMUP=0
export SGL_DG_USE_NVRTC=0
export DG_JIT_USE_NVRTC=0
export SGLANG_DEEPGEMM_SANITY_CHECK=0
export SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=128
export SGLANG_DG_CACHE_DIR="$OUT/deep_gemm_cache"
export DG_JIT_CACHE_DIR="$SGLANG_DG_CACHE_DIR"
export DG_JIT_WITH_LINEINFO=1
export DG_JIT_PTXAS_VERBOSE=1
export DG_JIT_DUMP_ASM=1
export DG_PRINT_CONFIGS=1
export DG_USE_NVIDIA_TOOLS=1
export PYTHONNOUSERSITE=1

failures=0
run_one() {
    local label="$1"
    shift
    local log="$OUT/logs/${label}.log"
    "$@" >"$log" 2>&1
    local status=$?
    printf '%s\t%s\t%s\n' "$label" "$status" "$log" >> "$MANIFEST"
    if [[ $status -ne 0 ]]; then
        failures=$((failures + 1))
    fi
    return 0
}

run_one runtime_probe \
    "$PY" "$ROOT/evidence/glm52_prod_07_moe_w2_decode_kernel/deepep_runtime_probe.py" \
    "$OUT/runtime_probe.json"
run_one topology nvidia-smi topo -m
run_one nvlink nvidia-smi nvlink --status
gpu_query='index,uuid,name,pci.bus_id,pstate,clocks.current.sm,clocks.current.memory,power.draw,temperature.gpu,driver_version'
run_one gpu_state_before nvidia-smi --query-gpu="$gpu_query" --format=csv,noheader,nounits

workloads=(
    ep4_deepep_ll_dispatch_decode_m16
    ep4_deepep_ll_combine_decode_m16
    ep4_deepep_ll_moe_region_decode_m16
    ep4_deepep_ll_dispatch_decode_m32
    ep4_deepep_ll_combine_decode_m32
    ep4_deepep_ll_moe_region_decode_m32
)
for trial in 1 2 3; do
    for workload in "${workloads[@]}"; do
        run_one "${workload}_trial${trial}" \
            "$ROOT/serving_native/run.sh" "$workload" \
            --warmup 3 --repeat 10 \
            --output "$OUT/results/${workload}_trial${trial}.json"
    done
done

for workload in "${workloads[@]}"; do
    run_one "nsys_${workload}" \
        nsys profile \
        --trace=cuda,nvtx,osrt,nccl \
        --cuda-event-trace=true \
        --sample=none \
        --cpuctxsw=none \
        --force-overwrite=true \
        --output "$OUT/profiles/${workload}" \
        "$ROOT/serving_native/run.sh" "$workload" \
        --warmup 1 --repeat 3 \
        --output "$OUT/results/nsys_${workload}.json"
done

run_one gpu_state_after nvidia-smi --query-gpu="$gpu_query" --format=csv,noheader,nounits

run_one validate_tp4 \
    "$PY" "$ROOT/evidence/glm52_prod_07_moe_w2_decode_kernel/validate_tp4_diagnostic.py" \
    "$OUT"

printf 'failures\t%s\n' "$failures" >> "$MANIFEST"
if [[ $failures -ne 0 ]]; then
    echo "TP4 diagnostic completed with $failures failed commands" >&2
    exit 2
fi
echo "TP4 diagnostic attempt completed: $ATTEMPT"
echo "artifacts: $OUT"
