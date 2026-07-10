#!/usr/bin/env bash
# Invoke sglang's OFFICIAL benchmark scripts for the kernels this harness covers.
# Requires sgl-kernel + deep_gemm.
#
# Usage: SGLANG_DIR=/path/to/sglang bash scripts/run_official_benchmarks.sh

set -euo pipefail
SGLANG_DIR="${SGLANG_DIR:-/mnt/public/qinhaiyan/sglang}"
cd "$SGLANG_DIR"

benchmarks=(
    # Dense FP8 GEMM (ops 1-5, 9-10, 14-16, 19)
    "benchmark/kernels/deepseek/benchmark_deepgemm_fp8_gemm.py"
    "benchmark/kernels/deepseek/benchmark_deepgemm_fp8_gemm_blackwell.py"

    # Grouped FP8 GEMM (ops 12, 13, 24, 25)
    "benchmark/kernels/deepseek/benchmark_deepgemm_fp8_group_gemm.py"

    # MoE fused Triton (Triton fallback for ops 12/13/24/25)
    "benchmark/kernels/fused_moe_triton/benchmark_sglang_fused_moe_triton.py"

    # DeepEP (ops 24/25 decode dispatch)
    "benchmark/kernels/deepep/tuning_deepep.py"

    # Router GEMM (ops 11, 23)
    "benchmark/kernels/deepseek/benchmark_deepgemm_dsv3_router_gemm_blackwell.py"
    "test/registered/jit/benchmark/bench_dsv3_router_gemm.py"

    # Fused A-GEMM small-M (op 14 decode)
    "test/registered/jit/benchmark/bench_dsv3_fused_a_gemm.py"

    # DSA paged MQA logits (op 22 decode Index_Score)
    "benchmark/kernels/deepseek/benchmark_cute_dsl_fp8_paged_mqa_logits.py"

    # topk_transform_512_v2 (ops 8, 22)
    "test/registered/jit/benchmark/bench_topk.py"

    # DSV4 FP4 indexer (V4 variant; useful if targeting V4)
    "test/registered/jit/benchmark/bench_dsv4_fp4_indexer.py"

    # MiniMax M3 kernels
    "test/registered/jit/benchmark/minimax/bench_minimax_qknorm_rope.py"
    "test/registered/jit/benchmark/minimax/bench_minimax_decode_topk.py"
    "test/registered/jit/benchmark/minimax/bench_minimax_store_kv_index.py"
)

for b in "${benchmarks[@]}"; do
    if [[ -f "$b" ]]; then
        echo -e "\n>>> $b"
        python3 "$b" 2>&1 | tail -30 || true
    else
        echo "  (MISSING: $b)"
    fi
done
