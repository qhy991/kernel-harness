#!/usr/bin/env bash
# Invoke sglang's OFFICIAL pytest suites for the kernels this harness covers.
# Requires sgl-kernel + deep_gemm + a checkout of the sglang repo at $SGLANG_DIR.
#
# Usage: SGLANG_DIR=/path/to/sglang bash scripts/run_official_tests.sh [gemm|moe|attention|dsa|jit|all]

set -euo pipefail
SGLANG_DIR="${SGLANG_DIR:-/mnt/public/qinhaiyan/sglang}"
SUITE="${1:-all}"

cd "$SGLANG_DIR"

gemm_tests=(
    "sgl-kernel/tests/test_dsv3_fused_a_gemm.py"          # op 1, 14: fused Q_a+KV_a fast path
    "sgl-kernel/tests/test_fp8_blockwise_gemm.py"         # ops 1-5, 9-10, 14-16, 19: block-fp8 GEMM
    "sgl-kernel/tests/test_bmm_fp8.py"                    # ops 17, 18: BMM fp8 (absorb)
    "test/registered/jit/test_dsv3_fused_a_gemm.py"
    "test/registered/jit/test_cutedsl_dsv3_fused_a_gemm.py"
    "test/registered/jit/test_dsv3_router_gemm.py"        # op 11, 23: MoE Router GEMM
    "sgl-kernel/tests/test_moe_topk_sigmoid.py"           # topk after router
    "sgl-kernel/tests/test_fused_qk_norm_rope.py"         # Q_b fused pre-op
    "sgl-kernel/tests/test_dsv4_norm_rope.py"             # DSA norm+rope
)

moe_tests=(
    "sgl-kernel/tests/test_fp8_blockwise_moe.py"          # ops 12, 13, 24, 25 grouped fp8
    "test/registered/moe/test_moe_runners_1gpu.py"        # MoeRunner smoke
    "test/registered/moe/test_fused_moe.py"               # Triton fused_experts
    "test/registered/moe/test_triton_fused_moe.py"
    "test/registered/moe/test_cutedsl_moe.py"
    "test/registered/moe/test_moe_ep.py"                  # EP MoE
    "test/registered/kernels/test_fused_topk_deepseek.py"
)

attention_tests=(
    "sgl-kernel/tests/test_flash_attention.py"            # op 27 FA3 causal MHA
    "sgl-kernel/tests/test_flash_attn_sparse.py"          # op 26 flash_mla_sparse_fwd
    "sgl-kernel/tests/test_flashmla.py"                   # op 26 flashmla_kv variant
    "test/registered/attention/test_flash_attention_4.py"
    "test/registered/mla/test_flashmla.py"                # end-to-end MLA MTP
    "test/registered/attention/test_hybrid_attn_backend.py"
)

dsa_tests=(
    "test/registered/kernels/test_dsa_indexer.py"         # ops 6-8, 20-22: Index_Q/K/Score
    "test/registered/kernels/test_deepgemm_paged_mqa_logits.py"  # op 22 decode paged
    "test/registered/kernels/test_cute_dsl_fp8_paged_mqa_logits.py"
    "test/registered/kernels/test_sm120_paged_mqa_logits.py"
    "test/registered/kernels/test_dsa_metadata.py"
    "test/registered/jit/test_dsv32_indexer_fusion.py"    # fused_q/k_indexer_*
    "test/registered/jit/deepseek_v4/test_topk_v2.py"     # topk_transform_512_v2
    "test/registered/jit/deepseek_v4/test_fp4_indexer.py"
)

jit_tests=(
    "test/registered/jit/minimax/test_minimax_qknorm_rope.py"      # M3 Q/K norm+RoPE (CUDA)
    "test/registered/jit/minimax/test_minimax_decode_topk.py"       # M3 decode indexer topk
    "test/registered/jit/minimax/test_minimax_store_kv_index.py"    # M3 fused KV+idx store
    "test/registered/jit/minimax/test_minimax_decode_topk_page_table.py"
    "python/sglang/srt/layers/attention/minimax_sparse_ops/tests/test_sparse_gqa.py"
    "python/sglang/srt/layers/attention/minimax_sparse_ops/tests/test_flash_with_topk_idx.py"
)

case "$SUITE" in
    gemm)      TESTS=("${gemm_tests[@]}") ;;
    moe)       TESTS=("${moe_tests[@]}") ;;
    attention) TESTS=("${attention_tests[@]}") ;;
    dsa)       TESTS=("${dsa_tests[@]}") ;;
    jit)       TESTS=("${jit_tests[@]}") ;;
    all)       TESTS=("${gemm_tests[@]}" "${moe_tests[@]}" "${attention_tests[@]}" "${dsa_tests[@]}" "${jit_tests[@]}") ;;
    *)         echo "unknown suite: $SUITE (choose gemm|moe|attention|dsa|jit|all)"; exit 2 ;;
esac

echo "# Running $SUITE: ${#TESTS[@]} test files under $SGLANG_DIR"
for t in "${TESTS[@]}"; do
    if [[ -f "$t" ]]; then
        echo -e "\n>>> pytest $t"
        pytest -q "$t" || echo "  (FAILED: $t)"
    else
        echo "  (MISSING: $t)"
    fi
done
