#!/usr/bin/env python3
"""Extract MiniMax-M3 ops (op 28-43) from kernel_api_mapping.csv into a
focused CSV with an added `test_result_on_b200` column.

CRITICAL CONTEXT: M3 code is NOT on sglang `main` -- it lives on the upstream
remote branch `amd_add_m3` (refs/heads/amd_add_m3 on github.com/sgl-project/sglang).
All M3 operator tests below were run against a git worktree checked out from
that branch at commit e28a0c81 ("Merge branch 'main' into amd_add_m3"), on an
NVIDIA B200 (SM100), harness venv (torch 2.12 dev + sgl_kernel + deep_gemm +
flashinfer; NO flash_attn). Kernels are JIT-compiled at runtime from .cuh via
load_jit -- they are NOT in the sgl_kernel wheel, so the wheel having no
minimax_* symbols is expected and not a blocker.

Reads:  docs/kernel_api_mapping.csv
Writes: docs/minimax_m3_operators.csv
"""
import csv
import os

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "kernel_api_mapping.csv")
DST = os.path.join(HERE, "minimax_m3_operators.csv")

# Per-op TEST RESULT on amd_add_m3 worktree @ e28a0c81, B200, harness venv.
# Verdicts prefixed PASS / SKIP / NOT-RUN / NO-OP-KERNEL.
TEST_RESULT = {
    28: "NO-OP-KERNEL (pass-by-design) - M3 main QKV is per-head GQA via standard "
        "Linear; no M3-specific kernel. Covered indirectly by op 37/42 infrastructure. "
        "Not unit-testable at op level.",
    29: "PASS - test/registered/jit/minimax/test_minimax_qknorm_rope.py: 79 passed. "
        "bench_minimax_qknorm_rope: fused 1.64-192.7us, peak 1372.8 GB/s @ T=1024; "
        "combined_one_launch beats two_launches (e.g. T=8192: 17.6us vs 42.9us).",
    30: "PASS - test/registered/jit/minimax/test_minimax_store_kv_index.py: 96 passed. "
        "bench_minimax_store_kv_index: fused 1.41-4.89us, peak 4815.5 GB/s @ T=16384; "
        "fused vs separate launches (e.g. T=16384: 4.89us vs 59.4us = ~12x faster).",
    31: "PASS - NEW test written: test/registered/jit/minimax/test_minimax_m3_cuda_config.py "
        "::test_m3_prefill_score_topk. flash_prefill_with_topk_index (Triton/CUDA) at "
        "M3 index shapes (idx_q [B,4,128], idx_k_cache [slots,1,128], block_size=128, "
        "topk=8, init=0, local=1, score_type=max). Oracle: proven decode reference "
        "(pytorch_reference) via the equivalence prefill(length-1, prefix=L-1, "
        "seq_len=L) == decode(seq_len=L). 1 case passed.",
    32: "PASS - test_minimax_decode_topk.py: 118 passed; test_minimax_decode_topk_page_table.py: "
        "30 passed. NEW M3-config case in test_minimax_m3_cuda_config.py: topk in "
        "{4,8,16,32} (closes gap -- existing grid was {16,32,64}; M3 MoE topk=4) with "
        "num_idx_heads=4, block_size=128. bench: JIT radix 2.25-7.50us beats Triton "
        "2-stage by 2-30x. 12 cases passed.",
    33: "PASS - NEW test written: test_minimax_m3_cuda_config.py::test_m3_prefill_main_sparse. "
        "flash_prefill_with_gqa_share_sparse (Triton) at M3 main shapes (q [B,64,128], "
        "k_cache [slots,4,128], GQA 16:1, block_size=128, topk=8). topk_idx flows from "
        "op31 (real M3 hand-off, idx_group_size=1 no reduce). Oracle: sparse-attention "
        "reference over topk blocks (gather paged cache, softmax+sink). 1 case passed.",
    34: "PASS - minimax_sparse_ops/tests/test_sparse_gqa.py: 20 passed in 77s. "
        "Triton flash_decode_with_gqa_share_sparse path verified. MSA fmha_sm100 path "
        "NOT exercised (needs external fmha_sm100 import + eager/non-cuda-graph).",
    35: "PASS (same as op 29) - CUDA fused Gemma-RMSNorm+RoPE covered by "
        "test_minimax_qknorm_rope.py (79 passed) + NEW M3 (64,4,4) grouped case in "
        "test_minimax_m3_cuda_config.py (closes grid gap -- existing was "
        "{(8,1,1),(8,1,4),(16,2,2)}). ROCm Triton variant not run (gfx95 only).",
    36: "SKIP - test_minimax_m3_rmsnorm.py: 1 skipped on B200 (ROCm gfx94x/gfx95x only; "
        "this is SM100). Note: layernorm.py uses sgl_kernel.gemma_rmsnorm (C++) not this "
        "Triton one, so this Triton rmsnorm has no production caller on CUDA.",
    37: "PASS - M3 reuses standard sglang FusedMoE router+topk. "
        "sgl-kernel/tests/test_moe_topk_sigmoid.py: 774 passed. No M3-specific kernel.",
    38: "SKIP - test_minimax_m3_mxfp8.py: 1 skipped on B200 (ROCm gfx950 hard gate). "
        "All MXFP8 MoE kernels are gfx95-only, not runnable on this B200.",
    39: "SKIP - covered by test_minimax_m3_mxfp8.py (1 skipped, ROCm gfx95x SwiGLU-OAI).",
    40: "SKIP - covered by test_minimax_m3_mxfp8.py (1 skipped, ROCm fused SwiGLU+MXFP8 quant).",
    41: "SKIP - covered by test_minimax_m3_mxfp8.py (combine kernel, ROCm only).",
    42: "PASS - test_minimax_sparse_pool_host_unit.py: 7 passed, 1 skipped, 3 subtests "
        "passed; test_minimax_sparse_pool_pd_unit.py: 2 passed. MiniMaxSparseKVPool "
        "(main K/V + index_kv + index_k sub-pools) verified.",
    43: "NOT-RUN (integration) - test_unified_radix_hicache_dispatch.py exists but is an "
        "end-to-end hybrid-cache dispatch test requiring the full model pipeline; out of "
        "scope for operator-only testing per user request.",
}

rows_out = []
with open(SRC, newline="") as f:
    reader = csv.DictReader(f)
    for r in reader:
        try:
            op_id = int(r["op_id"])
        except (ValueError, KeyError):
            continue
        if op_id < 28:
            continue
        rows_out.append({
            "op_id": op_id,
            "operator": r["operator"],
            "phase": r["phase"],
            "shape_or_role": r["shape_or_role"],
            "ultimate_kernel_or_api": r["ultimate_kernel_or_api"],
            "kernel_impl_location": r["kernel_impl_location"],
            "dispatch_switch": r["dispatch_switch"],
            "unit_test": r["unit_test"],
            "benchmark": r["benchmark"],
            "test_result_on_b200": TEST_RESULT.get(op_id, ""),
            "notes": r["notes"],
        })

# Put PASS first, then SKIP, then NOT-RUN/NO-OP-KERNEL, ordered by op_id within group
def sort_key(r):
    v = r["test_result_on_b200"]
    rank = 0 if v.startswith("PASS") else 1 if v.startswith("SKIP") else 2
    return (rank, r["op_id"])
rows_out.sort(key=sort_key)

fields = ["op_id", "operator", "phase", "shape_or_role", "ultimate_kernel_or_api",
          "kernel_impl_location", "dispatch_switch", "unit_test", "benchmark",
          "test_result_on_b200", "notes"]

with open(DST, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader()
    w.writerows(rows_out)

print(f"wrote {len(rows_out)} rows -> {DST}")
for r in rows_out:
    v = r["test_result_on_b200"].split(" - ")[0]
    print(f"  op {r['op_id']:>2} | {v:16s} | {r['operator']}")
