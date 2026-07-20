#!/usr/bin/env python3
"""GLM-5 PREFILL operator reward benchmark (13 ops, sglang-aligned, B200).

Two modes:
  1. BASELINE (default) — benchmark the 13 sglang/llm_flops reference prefill kernels
     (same ABI/dtype/layout, Rule 5) to establish the reward denominator.
  2. CANDIDATE (--kernels-dir DIR) — test a folder of GIVEN optimized operators
     (best-kernels tree: <cand>/solution.py [+ get_inputs] + META.md + task.json),
     benchmarking each prefill candidate directly and writing ONE big CSV. This is a
     PERFORMANCE-only module: correctness is a separate upstream gate we do not implement.

Both report a **bound-aware roofline-utilization reward** in [0,1] (Rule 3): compute-util
for compute-bound ops, HBM-bandwidth-util for memory-bound ops, auto-classified by
arithmetic intensity. Peaks (B200/SM100): HBM 8TB/s, FP8 4.5PF, BF16 2.25PF.

Prefill shapes: M (prefill token count) in {1024,2048,4096}; S (KV ctx) = 65536.
Run:  python bench_GLM5_ops_prefill.py                          # baseline, full M sweep
      python bench_GLM5_ops_prefill.py --m 4096                 # baseline, single M
      python bench_GLM5_ops_prefill.py --kernels-dir best-kernels-reward-bench   # candidates
"""
import argparse
import torch

import glm5_ops_common as C


def build_ops():
    """13 prefill ops: (name, category, backend, cost_fn(axes), run_builder(axes,dev))."""
    H, QL, KVL = C.HIDDEN_SIZE, C.Q_LORA_RANK, C.KV_LORA_RANK
    NH, QKH, VH, QKN = C.NUM_HEADS, C.QK_HEAD_DIM, C.V_HEAD_DIM, C.QK_NOPE_HEAD_DIM
    FUSED = C.FUSED_QKV_A_OUT
    IH, IHD = C.INDEX_N_HEADS, C.INDEX_HEAD_DIM
    MOE = C.MOE_INTERMEDIATE_SIZE

    return [
        # ── Attention GEMMs ──
        ("fused_qkv_a_proj", "Attention", "deep_gemm.fp8_gemm_nt",
         lambda a: C.gemm_fp8_cost(a["M"], H, FUSED),
         lambda a, d: C.build_fp8_gemm(a["M"], H, FUSED, d)),
        ("q_b_proj", "Attention", "deep_gemm.fp8_gemm_nt",
         lambda a: C.gemm_fp8_cost(a["M"], QL, NH * QKH),
         lambda a, d: C.build_fp8_gemm(a["M"], QL, NH * QKH, d)),
        # NOTE (dtype/hardware branches): absorbed_W_UK/UV, index_k_proj,
        # index_weights_proj, dsa_prefill_attn have branched backends in sglang
        # (fp8-KV vs bf16-KV; H800/SM90 vs B200/SM100 DEEPGEMM_BLACKWELL; indexer-fusion).
        # ACTIVE selection == llm_flops/bench_glm5_prefill.py (the H800+B200 tuned
        # deployment reference). The commented line under each is the alternate-dtype
        # baseline (uncomment to measure that branch); builders for both live in common.
        ("absorbed_W_UK", "Attention", "sgl_kernel.bmm_fp8",
         lambda a: C.bmm_fp8_cost(NH, a["M"], QKN, KVL),
         lambda a, d: C.build_bmm_fp8(NH, a["M"], QKN, KVL, d)),
        #   alt bf16 torch.bmm (B200 SM100 DEEPGEMM_BLACKWELL dequant path):
        #   C.bmm_bf16_cost(NH, a["M"], QKN, KVL) / C.build_bmm_bf16(NH, a["M"], QKN, KVL, d)
        ("absorbed_W_UV", "Attention", "sgl_kernel.bmm_fp8",
         lambda a: C.bmm_fp8_cost(NH, a["M"], KVL, VH),
         lambda a, d: C.build_bmm_fp8(NH, a["M"], KVL, VH, d)),
        #   alt bf16 torch.bmm: C.bmm_bf16_cost(NH, a["M"], KVL, VH) / C.build_bmm_bf16(NH, a["M"], KVL, VH, d)
        ("o_proj", "Attention", "deep_gemm.fp8_gemm_nt",
         lambda a: C.gemm_fp8_cost(a["M"], NH * VH, H),
         lambda a, d: C.build_fp8_gemm(a["M"], NH * VH, H, d)),
        # ── DSA prefill attention (sparse) ──
        ("dsa_prefill_attn", "DSA", "sgl_kernel.flash_mla_sparse_fwd",
         lambda a: C.sparse_mla_cost(a["M"], a["S"]),
         lambda a, d: C.build_sparse_mla(a["M"], a["S"], d)),
        #   alt fp8-KV trtllm-gen (B200 kv_cache_dtype=fp8 default):
        #   C.sparse_mla_trtllm_cost(a["M"], a["S"]) / C.build_sparse_mla_trtllm(a["M"], a["S"], d)
        # ── DSA indexer ──
        ("index_k_proj", "DSA-Indexer", "deep_gemm.fp8_gemm_nt",
         lambda a: C.gemm_fp8_cost(a["S"], H, IHD),          # M-axis = S (per-KV-token)
         lambda a, d: C.build_fp8_gemm(a["S"], H, IHD, d)),
        #   alt bf16 (GLM-5.2 indexer-fusion wk_weights_proj, torch F.linear):
        #   C.linear_bf16_cost(a["S"], H, IHD) / C.build_linear_bf16(a["S"], H, IHD, d)
        ("index_q_upproj", "DSA-Indexer", "deep_gemm.fp8_gemm_nt",
         lambda a: C.gemm_fp8_cost(a["M"], QL, IH * IHD),
         lambda a, d: C.build_fp8_gemm(a["M"], QL, IH * IHD, d)),
        ("index_weights_proj", "DSA-Indexer", "deep_gemm.bf16_gemm_nt",
         lambda a: C.gemm_bf16_cost(a["M"], H, IH),
         lambda a, d: C.build_bf16_gemm(a["M"], H, IH, d)),
        #   alt torch bf16->f32 (fusion path weights_proj):
        #   C.linear_bf16_cost(a["M"], H, IH, out_b=C.F32_B) / C.build_linear_bf16(a["M"], H, IH, d, out_f32=True)
        ("index_score", "DSA-Indexer", "deep_gemm.fp8_mqa_logits",
         lambda a: C.mqa_logits_ragged_cost(a["M"], a["S"]),
         lambda a, d: C.build_mqa_ragged(a["M"], a["S"], d)),
        # ── MoE grouped GEMMs (masked) ──
        ("moe_gate_proj", "MoE", "deep_gemm.fp8_m_grouped_gemm_nt_masked",
         lambda a: C.moe_grouped_cost(a["M"], H, MOE),
         lambda a, d: C.build_moe_grouped(a["M"], H, MOE, d)),
        ("moe_up_proj", "MoE", "deep_gemm.fp8_m_grouped_gemm_nt_masked",
         lambda a: C.moe_grouped_cost(a["M"], H, MOE),
         lambda a, d: C.build_moe_grouped(a["M"], H, MOE, d)),
        ("moe_down_proj", "MoE", "deep_gemm.fp8_m_grouped_gemm_nt_masked",
         lambda a: C.moe_grouped_cost(a["M"], MOE, H),
         lambda a, d: C.build_moe_grouped(a["M"], MOE, H, d)),
    ]


def main():
    ap = argparse.ArgumentParser(description="GLM-5 PREFILL operator reward benchmark")
    ap.add_argument("--m", type=int, default=None, help="single prefill M (default: sweep 1024,2048,4096)")
    ap.add_argument("--s", type=int, default=65536, help="KV context length")
    ap.add_argument("--csv", default="glm5_ops_prefill_reward.csv")
    # Candidate mode: point at a folder of GIVEN optimized operators (best-kernels tree).
    # Each <candidate>/solution.py is benchmarked directly; results -> one big CSV.
    ap.add_argument("--kernels-dir", default=None,
                    help="folder of optimized operators to test (prefill candidates only)")
    ap.add_argument("--repeat", type=int, default=1, help="candidate mode: timing repeats (take fastest)")
    ap.add_argument("--no-baseline", action="store_true", help="candidate mode: skip speedup baseline")
    ap.add_argument("--round", type=int, default=0, help="candidate mode: round index for the CSV")
    args = ap.parse_args()

    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    torch.backends.cuda.matmul.allow_tf32 = True

    if args.kernels_dir:
        # ── Candidate mode: test given optimized operators (phase == prefill) ──
        out = args.csv if args.csv != "glm5_ops_prefill_reward.csv" else "glm5_ops_prefill_candidates.csv"
        print("=" * 120)
        print("GLM-5 PREFILL candidate reward-bench [input folder of optimized ops -> one CSV, perf only]")
        print(f"  reward = bound-aware roofline utilization in [0,1]  (peaks: HBM 8TB/s, FP8 4.5PF, BF16 2.25PF)")
        print(f"  kernels-dir = {args.kernels_dir}")
        print("=" * 120)
        C.run_candidate_folder(args.kernels_dir, "prefill", device, out,
                               repeat=args.repeat, no_baseline=args.no_baseline, rnd=args.round)
        return

    # ── Baseline mode: the 13 sglang/llm_flops reference ops (reward denominator) ──
    m_list = [args.m] if args.m else [1024, 2048, 4096]
    sweep = [{"M": m, "S": args.s} for m in m_list]

    print("=" * 120)
    print("GLM-5 PREFILL operator reward benchmark [B200, sglang ABI, UE8M0 FP8, CUDA-graph timing]")
    print(f"  reward = bound-aware roofline utilization in [0,1]  (peaks: HBM 8TB/s, FP8 4.5PF, BF16 2.25PF)")
    print(f"  M sweep = {m_list}, S = {args.s}")
    print("=" * 120)
    C.run_ops(build_ops(), sweep, device, "prefill", args.csv)


if __name__ == "__main__":
    main()
