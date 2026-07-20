#!/usr/bin/env python3
"""GLM-5 DECODE operator reward benchmark (13 ops, sglang-aligned, B200).

Decode counterpart of bench_GLM5_ops_prefill.py — same two modes:
  1. BASELINE (default) — the 13 sglang/llm_flops reference decode kernels (M = batch,
     2D hidden states, index score via the PAGED kernel fp8_paged_mqa_logits).
  2. CANDIDATE (--kernels-dir DIR) — test a folder of GIVEN optimized operators,
     benchmarking each decode candidate directly into ONE big CSV (performance only;
     correctness is a separate upstream gate we do not implement).

At decode's tiny M the GEMMs become weight-memory-bound (skinny-M), so most rewards are
HBM-bandwidth utilization — the reward correctly rewards a kernel that streams weights
closer to 8 TB/s rather than one that maximizes (already-trivial) tensor-core work.

Decode shapes: M (batch) in {1,4,8,16,32,64}; S (KV ctx) = 65536.
Run:  python bench_GLM5_ops_decode.py                          # baseline, full batch sweep
      python bench_GLM5_ops_decode.py --m 32                   # baseline, single batch
      python bench_GLM5_ops_decode.py --kernels-dir best-kernels-reward-bench   # candidates
"""
import argparse
import torch

import glm5_ops_common as C


def build_ops():
    """13 decode ops: identical to prefill except index_score uses the PAGED kernel."""
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
        # index_weights_proj, dsa_decode_attn have branched backends in sglang
        # (fp8-KV vs bf16-KV; H800/SM90 vs B200/SM100 DEEPGEMM_BLACKWELL; indexer-fusion).
        # ACTIVE selection == llm_flops/bench_glm5_decode.py (the H800+B200 tuned
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
        # ── DSA decode attention (sparse) ──
        ("dsa_decode_attn", "DSA", "sgl_kernel.flash_mla_sparse_fwd",
         lambda a: C.sparse_mla_cost(a["M"], a["S"]),
         lambda a, d: C.build_sparse_mla(a["M"], a["S"], d)),
        #   alt fp8-KV trtllm-gen (B200 kv_cache_dtype=fp8 default):
        #   C.sparse_mla_trtllm_cost(a["M"], a["S"]) / C.build_sparse_mla_trtllm(a["M"], a["S"], d)
        # ── DSA indexer ──
        ("index_k_proj", "DSA-Indexer", "deep_gemm.fp8_gemm_nt",
         lambda a: C.gemm_fp8_cost(a["M"], H, IHD),          # decode: M-axis = batch
         lambda a, d: C.build_fp8_gemm(a["M"], H, IHD, d)),
        #   alt bf16 (GLM-5.2 indexer-fusion wk_weights_proj, torch F.linear):
        #   C.linear_bf16_cost(a["M"], H, IHD) / C.build_linear_bf16(a["M"], H, IHD, d)
        ("index_q_upproj", "DSA-Indexer", "deep_gemm.fp8_gemm_nt",
         lambda a: C.gemm_fp8_cost(a["M"], QL, IH * IHD),
         lambda a, d: C.build_fp8_gemm(a["M"], QL, IH * IHD, d)),
        ("index_weights_proj", "DSA-Indexer", "deep_gemm.bf16_gemm_nt",
         lambda a: C.gemm_bf16_cost(a["M"], H, IH),
         lambda a, d: C.build_bf16_gemm(a["M"], H, IH, d)),
        #   alt torch bf16->f32 (fusion path weights_proj):
        #   C.linear_bf16_cost(a["M"], H, IH, out_b=C.F32_B) / C.build_linear_bf16(a["M"], H, IH, d, out_f32=True)
        ("index_score", "DSA-Indexer", "deep_gemm.fp8_paged_mqa_logits",
         lambda a: C.paged_mqa_logits_cost(a["M"], a["S"]),
         lambda a, d: C.build_paged_mqa(a["M"], a["S"], d)),
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
    ap = argparse.ArgumentParser(description="GLM-5 DECODE operator reward benchmark")
    ap.add_argument("--m", type=int, default=None, help="single decode batch (default: sweep 1,4,8,16,32,64)")
    ap.add_argument("--s", type=int, default=65536, help="KV context length")
    ap.add_argument("--csv", default="glm5_ops_decode_reward.csv")
    # Candidate mode: point at a folder of GIVEN optimized operators (best-kernels tree).
    ap.add_argument("--kernels-dir", default=None,
                    help="folder of optimized operators to test (decode candidates only)")
    ap.add_argument("--repeat", type=int, default=1, help="candidate mode: timing repeats (take fastest)")
    ap.add_argument("--no-baseline", action="store_true", help="candidate mode: skip speedup baseline")
    ap.add_argument("--round", type=int, default=0, help="candidate mode: round index for the CSV")
    args = ap.parse_args()

    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    torch.backends.cuda.matmul.allow_tf32 = True

    if args.kernels_dir:
        # ── Candidate mode: test given optimized operators (phase == decode) ──
        out = args.csv if args.csv != "glm5_ops_decode_reward.csv" else "glm5_ops_decode_candidates.csv"
        print("=" * 120)
        print("GLM-5 DECODE candidate reward-bench [input folder of optimized ops -> one CSV, perf only]")
        print(f"  reward = bound-aware roofline utilization in [0,1]  (peaks: HBM 8TB/s, FP8 4.5PF, BF16 2.25PF)")
        print(f"  kernels-dir = {args.kernels_dir}")
        print("=" * 120)
        C.run_candidate_folder(args.kernels_dir, "decode", device, out,
                               repeat=args.repeat, no_baseline=args.no_baseline, rnd=args.round)
        return

    # ── Baseline mode: the 13 sglang/llm_flops reference ops (reward denominator) ──
    m_list = [args.m] if args.m else [1, 4, 8, 16, 32, 64]
    sweep = [{"M": m, "S": args.s} for m in m_list]

    print("=" * 120)
    print("GLM-5 DECODE operator reward benchmark [B200, sglang ABI, UE8M0 FP8, CUDA-graph timing]")
    print(f"  reward = bound-aware roofline utilization in [0,1]  (peaks: HBM 8TB/s, FP8 4.5PF, BF16 2.25PF)")
    print(f"  batch sweep = {m_list}, S = {args.s}")
    print("=" * 120)
    C.run_ops(build_ops(), sweep, device, "decode", args.csv)


if __name__ == "__main__":
    main()
