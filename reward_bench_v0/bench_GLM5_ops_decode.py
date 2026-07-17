#!/usr/bin/env python3
"""GLM-5 DECODE operator reward benchmark (13 ops, sglang-aligned, B200).

Decode counterpart of bench_GLM5_ops_prefill.py. Same 13 operators and same sglang
ABI, but the decode path: M = batch size (1 token/request, 2D hidden states), and the
indexer score uses the paged kernel `deep_gemm.fp8_paged_mqa_logits`. Reports the same
bound-aware roofline-utilization reward in [0,1] (Rule 3).

At decode's tiny M the GEMMs become weight-memory-bound (skinny-M), so most rewards are
HBM-bandwidth utilization — the reward correctly rewards a kernel that streams weights
closer to 8 TB/s rather than one that maximizes (already-trivial) tensor-core work.

Decode shapes: M (batch) in {1,4,8,16,32,64}; S (KV ctx) = 65536.
Run:  python bench_GLM5_ops_decode.py             # full batch sweep
      python bench_GLM5_ops_decode.py --m 32       # single batch
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
        ("absorbed_W_UK", "Attention", "torch.bmm (bf16)",
         lambda a: C.bmm_bf16_cost(NH, a["M"], QKN, KVL),
         lambda a, d: C.build_bmm_bf16(NH, a["M"], QKN, KVL, d)),
        ("absorbed_W_UV", "Attention", "torch.bmm (bf16)",
         lambda a: C.bmm_bf16_cost(NH, a["M"], KVL, VH),
         lambda a, d: C.build_bmm_bf16(NH, a["M"], KVL, VH, d)),
        ("o_proj", "Attention", "deep_gemm.fp8_gemm_nt",
         lambda a: C.gemm_fp8_cost(a["M"], NH * VH, H),
         lambda a, d: C.build_fp8_gemm(a["M"], NH * VH, H, d)),
        # ── DSA decode attention (sparse) — B200 default = trtllm-gen fp8 ──
        ("dsa_decode_attn", "DSA", "flashinfer trtllm-gen (fp8 KV)",
         lambda a: C.sparse_mla_trtllm_cost(a["M"], a["S"]),
         lambda a, d: C.build_sparse_mla_trtllm(a["M"], a["S"], d)),
        # ── DSA indexer ──
        ("index_k_proj", "DSA-Indexer", "torch.F.linear (bf16)",   # GLM-5.2 fused wk_weights_proj (bf16)
         lambda a: C.linear_bf16_cost(a["M"], H, IHD),             # decode: M-axis = batch
         lambda a, d: C.build_linear_bf16(a["M"], H, IHD, d)),
        ("index_q_upproj", "DSA-Indexer", "deep_gemm.fp8_gemm_nt",
         lambda a: C.gemm_fp8_cost(a["M"], QL, IH * IHD),
         lambda a, d: C.build_fp8_gemm(a["M"], QL, IH * IHD, d)),
        ("index_weights_proj", "DSA-Indexer", "torch.mm (bf16->f32)",  # weights_proj (folded into wk_weights_proj)
         lambda a: C.linear_bf16_cost(a["M"], H, IH, out_b=C.F32_B),
         lambda a, d: C.build_linear_bf16(a["M"], H, IH, d, out_f32=True)),
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
    ap = argparse.ArgumentParser()
    ap.add_argument("--m", type=int, default=None, help="single decode batch (default: sweep 1,4,8,16,32,64)")
    ap.add_argument("--s", type=int, default=65536, help="KV context length")
    ap.add_argument("--csv", default="glm5_ops_decode_reward.csv")
    args = ap.parse_args()

    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    torch.backends.cuda.matmul.allow_tf32 = True

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
