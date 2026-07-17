#!/usr/bin/env python3
"""GLM-5 PREFILL operator reward benchmark (13 ops, sglang-aligned, B200).

For each of the 13 prefill operators the model runs (confirmed against
llm_flops/bench_glm5_prefill.py and 算子baseline确认.xlsx), this measures the real
sglang backend kernel (same ABI/dtype/layout, Rule 5) and reports a **bound-aware
roofline-utilization reward** in [0,1] (Rule 3): compute-util for compute-bound ops,
HBM-bandwidth-util for memory-bound ops, auto-classified by arithmetic intensity.

Prefill shapes: M (prefill token count) in {1024,2048,4096}; S (KV ctx) = 65536.
Run:  python bench_GLM5_ops_prefill.py            # full M sweep
      python bench_GLM5_ops_prefill.py --m 4096   # single M
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
        ("absorbed_W_UK", "Attention", "torch.bmm (bf16)",
         lambda a: C.bmm_bf16_cost(NH, a["M"], QKN, KVL),
         lambda a, d: C.build_bmm_bf16(NH, a["M"], QKN, KVL, d)),
        ("absorbed_W_UV", "Attention", "torch.bmm (bf16)",
         lambda a: C.bmm_bf16_cost(NH, a["M"], KVL, VH),
         lambda a, d: C.build_bmm_bf16(NH, a["M"], KVL, VH, d)),
        ("o_proj", "Attention", "deep_gemm.fp8_gemm_nt",
         lambda a: C.gemm_fp8_cost(a["M"], NH * VH, H),
         lambda a, d: C.build_fp8_gemm(a["M"], NH * VH, H, d)),
        # ── DSA prefill attention (sparse) — B200 default = trtllm-gen fp8 ──
        ("dsa_prefill_attn", "DSA", "flashinfer trtllm-gen (fp8 KV)",
         lambda a: C.sparse_mla_trtllm_cost(a["M"], a["S"]),
         lambda a, d: C.build_sparse_mla_trtllm(a["M"], a["S"], d)),
        # ── DSA indexer ──
        ("index_k_proj", "DSA-Indexer", "torch.F.linear (bf16)",   # GLM-5.2 fused wk_weights_proj (bf16)
         lambda a: C.linear_bf16_cost(a["S"], H, IHD),             # M-axis = S (per-KV-token)
         lambda a, d: C.build_linear_bf16(a["S"], H, IHD, d)),
        ("index_q_upproj", "DSA-Indexer", "deep_gemm.fp8_gemm_nt",
         lambda a: C.gemm_fp8_cost(a["M"], QL, IH * IHD),
         lambda a, d: C.build_fp8_gemm(a["M"], QL, IH * IHD, d)),
        ("index_weights_proj", "DSA-Indexer", "torch.mm (bf16->f32)",  # weights_proj (folded into wk_weights_proj)
         lambda a: C.linear_bf16_cost(a["M"], H, IH, out_b=C.F32_B),
         lambda a, d: C.build_linear_bf16(a["M"], H, IH, d, out_f32=True)),
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
    ap = argparse.ArgumentParser()
    ap.add_argument("--m", type=int, default=None, help="single prefill M (default: sweep 1024,2048,4096)")
    ap.add_argument("--s", type=int, default=65536, help="KV context length")
    ap.add_argument("--csv", default="glm5_ops_prefill_reward.csv")
    args = ap.parse_args()

    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    torch.backends.cuda.matmul.allow_tf32 = True

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
