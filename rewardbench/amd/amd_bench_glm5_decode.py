#!/usr/bin/env python3
"""GLM-5.2 DECODE operator benchmark on **AMD MI300X (ROCm)**.

A-card port of llm_flops/bench_glm5_decode.py. Decode counterpart of
amd_bench_glm5_prefill.py: M = batch in {1,4,8,16,32,64}, S = 65536. At decode's tiny M
the dense GEMMs become weight-memory-bound (skinny-M), so most rewards are HBM-bandwidth
utilization — exactly the regime where MI300X's 5.3 TB/s HBM3 and the CK (vs ASM) kernel
choice matter (qhy: CK wins small-M, ASM wins M>=4096).

Only difference from prefill: index_score uses the PAGED cost model (paged fp8 KV read
dominates at decode). Same builders otherwise, including the fused SGLang MoE total
metric used to tie split MoE diagnostics back to the production ABI.

Run:  python amd_bench_glm5_decode.py                # batch sweep 1,4,8,16,32,64
      python amd_bench_glm5_decode.py --m 32         # single batch
"""
import argparse
import torch

import amd_glm5_ops_common as C


def build_ops():
    """GLM-5.2 decode ops (identical to prefill except index_score = paged)."""
    H, QL, KVL = C.HIDDEN_SIZE, C.Q_LORA_RANK, C.KV_LORA_RANK
    NH, QKH, VH, QKN = C.NUM_HEADS, C.QK_HEAD_DIM, C.V_HEAD_DIM, C.QK_NOPE_HEAD_DIM
    FUSED = C.FUSED_QKV_A_OUT
    IH, IHD = C.INDEX_N_HEADS, C.INDEX_HEAD_DIM
    MOE = C.MOE_INTERMEDIATE_SIZE

    return [
        ("fused_qkv_a_proj", "Attention", "fp8_gemm(aiter/hipBLASLt)",
         lambda a: C.gemm_fp8_cost(a["M"], H, FUSED),
         lambda a, d: C.build_fp8_gemm(a["M"], H, FUSED, d, tag="fused_qkv_a_proj")),
        ("q_b_proj", "Attention", "fp8_gemm(aiter/hipBLASLt)",
         lambda a: C.gemm_fp8_cost(a["M"], QL, NH * QKH),
         lambda a, d: C.build_fp8_gemm(a["M"], QL, NH * QKH, d, tag="q_b_proj")),
        ("absorbed_W_UK", "Attention", "bmm_fp8(hipBLASLt loop)",
         lambda a: C.bmm_fp8_cost(NH, a["M"], QKN, KVL),
         lambda a, d: C.build_bmm_fp8(NH, a["M"], QKN, KVL, d, tag="absorbed_W_UK")),
        ("absorbed_W_UV", "Attention", "bmm_fp8(hipBLASLt loop)",
         lambda a: C.bmm_fp8_cost(NH, a["M"], KVL, VH),
         lambda a, d: C.build_bmm_fp8(NH, a["M"], KVL, VH, d, tag="absorbed_W_UV")),
        ("o_proj", "Attention", "fp8_gemm(aiter/hipBLASLt)",
         lambda a: C.gemm_fp8_cost(a["M"], NH * VH, H),
         lambda a, d: C.build_fp8_gemm(a["M"], NH * VH, H, d, tag="o_proj")),
        ("dsa_decode_attn", "DSA", "sparse_mla(aiter/SDPA)",
         lambda a: C.sparse_mla_cost(a["M"], a["S"]),
         lambda a, d: C.build_sparse_mla(a["M"], a["S"], d, tag="dsa_decode_attn")),
        ("index_k_proj", "DSA-Indexer", "fp8_gemm(aiter/hipBLASLt)",
         lambda a: C.gemm_fp8_cost(a["M"], H, IHD),          # decode: M-axis = batch
         lambda a, d: C.build_fp8_gemm(a["M"], H, IHD, d, tag="index_k_proj")),
        ("index_q_upproj", "DSA-Indexer", "fp8_gemm(aiter/hipBLASLt)",
         lambda a: C.gemm_fp8_cost(a["M"], QL, IH * IHD),
         lambda a, d: C.build_fp8_gemm(a["M"], QL, IH * IHD, d, tag="index_q_upproj")),
        ("index_weights_proj", "DSA-Indexer", "bf16_gemm(torch.mm)",
         lambda a: C.gemm_bf16_cost(a["M"], H, IH),
         lambda a, d: C.build_bf16_gemm(a["M"], H, IH, d, tag="index_weights_proj")),
        ("index_score", "DSA-Indexer", "mqa_logits_paged(hipBLASLt)",
         lambda a: C.paged_mqa_logits_cost(a["M"], a["S"]),
         lambda a, d: C.build_mqa_logits(a["M"], a["S"], d, tag="index_score")),
        ("moe_gate_proj", "MoE", "moe_grouped(aiter/hipBLASLt)",
         lambda a: C.moe_grouped_cost(a["M"], H, MOE),
         lambda a, d: C.build_moe_grouped(a["M"], H, MOE, d, tag="moe_gate_proj")),
        ("moe_up_proj", "MoE", "moe_grouped(aiter/hipBLASLt)",
         lambda a: C.moe_grouped_cost(a["M"], H, MOE),
         lambda a, d: C.build_moe_grouped(a["M"], H, MOE, d, tag="moe_up_proj")),
        ("moe_down_proj", "MoE", "moe_grouped(aiter/hipBLASLt)",
         lambda a: C.moe_grouped_cost(a["M"], MOE, H),
         lambda a, d: C.build_moe_grouped(a["M"], MOE, H, d, tag="moe_down_proj")),
        ("moe_total", "MoE", "sglang.fused_moe(total)",
         lambda a: C.moe_fused_total_cost(a["M"]),
         lambda a, d: C.build_moe_fused_total(a["M"], d, tag="moe_total")),
    ]


def main():
    ap = argparse.ArgumentParser(description="GLM-5.2 DECODE operator benchmark (MI300X)")
    ap.add_argument("--m", type=int, default=None, help="single decode batch (default sweep 1,4,8,16,32,64)")
    ap.add_argument("--s", type=int, default=65536, help="KV context length")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--csv", default="amd_glm5_decode_perf.csv")
    args = ap.parse_args()

    device = torch.device(args.device)
    torch.cuda.set_device(device)
    C.print_env_banner()

    m_list = [args.m] if args.m else [1, 4, 8, 16, 32, 64]
    sweep = [{"M": m, "S": args.s} for m in m_list]
    print(f"GLM-5.2 DECODE | batch sweep = {m_list}, S = {args.s}")
    C.run_ops(build_ops(), sweep, device, "decode", args.csv)


if __name__ == "__main__":
    main()
