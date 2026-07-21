#!/usr/bin/env python3
"""GLM-5.2 PREFILL operator benchmark on **AMD MI300X (ROCm)**.

A-card port of llm_flops/bench_glm5_prefill.py. Same GLM-5.2 shapes,
same M sweep {1024,2048,4096} / S=65536 — but every kernel is the sglang-ROCm / AMD
backend (aiter or hipBLASLt via torch._scaled_mm) instead of DeepGEMM / FlashMLA /
sgl_kernel. It keeps the original split MoE diagnostics and adds the fused SGLang MoE
total metric for objective leaderboard/rollup use. Emits latency, TFLOP/s, GB/s,
arithmetic intensity, roofline bound and the MI300X bound-aware roofline reward per
operator, and writes a CSV.

Operator -> AMD backend map (see operator_mapping.md for the full table):
  fused_qkv_a_proj / q_b_proj / o_proj   deep_gemm.fp8_gemm_nt   -> aiter.gemm_a8w8_blockscale | hipBLASLt _scaled_mm
  absorbed_W_UK / absorbed_W_UV          sgl_kernel.bmm_fp8      -> per-head hipBLASLt _scaled_mm loop
  dsa_prefill_attn                       flash_mla_sparse_fwd    -> aiter MLA | gather+SDPA (bf16)
  index_k_proj / index_q_upproj          deep_gemm.fp8_gemm_nt   -> aiter.gemm_a8w8_blockscale | hipBLASLt _scaled_mm
  index_weights_proj                     deep_gemm.bf16_gemm_nt  -> torch.mm (bf16->f32)
  index_score                            deep_gemm.fp8_mqa_logits-> aiter.ops.triton.fp8_mqa_logits
  moe_gate/up/down_proj                  fp8_m_grouped_gemm...   -> aiter fmoe | per-expert hipBLASLt loop
  moe_total                              fused runtime MoE       -> sglang fused_moe

Run:  python amd_bench_glm5_prefill.py                       # full M sweep
      python amd_bench_glm5_prefill.py --m 4096              # single M
      AMD_BENCH_NO_GRAPH=1 python amd_bench_glm5_prefill.py  # event timing (no hipGraph)
"""
import argparse
import torch

import amd_glm5_ops_common as C


def build_ops():
    """GLM-5.2 prefill ops as (name, category, backend_label, cost_fn, run_builder)."""
    H, QL, KVL = C.HIDDEN_SIZE, C.Q_LORA_RANK, C.KV_LORA_RANK
    NH, QKH, VH, QKN = C.NUM_HEADS, C.QK_HEAD_DIM, C.V_HEAD_DIM, C.QK_NOPE_HEAD_DIM
    FUSED = C.FUSED_QKV_A_OUT
    IH, IHD = C.INDEX_N_HEADS, C.INDEX_HEAD_DIM
    MOE = C.MOE_INTERMEDIATE_SIZE

    return [
        # ── Attention GEMMs ──
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
        # ── DSA prefill attention (sparse) ──
        ("dsa_prefill_attn", "DSA", "sparse_mla(aiter/SDPA)",
         lambda a: C.sparse_mla_cost(a["M"], a["S"]),
         lambda a, d: C.build_sparse_mla(a["M"], a["S"], d, tag="dsa_prefill_attn")),
        # ── DSA indexer ──
        ("index_k_proj", "DSA-Indexer", "fp8_gemm(aiter/hipBLASLt)",
         lambda a: C.gemm_fp8_cost(a["S"], H, IHD),          # M-axis = S (per-KV-token)
         lambda a, d: C.build_fp8_gemm(a["S"], H, IHD, d, tag="index_k_proj")),
        ("index_q_upproj", "DSA-Indexer", "fp8_gemm(aiter/hipBLASLt)",
         lambda a: C.gemm_fp8_cost(a["M"], QL, IH * IHD),
         lambda a, d: C.build_fp8_gemm(a["M"], QL, IH * IHD, d, tag="index_q_upproj")),
        ("index_weights_proj", "DSA-Indexer", "bf16_gemm(torch.mm)",
         lambda a: C.gemm_bf16_cost(a["M"], H, IH),
         lambda a, d: C.build_bf16_gemm(a["M"], H, IH, d, tag="index_weights_proj")),
        ("index_score", "DSA-Indexer", "mqa_logits(hipBLASLt)",
         lambda a: C.mqa_logits_ragged_cost(a["M"], a["S"]),
         lambda a, d: C.build_mqa_logits(a["M"], a["S"], d, tag="index_score")),
        # ── MoE grouped GEMMs ──
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
    ap = argparse.ArgumentParser(description="GLM-5.2 PREFILL operator benchmark (MI300X)")
    ap.add_argument("--m", type=int, default=None, help="single prefill M (default sweep 1024,2048,4096)")
    ap.add_argument("--s", type=int, default=65536, help="KV context length")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--csv", default="amd_glm5_prefill_perf.csv")
    args = ap.parse_args()

    device = torch.device(args.device)
    torch.cuda.set_device(device)
    C.print_env_banner()

    m_list = [args.m] if args.m else [1024, 2048, 4096]
    sweep = [{"M": m, "S": args.s} for m in m_list]
    print(f"GLM-5.2 PREFILL | M sweep = {m_list}, S = {args.s}")
    C.run_ops(build_ops(), sweep, device, "prefill", args.csv)


if __name__ == "__main__":
    main()
