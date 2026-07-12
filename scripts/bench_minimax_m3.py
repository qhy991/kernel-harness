#!/usr/bin/env python3
"""MiniMax-M3 model-agnostic kernel baselines + sweeps (B200), to bring M3 to
parity with the Kimi-K2.7 coverage for the ops that ARE runnable in this
sglang checkout.

MiniMax-M3 config (HF text_config): hidden=6144, GQA num_q_heads=64,
num_kv_heads=4, head_dim=128; MoE n_experts=128, top_k=4, moe_inter=3072,
sigmoid routing (routed_scaling_factor=2.0); SwiGLU-OAI activation.

NOT covered here (M3-specific kernels live in the amd_add_m3 worktree, absent
from this checkout): sparse indexer (op29-34), fused_gemma_qknorm_rope,
store_kv_index, minimax_decode_topk, gqa_share_sparse, SwiGLU-OAI/MXFP8 fused,
mega_moe whole-op — see docs/minimax_m3_operator_backend_inventory.csv.

Caveats: FP8 GEMMs use deep_gemm block-fp8 as a proxy for the MXFP8 checkpoint;
SwiGLU uses standard silu_and_mul as a proxy for SwiGLU-OAI (clamped); main
attention is benched DENSE (GQA) as an upper bound — M3 runs it sparse (topk).
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from bench_kimi_real_kernels import bench, bench_fp8_gemm, bench_grouped_fp8, gemm_tflops  # noqa: E402
from bench_kimi_untested import bench_grouped_fp8_masked, bench_router_prefill  # noqa: E402
from bench_kimi_missing_ops import bench_act_quant, bench_silu  # noqa: E402

# ----- M3 config -----
HID, NQ, NKV, HD = 6144, 64, 4, 128
QKV_OUT = (NQ + NKV + NKV) * HD  # 9216
O_K = NQ * HD                     # 8192
E, TOPK, INTER = 128, 4, 3072
M_PREFILL, M_DECODE = 16384, 16

PREFILL_M = [512, 1024, 2048, 4096, 8192, 16384, 32768]
DECODE_B = [1, 2, 4, 8, 16, 32, 64, 128, 256]
EXPERT_M = [16, 32, 64, 128, 256, 512, 1024]
MASKED_M = [1, 2, 4, 8, 16, 32, 64, 128]
ATTN_SEQ = [1024, 2048, 4096, 8192, 16384]

results: list[dict] = []


def rec(r, op, mtype, axis="", val="ref"):
    r.update({"op": op, "mtype": mtype, "axis": axis, "axis_value": val})
    results.append(r)
    print(f"  {op:28s} | {mtype:9s} | {axis or '-':14s}={str(val):>7s} | {r['us']:9.2f} us | {str(r.get('TFLOPS','')):>8}")


def guard(fn, *a, op="", mtype="reference", axis="", val="ref"):
    try:
        rec(fn(*a), op, mtype, axis, val)
    except Exception as e:
        print(f"  FAIL {op} {axis}={val}: {type(e).__name__}: {e}")
        results.append({"op": op, "mtype": mtype, "axis": axis, "axis_value": val,
                        "error": f"{type(e).__name__}: {e}"})


def bench_gemma_rmsnorm(M, phase):
    from sgl_kernel import gemma_rmsnorm
    x = torch.randn(M, HID, device="cuda", dtype=torch.bfloat16)
    w = torch.ones(HID, device="cuda", dtype=torch.bfloat16)

    def run():
        gemma_rmsnorm(x, w, eps=1e-6)

    return {"name": "Gemma-RMSNorm", "phase": phase, "backend": "sgl_kernel.gemma_rmsnorm",
            "shape": f"[{M},{HID}]", "us": round(bench(run, 10, 50), 2)}


def bench_gate_sigmoid(M, phase):
    logits = torch.randn(M, E, device="cuda", dtype=torch.float32)

    def run():
        s = torch.sigmoid(logits)
        torch.topk(s, TOPK, dim=-1)

    return {"name": "MoE Gate (sigmoid+top4)", "phase": phase,
            "backend": "sigmoid+topk (torch; moe_fused_gate caps 32 exp/grp)",
            "shape": f"[{M},{E}] top{TOPK}", "us": round(bench(run, 10, 50), 2)}


def bench_moe_combine(M, phase):
    from sgl_kernel import moe_sum
    ic = torch.randn(M, TOPK, HID, device="cuda", dtype=torch.bfloat16)
    out = torch.empty(M, HID, device="cuda", dtype=torch.bfloat16)

    def run():
        moe_sum(ic, out)

    return {"name": "MoE Combine (moe_sum)", "phase": phase, "backend": "sgl_kernel.moe_sum",
            "shape": f"[{M},{TOPK},{HID}]->[{M},{HID}]", "us": round(bench(run, 10, 50), 2)}


def bench_gqa_prefill(seqlen, phase="prefill"):
    import flashinfer
    q = torch.randn(seqlen, NQ, HD, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(seqlen, NKV, HD, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(seqlen, NKV, HD, device="cuda", dtype=torch.bfloat16)

    def run():
        flashinfer.single_prefill_with_kv_cache(q, k, v, causal=True)

    us = bench(run, 3, 10)
    flops = 2.0 * NQ * seqlen * seqlen * (HD + HD) * 0.5
    return {"name": "GQA attention (dense)", "phase": phase,
            "backend": "flashinfer single_prefill (GQA 64/4, causal)",
            "shape": f"q[{seqlen},{NQ},{HD}] kv[{seqlen},{NKV},{HD}]",
            "us": round(us, 2), "TFLOPS": round(flops / (us * 1e-6) / 1e12, 2)}


def main():
    print(f"# GPU: {torch.cuda.get_device_name(0)}  (MiniMax-M3 model-agnostic baselines)")

    print("\n### reference (M3 shapes) ###")
    for M, ph in ((M_PREFILL, "prefill"), (M_DECODE, "decode")):
        guard(bench_fp8_gemm, M, HID, QKV_OUT, "Main QKV proj (fp8)", -1, ph, op="Main QKV proj (fp8)", val=ph)
        guard(bench_fp8_gemm, M, O_K, HID, "O_proj (fp8)", -1, ph, op="O_proj (fp8)", val=ph)
        guard(bench_router_prefill, M, HID, E, "MoE Router", op="MoE Router (cuBLAS)", val=ph)
        guard(bench_gate_sigmoid, M, ph, op="MoE Gate (sigmoid+top4)", val=ph)
        guard(bench_silu, M, 2 * INTER, "M3", ph, op="SwiGLU act (silu proxy)", val=ph)
        guard(bench_moe_combine, M, ph, op="MoE Combine (moe_sum)", val=ph)
        guard(bench_gemma_rmsnorm, M, ph, op="Gemma-RMSNorm", val=ph)
        guard(bench_act_quant, M, HID, "hidden", ph, op="Act FP8 quant", val=ph)
    # MoE grouped (the mega_moe core, real M3 shapes)
    guard(bench_grouped_fp8, E, 512, HID, 2 * INTER, "MoE GateUp grouped", -1, "prefill", op="MoE GateUp grouped", val="prefill")
    guard(bench_grouped_fp8, E, 512, INTER, HID, "MoE Down grouped", -1, "prefill", op="MoE Down grouped", val="prefill")
    guard(bench_grouped_fp8_masked, 16, 16, HID, 2 * INTER, "MoE GateUp masked", "decode", op="MoE GateUp masked", val="decode")
    guard(bench_grouped_fp8_masked, 16, 16, INTER, HID, "MoE Down masked", "decode", op="MoE Down masked", val="decode")
    guard(bench_gqa_prefill, 4096, op="GQA attention (dense)", val="prefill")

    print("\n### sweep A: prefill token count M ###")
    for M in PREFILL_M:
        guard(bench_fp8_gemm, M, HID, QKV_OUT, "QKV", -1, "prefill", op="Main QKV proj (fp8)", mtype="sweep", axis="prefill_M", val=M)
        guard(bench_router_prefill, M, HID, E, "Router", op="MoE Router (cuBLAS)", mtype="sweep", axis="prefill_M", val=M)
        guard(bench_silu, M, 2 * INTER, "M3", "prefill", op="SwiGLU act (silu proxy)", mtype="sweep", axis="prefill_M", val=M)
        guard(bench_moe_combine, M, "prefill", op="MoE Combine (moe_sum)", mtype="sweep", axis="prefill_M", val=M)
        guard(bench_gemma_rmsnorm, M, "prefill", op="Gemma-RMSNorm", mtype="sweep", axis="prefill_M", val=M)
        guard(bench_act_quant, M, HID, "hidden", "prefill", op="Act FP8 quant", mtype="sweep", axis="prefill_M", val=M)

    print("\n### sweep B: decode batch size B ###")
    for B in DECODE_B:
        guard(bench_fp8_gemm, B, HID, QKV_OUT, "QKV", -1, "decode", op="Main QKV proj (fp8)", mtype="sweep", axis="decode_B", val=B)
        guard(bench_gate_sigmoid, B, "decode", op="MoE Gate (sigmoid+top4)", mtype="sweep", axis="decode_B", val=B)
        guard(bench_moe_combine, B, "decode", op="MoE Combine (moe_sum)", mtype="sweep", axis="decode_B", val=B)
        guard(bench_gemma_rmsnorm, B, "decode", op="Gemma-RMSNorm", mtype="sweep", axis="decode_B", val=B)

    print("\n### sweep C: tokens per expert (contiguous grouped, prefill) ###")
    for me in EXPERT_M:
        guard(bench_grouped_fp8, E, me, HID, 2 * INTER, "MoE GateUp grouped", -1, "prefill",
              op="MoE GateUp grouped", mtype="sweep", axis="tokens_per_expert", val=me)

    print("\n### sweep D: masked_m (masked grouped, decode) ###")
    for mm in MASKED_M:
        guard(bench_grouped_fp8_masked, 16, mm, HID, 2 * INTER, "MoE GateUp masked", "decode",
              op="MoE GateUp masked", mtype="sweep", axis="masked_m", val=mm)

    print("\n### sweep E: GQA attention prefill seqlen ###")
    for s in ATTN_SEQ:
        guard(bench_gqa_prefill, s, op="GQA attention (dense)", mtype="sweep", axis="prefill_seqlen", val=s)

    out_json = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "logs" / "minimax_m3_bench.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(results, indent=2))
    ok = sum(1 for r in results if "us" in r)
    print(f"\n# wrote {out_json} ({ok}/{len(results)} ok)")


if __name__ == "__main__":
    main()
