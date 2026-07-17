#!/usr/bin/env python3
"""Baseline microbench for Kimi-K2.7 per-layer kernels missing from kimi_k27.csv.

Complements bench_kimi_untested.py (GEMM/grouped/attention). These are the
small-but-every-layer element-wise / routing / norm kernels that the CSV's
GEMM-centric coverage skipped, using the real sglang / sgl_kernel APIs:

  MoE routing gate (Kimi-specific) -> sgl_kernel.kimi_k2_moe_fused_gate
  SwiGLU activation                -> sgl_kernel.silu_and_mul
  Activation FP8 quant             -> sglang_per_token_group_quant_fp8
  Residual + RMSNorm (fused)       -> sgl_kernel.fused_add_rmsnorm
  MLA latent RMSNorms              -> sgl_kernel.rmsnorm  (q_a=1536, kv_a=512)
  RoPE (rotary embedding)          -> sglang.jit_kernel.rope.apply_rope_with_cos_sin_cache_inplace
  MoE top-k weighted reduce        -> sgl_kernel.moe_sum
  Absorb BMM (H=64 fix)            -> torch.bmm  (CSV used H=8; DP decode = 64 heads)

Kimi-K2.7: hidden=7168, 384 experts, top_k=8, dense inter/TP=2304,
moe inter/TP=256, shared inter/TP=256, q_lora=1536, kv_lora=512, qk_rope=64,
num_heads=64. Prefill M=16384, decode M=16.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from bench_kimi_real_kernels import bench, bench_bmm, gemm_tflops, sync  # noqa: E402
from bench_kimi_layer_coverage import bench_rmsnorm  # noqa: E402

from sglang.srt.layers.quantization.fp8_kernel import sglang_per_token_group_quant_fp8  # noqa: E402

M_PREFILL, M_DECODE = 16384, 16
HIDDEN, N_EXPERTS, TOPK = 7168, 384, 8


# --------------------------------------------------------------------------- #
def bench_gate(M, phase):
    """Kimi-K2 fused routing gate: sigmoid + biased top-8 over 384 experts."""
    from sgl_kernel import kimi_k2_moe_fused_gate

    g = torch.randn(M, N_EXPERTS, device="cuda", dtype=torch.float32)
    bias = torch.randn(N_EXPERTS, device="cuda", dtype=torch.float32)

    def run():
        kimi_k2_moe_fused_gate(
            g, bias, topk=TOPK, renormalize=True,
            routed_scaling_factor=2.5, apply_routed_scaling_factor_on_output=False,
        )

    us = bench(run, warmup=10, iters=50)
    return {"name": "MoE Gate (fused topk)", "phase": phase,
            "backend": "sgl_kernel.kimi_k2_moe_fused_gate",
            "shape": f"[{M},{N_EXPERTS}] topk={TOPK}", "us": round(us, 2)}


def bench_silu(M, two_n, tag, phase):
    """SwiGLU: silu(x[:d]) * x[d:], d = two_n//2."""
    from sgl_kernel import silu_and_mul

    d = two_n // 2
    x = torch.randn(M, two_n, device="cuda", dtype=torch.bfloat16)
    out = torch.empty(M, d, device="cuda", dtype=torch.bfloat16)

    def run():
        silu_and_mul(x, out)

    us = bench(run, warmup=10, iters=50)
    return {"name": f"SwiGLU act ({tag})", "phase": phase,
            "backend": "sgl_kernel.silu_and_mul",
            "shape": f"[{M},{two_n}]->[{M},{d}]", "us": round(us, 2)}


def bench_act_quant(M, K, tag, phase):
    """Per-token-group FP8 quant of activations (runs before every FP8 GEMM)."""
    x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)

    def run():
        sglang_per_token_group_quant_fp8(
            x, 128, column_major_scales=True, scale_tma_aligned=True, scale_ue8m0=True
        )

    us = bench(run, warmup=10, iters=50)
    return {"name": f"Act FP8 quant ({tag})", "phase": phase,
            "backend": "sglang_per_token_group_quant_fp8 (UE8M0)",
            "shape": f"[{M},{K}] group=128", "us": round(us, 2)}


def bench_fused_add_rmsnorm(M, phase):
    """Residual add + RMSNorm fused — the real per-layer norm path."""
    from sgl_kernel import fused_add_rmsnorm

    x = torch.randn(M, HIDDEN, device="cuda", dtype=torch.bfloat16)
    res = torch.randn(M, HIDDEN, device="cuda", dtype=torch.bfloat16)
    w = torch.ones(HIDDEN, device="cuda", dtype=torch.bfloat16)

    def run():
        fused_add_rmsnorm(x, res, w, 1e-6)

    us = bench(run, warmup=10, iters=50)
    return {"name": "Fused add+RMSNorm", "phase": phase,
            "backend": "sgl_kernel.fused_add_rmsnorm",
            "shape": f"[{M},{HIDDEN}]", "us": round(us, 2)}


def bench_rope(tok, phase):
    """RoPE on q (64 heads) + k (1 head), rotary_dim=64."""
    from sglang.jit_kernel.rope import apply_rope_with_cos_sin_cache_inplace as rope

    H, D = 64, 64
    pos = torch.arange(tok, device="cuda", dtype=torch.int64)
    q = torch.randn(tok, H, D, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(tok, 1, D, device="cuda", dtype=torch.bfloat16)
    cache = torch.randn(max(tok, 4096), D, device="cuda", dtype=torch.float32)

    def run():
        rope(positions=pos, q=q, k=k, cos_sin_cache=cache, is_neox=True)

    us = bench(run, warmup=10, iters=50)
    return {"name": "RoPE (q+k)", "phase": phase,
            "backend": "sglang.jit_kernel.rope apply_rope_with_cos_sin_cache_inplace",
            "shape": f"q:[{tok},{H},{D}] k:[{tok},1,{D}]", "us": round(us, 2)}


def bench_moe_sum(M, phase):
    """Top-k weighted reduce of expert outputs -> hidden."""
    from sgl_kernel import moe_sum

    ic = torch.randn(M, TOPK, HIDDEN, device="cuda", dtype=torch.bfloat16)
    out = torch.empty(M, HIDDEN, device="cuda", dtype=torch.bfloat16)

    def run():
        moe_sum(ic, out)

    us = bench(run, warmup=10, iters=50)
    return {"name": "MoE Combine (moe_sum)", "phase": phase,
            "backend": "sgl_kernel.moe_sum",
            "shape": f"[{M},{TOPK},{HIDDEN}]->[{M},{HIDDEN}]", "us": round(us, 2)}


# --------------------------------------------------------------------------- #
def main():
    print(f"# GPU: {torch.cuda.get_device_name(0)}  torch={torch.__version__}")
    results = []

    def add(fn, *a, label=""):
        try:
            r = fn(*a)
            results.append(r)
            print(f"  OK  | {r['name']:26s} | {r['phase']:7s} | {r['us']:9.2f} us | {r['backend']}")
        except Exception as e:
            print(f"  FAIL| {label:26s} | {type(e).__name__}: {e}")
            results.append({"name": label, "error": f"{type(e).__name__}: {e}"})

    print("\n### MoE routing gate (Kimi-specific) ###")
    add(bench_gate, M_PREFILL, "prefill", label="gate prefill")
    add(bench_gate, M_DECODE, "decode", label="gate decode")

    print("\n### SwiGLU activation ###")
    #   dense inter/TP=2304 (2N=4608); shared inter/TP=256 (2N=512); moe expert 2N=512
    add(bench_silu, M_PREFILL, 4608, "dense", "prefill", label="silu dense prefill")
    add(bench_silu, M_DECODE, 4608, "dense", "decode", label="silu dense decode")
    add(bench_silu, N_EXPERTS * 512, 512, "moe grouped", "prefill", label="silu moe prefill")
    add(bench_silu, 8 * 128, 2048, "moe masked", "decode", label="silu moe decode")

    print("\n### Activation FP8 quantization ###")
    add(bench_act_quant, M_PREFILL, HIDDEN, "hidden", "prefill", label="quant hidden prefill")
    add(bench_act_quant, M_DECODE, HIDDEN, "hidden", "decode", label="quant hidden decode")
    add(bench_act_quant, M_DECODE, 2304, "dense-down-in", "decode", label="quant down decode")

    print("\n### Residual + RMSNorm (fused) + MLA latent norms ###")
    add(bench_fused_add_rmsnorm, M_PREFILL, "prefill", label="far prefill")
    add(bench_fused_add_rmsnorm, M_DECODE, "decode", label="far decode")
    for M, ph in ((M_PREFILL, "prefill"), (M_DECODE, "decode")):
        for dim, nm in ((1536, "q_a_layernorm"), (512, "kv_a_layernorm")):
            try:
                r = bench_rmsnorm(M, dim)
                r.update({"name": nm, "phase": ph})
                results.append(r)
                print(f"  OK  | {nm:26s} | {ph:7s} | {r['us']:9.2f} us | {r['backend']}")
            except Exception as e:
                print(f"  FAIL| {nm} {ph}: {e}")

    print("\n### RoPE ###")
    add(bench_rope, M_PREFILL, "prefill", label="rope prefill")
    add(bench_rope, M_DECODE, "decode", label="rope decode")

    print("\n### MoE combine (weighted reduce) ###")
    add(bench_moe_sum, M_PREFILL, "prefill", label="moe_sum prefill")
    add(bench_moe_sum, M_DECODE, "decode", label="moe_sum decode")

    print("\n### Absorb BMM — H=64 fix (CSV used H=8; DP decode runs all 64 heads) ###")
    for IN, OUT, nm in ((128, 512, "KV_b absorb BMM (H=64)"), (512, 128, "V absorb BMM (H=64)")):
        try:
            r = bench_bmm(M_DECODE, 64, IN, OUT, nm, -1, "decode")
            results.append(r)
            print(f"  OK  | {nm:26s} | decode  | {r['us']:9.2f} us | {r['TFLOPS']} TFLOPS | {r['backend']}")
        except Exception as e:
            print(f"  FAIL| {nm}: {e}")

    out = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "logs" / "kimi_missing_ops.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    print(f"\n# wrote {out} ({len(results)} entries)")


if __name__ == "__main__":
    main()
