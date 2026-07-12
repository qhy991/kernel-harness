#!/usr/bin/env python3
"""Sweep the *non-config* axes of the Kimi-K2.7 kernel baselines.

Everything that isn't a fixed model-config dimension is swept over a range
instead of a single operating point:

  A. Prefill token count M      [512 .. 32768]     (chunked-prefill sizes)
  B. Decode batch size B        [1 .. 256]         (tokens = B, 1 step each)
  C. Per-expert tokens (contig) [16 .. 1024] + realistic non-uniform routing
  D. masked_m (decode masked)   [1 .. 128]  + non-uniform masked_m vector
  E. Attention prefill seqlen   [1024 .. 16384]
  F. Attention decode KV len    [1024 .. 32768]  and decode batch [1 .. 256]

Reuses the validated per-op benches from bench_kimi_untested.py /
bench_kimi_missing_ops.py. Emits long-form rows (one per op×axis×value) to
kimi_sweeps.json and kimi_k27_sweeps.csv.

NOTE: decode-side wall-clock for tiny ops floors at ~49us (Python launch +
perf_counter overhead at small B); read the *shape* of each curve and the
prefill/large-B end, not the absolute small-B floor.
"""
from __future__ import annotations

import csv
import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from bench_kimi_real_kernels import bench, bench_bmm, bench_fp8_gemm, gemm_tflops  # noqa: E402
from bench_kimi_untested import (  # noqa: E402
    bench_grouped_fp8_masked,
    bench_mla_decode,
    bench_mla_prefill,
    bench_router_decode,
    bench_router_prefill,
)
from bench_kimi_missing_ops import (  # noqa: E402
    bench_act_quant,
    bench_fused_add_rmsnorm,
    bench_gate,
    bench_moe_sum,
    bench_silu,
)

import deep_gemm  # noqa: E402
from deep_gemm.utils.math import align, per_block_cast_to_fp8, per_token_cast_to_fp8  # noqa: E402

# ------------------------------- sweep grids -------------------------------- #
PREFILL_M = [512, 1024, 2048, 4096, 8192, 16384, 32768]
DECODE_B = [1, 2, 4, 8, 16, 32, 64, 128, 256]
EXPERT_M = [16, 32, 64, 128, 256, 341, 512, 1024]   # tokens per expert (contiguous)
MASKED_M = [1, 2, 4, 8, 16, 32, 64, 128]            # valid rows per group (masked)
ATTN_PREFILL_SEQ = [1024, 2048, 4096, 8192, 16384]
ATTN_DECODE_KV = [1024, 2048, 4096, 8192, 16384, 32768]

HIDDEN, N_EXP, TOPK = 7168, 384, 8
results: list[dict] = []


def rec(r, op, axis, val):
    r.update({"op": op, "axis": axis, "axis_value": val})
    results.append(r)
    tf = r.get("TFLOPS", "")
    print(f"  {op:28s} | {axis:14s}={str(val):>7s} | {r['us']:9.2f} us | {str(tf):>8s}")
    return r


def guard(fn, *a, op="", axis="", val=""):
    try:
        return rec(fn(*a), op, axis, val)
    except Exception as e:
        print(f"  FAIL {op} {axis}={val}: {type(e).__name__}: {e}")
        results.append({"op": op, "axis": axis, "axis_value": val,
                        "error": f"{type(e).__name__}: {e}"})


# --------- C/D non-uniform grouped helpers (realistic routing) -------------- #
def bench_grouped_contig_counts(counts, K, N, op, note):
    """Contiguous grouped FP8 GEMM with a per-expert token-count vector.
    Each expert's block is padded up to the 128 alignment (as in real MoE)."""
    ALIGN = deep_gemm.get_m_alignment_for_contiguous_layout()
    E = len(counts)
    Ka, Na = align(K, 128), align(N, 128)
    padded = [align(int(c), ALIGN) if c > 0 else 0 for c in counts]
    m_sum = sum(padded)
    a = torch.randn(m_sum, Ka, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(E, Na, Ka, device="cuda", dtype=torch.bfloat16)
    a_fp8, a_s = deep_gemm.per_token_cast_to_fp8(a, False)
    b_fp8, b_s = zip(*[per_block_cast_to_fp8(b[e], use_ue8m0=False) for e in range(E)])
    b_fp8, b_s = torch.stack(b_fp8), torch.stack(b_s)
    out = torch.empty(m_sum, Na, device="cuda", dtype=torch.bfloat16)
    m_idx = torch.repeat_interleave(
        torch.arange(E, device="cuda", dtype=torch.int32),
        torch.tensor(padded, device="cuda", dtype=torch.int64),
    )

    def run():
        deep_gemm.m_grouped_fp8_gemm_nt_contiguous((a_fp8, a_s), (b_fp8, b_s), out, m_idx)

    us = bench(run, warmup=5, iters=20)
    return {"name": op, "phase": "prefill", "backend": "deep_gemm contiguous (non-uniform)",
            "shape": f"E={E} m_sum={m_sum} K={K} N={N} ({note})",
            "us": round(us, 2), "TFLOPS": round(gemm_tflops(m_sum, Ka, Na, us), 2)}


def bench_grouped_masked_vec(masked_vec, K, N, op, note):
    """Masked grouped FP8 GEMM with a per-group masked_m vector."""
    E = len(masked_vec)
    Ka, Na = align(K, 128), align(N, 128)
    Mp = align(max(int(max(masked_vec)), 1), 128)
    a = torch.randn(E, Mp, Ka, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(E, Na, Ka, device="cuda", dtype=torch.bfloat16)
    a_fp8, a_s = zip(*[per_token_cast_to_fp8(a[e], use_ue8m0=True) for e in range(E)])
    b_fp8, b_s = zip(*[per_block_cast_to_fp8(b[e], use_ue8m0=True) for e in range(E)])
    a_fp8, a_s = torch.stack(a_fp8), torch.stack(a_s)
    b_fp8, b_s = torch.stack(b_fp8), torch.stack(b_s)
    a_s = deep_gemm.transform_sf_into_required_layout(a_s, mn=Mp, k=Ka, recipe=(1, 128, 128), num_groups=E, is_sfa=True)
    b_s = deep_gemm.transform_sf_into_required_layout(b_s, mn=Na, k=Ka, recipe=(1, 128, 128), num_groups=E, is_sfa=False)
    out = torch.empty(E, Mp, Na, device="cuda", dtype=torch.bfloat16)
    masked = torch.tensor(masked_vec, device="cuda", dtype=torch.int32)
    exp_m = int(sum(masked_vec) / E)

    def run():
        deep_gemm.fp8_m_grouped_gemm_nt_masked((a_fp8, a_s), (b_fp8, b_s), out, masked, exp_m)

    us = bench(run, warmup=10, iters=50)
    tot = int(sum(masked_vec))
    return {"name": op, "phase": "decode", "backend": "deep_gemm masked (non-uniform)",
            "shape": f"E={E} masked_m={note} K={K} N={N}",
            "us": round(us, 2), "TFLOPS": round(gemm_tflops(tot, Ka, Na, us), 2)}


def routing_counts(M, E=N_EXP, topk=TOPK, seed=0):
    torch.manual_seed(seed)
    sel = torch.randn(M, E).topk(topk, dim=1).indices.reshape(-1)
    return torch.bincount(sel, minlength=E).tolist()


# ---------------------------------- main ------------------------------------ #
def main():
    print(f"# GPU: {torch.cuda.get_device_name(0)}  torch={torch.__version__}")

    print("\n### A. Prefill token-count sweep (M) ###")
    for M in PREFILL_M:
        guard(bench_fp8_gemm, M, 7168, 2112, "Q_a fused", -1, "prefill", op="Q_a fused (fp8)", axis="prefill_M", val=M)
        guard(bench_fp8_gemm, M, 1024, 7168, "O_proj", -1, "prefill", op="O_proj (fp8)", axis="prefill_M", val=M)
        guard(bench_fp8_gemm, M, 7168, 4608, "Dense GateUp", -1, "prefill", op="Dense GateUp (fp8)", axis="prefill_M", val=M)
        guard(bench_router_prefill, M, 7168, 384, "MoE Router", op="MoE Router (cuBLAS)", axis="prefill_M", val=M)
        guard(bench_fused_add_rmsnorm, M, "prefill", op="Fused add+RMSNorm", axis="prefill_M", val=M)
        guard(bench_silu, M, 4608, "dense", "prefill", op="SwiGLU dense", axis="prefill_M", val=M)
        guard(bench_act_quant, M, 7168, "hidden", "prefill", op="Act FP8 quant", axis="prefill_M", val=M)
        guard(bench_moe_sum, M, "prefill", op="MoE Combine (moe_sum)", axis="prefill_M", val=M)

    print("\n### B. Decode batch-size sweep (B) ###")
    for B in DECODE_B:
        guard(bench_fp8_gemm, B, 7168, 2112, "Q_a fused", -1, "decode", op="Q_a fused (fp8)", axis="decode_B", val=B)
        guard(bench_fp8_gemm, B, 1536, 12288, "Q_b", -1, "decode", op="Q_b (fp8)", axis="decode_B", val=B)
        guard(bench_fp8_gemm, B, 8192, 7168, "O_proj", -1, "decode", op="O_proj (fp8)", axis="decode_B", val=B)
        guard(bench_router_prefill, B, 7168, 384, "MoE Router", op="MoE Router (cuBLAS)", axis="decode_B", val=B)
        if B <= 16:
            guard(bench_router_decode, B, 7168, 384, "MoE Router", op="MoE Router (dsv3_router_gemm)", axis="decode_B", val=B)
        guard(bench_bmm, B, 64, 128, 512, "KV_b absorb BMM (H=64)", -1, "decode", op="KV_b absorb BMM H=64", axis="decode_B", val=B)
        guard(bench_gate, B, "decode", op="MoE Gate (fused)", axis="decode_B", val=B)
        guard(bench_fused_add_rmsnorm, B, "decode", op="Fused add+RMSNorm", axis="decode_B", val=B)

    print("\n### C. Per-expert token sweep — contiguous grouped (prefill) ###")
    for me in EXPERT_M:
        guard(lambda me=me: bench_grouped_contig_counts([me] * N_EXP, 7168, 256, "MoE GateUp grouped", f"uniform={me}"),
              op="MoE GateUp grouped (contig)", axis="tokens_per_expert", val=me)
    # realistic non-uniform routing for a few prefill sizes
    for M in (4096, 16384):
        c = routing_counts(M)
        guard(lambda c=c, M=M: bench_grouped_contig_counts(c, 7168, 256, "MoE GateUp grouped", f"routed M={M} min{min(c)}/max{max(c)}"),
              op="MoE GateUp grouped (contig)", axis="tokens_per_expert", val=f"routed_M{M}")

    print("\n### D. masked_m sweep — masked grouped (decode) ###")
    for mm in MASKED_M:
        guard(bench_grouped_fp8_masked, 8, mm, 7168, 2048, "MoE GateUp GroupGEMM", "decode",
              op="MoE GateUp masked", axis="masked_m", val=mm)
    # non-uniform masked_m vector (skewed EP load)
    for label, vec in [("skew8", [1, 2, 4, 8, 16, 32, 64, 128]), ("ep12", [16] * 12)]:
        guard(lambda vec=vec, label=label: bench_grouped_masked_vec(vec, 7168, 2048, "MoE GateUp GroupGEMM", label),
              op="MoE GateUp masked", axis="masked_m", val=label)

    print("\n### E. Attention prefill seqlen sweep ###")
    for s in ATTN_PREFILL_SEQ:
        guard(bench_mla_prefill, s, f"MLA prefill seq={s}", op="MLA attention prefill", axis="prefill_seqlen", val=s)

    print("\n### F. Attention decode sweep (KV len @B=16, then batch @seq=4096) ###")
    for s in ATTN_DECODE_KV:
        guard(bench_mla_decode, s, 16, f"MLA decode kv={s}", op="MLA attention decode", axis="decode_kvlen", val=s)
    for B in DECODE_B:
        guard(bench_mla_decode, 4096, B, f"MLA decode B={B}", op="MLA attention decode", axis="decode_B", val=B)

    out_json = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "logs" / "kimi_sweeps.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(results, indent=2))

    out_csv = out_json.with_name("kimi_k27_sweeps.csv")
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["op", "phase", "axis", "axis_value", "latency_us", "TFLOPS", "shape", "backend"])
        for r in results:
            if "error" in r:
                w.writerow([r["op"], "", r["axis"], r["axis_value"], "ERR", "", r["error"], ""])
            else:
                w.writerow([r["op"], r.get("phase", ""), r["axis"], r["axis_value"],
                            r["us"], r.get("TFLOPS", ""), r.get("shape", ""), r.get("backend", "")])
    ok = sum(1 for r in results if "us" in r)
    print(f"\n# wrote {out_json} and {out_csv} ({ok}/{len(results)} ok)")


if __name__ == "__main__":
    main()
