#!/usr/bin/env python3
"""Recover important Kimi-K2.7 kernels missed by the GEMM/elementwise rounds.

A second coverage audit against the real sglang forward path found these on the
critical path but absent from kimi_k27.csv:

  KV-cache store (per layer)   -> set_mla_kv_buffer_triton (JIT CUDA / Triton)
  MoE scale TMA-align (per MoE)-> tma_align_input_scale (Triton)
  Token sampling (per step)    -> softmax + flashinfer top_k_top_p / argmax
  (identified, not benched: ep_scatter/ep_gather MoE permute — needs the DeepEP
   dispatch context to feed realistic inputs; TP/DP/EP collectives — multi-GPU.)

Kimi-K2.7: kv_lora=512, qk_rope=64 (MLA KV row=576), hidden=7168, vocab=163840.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from bench_kimi_real_kernels import bench  # noqa: E402

D_CKV, D_ROPE = 512, 64
HIDDEN, VOCAB = 7168, 163840


def bench_kv_store(n_loc, phase):
    from sglang.srt.mem_cache.triton_ops.mla_buffer import set_mla_kv_buffer_triton

    buf = torch.zeros(n_loc + 8, 1, D_CKV + D_ROPE, device="cuda", dtype=torch.bfloat16)
    loc = torch.arange(n_loc, device="cuda", dtype=torch.int64)
    knope = torch.randn(n_loc, D_CKV, device="cuda", dtype=torch.bfloat16)
    krope = torch.randn(n_loc, D_ROPE, device="cuda", dtype=torch.bfloat16)

    def run():
        set_mla_kv_buffer_triton(buf, loc, knope, krope)

    us = bench(run, warmup=10, iters=50)
    path = "JIT CUDA (n>=768)" if n_loc >= 768 else "Triton"
    return {"name": "KV-cache store (MLA)", "phase": phase,
            "backend": f"set_mla_kv_buffer_triton [{path}]",
            "shape": f"n_loc={n_loc} row={D_CKV}+{D_ROPE}", "us": round(us, 2)}


def bench_tma_align(m, phase):
    from sglang.srt.layers.moe.ep_moe.kernels import tma_align_input_scale

    s = torch.randn(m, HIDDEN // 128, device="cuda", dtype=torch.float32)

    def run():
        tma_align_input_scale(s)

    us = bench(run, warmup=10, iters=50)
    return {"name": "MoE scale TMA-align", "phase": phase,
            "backend": "tma_align_input_scale (Triton)",
            "shape": f"[{m},{HIDDEN // 128}]", "us": round(us, 2)}


def bench_softmax(B, phase):
    x = torch.randn(B, VOCAB, device="cuda", dtype=torch.float32)

    def run():
        torch.softmax(x, dim=-1)

    us = bench(run, warmup=10, iters=50)
    return {"name": "Sampler softmax", "phase": phase,
            "backend": "torch.softmax", "shape": f"[{B},{VOCAB}]", "us": round(us, 2)}


def bench_greedy(B, phase):
    x = torch.randn(B, VOCAB, device="cuda", dtype=torch.float32)

    def run():
        torch.argmax(x, dim=-1)

    us = bench(run, warmup=10, iters=50)
    return {"name": "Sampler greedy (argmax)", "phase": phase,
            "backend": "torch.argmax", "shape": f"[{B},{VOCAB}]", "us": round(us, 2)}


def bench_topk_topp(B, phase):
    from flashinfer.sampling import top_k_top_p_sampling_from_probs

    probs = torch.softmax(torch.randn(B, VOCAB, device="cuda"), dim=-1)
    tk = torch.full((B,), 50, device="cuda", dtype=torch.int32)
    tp = torch.full((B,), 0.9, device="cuda")

    def run():
        top_k_top_p_sampling_from_probs(probs, tk, tp)

    us = bench(run, warmup=10, iters=50)
    return {"name": "Sampler top_k_top_p", "phase": phase,
            "backend": "flashinfer.top_k_top_p_sampling_from_probs",
            "shape": f"[{B},{VOCAB}] k=50 p=0.9", "us": round(us, 2)}


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

    print("\n### KV-cache store (per layer) ###")
    add(bench_kv_store, 16384, "prefill", label="kv_store prefill")
    add(bench_kv_store, 16, "decode", label="kv_store decode")

    print("\n### MoE scale TMA-align (per MoE layer, deepgemm path) ###")
    add(bench_tma_align, 16384, "prefill", label="tma_align prefill")
    add(bench_tma_align, 16, "decode", label="tma_align decode")

    print("\n### Token sampling (per step, critical path) ###")
    for B, ph in ((16, "decode"), (256, "decode")):
        add(bench_softmax, B, ph, label=f"softmax B={B}")
        add(bench_greedy, B, ph, label=f"greedy B={B}")
        add(bench_topk_topp, B, ph, label=f"top_k_top_p B={B}")

    out = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "logs" / "kimi_recovered.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    print(f"\n# wrote {out} ({len(results)} entries)")


if __name__ == "__main__":
    main()
