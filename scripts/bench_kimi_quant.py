#!/usr/bin/env python3
"""Recover the missed QUANT kernels for Kimi-K2.7 (add to kimi_k27_all.csv).

Prior rounds benched only the block-FP8 activation quant standalone. These are
the other runtime quant kernels on the Kimi path:

  KV-cache FP8 quant (trtllm FP8 KV)  -> scaled_fp8_quant (per-tensor)
  MoE masked grouped 8-bit quant      -> sglang_per_token_group_quant_8bit
     (the fused silu+quant of the MoE down-input; previously only seen via the
      silu family — timed standalone here)
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

from sglang.srt.layers.quantization.fp8_kernel import (  # noqa: E402
    scaled_fp8_quant,
    sglang_per_token_group_quant_8bit,
)

KV_ROW = 576  # kv_lora(512) + qk_rope(64)


def bench_kv_quant(n, phase):
    k = torch.randn(n, KV_ROW, device="cuda", dtype=torch.bfloat16)
    scale = torch.tensor([1.0], device="cuda")

    def run():
        scaled_fp8_quant(k, scale)

    us = bench(run, warmup=10, iters=50)
    return {"name": "KV-cache FP8 quant", "phase": phase,
            "backend": "scaled_fp8_quant (per-tensor)",
            "shape": f"[{n},{KV_ROW}]", "us": round(us, 2)}


def bench_moe_masked_quant(E, Mp, two_n, fuse_silu, phase):
    x = torch.randn(E, Mp, two_n, device="cuda", dtype=torch.bfloat16)
    mm = torch.full((E,), 16, device="cuda", dtype=torch.int32)

    def run():
        sglang_per_token_group_quant_8bit(
            x, 128, torch.float8_e4m3fn, masked_m=mm,
            column_major_scales=True, scale_tma_aligned=True,
            fuse_silu_and_mul=fuse_silu,
        )

    us = bench(run, warmup=10, iters=50)
    tag = "fused silu+quant" if fuse_silu else "quant only"
    return {"name": f"MoE masked 8bit quant ({tag})", "phase": phase,
            "backend": "sglang_per_token_group_quant_8bit",
            "shape": f"[{E},{Mp},{two_n}] masked_m=16", "us": round(us, 2)}


def main():
    print(f"# GPU: {torch.cuda.get_device_name(0)}")
    results = []

    def add(fn, *a, label=""):
        try:
            r = fn(*a); results.append(r)
            print(f"  OK  | {r['name']:34s} | {r['phase']:7s} | {r['us']:9.2f} us | {r['backend']}")
        except Exception as e:
            print(f"  FAIL| {label}: {type(e).__name__}: {e}")
            results.append({"name": label, "error": f"{type(e).__name__}: {e}"})

    print("\n### KV-cache FP8 quant (trtllm FP8 KV path) ###")
    add(bench_kv_quant, 16384, "prefill", label="kv quant prefill")
    add(bench_kv_quant, 16, "decode", label="kv quant decode")

    print("\n### MoE masked grouped 8-bit quant (down-input) ###")
    add(bench_moe_masked_quant, 8, 128, 2048, True, "decode", label="moe quant fused")
    add(bench_moe_masked_quant, 8, 128, 2048, False, "decode", label="moe quant only")

    out = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "logs" / "kimi_quant.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"\n# wrote {out} ({len(results)} entries)")


if __name__ == "__main__":
    main()
