#!/usr/bin/env python3
"""MiniMax-M3 / DeepSeek-V3.2-style DSA indexer QUANT kernels (2nd model).

These run every layer in the DSA sparse-attention model but are ABSENT from the
Kimi CSV (Kimi-K2.7 has no indexer) and are done OUTSIDE the timed loop in the
harness's own indexer bench. Output -> minimax_m3_all.csv (separate model).

DSA/MiniMax-M3 profile (harness DSA_V32): hidden=6144, num_index_heads=32,
index_head_dim=128, kv_lora=512, qk_rope=64. Prefill 16384 tok, decode 16.

  Indexer q/k FP8 act-quant     -> dsa.triton_kernel.act_quant
  Hadamard rotation (pre-quant) -> dsa_indexer.rotate_activation (hadamard_transform)
  Index K-cache FP8 quant       -> dsa.quant_k_cache.quantize_k_cache_separate
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
from bench_kimi_real_kernels import bench  # noqa: E402

from sglang.srt.layers.attention.dsa.triton_kernel import act_quant  # noqa: E402
from sglang.srt.layers.attention.dsa.dsa_indexer import rotate_activation  # noqa: E402
from sglang.srt.layers.attention.dsa.quant_k_cache import quantize_k_cache_separate  # noqa: E402

H_IDX, D_IDX = 32, 128          # num_index_heads, index_head_dim
D_CKV, D_ROPE = 512, 64


def bench_act_quant_q(n, phase):
    x = torch.randn(n, H_IDX * D_IDX, device="cuda", dtype=torch.bfloat16)

    def run():
        act_quant(x, 128)

    us = bench(run, warmup=10, iters=50)
    return {"name": "Indexer Q FP8 act-quant", "phase": phase,
            "backend": "dsa.act_quant (Triton, block=128)",
            "shape": f"[{n},{H_IDX}x{D_IDX}]", "us": round(us, 2)}


def bench_act_quant_k(n, phase):
    x = torch.randn(n, D_IDX, device="cuda", dtype=torch.bfloat16)

    def run():
        act_quant(x, 128)

    us = bench(run, warmup=10, iters=50)
    return {"name": "Indexer K FP8 act-quant", "phase": phase,
            "backend": "dsa.act_quant (Triton, block=128)",
            "shape": f"[{n},{D_IDX}] (MQA index-K)", "us": round(us, 2)}


def bench_hadamard(n, phase):
    x = torch.randn(n, D_IDX, device="cuda", dtype=torch.bfloat16)

    def run():
        rotate_activation(x)

    us = bench(run, warmup=10, iters=50)
    return {"name": "Indexer Hadamard rotate", "phase": phase,
            "backend": "dsa_indexer.rotate_activation (hadamard_transform)",
            "shape": f"[{n},{D_IDX}]", "us": round(us, 2)}


def bench_kcache_quant(n, phase):
    kn = torch.randn(n, D_CKV, device="cuda", dtype=torch.bfloat16)
    kr = torch.randn(n, D_ROPE, device="cuda", dtype=torch.bfloat16)

    def run():
        quantize_k_cache_separate(kn, kr)

    us = bench(run, warmup=10, iters=50)
    return {"name": "Index K-cache FP8 quant", "phase": phase,
            "backend": "dsa.quantize_k_cache_separate (tiled fp8)",
            "shape": f"nope[{n},{D_CKV}] rope[{n},{D_ROPE}]", "us": round(us, 2)}


def main():
    print(f"# GPU: {torch.cuda.get_device_name(0)}  (MiniMax-M3 / DSA indexer quant)")
    results = []

    def add(fn, *a, label=""):
        try:
            r = fn(*a); results.append(r)
            print(f"  OK  | {r['name']:26s} | {r['phase']:7s} | {r['us']:9.2f} us | {r['backend']}")
        except Exception as e:
            print(f"  FAIL| {label}: {type(e).__name__}: {e}")
            results.append({"name": label, "error": f"{type(e).__name__}: {e}"})

    for n, ph in ((16384, "prefill"), (16, "decode")):
        add(bench_act_quant_q, n, ph, label=f"q act_quant {ph}")
        add(bench_act_quant_k, n, ph, label=f"k act_quant {ph}")
        add(bench_hadamard, n, ph, label=f"hadamard {ph}")
        add(bench_kcache_quant, n, ph, label=f"kcache quant {ph}")

    out_json = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "logs" / "minimax_indexer_quant.json"
    out_json.write_text(json.dumps(results, indent=2))

    # write a standalone minimax_m3_all.csv (same 17-col schema as kimi_k27_all.csv)
    out_csv = out_json.with_name("minimax_m3_all.csv")
    HEADER = ["模型", "层类型", "算子名称", "执行阶段", "数据精度", "测量类型",
              "sweep轴", "sweep取值", "算子规模/shape", "latency_us", "TFLOPS",
              "sglang入口", "底层Kernel", "实现类型", "B200实测后端", "备选实现", "备注/发现"]
    ARCH_NOTE = "DSA稀疏注意力indexer量化; hidden=6144 index_heads=32 index_dim=128; Kimi-K2.7无此路径"
    rows = [HEADER]
    for r in results:
        if "us" not in r:
            continue
        rows.append(["MiniMax-M3", "dsa_indexer", r["name"], r["phase"], "BF16->FP8",
                     "recovered", "", "", r["shape"], r["us"], "",
                     "nsa_indexer / dsa", r["backend"], "Triton/JIT", r["backend"], "",
                     ARCH_NOTE])
    csv.writer(open(out_csv, "w", newline="")).writerows(rows)
    print(f"\n# wrote {out_json} and {out_csv} ({len(rows)-1} rows)")


if __name__ == "__main__":
    main()
