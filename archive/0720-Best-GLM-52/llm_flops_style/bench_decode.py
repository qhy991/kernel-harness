#!/usr/bin/env python3
"""llm_flops DECODE bench — true drop-in: stock & candidate share the same tensors.

Unswapped ops: llm_flops bench_* verbatim.
Swapped ops: build with llm_flops quant helpers, time stock deep_gemm AND
archive candidate.run on those frozen tensors (CUDA Graph).

Usage:
  CUDA_VISIBLE_DEVICES=0 .../python bench_decode.py
  .../bench_decode.py --stock-only
"""
from __future__ import annotations

import argparse
import csv
import json
import time

import torch

from _common import (
    DECODE_SWAPS,
    NUM_RUNS,
    NUM_WARMUP,
    _HERE,
    _LLM_FLOPS,
    _load_llm_flops_module,
    bench_dropin,
    gemm_dims_for,
    moe_dims_for,
    print_summary,
)

lf = _load_llm_flops_module("bench_glm5_decode.py")
lf.NUM_WARMUP = NUM_WARMUP
lf.NUM_RUNS = NUM_RUNS

M_LIST = [16, 32]
OUT_DIR = _HERE / "results"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stock-only", action="store_true")
    ap.add_argument("--M", type=int, nargs="*", default=M_LIST)
    ap.add_argument("--S", type=int, default=65536)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 130)
    print("GLM-5.2 DECODE — llm_flops DROP-IN (same tensors for stock & candidate)")
    print("=" * 130)
    print(f"Timing: CUDA Graph (fallback Event)  warmup={NUM_WARMUP} runs={NUM_RUNS}")
    print(f"llm_flops: {_LLM_FLOPS}  seed={args.seed}")
    print(f"M={args.M}  S={args.S}")
    print(f"Mode: {'STOCK ONLY' if args.stock_only else 'DROP-IN SWAPS'}")
    if not args.stock_only:
        active = [k for k, (_, a, _) in DECODE_SWAPS.items() if a]
        print(f"Swaps: {', '.join(active)}")
    print("=" * 130)

    all_results = []

    for M in args.M:
        S = args.S
        print(f"\n{'=' * 130}")
        print(f"  batch={M}, S={S}")
        print(f"{'=' * 130}")
        print(f"  {'name':<24s} {'impl':<28s} {'avg(ms)':>10s} {'stock(ms)':>10s} "
              f"{'spd':>7s} {'same':>5s} {'proto':<12s}")
        print(f"  {'-' * 110}")

        for name, category, bench_fn, shape_str in lf.get_all_operators_decode(M, S):
            torch.cuda.empty_cache()
            harness_op, archive, kind = DECODE_SWAPS.get(name, (None, None, None))
            use_swap = (not args.stock_only) and archive is not None and kind is not None

            try:
                if use_swap:
                    if kind == "fp8_gemm":
                        dims = gemm_dims_for(lf, name, M, S, "decode")
                    else:
                        dims = moe_dims_for(lf, name, M)
                    info = bench_dropin(
                        lf, kind, harness_op, "decode", archive, dims,
                        device, seed=args.seed + M)
                    avg_ms = float(info["avg_ms"])
                    stock_ms = float(info["stock_ms"])
                    protocol = info["protocol"]
                    impl = info["impl"]
                    same = True
                    source = info["source"]
                else:
                    stock_ms = float(bench_fn(device))
                    avg_ms = stock_ms
                    protocol = "cuda_graph"
                    impl = "llm_flops_stock"
                    same = True
                    source = "llm_flops"
                    archive = None

                spd = stock_ms / avg_ms if avg_ms > 0 else 0.0
                print(f"  {name:<24s} {impl:<28s} {avg_ms:>10.4f} {stock_ms:>10.4f} "
                      f"{spd:>6.2f}x {'Y' if same else 'N':>5s} {protocol:<12s}")
                all_results.append({
                    "name": name, "category": category, "M": M, "S": S,
                    "shape": shape_str, "avg_ms": avg_ms, "stock_ms": stock_ms,
                    "impl": impl, "protocol": protocol, "source": source,
                    "archive": archive if use_swap else None,
                    "same_inputs": True, "dropin": True,
                })
            except Exception as e:
                print(f"  {name:<24s} FAILED: {type(e).__name__}: {e}")
                all_results.append({
                    "name": name, "category": category, "M": M, "S": S,
                    "shape": shape_str, "avg_ms": 0, "error": str(e),
                })
            time.sleep(0.05)

    print_summary(all_results, args.M, [args.S], "DECODE")

    csv_path = OUT_DIR / "glm5_decode_swapped_perf.csv"
    with csv_path.open("w") as f:
        f.write("name,category,batch,S,shape,avg_ms,stock_ms,impl,protocol,"
                "archive,same_inputs,dropin\n")
        for r in all_results:
            f.write(
                f"{r['name']},{r.get('category','')},{r['M']},{r['S']},"
                f"\"{r.get('shape','')}\",{r.get('avg_ms',0):.6f},"
                f"{r.get('stock_ms', r.get('avg_ms',0)):.6f},"
                f"{r.get('impl','')},{r.get('protocol','')},{r.get('archive') or ''},"
                f"{r.get('same_inputs', False)},{r.get('dropin', False)}\n")
    json_path = OUT_DIR / "glm5_decode_swapped_perf.json"
    json_path.write_text(json.dumps(all_results, indent=2))
    print(f"\nResults: {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
