#!/usr/bin/env python3
"""llm_flops PREFILL bench — drop-in with PR#3 + prior archive winners.

Unswapped ops: llm_flops bench_* verbatim.
Swapped ops: same llm_flops tensors for stock and candidate.
"""
from __future__ import annotations

import argparse
import csv
import json
import time

import torch

from _common import (
    NUM_RUNS,
    NUM_WARMUP,
    PREFILL_SWAPS,
    _HERE,
    _LLM_FLOPS,
    _load_llm_flops_module,
    bench_dropin,
    bf16_dims_for,
    bmm_dims_for,
    gemm_dims_for,
    moe_dims_for,
    print_summary,
)

lf = _load_llm_flops_module("bench_glm5_prefill.py")
lf.NUM_WARMUP = NUM_WARMUP
lf.NUM_RUNS = NUM_RUNS

M_LIST = [1024, 2048, 4096]
OUT_DIR = _HERE / "results"


def _dims_for(name: str, kind: str, M: int, S: int):
    if kind == "fp8_gemm":
        return gemm_dims_for(lf, name, M, S, "prefill")
    if kind == "moe_masked":
        return moe_dims_for(lf, name, M)
    if kind == "bmm":
        return bmm_dims_for(lf, name, M)
    if kind in ("dsa", "score_mqa"):
        return (M,)
    if kind == "bf16_gemm":
        return bf16_dims_for(lf, name, M)
    raise ValueError(kind)


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
    print("GLM-5.2 PREFILL — llm_flops DROP-IN (+ PR#3 胡延 kernels)")
    print("=" * 130)
    print(f"Timing: CUDA Graph (fallback Event)  warmup={NUM_WARMUP} runs={NUM_RUNS}")
    print(f"llm_flops: {_LLM_FLOPS}  seed={args.seed}")
    print(f"M={args.M}  S={args.S}")
    print(f"Mode: {'STOCK ONLY' if args.stock_only else 'DROP-IN SWAPS'}")
    if not args.stock_only:
        active = [f"{k}←{a}" for k, (_, a, _) in PREFILL_SWAPS.items() if a]
        print("Swaps:")
        for line in active:
            print(f"  {line}")
    print("=" * 130)

    all_results = []

    for M in args.M:
        S = args.S
        print(f"\n{'=' * 130}")
        print(f"  M={M}, S={S}")
        print(f"{'=' * 130}")
        print(f"  {'name':<24s} {'impl':<40s} {'avg(ms)':>10s} {'stock(ms)':>10s} "
              f"{'spd':>7s} {'proto':<12s}")
        print(f"  {'-' * 120}")

        for name, category, bench_fn, shape_str in lf.get_all_operators(M, S):
            torch.cuda.empty_cache()
            harness_op, archive, kind = PREFILL_SWAPS.get(name, (None, None, None))
            use_swap = (not args.stock_only) and archive is not None and kind is not None

            try:
                if use_swap:
                    dims = _dims_for(name, kind, M, S)
                    info = bench_dropin(
                        lf, kind, harness_op, "prefill", archive, dims,
                        device, seed=args.seed + M, S=S)
                    avg_ms = float(info["avg_ms"])
                    stock_ms = float(info["stock_ms"])
                    protocol = info["protocol"]
                    impl = info["impl"]
                    source = info["source"]
                else:
                    stock_ms = float(bench_fn(device))
                    avg_ms = stock_ms
                    protocol = "cuda_graph"
                    impl = "llm_flops_stock"
                    source = "llm_flops"
                    archive = None

                spd = stock_ms / avg_ms if avg_ms > 0 else 0.0
                print(f"  {name:<24s} {impl:<40s} {avg_ms:>10.4f} {stock_ms:>10.4f} "
                      f"{spd:>6.2f}x {protocol:<12s}")
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

    print_summary(all_results, args.M, [args.S], "PREFILL")

    csv_path = OUT_DIR / "glm5_prefill_swapped_perf.csv"
    with csv_path.open("w") as f:
        f.write("name,category,M,S,shape,avg_ms,stock_ms,impl,protocol,"
                "archive,same_inputs,dropin\n")
        for r in all_results:
            f.write(
                f"{r['name']},{r.get('category','')},{r['M']},{r['S']},"
                f"\"{r.get('shape','')}\",{r.get('avg_ms',0):.6f},"
                f"{r.get('stock_ms', r.get('avg_ms',0)):.6f},"
                f"{r.get('impl','')},{r.get('protocol','')},{r.get('archive') or ''},"
                f"{r.get('same_inputs', False)},{r.get('dropin', False)}\n")
    json_path = OUT_DIR / "glm5_prefill_swapped_perf.json"
    json_path.write_text(json.dumps(all_results, indent=2))
    print(f"\nResults: {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
