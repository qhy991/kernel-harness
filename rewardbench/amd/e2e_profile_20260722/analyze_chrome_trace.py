#!/usr/bin/env python3
"""Recompute GLM-5.2 e2e prefill op-share tables from chrome traces.

Example:
  python analyze_chrome_trace.py \\
    --trace-dir /mnt/public/qinhaiyan/glm52_profile_traces \\
    --glob 'glm52_prefill_20260722_172150_*_prefill.trace.json.gz' \\
    --out-csv e2e_prefill_op_share.csv
"""
from __future__ import annotations

import argparse
import gzip
import json
import re
from collections import defaultdict
from pathlib import Path

SKIP = re.compile(
    r"hipMemset|hipMemcpy|Memcpy |hipLaunch|hipModule|hipExt|hipPointer|"
    r"hipStream|hipEvent|hipDevice|cudaDevice|cudaLaunch|cudaMemcpy|"
    r"ProfilerStep|^step\[",
    re.I,
)


def group_of(name: str) -> str:
    if "cross_device_reduce" in name or "all_reduce" in name.lower():
        return "AllReduce (TP)"
    if "fused_moe_kernel" in name:
        return "MoE fused GEMM"
    if "mla_dec" in name or "_fwd_kernel_stage2" in name or "set_mla_kv" in name:
        return "MLA attention"
    if "gluon_deepgemm" in name or "pa_mqa" in name or "paged_mqa" in name:
        return "DSA indexer"
    if "get_valid_kv_indices" in name:
        return "DSA get_valid_kv_indices"
    if (
        "fp8gemm" in name
        or "blockscale_b_preshuffle" in name
        or "blockscale_BpreShuffle" in name
        or "BBS_BH" in name
    ):
        return "FP8 GEMM (linear)"
    if (
        "per_token_group_quant" in name
        or "scaled_quant" in name
        or "act_and_mul" in name
    ):
        return "Quant / act"
    if "rmsnorm" in name.lower() or "Layernorm" in name:
        return "RMSNorm"
    if "topk" in name.lower() or "mbtopk" in name or "grouped_topk" in name:
        return "TopK"
    if (
        "elementwise" in name
        or "CatArray" in name
        or "reduce_kernel" in name
        or "scatter_gather" in name
        or "masked_fill" in name
        or "FillFunctor" in name
        or "arange" in name
    ):
        return "ATen elementwise/misc"
    if "rocprim" in name:
        return "rocPRIM"
    if "Cijk_" in name or "gemm" in name.lower() or "Gemm" in name:
        return "Other GEMM"
    return "Other"


def analyze_one(path: Path) -> tuple[int, dict[str, float]]:
    m = re.search(r"input(\d+)", path.name)
    if not m:
        raise ValueError(f"cannot parse input_len from {path.name}")
    L = int(m.group(1))
    with gzip.open(path, "rt") as f:
        events = json.load(f)["traceEvents"]
    by_g: dict[str, float] = defaultdict(float)
    for e in events:
        if e.get("ph") != "X" or e.get("cat") != "kernel":
            continue
        name = e.get("name") or ""
        if SKIP.search(name):
            continue
        dur = float(e.get("dur") or 0)
        if dur <= 0:
            continue
        by_g[group_of(name)] += dur
    return L, dict(by_g)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace-dir", type=Path, required=True)
    ap.add_argument("--glob", default="*_prefill.trace.json.gz")
    ap.add_argument("--out-csv", type=Path, default=Path("e2e_prefill_op_share.csv"))
    ap.add_argument(
        "--wall-ms",
        action="append",
        default=[],
        help="optional INPUT:WALL_MS e.g. 4096:738.21 (repeatable)",
    )
    args = ap.parse_args()
    wall = {}
    for item in args.wall_ms:
        k, v = item.split(":")
        wall[int(k)] = float(v)

    rows = []
    for path in sorted(args.trace_dir.glob(args.glob)):
        L, by_g = analyze_one(path)
        total = sum(by_g.values())
        for g, d in sorted(by_g.items(), key=lambda x: -x[1]):
            rows.append(
                {
                    "input_len": L,
                    "wall_ms": wall.get(L, ""),
                    "gpu_kernel_ms": round(total / 1000, 2),
                    "group": g,
                    "gpu_ms": round(d / 1000, 2),
                    "pct_of_gpu": round(100 * d / total, 2) if total else 0.0,
                }
            )
        print(f"[ok] input={L} gpu_ms={total/1000:.2f} file={path.name}")

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.out_csv.open("w") as f:
        f.write("input_len,wall_ms,gpu_kernel_ms,group,gpu_ms,pct_of_gpu\n")
        for r in rows:
            f.write(
                f"{r['input_len']},{r['wall_ms']},{r['gpu_kernel_ms']},"
                f"{r['group']},{r['gpu_ms']},{r['pct_of_gpu']}\n"
            )
    print(f"wrote {args.out_csv}")


if __name__ == "__main__":
    main()
