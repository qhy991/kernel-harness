#!/usr/bin/env python3
"""Compare baseline latency under three (+ optional CUPTI) timing methodologies.

Methods:
  1. graph     — CUDA-graph, warm L2, mean over captured replays
                 (rewardbench cuda_graph_bench)
  2. warm_evt  — CUDA-event, warm L2, no L2 flush, no per-iter clone
                 (rewardbench event_bench)
  3. cold_l2   — L2 flush + per-iter clone + median CUDA-event
                 (what opbench's time_cold_l2 did)
  4. cupti     — L2 flush + per-iter clone + median CUPTI device-kernel
                 (testbench evaluate.py authoritative path)

Uses the same glm52_ops frozen inputs + reference backend for every method.
"""
from __future__ import annotations

import argparse
import importlib.util
import statistics
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# glm52_ops_cuda is the CUDA-side operator definition; this script is
# CUDA-specific (compares CUPTI/CUDA-graph/CUDA-event timers). Load by path so
# nothing needs to be on sys.path.
specs = _load("glm52_ops_cmp", REPO / "testbench" / "harness" / "glm52_ops_cuda.py")
from harness.timing import time_callable, time_cold_l2, clone_inputs  # noqa: E402
sys.path.pop(0)

tb_timing = _load("tb_timing_cmp", REPO / "testbench" / "harness" / "timing.py")
_HAVE_CUPTI = tb_timing._HAVE_CUPTI
tb_time_runnable = tb_timing.time_runnable


def warm_event_ms(run_fn, inputs, warmup=5, iters=20) -> float:
    """rewardbench event_bench: warm L2, no flush, no clone, mean over back-to-back calls."""
    for _ in range(warmup):
        run_fn(inputs)
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        run_fn(inputs)
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


def cupti_cold_ms(run_fn, inputs, warmup=10, rep=50) -> float:
    """testbench authoritative: cold L2 + clone + CUPTI/event median."""
    device = next(v.device for v in inputs.values() if torch.is_tensor(v))
    return tb_time_runnable(
        fn=lambda ins: run_fn(ins),
        setup=lambda: clone_inputs(inputs),
        warmup=warmup,
        rep=rep,
        device=str(device),
    )


CASES = [
    ("o_proj", 4096),
    ("q_b", 4096),
    ("fused_qkv_a", 4096),
    ("moe_gate", 4096),
    ("o_proj", 32),
    ("o_proj", 16),
    ("q_b", 32),
    ("moe_gate", 32),
    ("dsa_attn", 4096),
    ("dsa_attn", 32),
    ("index_score", 1024),
    ("index_score", 32),
]


def measure(op: str, M: int, device: torch.device, rep_cold: int, rep_graph: int):
    phase = specs.infer_phase(M)
    inputs = specs.build_inputs(op, phase, M, specs.DEFAULT_S, device, seed=0)

    def run(ins):
        return specs.reference(op, phase, ins)

    # shared warmup so first-method doesn't alone pay compile/autotune
    run(clone_inputs(inputs))
    torch.cuda.synchronize()

    row = {"op": op, "phase": phase, "M": M}
    row["graph"] = time_callable(lambda: run(inputs), num_warmup=5, num_runs=rep_graph)
    row["warm_evt"] = warm_event_ms(run, inputs, warmup=5, iters=rep_graph)
    row["cold_l2"] = time_cold_l2(run, inputs, warmup=10, rep=rep_cold, device=device)
    if _HAVE_CUPTI:
        try:
            row["cupti"] = cupti_cold_ms(run, inputs, warmup=10, rep=rep_cold)
        except Exception as e:
            row["cupti"] = None
            row["cupti_err"] = str(e)[:120]
    else:
        row["cupti"] = None
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--rep-cold", type=int, default=50)
    ap.add_argument("--rep-graph", type=int, default=20)
    ap.add_argument("--ops", nargs="*", default=None, help="subset like o_proj:32 q_b:4096")
    args = ap.parse_args()

    device = torch.device(args.device)
    torch.cuda.set_device(device)

    cases = CASES
    if args.ops:
        cases = []
        for tok in args.ops:
            op, m = tok.split(":")
            cases.append((op, int(m)))

    print(f"device={device} ({torch.cuda.get_device_name(device)})")
    print(f"cupti_available={_HAVE_CUPTI}")
    print(f"rep_cold={args.rep_cold}  rep_graph/warm={args.rep_graph}")
    print()
    hdr = (
        f"{'op':<14s} {'ph':<7s} {'M':>5s} "
        f"{'graph_ms':>10s} {'warm_evt':>10s} {'cold_l2':>10s} {'cupti_ms':>10s} "
        f"{'w/g':>7s} {'c/g':>7s} {'u/g':>7s}"
    )
    print(hdr)
    print("-" * len(hdr))

    rows = []
    for op, M in cases:
        try:
            r = measure(op, M, device, args.rep_cold, args.rep_graph)
        except Exception as e:
            print(f"{op:<14s} {specs.infer_phase(M):<7s} {M:>5d}  FAILED: {e}")
            continue
        rows.append(r)
        g, w, c, u = r["graph"], r["warm_evt"], r["cold_l2"], r["cupti"]
        print(
            f"{r['op']:<14s} {r['phase']:<7s} {r['M']:>5d} "
            f"{g:10.4f} {w:10.4f} {c:10.4f} "
            f"{(u if u is not None else float('nan')):10.4f} "
            f"{w/g:7.3f} {c/g:7.3f} {(u/g if u else float('nan')):7.3f}"
        )
        if r.get("cupti_err"):
            print(f"  cupti_err: {r['cupti_err']}")

    if not rows:
        return 1

    print()
    print("ratio summary (method / graph):")
    for key in ("warm_evt", "cold_l2", "cupti"):
        ratios = [r[key] / r["graph"] for r in rows if r.get(key)]
        if not ratios:
            continue
        print(
            f"  {key:8s}: median={statistics.median(ratios):.3f}x  "
            f"min={min(ratios):.3f}x  max={max(ratios):.3f}x  n={len(ratios)}"
        )
    for phase in ("prefill", "decode"):
        ratios = [r["cold_l2"] / r["graph"] for r in rows if r["phase"] == phase]
        if ratios:
            print(
                f"  cold_l2/{phase}: median={statistics.median(ratios):.3f}x  "
                f"min={min(ratios):.3f}x  max={max(ratios):.3f}x"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
