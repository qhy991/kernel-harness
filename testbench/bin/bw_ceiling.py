#!/usr/bin/env python3
"""Attainable-bandwidth ceiling — the physical BW a pure copy sustains over a byte
footprint, so "distance to roof" is honest instead of spec-peak arithmetic.

Small transfers cannot reach spec HBM (<64 MB may top out at a small fraction of
the nominal peak because fill/drain dominates). So a memory-bound op sitting near its
*attainable* ceiling is at its real roof — an evidence-backed NO-GO by constructive
proof, not the hand-computed estimate the SOP warns against. Feed the measured
bytes/s into glm52_ops.reward(attainable_bw=...) for the honest utilisation.

    python3 testbench/bin/bw_ceiling.py --mb 8
    python3 testbench/bin/bw_ceiling.py --bytes 138000000 --peak-tbps 5.3
    python3 testbench/bin/bw_ceiling.py --sweep

Requires a GPU + the venv (imports torch and the active harness timer).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "harness"))
import torch  # noqa: E402
import timing as tb_timing  # noqa: E402  (harness/timing.py — same CUPTI cold-L2 timer)

# 2 MB … 4 GB, matching the SOP's ramp-curve sample points.
_SWEEP = [2, 6, 9, 16, 64, 138, 256, 1024, 4096]  # MiB


def measure(total_bytes: int, peak_bps: float, warmup: int, rep: int):
    """Copy `total_bytes` of traffic (read half + write half) and report sustained BW."""
    n = max(1, total_bytes // 2)
    src = torch.empty(n, dtype=torch.int8, device="cuda")
    dst = torch.empty(n, dtype=torch.int8, device="cuda")

    def fn(_=None):
        dst.copy_(src)

    t_ms = tb_timing.time_runnable(fn, warmup=warmup, rep=rep, device="cuda")
    bw = (2 * n) / (t_ms * 1e-3)  # bytes/s (read + write)
    return t_ms, bw, bw / peak_bps


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--bytes", type=int, help="footprint in bytes (e.g. an op's bytes_hbm)")
    g.add_argument("--mb", type=float, help="footprint in MiB")
    g.add_argument("--sweep", action="store_true", help="the 2 MB … 4 GB ramp curve")
    ap.add_argument("--peak-tbps", type=float, default=5.3, help="spec HBM peak (MI300X=5.3)")
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--rep", type=int, default=50)
    args = ap.parse_args()

    if not torch.cuda.is_available():
        print("error: no CUDA device (run on a GPU node in the venv)", file=sys.stderr)
        return 2

    peak = args.peak_tbps * 1e12
    if args.sweep:
        sizes = [int(mb * 2 ** 20) for mb in _SWEEP]
    elif args.bytes is not None:
        sizes = [args.bytes]
    elif args.mb is not None:
        sizes = [int(args.mb * 2 ** 20)]
    else:
        ap.error("give one of --bytes / --mb / --sweep")

    print(f"{'footprint':>12}  {'median':>9}  {'attainable':>12}  {'% of spec':>10}")
    for b in sizes:
        t_ms, bw, frac = measure(b, peak, args.warmup, args.rep)
        print(f"{b / 2 ** 20:9.1f} MiB  {t_ms:7.3f}ms  {bw / 1e9:9.1f} GB/s  {frac * 100:8.1f}%")
    print(f"(spec peak {args.peak_tbps} TB/s; a memory-bound op near these numbers is at "
          f"its physical roof — a NO-GO needs no further search.)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
