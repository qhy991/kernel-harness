#!/usr/bin/env python3
"""Measure the opbench REFERENCE baseline for the GLM-5.2-on-MI300X target ops and
emit the optimization target = min(hardware roofline, 1.5x baseline).

The baseline is the exact latency denominator the opbench gate uses (glm52_ops.reference
timed under the HIP-event cold-L2 protocol), so "target" here is the number a KDA-Pilot
candidate is chasing in run.sh. In reward terms (reward = achieved/roofline, 1.0 = at the
hardware roofline):

    target_reward   = min(1.0, 1.5 * baseline_reward)         # roofline caps it
    target_latency  = max(roofline_latency, baseline_latency / 1.5)

Writes amd_glm5_targets.csv (baseline | target columns side by side) and, with --augment,
appends a `target_reward`/`target_desc` column to the right of `reward` in the existing
rewardbench reward CSVs.

Run:  .venv/bin/python rewardbench/amd/emit_targets.py [--augment] [--device cuda:0]
      [--repeat 7] [--iterations 20]
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import os
import statistics
import sys
from pathlib import Path

os.environ.setdefault("KERNEL_HARNESS_PLATFORM", "rocm")
os.environ.setdefault("KERNEL_HARNESS_PROFILE", "rocm-mi300x")
os.environ.setdefault("KERNEL_HARNESS_PROVIDER", "torch-triton-rocm")
os.environ.setdefault("KERNEL_HARNESS_TIMER", "event")

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parents[1]
sys.path.insert(0, str(_REPO))

import torch  # noqa: E402
from functools import partial  # noqa: E402


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


ops = _load("glm52_ops_amd", _REPO / "testbench" / "harness" / "glm52_ops_amd.py")
tb_timing = _load("tb_timing", _REPO / "testbench" / "harness" / "timing.py")

# The 3 optimization targets (amd-glm-object.csv): o_proj+index_k prefill, MLA decode.
TARGETS = [
    ("o_proj", "prefill", (1024, 2048, 4096)),
    ("index_k", "prefill", (1024, 2048, 4096)),
    ("dsa_attn", "decode", (16, 32)),
]
TARGET_MULT = 1.5


def measure(op, phase, M, device, repeat, iters):
    S = ops.DEFAULT_S
    inputs = ops.build_inputs(op, phase, M, S, device, 0)
    ref_fn = partial(ops.reference, op, phase)
    setup = lambda: {k: (v.clone() if torch.is_tensor(v) else v) for k, v in inputs.items()}  # noqa: E731
    samples = [tb_timing.time_runnable(ref_fn, setup=setup, warmup=3, rep=iters, device=device)
               for _ in range(repeat)]
    lat_ms = statistics.median(samples)
    flops, byts, dtype = ops.cost(op, phase, M, S)
    r = ops.reward(lat_ms, flops, byts, dtype)
    ceiling = min(ops.PEAK_FLOPS[dtype], (flops / byts) * ops.HBM_BYTES_PER_S)
    roofline_lat_ms = (flops / ceiling) * 1e3          # reward == 1.0
    base_reward = r["reward"]
    target_reward = min(1.0, TARGET_MULT * base_reward)
    target_lat_ms = max(roofline_lat_ms, lat_ms / TARGET_MULT)
    return {
        "op": op, "phase": phase, "M": M, "S": S,
        "family": ops.family(op), "flops": flops, "bytes_hbm": byts,
        "ai": round(r["arithmetic_intensity"], 2), "ridge": round(r["ridge"], 2),
        "bound": r["bound"], "compute_dtype": dtype,
        "baseline_lat_ms": round(lat_ms, 5), "baseline_reward": round(base_reward, 5),
        "target_reward": round(target_reward, 5),
        "target_lat_ms": round(target_lat_ms, 5),
        "target_speedup_x": round(lat_ms / target_lat_ms, 4),
        "roofline_lat_ms": round(roofline_lat_ms, 5),
        "target_desc": (f"min(roofline=1.000, 1.5x_base={TARGET_MULT*base_reward:.4f}) "
                        f"-> reward>={target_reward:.4f} @ <= {target_lat_ms*1e3:.1f}us"),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--repeat", type=int, default=7)
    ap.add_argument("--iterations", type=int, default=20)
    ap.add_argument("--augment", action="store_true",
                    help="also add a target_reward/target_desc column to the existing "
                         "rewardbench reward CSVs, right of `reward`")
    args = ap.parse_args()
    device = torch.device(args.device)
    torch.cuda.set_device(device)

    rows = []
    print(f"measuring opbench reference baselines on {args.device} "
          f"(repeat={args.repeat} iters={args.iterations})\n")
    hdr = f"{'op':>10} {'phase':>7} {'M':>5} {'bound':>7} {'base_us':>9} {'base_rw':>8} {'target_rw':>9} {'tgt_us':>9} {'tgt_x':>6}"
    print(hdr)
    for op, phase, Ms in TARGETS:
        for M in Ms:
            r = measure(op, phase, M, device, args.repeat, args.iterations)
            rows.append(r)
            print(f"{op:>10} {phase:>7} {M:>5} {r['bound']:>7} "
                  f"{r['baseline_lat_ms']*1e3:>9.1f} {r['baseline_reward']:>8.4f} "
                  f"{r['target_reward']:>9.4f} {r['target_lat_ms']*1e3:>9.1f} {r['target_speedup_x']:>6.2f}")

    out = _HERE / "amd_glm5_targets.csv"
    cols = list(rows[0].keys())
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {out.relative_to(_REPO)}  ({len(rows)} rows)")

    if args.augment:
        for name in ("amd_glm5_ops_prefill_reward.csv", "amd_glm5_ops_decode_reward.csv"):
            _augment(_HERE / name)


def _augment(path: Path):
    """Add target_reward = min(1.0, 1.5*reward) + target_desc right of `reward`."""
    if not path.exists():
        print(f"  (skip {path.name}: not found)")
        return
    with open(path) as f:
        r = list(csv.reader(f))
    hdr = r[0]
    if "target_reward" in hdr:
        ridx = hdr.index("reward")
        rows = r[1:]
    else:
        ridx = hdr.index("reward")
        hdr = hdr[:ridx + 1] + ["target_reward", "target_desc"] + hdr[ridx + 1:]
        rows = []
        for row in r[1:]:
            try:
                base = float(row[ridx])
                tr = min(1.0, 1.5 * base)
                desc = f"min(1.000,1.5x{base:.4f})={tr:.4f}"
            except (ValueError, IndexError):
                tr, desc = "", ""
            rows.append(row[:ridx + 1] + [f"{tr:.5f}" if tr != "" else "", desc] + row[ridx + 1:])
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(hdr)
        w.writerows(rows)
    print(f"  augmented {path.name} with target_reward/target_desc (right of reward)")


if __name__ == "__main__":
    main()
