"""Fast, in-process ADVISORY profiler for the agent's inner optimization loop.

`quick_latency()` gives an approximate median/min microsecond latency using CUDA events
with almost no fixed overhead — no subprocess, no CUPTI, few reps, warm L2 by default.
`profile_task()` loads a task's recipe once and profiles one shape, and adds a roofline
hint (achieved GB/s vs B200 HBM peak, arithmetic intensity, memory-vs-compute bound) so
the agent knows *which direction* to optimize, not just the current number.

ADVISORY ONLY — a compass for exploration, never the verdict. The win/lose gate is always
evaluate.py's authoritative CUPTI / cold-L2 / 100-rep measurement, which can legitimately
disagree: warm-vs-cold L2 flips rankings for memory-bound kernels, and few-rep variance
can swamp a real few-% delta at ~µs latencies. Profile to explore; evaluate to confirm.

    from harness.profile import quick_latency, profile_task
    r = profile_task(task_dir, "solution.py", shape=16)   # {'median_us', 'min_us', 'roofline': {...}}

    python -m harness.profile <task_dir> [--solution X] [--shape M] [--reps N] [--cold-l2]
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Callable, Optional

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from harness.inputs import resolve_axes, build_inputs, normalize_outputs, eval_expr
from harness.timing import clone_args, _empty_cache, _clear_cache

# Approximate NVIDIA B200 peaks (label as approximate; used only for the % context).
HBM_GBPS = 8000.0          # ~8 TB/s HBM3e
FP_PEAK_TFLOPS = 2250.0    # dense bf16 tensor-core (fp8 ~2x); rough, for context only


def quick_latency(fn: Callable, setup: Optional[Callable] = None, reps: int = 20,
                  warmup: int = 5, cold_l2: bool = False, device: str = "cuda") -> dict:
    """Approximate median/min latency (µs) via CUDA events, in-process, low overhead.

    fn(args) runs the kernel; setup() returns fresh args each iteration (untimed). Warm L2
    by default (fastest); pass cold_l2=True to match the verdict's cold-cache regime.

    Note: event record-to-record includes kernel-launch overhead (a ~few-µs additive floor)
    that CUPTI isolates out, so these numbers run a few µs above the verdict's device-kernel
    time and that floor compresses small deltas. Trust this for direction and large wins;
    use evaluate.py to confirm fine (~few-%) differences.
    """
    if setup is None:
        _fn = fn
        def fn(_):  # noqa: E731
            return _fn()
        def setup():  # noqa: E731
            return None
    buf = _empty_cache(device) if cold_l2 else None
    torch.cuda.synchronize()
    for _ in range(warmup):
        if cold_l2:
            _clear_cache(buf)
        fn(setup())
    torch.cuda.synchronize()
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(reps)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(reps)]
    for i in range(reps):
        args = setup()
        if cold_l2:
            _clear_cache(buf)
        torch.cuda.synchronize()
        starts[i].record()
        fn(args)
        ends[i].record()
    torch.cuda.synchronize()
    ms = [starts[i].elapsed_time(ends[i]) for i in range(reps)]
    return {"median_us": statistics.median(ms) * 1e3, "min_us": min(ms) * 1e3, "reps": reps}


def _bytes(tensors) -> int:
    return sum(t.numel() * t.element_size() for t in tensors if isinstance(t, torch.Tensor))


def roofline(input_tensors, output_tensors, latency_us: float, flops: Optional[int] = None) -> dict:
    """Bandwidth roofline from actual tensor traffic; optional compute roofline if flops given.

    'moved' counts every input tensor read + every output tensor written once (approx: it
    ignores cache reuse and DPS in/out overlap). Good enough for a memory-bound signal.
    """
    moved = _bytes(input_tensors) + _bytes(output_tensors)
    gbps = moved / (latency_us * 1e3) if latency_us > 0 else 0.0   # bytes / (µs*1e3) = GB/s
    pct_hbm = 100.0 * gbps / HBM_GBPS
    out = {"bytes_moved": moved, "achieved_gbps": round(gbps, 1),
           "hbm_peak_gbps": HBM_GBPS, "pct_of_hbm_peak": round(pct_hbm, 1)}
    if flops:
        tflops = flops / (latency_us * 1e-6) / 1e12 if latency_us > 0 else 0.0
        ridge = FP_PEAK_TFLOPS * 1e12 / (HBM_GBPS * 1e9)   # FLOP/byte machine balance
        ai = flops / moved if moved else 0.0
        out.update({"flops": flops, "achieved_tflops": round(tflops, 1),
                    "arithmetic_intensity": round(ai, 2), "ridge_point": round(ridge, 1),
                    "pct_of_fp_peak": round(100.0 * tflops / FP_PEAK_TFLOPS, 1)})
        # roofline region: AI above the ridge -> compute-bound, below -> memory-bound.
        # ridge uses the bf16 FP peak, so a "compute-bound" call is conservative for fp8.
        out["bound"] = "compute-bound" if ai >= ridge else "memory-bound"
    else:
        out["bound"] = ("memory-bound (near HBM peak)" if pct_hbm >= 50 else
                        f"not memory-bound ({out['pct_of_hbm_peak']}% HBM) — compute- or "
                        "launch-bound; declare flops_expr to disambiguate")
    return out


def _load_run_and_inputs(task_dir: Path, solution: str):
    defn = json.loads((task_dir / "definition.json").read_text())
    ref_ns: dict = {"__name__": "<reference>"}
    exec(compile(defn["reference"], "<reference>", "exec"), ref_ns)
    get_inputs = ref_ns[defn["custom_inputs_entrypoint"]]
    cand_ns: dict = {"__name__": solution}
    exec(compile((task_dir / solution).read_text(), solution, "exec"), cand_ns)
    return defn, get_inputs, cand_ns["run"]


def _pick_shape(task_dir: Path, shape: Optional[int]) -> int:
    if shape is not None:
        return shape
    sweep = json.loads((task_dir / "task.json").read_text()).get("sweep", [16])
    return sweep[len(sweep) // 2]   # a representative mid-sweep shape


def profile_task(task_dir, solution: str = "solution.py", shape: Optional[int] = None,
                 reps: int = 20, cold_l2: bool = False, sglang_python: Optional[str] = None) -> dict:
    """Load a task's recipe once, profile one shape's candidate run(), + roofline hint.

    The in-process caller (an agent loop under evaluate.py's env) already resolves the
    reference's imports; pass sglang_python only for standalone use where a specific
    sglang checkout's `python/` dir must shadow a mismatched installed one.
    """
    task_dir = Path(task_dir)
    if sglang_python:
        sys.path.insert(0, sglang_python)
    defn, get_inputs, run = _load_run_and_inputs(task_dir, solution)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    m = _pick_shape(task_dir, shape)
    resolved = resolve_axes(defn, {"M": m})
    inputs = build_inputs(get_inputs, defn, resolved, device)

    out_tensors = normalize_outputs(run(*clone_args(inputs)), defn, device)
    lat = quick_latency(fn=lambda a: run(*a), setup=lambda: clone_args(inputs),
                        reps=reps, cold_l2=cold_l2, device=device)

    flops = None
    fexpr = defn.get("flops_expr")   # optional per-task; e.g. "2*M*K*N" for a GEMM
    if fexpr:
        try:
            flops = eval_expr(fexpr, resolved)
        except Exception:
            flops = None
    rl = roofline(inputs, out_tensors, lat["median_us"], flops)
    return {"shape": m, "cold_l2": cold_l2, **lat, "roofline": rl}


def main():
    ap = argparse.ArgumentParser(description="Fast advisory kernel profiler (NOT the verdict).")
    ap.add_argument("task_dir", type=Path)
    ap.add_argument("--solution", default="solution.py")
    ap.add_argument("--shape", type=int, default=None, help="M (default: mid-sweep)")
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--cold-l2", action="store_true", help="match the verdict's cold-cache regime")
    ap.add_argument("--sglang-python", default=None,
                    help="a sglang checkout's python/ dir to shadow a mismatched installed sglang")
    args = ap.parse_args()
    r = profile_task(args.task_dir, args.solution, args.shape, args.reps,
                     args.cold_l2, args.sglang_python)
    rl = r["roofline"]
    print(f"{args.task_dir.name}  M={r['shape']}  ({args.solution}, {r['reps']} reps, "
          f"{'cold' if r['cold_l2'] else 'warm'} L2)")
    print(f"  latency: median {r['median_us']:.2f} us   min {r['min_us']:.2f} us")
    line = (f"  roofline: {rl['achieved_gbps']} GB/s ({rl['pct_of_hbm_peak']}% HBM)"
            f"  bound={rl['bound']}")
    if "achieved_tflops" in rl:
        line += f"  {rl['achieved_tflops']} TFLOP/s ({rl['pct_of_fp_peak']}% peak), AI={rl['arithmetic_intensity']}"
    print(line)
    print("  (ADVISORY — use evaluate.py for the authoritative correctness+latency verdict)")


if __name__ == "__main__":
    main()
