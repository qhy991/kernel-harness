"""Self-contained kernel correctness and performance driver.

For a task dir + a candidate file it: resolves each workload's axes, builds inputs via the
reference's get_inputs, runs the reference (oracle) and the candidate on independent input
clones, checks correctness, times the candidate (median device-kernel ms), and writes
traces.json in the shape evaluate.py consumes. Depends only on torch (+ optional cupti).

    python testbench/harness/driver.py <task_dir> --solution-name solution.py -o OUT [--iterations N] [--max-workloads N]
"""
from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from harness import correctness as C
from harness import reward_hack as RH
from harness.inputs import resolve_axes, build_inputs, normalize_outputs
from harness.timing import clone_args, time_runnable


def _load_ns(source: str, name: str) -> dict:
    ns: dict = {"__name__": name, "__file__": name}
    exec(compile(source, name, "exec"), ns)
    return ns


def _eval_workload(defn, wl, get_inputs, ref_run, cand_run, device, iters):
    axes = wl.get("axes", {})
    tol = wl.get("tolerance", C.DEFAULT_TOL)
    resolved = resolve_axes(defn, axes)

    def trace(status, corr=None, latency=None, log=""):
        return {"workload": {"axes": axes},
                "evaluation": {"status": status,
                               "performance": {"latency_ms": latency} if latency is not None else None,
                               "correctness": corr, "log": log}}

    try:
        inputs = build_inputs(get_inputs, defn, resolved, device)
    except Exception as e:
        return trace("RUNTIME_ERROR", log=f"gen_inputs failed: {e}\n{traceback.format_exc()[:400]}")

    # Reference on a private clone so any in-place mutation cannot leak to the candidate.
    try:
        ref_out = normalize_outputs(ref_run(*clone_args(inputs)), defn, device)
    except Exception as e:
        return trace("RUNTIME_ERROR", log=f"Reference run() failed: {e}\n{traceback.format_exc()[:400]}")

    RH.check_monkey_patch()
    try:
        raw = cand_run(*clone_args(inputs))
    except Exception as e:
        return trace("RUNTIME_ERROR", log=f"Solution run() failed: {e}\n{traceback.format_exc()[:400]}")
    try:
        cand_out = normalize_outputs(raw, defn, device)
        RH.check_lazy_outputs(cand_out)
    except RH.RewardHackDetected as e:
        return trace("REWARD_HACK", log=str(e))
    except Exception as e:
        return trace("RUNTIME_ERROR", log=f"output normalization failed: {e}")

    if len(cand_out) != len(ref_out):
        return trace("INCORRECT", log=f"output count {len(cand_out)} != reference {len(ref_out)}")
    agg = {"max_absolute_error": 0.0, "max_relative_error": 0.0, "has_nan": False, "has_inf": False}
    exceeded = False
    for c, r in zip(cand_out, ref_out):
        if c.shape != r.shape:
            return trace("INCORRECT", corr=agg, log=f"shape {tuple(c.shape)} != {tuple(r.shape)}")
        stats, ex = C.compute_error_stats(c, r, tol)
        exceeded = exceeded or ex
        agg["has_nan"] |= bool(stats["has_nan"]); agg["has_inf"] |= bool(stats["has_inf"])
        for k in ("max_absolute_error", "max_relative_error"):
            if stats[k] is not None:
                agg[k] = max(agg[k] or 0.0, stats[k])
    if exceeded or agg["has_nan"] or agg["has_inf"]:
        return trace("INCORRECT", corr=agg, log="tolerance exceeded" if exceeded else "nan/inf")

    # Correct -> time the candidate (median device-kernel ms), fresh clone per iteration.
    latency = time_runnable(fn=lambda a: cand_run(*a),
                            setup=lambda: clone_args(inputs), rep=iters)

    # Post-timing guards. The timed calls above are not output-checked, so (a) re-verify
    # one fresh call against the oracle — a stateful candidate that computes honestly
    # while checked but goes lazy under timing fails here; (b) re-check the timer
    # identity — a patch installed inside run() itself would otherwise evade the pre-run
    # check for this workload's CUDA-events fallback timing.
    try:
        RH.check_monkey_patch()
        recheck = normalize_outputs(cand_run(*clone_args(inputs)), defn, device)
        RH.check_lazy_outputs(recheck)
        if len(recheck) != len(ref_out):
            raise RH.RewardHackDetected("post-timing output count changed")
        for c, r in zip(recheck, ref_out):
            if c.shape != r.shape or C.compute_error_stats(c, r, tol)[1]:
                raise RH.RewardHackDetected(
                    "output no longer matches the oracle after timing — "
                    "stateful/lazy candidate rejected")
    except RH.RewardHackDetected as e:
        return trace("REWARD_HACK", log=str(e))
    except Exception as e:
        return trace("RUNTIME_ERROR", log=f"post-timing recheck failed: {e}")

    return trace("PASSED", corr=agg, latency=latency)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("task_dir", type=Path)
    ap.add_argument("--solution-name", default="solution.py")
    ap.add_argument("--iterations", type=int, default=50)
    ap.add_argument("--max-workloads", type=int, default=None)
    ap.add_argument("-o", "--out", type=Path, required=True)
    args = ap.parse_args()

    defn = json.loads((args.task_dir / "definition.json").read_text())
    ref_ns = _load_ns(defn["reference"], "<reference>")
    get_inputs = ref_ns[defn["custom_inputs_entrypoint"]]
    ref_run = ref_ns["run"]
    cand_ns = _load_ns((args.task_dir / args.solution_name).read_text(), args.solution_name)
    cand_run = cand_ns["run"]

    workloads = [json.loads(l) for l in (args.task_dir / "workload.jsonl").read_text().splitlines() if l.strip()]
    if args.max_workloads:
        workloads = workloads[:args.max_workloads]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    traces = []
    for wl in workloads:
        try:
            traces.append(_eval_workload(defn, wl, get_inputs, ref_run, cand_run, device, args.iterations))
        except Exception as e:
            traces.append({"workload": {"axes": wl.get("axes", {})},
                           "evaluation": {"status": "RUNTIME_ERROR", "performance": None,
                                          "correctness": None, "log": f"driver error: {e}"}})

    out_dir = args.out / args.task_dir.name
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "traces.json").write_text(json.dumps(traces, indent=2))
    n_pass = sum(1 for t in traces if t["evaluation"]["status"] == "PASSED")
    print(f"{args.task_dir.name}: {n_pass}/{len(traces)} PASSED ({args.solution_name})")


if __name__ == "__main__":
    main()
