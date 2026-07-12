#!/usr/bin/env python3
"""Evaluate a candidate kernel against the sglang baseline — framework-agnostic.

Any agent (or human) edits `solution.py` in a task dir, then runs:

    python testbench/bin/evaluate.py <task_dir>

It measures the sglang baseline (reference.py, cached), runs the candidate, checks
correctness against the sglang output per shape, benchmarks it, and prints a verdict
as a human table AND a machine-readable JSON block (between VERDICT_JSON markers) so
any optimization loop can parse the correctness + efficiency feedback.

Exit code:  0 = WIN (correct on every shape AND faster on every shape)
            1 = correct but not faster everywhere
            2 = incorrect / error

Options:
    --solution NAME     candidate file in the task dir (default: solution.py)
    --iterations N      timing iterations per shape (default: 50)
    --repeat K          re-time candidate AND baseline over K independent process
                        runs and gate the win on the worst-case margin, so a small
                        speedup is judged against run-to-run noise (default: 1)
    --max-workloads N   evaluate only the first N shapes (quick check)
    --refresh-baseline  re-measure the baseline instead of using the cache
    --no-baseline       correctness-only (skip baseline; speedups reported as null)
"""
import argparse
import glob
import hashlib
import json
import math
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import VENV, SGLANG_DIR as SGLANG, CUDA_HOME, resolve_sglang_dir


def _env(sglang_dir=None):
    e = dict(os.environ)
    e["PATH"] = f"{VENV}/bin:" + e.get("PATH", "")
    e["PYTHONPATH"] = f"{sglang_dir or SGLANG}/python:" + e.get("PYTHONPATH", "")
    e.setdefault("CUDA_HOME", str(CUDA_HOME))
    return e


def _sglang_dir_for(task_dir: Path):
    """A task may pin its own sglang build (e.g. DSA tasks need the amd_add_m3 tree)
    via task.json 'sglang_dir'. Falls back to SGLANG_DIR / the default checkout."""
    tj = task_dir / "task.json"
    if tj.exists():
        pinned = json.loads(tj.read_text()).get("sglang_dir")
        if pinned:
            return resolve_sglang_dir(pinned)
    return SGLANG


def _tmp_base(task_dir: Path) -> str:
    """Unique /tmp scratch base for a task, keyed on its FULL resolved path.

    Two benchmark kits can contain dirs with the same leaf name (e.g. both a
    bmm and a moe kit have `task_official`). Keying scratch on the leaf name
    alone made their outputs share one /tmp tree, and _run()'s recursive
    `**/traces.json` glob then pulled the other task's shapes in via dedup-by-axes
    (e.g. a stale M=256 row leaking into an M<=128 sweep). The path hash prevents
    that cross-task leak.
    """
    h = hashlib.sha1(str(task_dir.resolve()).encode()).hexdigest()[:8]
    return f"/tmp/kernel-harness/{task_dir.name}-{h}"


_DRIVER = Path(__file__).resolve().parent.parent / "harness" / "driver.py"


def _run(task_dir: Path, solution: str, out: Path, iterations: int, max_workloads):
    out.mkdir(parents=True, exist_ok=True)
    cmd = [
        str(VENV / "bin" / "python"), str(_DRIVER), str(task_dir),
        "--solution-name", solution, "--iterations", str(iterations), "-o", str(out),
    ]
    if max_workloads:
        cmd += ["--max-workloads", str(max_workloads)]
    proc = subprocess.run(cmd, env=_env(_sglang_dir_for(task_dir)), check=False)
    traces = {}
    for f in glob.glob(f"{out}/**/traces.json", recursive=True):
        for t in json.load(open(f)):
            ax = json.dumps(t["workload"].get("axes", {}), sort_keys=True)
            e = t["evaluation"]
            perf = e.get("performance") or {}
            corr = e.get("correctness") or {}
            traces[ax] = {
                "status": e["status"],
                "latency_ms": perf.get("latency_ms"),
                "max_abs_err": corr.get("max_absolute_error"),
                "max_rel_err": corr.get("max_relative_error"),
                "has_nan": corr.get("has_nan"),
                "has_inf": corr.get("has_inf"),
                "log": (e.get("log") or "")[:300],
            }
    # Surface a non-zero harness exit when it produced NO traces at all — a silent
    # crash (import error, OOM) otherwise looks like "0 shapes evaluated" downstream.
    if proc.returncode != 0 and not traces:
        print(f"warning: harness driver exited {proc.returncode} with no traces "
              f"({solution})", file=sys.stderr)
    return traces


def _expected_axes(task_dir: Path):
    """The full set of workload axes (sorted-json keys) from workload.jsonl."""
    out = set()
    wl = task_dir / "workload.jsonl"
    if wl.exists():
        for line in wl.read_text().splitlines():
            line = line.strip()
            if line:
                ax = json.loads(line).get("axes", {})
                out.add(json.dumps(ax, sort_keys=True))
    return out


def _baseline_fingerprint(task_dir: Path, iterations) -> str:
    """Cache key that changes when anything affecting the baseline changes:
    iterations, the reference.py bytes, and the sglang build (commit). A stale cache
    from a different sglang or an edited reference must NOT be reused as the denominator.
    """
    ref = task_dir / "reference.py"
    ref_hash = hashlib.sha1(ref.read_bytes()).hexdigest()[:12] if ref.exists() else "none"
    sglang_dir = _sglang_dir_for(task_dir)
    try:
        commit = subprocess.run(["git", "-C", str(sglang_dir), "rev-parse", "--short", "HEAD"],
                                capture_output=True, text=True).stdout.strip() or "nogit"
    except Exception:
        commit = "nogit"
    return f"iters={iterations};ref={ref_hash};sglang={commit}"


def _baseline(task_dir: Path, iterations, max_workloads, refresh):
    cache = task_dir / ".baseline_cache.json"
    key = _baseline_fingerprint(task_dir, iterations)
    expected = _expected_axes(task_dir) if not max_workloads else None
    if cache.exists() and not refresh:
        data = json.loads(cache.read_text())
        cached = data.get(key)
        # Only trust a cache entry that (a) matches the fingerprint AND (b) covers the
        # full sweep. A partial cache would silently shrink the comparison denominator.
        if cached and (expected is None or set(cached) >= expected):
            return {a: v for a, v in cached.items()}
    tr = _run(task_dir, "reference.py", Path(f"{_tmp_base(task_dir)}-baseline"),
              iterations, max_workloads)
    base = {a: v["latency_ms"] for a, v in tr.items() if v["status"] == "PASSED"}
    # Never cache an empty/failed/partial baseline — it would poison every later run
    # (0-latency denominator, or a shrunk sweep).
    if base and (expected is None or set(base) >= expected):
        data = json.loads(cache.read_text()) if cache.exists() else {}
        data[key] = base
        cache.write_text(json.dumps(data, indent=2))
    return base


def _median(xs):
    s = sorted(xs)
    n = len(s)
    if not n:
        return None
    return s[n // 2] if n % 2 else 0.5 * (s[n // 2 - 1] + s[n // 2])


def _geo(xs):
    return math.exp(sum(math.log(x) for x in xs) / len(xs)) if xs else None


def _run_samples(task_dir, solution, tag, iterations, max_workloads, repeat):
    """Run *solution* `repeat` times as independent processes; aggregate per shape.

    Each run is a fresh harness-driver invocation (its own CUPTI median over
    `iterations` iters), so the samples capture process-level noise the within-run
    median cannot. A shape counts as correct only if it PASSED on every run.
    """
    agg = {}
    for i in range(repeat):
        out = Path(f"{_tmp_base(task_dir)}-{tag}/run{i}")
        tr = _run(task_dir, solution, out, iterations, max_workloads)
        for ax, v in tr.items():
            rec = agg.setdefault(ax, {"samples": [], "runs": 0, "passed_runs": 0,
                                      "max_abs_err": None, "max_rel_err": None,
                                      "has_nan": False, "has_inf": False, "log": ""})
            rec["runs"] += 1
            ok = v["status"] == "PASSED" and not v["has_nan"] and not v["has_inf"]
            if ok and v["latency_ms"]:
                rec["passed_runs"] += 1
                rec["samples"].append(v["latency_ms"])
            elif not rec["log"]:
                rec["log"] = v["log"]
            rec["has_nan"] = rec["has_nan"] or bool(v["has_nan"])
            rec["has_inf"] = rec["has_inf"] or bool(v["has_inf"])
            for k in ("max_abs_err", "max_rel_err"):
                if v[k] is not None:
                    rec[k] = v[k] if rec[k] is None else max(rec[k], v[k])
    return agg


def _alias_probe(task_dir: Path, solution: str):
    """Independent second-layer defense against the input-aliasing reward hack.

    The primary fix lives in the harness driver (testbench/harness/driver.py: the
    reference runs on a clone, so the candidate never sees reference-mutated inputs).
    This probe re-checks that guarantee locally — a redundant second layer — for
    in-place / interface-exact families where a candidate could return an input buffer
    instead of computing.

    Method: generate ONE input set, then run the reference and the candidate each on
    their OWN isolated clone, and compare. Because the buffers are isolated, a candidate
    that just returns its input buffer (aliasing) cannot see the reference's result and
    will mismatch; a candidate that genuinely computes matches. This is robust for
    in-place ops (where storage-sharing is legitimate) and has no false positives —
    unlike a poison-magnitude heuristic, which flags residual-add/rope outputs that
    legitimately preserve input magnitude.

    Returns (ok, message). ok=False means the probe caught an aliasing cheat.
    """
    meta = json.loads((task_dir / "task.json").read_text())
    family = meta.get("family", "")
    interface_exact = bool(meta.get("interface_exact"))
    ALIAS_FAMILIES = {"fused-add-rmsnorm", "gemma-fused-add-rmsnorm", "rope", "moe-combine"}
    if not (interface_exact or family in ALIAS_FAMILIES):
        return True, "n/a"

    tol = meta.get("tolerance", {"max_atol": 0.1, "max_rtol": 0.05, "required_matched_ratio": 0.98})
    probe_src = f'''
import importlib.util, json
import torch

TASK = r"{task_dir}"
ATOL, RTOL, RATIO = {tol["max_atol"]}, {tol["max_rtol"]}, {tol["required_matched_ratio"]}

def _load(p, name):
    s = importlib.util.spec_from_file_location(name, p)
    m = importlib.util.module_from_spec(s); s.loader.exec_module(m); return m

ref = _load(TASK + "/reference.py", "ref")
sol = _load(TASK + "/{solution}", "sol")
defn = json.load(open(TASK + "/definition.json"))
meta = json.load(open(TASK + "/task.json"))

wl = [ json.loads(l) for l in open(TASK + "/workload.jsonl") if l.strip() ]
scalars = dict(wl[0]["axes"])
for name, spec in defn.get("axes", {{}}).items():
    if spec.get("type") == "const":
        scalars[name] = spec["value"]
for k in ("top_k","num_experts","topk","H","V","K","N","E","Bh"):
    if k in meta: scalars.setdefault(k, meta[k])

dev = torch.device("cuda")
def _clone(inp):
    return {{k: (v.clone() if torch.is_tensor(v) else v) for k, v in inp.items()}}
def _lst(o):
    return list(o) if isinstance(o, (tuple, list)) else [o]

inp = ref.get_inputs(scalars, dev)      # one input set; each side gets its own clone
r = _lst(ref.run(**_clone(inp)))
c = _lst(sol.run(**_clone(inp)))
ok = (len(r) == len(c))
for a, b in zip(r, c):
    if not (torch.is_tensor(a) and torch.is_tensor(b) and a.shape == b.shape):
        ok = False; break
    if a.is_floating_point():
        match = ((b.float() - a.float()).abs() <= (ATOL + RTOL * a.float().abs()))
        ok = ok and (match.float().mean().item() >= RATIO)
    else:
        ok = ok and bool((a == b).all().item())
print("ALIAS_PROBE:" + ("OK" if ok else "ALIASED"))
'''
    r = subprocess.run([str(VENV / "bin" / "python"), "-c", probe_src],
                       env=_env(_sglang_dir_for(task_dir)),
                       capture_output=True, text=True)
    if "ALIAS_PROBE:OK" in r.stdout:
        return True, "ALIAS_PROBE:OK"
    if "ALIAS_PROBE:ALIASED" in r.stdout:
        return False, "ALIAS_PROBE:ALIASED"
    # Probe couldn't run (import/shape issue) — don't block, just note it.
    return True, f"probe inconclusive ({r.stderr.strip()[-160:]})"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("task_dir", type=Path)
    ap.add_argument("--solution", default="solution.py")
    ap.add_argument("--iterations", type=int, default=50)
    ap.add_argument("--repeat", type=int, default=1,
                    help="independent timing runs of candidate and baseline (default 1)")
    ap.add_argument("--max-workloads", type=int, default=None)
    ap.add_argument("--refresh-baseline", action="store_true")
    ap.add_argument("--no-baseline", action="store_true")
    args = ap.parse_args()

    task_dir = args.task_dir.resolve()
    if not (task_dir / "definition.json").exists():
        print(f"error: {task_dir} is not a task dir", file=sys.stderr)
        sys.exit(2)

    repeat = max(1, args.repeat)

    # Second-layer aliasing defense (see _alias_probe): reject a candidate that
    # returns an input buffer instead of computing, before we trust any WIN.
    alias_ok, alias_msg = _alias_probe(task_dir, args.solution)
    if not alias_ok:
        print(f"\n== {task_dir.name}  (solution={args.solution}) ==")
        print(f"ALIASING REWARD HACK DETECTED: {alias_msg} — candidate returns an "
              f"input buffer without computing. Rejected.")
        print("VERDICT_JSON_BEGIN")
        print(json.dumps({"task": task_dir.name, "solution": args.solution,
                          "correct": False, "win": False, "reason": "input_aliasing"}))
        print("VERDICT_JSON_END")
        sys.exit(2)

    sol_agg = _run_samples(task_dir, args.solution, args.solution,
                           args.iterations, args.max_workloads, repeat)

    # Workload completeness: unless the caller explicitly asked for a subset
    # (--max-workloads), the candidate must have been evaluated on the FULL sweep.
    # A silent partial run (harness crash mid-sweep) otherwise looks like a pass on
    # whatever shapes happened to complete.
    incomplete = None
    if not args.max_workloads:
        expected = _expected_axes(task_dir)
        got = set(sol_agg)
        missing = expected - got
        if missing:
            incomplete = f"{len(missing)}/{len(expected)} shapes missing"

    # Baseline as (lo, med, hi) latency per shape. K==1 keeps the cached fast path
    # (lo==med==hi); K>1 re-measures live so noise is compared against noise.
    if args.no_baseline:
        base_stats = {}
    elif repeat > 1:
        base_agg = _run_samples(task_dir, "reference.py", "baseline",
                                args.iterations, args.max_workloads, repeat)
        base_stats = {ax: (min(r["samples"]), _median(r["samples"]), max(r["samples"]))
                      for ax, r in base_agg.items() if r["samples"]}
    else:
        base = _baseline(task_dir, args.iterations, args.max_workloads, args.refresh_baseline)
        base_stats = {ax: (v, v, v) for ax, v in base.items()}

    rows, sp_med, sp_cons, sp_opt = [], [], [], []
    correct = bool(sol_agg)
    for ax in sorted(sol_agg, key=lambda a: json.loads(a).get("M", 0)):
        r = sol_agg[ax]
        ok = r["runs"] > 0 and r["passed_runs"] == r["runs"] and bool(r["samples"])
        correct = correct and ok
        c_lo = c_med = c_hi = None
        if r["samples"]:
            c_lo, c_med, c_hi = min(r["samples"]), _median(r["samples"]), max(r["samples"])
        b = base_stats.get(ax)
        s_med = s_cons = s_opt = None
        if ok and b and c_med:
            b_lo, b_med, b_hi = b
            s_med = b_med / c_med          # headline
            s_cons = b_lo / c_hi           # candidate worst vs baseline best
            s_opt = b_hi / c_lo
            sp_med.append(s_med); sp_cons.append(s_cons); sp_opt.append(s_opt)
        rows.append({"axes": json.loads(ax), "passed": ok, "status": r["log"] and "FAIL" or "PASSED",
                     "runs": r["runs"], "passed_runs": r["passed_runs"],
                     "max_abs_err": r["max_abs_err"], "max_rel_err": r["max_rel_err"],
                     "sol_us": round(c_med * 1e3, 2) if c_med else None,
                     "sol_us_lo": round(c_lo * 1e3, 2) if c_lo else None,
                     "sol_us_hi": round(c_hi * 1e3, 2) if c_hi else None,
                     "base_us": round(b[1] * 1e3, 2) if b else None,
                     "speedup": round(s_med, 3) if s_med else None,
                     "speedup_conservative": round(s_cons, 3) if s_cons else None,
                     "speedup_optimistic": round(s_opt, 3) if s_opt else None,
                     "log": r["log"] if not ok else ""})

    geo, geo_c, geo_o = _geo(sp_med), _geo(sp_cons), _geo(sp_opt)
    mn = min(sp_med) if sp_med else None
    mn_c = min(sp_cons) if sp_cons else None
    # An incomplete sweep (missing shapes on a full run) is NOT correct and NOT a win —
    # otherwise a partial harness crash reads as a pass on the shapes that completed.
    if incomplete:
        correct = False
    # Win gates on the worst-case margin. At K==1, conservative==median so this is
    # identical to the previous min_speedup>1 rule (and the same exit code).
    win = bool(correct and not incomplete and mn_c is not None and mn_c > 1.0)

    print(f"\n== {task_dir.name}  (solution={args.solution}, repeat={repeat}) ==")
    hdr = f"{'axes':22s} {'ok':>4} {'sol_us':>9} {'base_us':>9} {'sp_med':>7} {'sp_cons':>8} {'max_rel_err':>12}"
    print(hdr)
    for r in rows:
        print(f"{json.dumps(r['axes']):22s} {str(r['passed']):>4} "
              f"{(r['sol_us'] if r['sol_us'] is not None else 0):>9.2f} "
              f"{(r['base_us'] if r['base_us'] is not None else 0):>9.2f} "
              f"{(r['speedup'] if r['speedup'] is not None else 0):>7.3f} "
              f"{(r['speedup_conservative'] if r['speedup_conservative'] is not None else 0):>8.3f} "
              f"{(r['max_rel_err'] if r['max_rel_err'] is not None else 0):>12.2e}"
              + (f"   {r['log']}" if not r['passed'] else ""))
    verdict = {
        "task": task_dir.name, "solution": args.solution, "repeat": repeat,
        "correct": correct, "win": win,
        "geomean_speedup": round(geo, 4) if geo else None,
        "min_speedup": round(mn, 4) if mn else None,
        "geomean_speedup_conservative": round(geo_c, 4) if geo_c else None,
        "min_speedup_conservative": round(mn_c, 4) if mn_c else None,
        "geomean_speedup_optimistic": round(geo_o, 4) if geo_o else None,
        "num_shapes": len(rows), "per_shape": rows,
    }
    if incomplete:
        verdict["incomplete"] = incomplete
    tag = "WIN" if win else ("CORRECT (not faster)" if correct else "INCORRECT")
    if incomplete:
        tag = f"INCOMPLETE ({incomplete})"
    print(f"\n{tag}: correct={correct} geomean_speedup={verdict['geomean_speedup']} "
          f"min_speedup={verdict['min_speedup']} "
          f"min_speedup_conservative={verdict['min_speedup_conservative']}")
    print("VERDICT_JSON_BEGIN")
    print(json.dumps(verdict))
    print("VERDICT_JSON_END")
    sys.exit(0 if win else (1 if correct else 2))


if __name__ == "__main__":
    main()
