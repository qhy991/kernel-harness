#!/usr/bin/env python3
"""Summarize a testbench run: per-shape pass/latency, and (if two out-dirs given)
solution-vs-baseline speedup.

  report.py <solution_out_dir> [baseline_out_dir]
"""
import glob
import json
import sys


def load(out_dir):
    res = {}
    for f in glob.glob(f"{out_dir}/**/traces.json", recursive=True):
        for t in json.load(open(f)):
            ax = json.dumps(t["workload"].get("axes", {}), sort_keys=True)
            e = t["evaluation"]
            perf = e.get("performance") or {}
            res[ax] = {"status": e["status"], "latency_ms": perf.get("latency_ms")}
    return res


def main():
    sol = load(sys.argv[1])
    base = load(sys.argv[2]) if len(sys.argv) > 2 else {}
    print(f"{'axes':32s} {'status':10s} {'sol_us':>9} {'base_us':>9} {'speedup':>8}")
    gm, n = 1.0, 0
    for ax in sorted(sol, key=lambda a: json.loads(a).get("M", 0)):
        s = sol[ax]
        su = (s["latency_ms"] or 0) * 1e3
        bu = (base.get(ax, {}).get("latency_ms") or 0) * 1e3
        sp = bu / su if su and bu else 0.0
        if sp:
            gm *= sp
            n += 1
        print(f"{ax:32s} {s['status']:10s} {su:>9.2f} {bu:>9.2f} {sp:>8.3f}")
    if n:
        print(f"\ngeomean speedup over {n} shapes: {gm ** (1/n):.3f}x")


if __name__ == "__main__":
    main()
