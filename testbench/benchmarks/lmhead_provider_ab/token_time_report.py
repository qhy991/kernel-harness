#!/usr/bin/env python3
"""Report wall-clock + token cost for the two provider runs of a benchmark.

Auto-detects the infini/official session transcripts under ~/.claude/projects/ by
matching the task-dir path each run referenced, then prints a side-by-side of
duration and token usage (output / cache-creation / cache-read).

  python token_time_report.py <benchmark_dir_name>   # e.g. lmhead_provider_ab
"""
import datetime
import glob
import json
import os
import sys

BENCH_NAME = sys.argv[1] if len(sys.argv) > 1 else "lmhead_provider_ab"
PROJ = os.path.expanduser("~/.claude/projects/-home-qinhaiyan-kernel-harness-testbench")


def _parse_ts(x):
    try:
        return datetime.datetime.fromisoformat(x.replace("Z", "+00:00"))
    except Exception:
        return None


def find_session(provider):
    marker = f"{BENCH_NAME}/task_{provider}"
    best, best_hits, best_mtime = None, 0, 0
    for f in glob.glob(f"{PROJ}/*.jsonl"):
        try:
            txt = open(f, errors="ignore").read()
        except Exception:
            continue
        hits = txt.count(marker)
        # a session "belongs" to a provider if it references that task dir AND not
        # the other one (the prompt hardcodes exactly one task dir).
        other = txt.count(f"{BENCH_NAME}/task_{'official' if provider=='infini' else 'infini'}")
        if hits > 0 and hits > other:
            mt = os.path.getmtime(f)
            if hits > best_hits or (hits == best_hits and mt > best_mtime):
                best, best_hits, best_mtime = f, hits, mt
    return best


def summarize(f):
    ins = outs = cc = cr = msgs = 0
    ts, models = [], set()
    for line in open(f, errors="ignore"):
        line = line.strip()
        if not line:
            continue
        try:
            o = json.loads(line)
        except Exception:
            continue
        if o.get("timestamp"):
            ts.append(o["timestamp"])
        m = o.get("message") or {}
        u = (m.get("usage") or {}) if isinstance(m, dict) else {}
        if u:
            ins += u.get("input_tokens", 0) or 0
            outs += u.get("output_tokens", 0) or 0
            cc += u.get("cache_creation_input_tokens", 0) or 0
            cr += u.get("cache_read_input_tokens", 0) or 0
            msgs += 1
            if m.get("model"):
                models.add(m["model"])
    tp = [t for t in (_parse_ts(x) for x in ts) if t]
    dur = (max(tp) - min(tp)).total_seconds() if len(tp) >= 2 else None
    return {"file": os.path.basename(f), "models": models, "turns": msgs,
            "in": ins, "out": outs, "cache_create": cc, "cache_read": cr,
            "dur_min": dur / 60 if dur else None,
            "first": min(tp) if tp else None, "last": max(tp) if tp else None}


def main():
    rows = {}
    for p in ("infini", "official"):
        f = find_session(p)
        rows[p] = summarize(f) if f else None

    print(f"\n=== {BENCH_NAME}: call-method timing + token cost (same model, 2 providers) ===")
    hdr = f"{'metric':22s} {'infini':>16s} {'official':>16s}"
    print(hdr); print("-" * len(hdr))
    def g(p, k): return rows[p][k] if rows[p] else None
    def fnum(v): return f"{v:,}" if isinstance(v, int) else ("%.1f" % v if isinstance(v, float) else str(v))
    for label, key in [("wall-clock (min)", "dur_min"), ("assistant turns", "turns"),
                       ("output tokens", "out"), ("cache-creation tokens", "cache_create"),
                       ("cache-read tokens", "cache_read"), ("input (uncached)", "in")]:
        print(f"{label:22s} {fnum(g('infini',key)):>16s} {fnum(g('official',key)):>16s}")
    for p in ("infini", "official"):
        r = rows[p]
        if r:
            print(f"\n  {p}: model={r['models']} session={r['file']}")
            print(f"        {r['first']} -> {r['last']}")
        else:
            print(f"\n  {p}: SESSION NOT FOUND")


if __name__ == "__main__":
    main()
