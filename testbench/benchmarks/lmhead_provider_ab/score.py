#!/usr/bin/env python3
"""Fair, anti-cheat A/B scorer for the provider benchmark.

For each provider, take ONLY its final solution.py, drop it into a fresh copy of the
pristine canonical task (so any edits it made to reference.py / workload.jsonl /
definition.json are discarded — the oracle of record wins), then re-benchmark both
providers back-to-back on this GPU with the same evaluator settings. This removes
cross-session timing drift and makes tampering impossible to convert into a score.

Usage:
    python score.py [--iterations 50] [--repeat 5]
"""
import argparse
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

BENCH = Path(__file__).resolve().parent
PRISTINE = BENCH / "_pristine"
EVAL = Path("/home/qinhaiyan/kernel-harness/testbench/bin/evaluate.py")
PY = "/home/qinhaiyan/kernel-harness/.venv/bin/python"
PROVIDERS = ["infini", "official"]


def score_one(provider: str, iterations: int, repeat: int) -> dict:
    src_sol = BENCH / f"task_{provider}" / "solution.py"
    if not src_sol.exists():
        return {"provider": provider, "error": f"no solution.py in task_{provider}"}
    # fresh scoring copy from the pristine oracle; inject ONLY the candidate solution
    scored = BENCH / "results" / f"scored_{provider}"
    if scored.exists():
        shutil.rmtree(scored)
    shutil.copytree(PRISTINE, scored)
    shutil.copyfile(src_sol, scored / "solution.py")

    r = subprocess.run(
        [PY, str(EVAL), str(scored), "--iterations", str(iterations), "--repeat", str(repeat)],
        capture_output=True, text=True,
    )
    out = r.stdout
    m = re.search(r"VERDICT_JSON_BEGIN\s*(\{.*?\})\s*VERDICT_JSON_END", out, re.S)
    if not m:
        m = re.search(r'(\{"task".*\})', out, re.S)  # fallback: last json-ish line
    verdict = json.loads(m.group(1)) if m else {"parse_error": True, "tail": out[-500:]}
    verdict["provider"] = provider
    # Did the candidate tamper with the correctness oracle? The real oracle is
    # definition.json's inline `reference` field + tolerance, plus reference.py
    # (baseline) and workload.jsonl (shapes/tol). Compare content, ignoring the
    # `name` field we intentionally suffixed per provider.
    tampered = []
    for f in ["reference.py", "workload.jsonl"]:
        if (BENCH / f"task_{provider}" / f).read_bytes() != (PRISTINE / f).read_bytes():
            tampered.append(f)
    def _defn_no_name(p):
        d = json.loads(Path(p).read_text()); d.pop("name", None)
        return json.dumps(d, sort_keys=True)
    if _defn_no_name(BENCH / f"task_{provider}" / "definition.json") != \
       _defn_no_name(PRISTINE / "definition.json"):
        tampered.append("definition.json")
    verdict["tampered_oracle_files"] = tampered
    return verdict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iterations", type=int, default=50)
    ap.add_argument("--repeat", type=int, default=5)
    args = ap.parse_args()

    results = [score_one(p, args.iterations, args.repeat) for p in PROVIDERS]
    (BENCH / "results" / "scoreboard.json").write_text(json.dumps(results, indent=2))

    print("\n" + "=" * 74)
    print(f"{'provider':10s} {'correct':>8} {'win':>5} {'geomean':>9} {'min_cons':>9} {'tampered':>9}")
    print("-" * 74)
    for v in results:
        if v.get("error"):
            print(f"{v['provider']:10s}  ERROR: {v['error']}")
            continue
        print(f"{v['provider']:10s} {str(v.get('correct')):>8} {str(v.get('win')):>5} "
              f"{str(v.get('geomean_speedup')):>9} {str(v.get('min_speedup_conservative')):>9} "
              f"{str(bool(v.get('tampered_oracle_files'))):>9}")
    print("=" * 74)
    # winner = correct AND highest conservative min-speedup
    valid = [v for v in results if v.get("correct") and not v.get("error")]
    if valid:
        best = max(valid, key=lambda v: v.get("min_speedup_conservative") or 0)
        print(f"WINNER (correct + best worst-case speedup): {best['provider']}  "
              f"min_speedup_conservative={best.get('min_speedup_conservative')}")
    else:
        print("No provider produced a correct solution.")
    print(f"\nscoreboard.json -> {BENCH/'results'/'scoreboard.json'}")


if __name__ == "__main__":
    main()
