#!/usr/bin/env python3
"""The MI300X optimization flow, one command: correctness gate -> performance -> target.

This is the bridge between KDA-Pilot (which generates a candidate kernel) and the
kernel-harness marks (which score it). It runs the real opbench gate — the SAME
evaluate_task.py run.sh uses, so poison / anti-cheat / post-timing recheck all apply —
parses its RESULT_JSON, compares the achieved roofline reward against the frozen target
(min(hardware roofline, 1.5x baseline) from amd_glm5_targets.csv), and writes a
flotilla-compatible status.json.

    correctness FIRST:  a wrong candidate (exit 2) never reaches the perf comparison.
    performance THEN:   speedup vs the reference + roofline reward vs the target.

Usage:
    .venv/bin/python rewardbench/amd/run_flow.py <op> --candidate <path> [--round N]
        [--repeat 10] [--iterations 30] [--status-dir DIR] [--device cuda:0]

    <op> is a task name (o_proj_prefill / index_k_prefill / dsa_attn_decode) or any
    testbench/tasks/glm52 task. Exit: 0 correct & target met · 1 correct, target not
    met · 2 incorrect · 3 infra.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parents[1]
# Post platform-split: the AMD task tree is glm52_amd (glm52_ops_amd inputs). The old
# pre-split `glm52` tree still exists but carries CUDA-schema seeds.
_TASKS = _REPO / "testbench" / "tasks" / "glm52_amd"
_TARGETS = _HERE / "amd_glm5_targets.csv"

_OP_TO_TASK = {  # op -> (task_dir_name, op, phase)
    "o_proj_prefill": ("o_proj_prefill", "o_proj", "prefill"),
    "index_k_prefill": ("index_k_prefill", "index_k", "prefill"),
    "dsa_attn_decode": ("dsa_attn_decode", "dsa_attn", "decode"),
}


def _resolve_task(name: str):
    if name in _OP_TO_TASK:
        return _OP_TO_TASK[name]
    d = _TASKS / name
    if (d / "task.json").is_file():
        meta = json.loads((d / "task.json").read_text())
        return (name, meta["operator"], meta["phase"])
    raise SystemExit(f"unknown task {name!r}; known: {', '.join(_OP_TO_TASK)}")


def _load_targets(op: str, phase: str) -> dict[int, dict]:
    """{M: target-row} for this op/phase from amd_glm5_targets.csv."""
    out = {}
    if _TARGETS.exists():
        with open(_TARGETS) as f:
            for r in csv.DictReader(f):
                if r["op"] == op and r["phase"] == phase:
                    out[int(r["M"])] = r
    return out


def _run_gate(task_dir: Path, candidate: str, repeat: int, iters: int, device: str) -> dict:
    """Run evaluate_task.py and parse its RESULT_JSON block."""
    # The venv on this node lives on tmpfs and is rebuilt per session; honor KH_PYTHON
    # (or fall back to the interpreter running this flow) instead of a frozen .venv path.
    py = os.environ.get("KH_PYTHON") or sys.executable
    cmd = [py,
           str(_REPO / "testbench" / "harness" / "evaluate_task.py"), str(task_dir),
           "--candidate", candidate, "--repeat", str(repeat),
           "--iterations", str(iters), "--device", device, "--no-persist"]
    env = dict(os.environ)
    # AMD/MI300X production-baseline bundle (aiter): correctness is gated against a math
    # oracle, latency against aiter's gfx942 kernel (Triton fallback when CK/ASM absent).
    env.setdefault("KERNEL_HARNESS_PLATFORM", "rocm")
    env.setdefault("KERNEL_HARNESS_PROFILE", "amd-mi300x")
    env.setdefault("KERNEL_HARNESS_PROVIDER", "aiter-torch-reference")
    env.setdefault("KERNEL_HARNESS_TIMER", "event")
    p = subprocess.run(cmd, capture_output=True, text=True, env=env)
    out = p.stdout
    if "RESULT_JSON_BEGIN" not in out:
        raise RuntimeError(f"gate produced no RESULT_JSON (exit {p.returncode}):\n"
                           f"{p.stdout[-1500:]}\n{p.stderr[-1500:]}")
    blob = out.split("RESULT_JSON_BEGIN", 1)[1].split("RESULT_JSON_END", 1)[0].strip()
    return json.loads(blob)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("op")
    ap.add_argument("--candidate", required=True)
    ap.add_argument("--round", type=int, default=0)
    ap.add_argument("--repeat", type=int, default=10)
    ap.add_argument("--iterations", type=int, default=30)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--status-dir", default=None,
                    help="write status.json here (default: alongside the candidate)")
    args = ap.parse_args()

    task_name, op, phase = _resolve_task(args.op)
    task_dir = _TASKS / task_name
    targets = _load_targets(op, phase)
    cand = str(Path(args.candidate).expanduser().resolve())

    result = _run_gate(task_dir, cand, args.repeat, args.iterations, args.device)
    verdict = result["verdict"]
    agg = result["aggregate"]
    correct = verdict["correct"]

    # per-shape: achieved reward vs target reward
    shapes = []
    target_met_all = correct
    best_ratio = 0.0
    for row in result["per_shape"]:
        if "reward" not in row:
            continue
        M = row["axes"]["M"]
        tr = targets.get(M, {})
        target_reward = float(tr["target_reward"]) if tr else None
        reward = row["reward"]
        met = (target_reward is not None and reward >= target_reward)
        ratio = (reward / target_reward) if target_reward else 0.0
        best_ratio = max(best_ratio, ratio)
        if not met:
            target_met_all = False
        shapes.append({
            "M": M, "correct": row.get("correct"), "reward": round(reward, 5),
            "reference_reward": round(row.get("reference_reward", 0.0), 5),
            "target_reward": target_reward,
            "speedup": row.get("speedup"), "shape_verdict": row.get("shape_verdict"),
            "bound": row.get("bound"), "target_met": met,
            "pct_of_target": round(100 * ratio, 1),
        })

    if not correct:
        state, exit_code = "INCORRECT", 2
    elif target_met_all and agg.get("shapes_regressed", 1) == 0:
        state, exit_code = "TARGET_MET", 0
    else:
        state, exit_code = "CORRECT_BELOW_TARGET", 1

    status = {
        "task": task_name, "op": op, "phase": phase,
        "round": args.round,
        "candidate": cand,
        "candidate_sha256": result["candidate"].get("sha256"),
        "state": state,
        "correct": correct,
        "shapes_won": agg.get("shapes_won"),
        "shapes_regressed": agg.get("shapes_regressed"),
        "geomean_speedup": agg.get("geomean_speedup"),
        "best_reward": agg.get("best_reward"),
        "worst_reward": agg.get("worst_reward"),
        "target_met": target_met_all,
        "best_pct_of_target": round(100 * best_ratio, 1),
        "per_shape": shapes,
        "backend": result.get("backend", {}),
        "timing_protocol": result["run"].get("timing_protocol"),
        "message": _message(state, correct, shapes, agg),
    }

    status_dir = Path(args.status_dir).expanduser() if args.status_dir else Path(cand).parent
    status_dir.mkdir(parents=True, exist_ok=True)
    status_path = status_dir / "status.json"
    status_path.write_text(json.dumps(status, indent=2))

    print(json.dumps({k: status[k] for k in
                      ("task", "state", "correct", "geomean_speedup", "best_reward",
                       "target_met", "best_pct_of_target", "message")}, indent=2))
    print(f"status -> {status_path}")
    return exit_code


def _message(state, correct, shapes, agg):
    if not correct:
        bad = next((s for s in shapes if not s.get("correct")), None)
        return f"INCORRECT — {bad['M'] if bad else '?'} failed correctness gate"
    if state == "TARGET_MET":
        return (f"target met on all shapes ({agg.get('shapes_won')} win, "
                f"{agg.get('shapes_regressed')} regress); best_reward={agg.get('best_reward')}")
    worst = min(shapes, key=lambda s: s["pct_of_target"]) if shapes else None
    return (f"correct but below target — worst shape M={worst['M']} at "
            f"{worst['pct_of_target']}% of target reward "
            f"({worst['reward']} vs {worst['target_reward']})" if worst else "correct")


if __name__ == "__main__":
    raise SystemExit(main())
