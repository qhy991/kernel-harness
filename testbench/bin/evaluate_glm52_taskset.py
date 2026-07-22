#!/usr/bin/env python3
"""Run kernel-harness testbench over a shared GLM-5.2 taskset."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parents[2]
DEFAULT_TASKSET = REPO / "tasksets" / "glm52_rocm_local.json"
EVALUATE_TASK = REPO / "testbench" / "harness" / "evaluate_task.py"

# Task root is derived from the taskset's hardware.platform:
#   rocm → testbench/tasks/glm52_amd/
#   cuda → testbench/tasks/glm52_cuda/
# A taskset may override with "task_root": "<repo-relative path>" if it points at
# the legacy glm52/ tree or a custom fork.
_PLATFORM_TASK_ROOT = {
    "rocm": REPO / "testbench" / "tasks" / "glm52_amd",
    "cuda": REPO / "testbench" / "tasks" / "glm52_cuda",
}


def _resolve_task_root(taskset: dict[str, Any]) -> Path:
    override = taskset.get("task_root")
    if override:
        p = Path(override)
        return p if p.is_absolute() else (REPO / p)
    platform = ((taskset.get("hardware") or {}).get("platform") or "").lower()
    root = _PLATFORM_TASK_ROOT.get(platform)
    if root is None:
        raise SystemExit(
            f"taskset has no hardware.platform (or an unknown one: {platform!r}); "
            f"add a 'task_root' field pointing at the task directory."
        )
    return root


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--taskset", type=Path, default=DEFAULT_TASKSET)
    ap.add_argument("--task", action="append", default=None,
                    help="task id or harness task to run; can be passed more than once")
    ap.add_argument("--phase", choices=("all", "prefill", "decode"), default="all")
    ap.add_argument("--smoke", action="store_true",
                    help="one low-cost shape per task: prefill M=1024, decode M=16")
    ap.add_argument("--M", type=int, default=None, help="single M for every selected task")
    ap.add_argument("--repeat", type=int, default=None)
    ap.add_argument("--iterations", type=int, default=None)
    ap.add_argument("--warmup", type=int, default=None)
    ap.add_argument("--candidate-reference", action="store_true",
                    help="run backend reference instead of task candidate.py")
    ap.add_argument("--candidate-root", type=Path, default=None,
                    help="root containing <harness_task>/candidate.py for external candidates")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--no-gpu-lock", action="store_true")
    ap.add_argument("--json-out", type=Path, default=None)
    args = ap.parse_args()

    taskset = load_taskset(args.taskset)
    task_root = _resolve_task_root(taskset)
    selected = select_tasks(taskset["tasks"], args.phase, args.task)
    defaults = taskset.get("defaults", {})
    env = rocm_env(taskset.get("hardware", {}))

    results = []
    for task in selected:
        for m_value in m_values_for_task(task, args, defaults):
            result = run_task(task, args, env, m_value, task_root)
            results.append(result)
            status = result.get("status")
            speedup = result.get("geomean_speedup")
            suffix = f" geomean={speedup:.4g}" if isinstance(speedup, (int, float)) else ""
            print(f"{task['id']}[M={m_value}]: {status}{suffix}")

    summary = summarize(results)
    report = {
        "taskset": taskset["name"],
        "task_count": len(selected),
        "case_count": len(results),
        "summary": summary,
        "results": results,
    }
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"summary": summary}, indent=2, sort_keys=True))
    return 0 if summary["infra_failed"] == 0 and summary["incorrect"] == 0 else 1


def load_taskset(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text())
    tasks = data.get("tasks") or []
    if not tasks:
        raise SystemExit(f"taskset has no tasks: {path}")
    seen = set()
    for task in tasks:
        for key in ("id", "phase", "harness_task", "reward_operator"):
            if key not in task:
                raise SystemExit(f"taskset task missing {key}: {task}")
        if task["id"] in seen:
            raise SystemExit(f"duplicate task id: {task['id']}")
        seen.add(task["id"])
    return data


def select_tasks(tasks: list[dict[str, Any]], phase: str, requested: list[str] | None):
    selected = [t for t in tasks if phase == "all" or t["phase"] == phase]
    if requested:
        wanted = set(requested)
        selected = [
            t for t in selected
            if t["id"] in wanted or t["harness_task"] in wanted or t["reward_operator"] in wanted
        ]
    if not selected:
        raise SystemExit("no tasks selected")
    return selected


def rocm_env(hardware: dict[str, Any]) -> dict[str, str]:
    env = os.environ.copy()
    env.setdefault("KERNEL_HARNESS_PLATFORM", hardware.get("platform", "rocm"))
    env.setdefault("KERNEL_HARNESS_PROFILE", hardware.get("profile", "amd-mi300x"))
    env.setdefault("KERNEL_HARNESS_PROVIDER", hardware.get("provider", "aiter-torch-reference"))
    env.setdefault("KERNEL_HARNESS_TIMER", hardware.get("timer", "event"))
    env.setdefault("SGLANG_USE_AITER", str(hardware.get("sglang_use_aiter", "1")))
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    return env


def m_values_for_task(task: dict[str, Any], args, defaults: dict[str, Any]) -> list[int]:
    if args.M is not None:
        return [args.M]
    phase = task["phase"]
    if args.smoke:
        smoke_defaults = defaults.get("smoke") or {}
        key = "prefill_M" if phase == "prefill" else "decode_M"
        return [int(smoke_defaults.get(key, 1024 if phase == "prefill" else 16))]
    key = "prefill_M" if phase == "prefill" else "decode_M"
    fallback = [1024, 2048, 4096] if phase == "prefill" else [1, 4, 8, 16, 32, 64]
    return [int(v) for v in defaults.get(key, fallback)]


def run_task(task: dict[str, Any], args, env: dict[str, str], m_value: int,
             task_root: Path) -> dict[str, Any]:
    harness_task = task["harness_task"]
    task_dir = task_root / harness_task
    cmd = [
        sys.executable,
        str(EVALUATE_TASK),
        str(task_dir),
        "--device",
        args.device,
        "--no-persist",
        "--M",
        str(m_value),
    ]
    if args.no_gpu_lock or args.smoke:
        cmd.append("--no-gpu-lock")
    if args.candidate_reference:
        cmd += ["--candidate", "reference"]
    elif args.candidate_root:
        cmd += ["--candidate", str(args.candidate_root / harness_task / "candidate.py")]

    if args.repeat is not None:
        cmd += ["--repeat", str(args.repeat)]
    elif args.smoke:
        cmd += ["--repeat", "1"]
    if args.iterations is not None:
        cmd += ["--iterations", str(args.iterations)]
    elif args.smoke:
        cmd += ["--iterations", "1"]
    if args.warmup is not None:
        cmd += ["--warmup", str(args.warmup)]
    elif args.smoke:
        cmd += ["--warmup", "0"]

    proc = subprocess.run(
        cmd,
        cwd=REPO,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=7200,
    )
    parsed = extract_result(proc.stdout)
    status = classify(proc.returncode, parsed)
    aggregate = (parsed or {}).get("aggregate") or {}
    verdict = (parsed or {}).get("verdict") or {}
    return {
        "id": task["id"],
        "harness_task": harness_task,
        "phase": task["phase"],
        "score_scope": task.get("score_scope", "task"),
        "metric_group": task.get("metric_group"),
        "metric_component": task.get("metric_component"),
        "production_equivalent": task.get("production_equivalent"),
        "M": m_value,
        "returncode": proc.returncode,
        "status": status,
        "correct": bool(verdict.get("correct")),
        "performance_ok": bool(verdict.get("performance_ok")),
        "geomean_speedup": aggregate.get("geomean_speedup"),
        "min_speedup_conservative": aggregate.get("min_speedup_conservative"),
        "stdout_tail": proc.stdout[-2000:] if status in {"infra_failed", "incorrect"} else "",
        "stderr_tail": proc.stderr[-2000:] if proc.stderr and status == "infra_failed" else "",
    }


def extract_result(stdout: str) -> dict[str, Any] | None:
    match = re.search(r"RESULT_JSON_BEGIN\s*(\{.*?\})\s*RESULT_JSON_END", stdout, re.S)
    if not match:
        return None
    return json.loads(match.group(1))


def classify(returncode: int, parsed: dict[str, Any] | None) -> str:
    if returncode == 3 or parsed is None:
        return "infra_failed"
    verdict = parsed.get("verdict", {})
    if returncode == 2 or not verdict.get("correct", False):
        return "incorrect"
    if verdict.get("performance_ok", False):
        return "passed"
    return "correct_not_faster"


def summarize(results: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "total": len(results),
        "passed": sum(r["status"] == "passed" for r in results),
        "correct_not_faster": sum(r["status"] == "correct_not_faster" for r in results),
        "incorrect": sum(r["status"] == "incorrect" for r in results),
        "infra_failed": sum(r["status"] == "infra_failed" for r in results),
    }


if __name__ == "__main__":
    raise SystemExit(main())
