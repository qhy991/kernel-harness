#!/usr/bin/env python3
"""GPU-free structural pre-flight over the GLM-5.2 task directories.

Standard library only — no torch, no GPU, no venv — so it runs anywhere: a laptop
before pushing, CI, or the GPU node before a session.

    python3 testbench/bin/selftest.py             # every task
    python3 testbench/bin/selftest.py <task_dir>  # one task

Per task: the five contract files exist and no obsolete ones linger; task.json parses,
carries its keys, and restates nothing glm52_ops owns; workload.jsonl parses with an M
axis on every line; candidate.py defines a top-level run() and compiles.

Semantic checks — the operator exists, the family matches, the sweep matches — need
glm52_ops and therefore torch, so evaluate_task does them at run time and exits 3 on a
mismatch; `sync_glm52_tasks.py --check` covers them in CI.

The Kimi-K2.7 / MiniMax-M3 contract this file used to validate (definition.json +
reference.py + solution.py + tolerance blocks, plus the shared SGLANG_DIR wiring)
retired with those tasks. That version lives on at legacy/testbench/bin/selftest.py.
"""
from __future__ import annotations

import ast
import json
import sys
from pathlib import Path

CONTRACT_FILES = ("task.json", "problem.json", "workload.jsonl", "candidate.py",
                  "run.sh", "README.md")
OBSOLETE = ("definition.json", "reference.py", "solution.py", "impl.py", "verify.py")
TASK_JSON_KEYS = ("name", "model", "operator", "phase", "family", "performance_gate")


def _defines(source: str, filename: str, names: set[str]) -> set[str]:
    """Top-level function names from `names` defined in the source."""
    tree = ast.parse(source, filename)
    return {n.name for n in tree.body
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name in names}


def check_task(task: Path) -> list[str]:
    """Structural check for the glm52 contract (stdlib only, like the rest of this
    file). Semantic checks — operator exists, family matches, sweep matches — need
    glm52_ops and therefore torch, so evaluate_task does them at run time and
    exits 3 on a mismatch; `sync_glm52_tasks.py --check` covers them in CI."""
    problems = []
    missing = [f for f in CONTRACT_FILES if not (task / f).is_file()]
    if missing:
        return [f"missing contract files: {missing}"]
    left = [f for f in OBSOLETE if (task / f).exists()]
    if left:
        problems.append(f"obsolete files from a superseded stack: {left} "
                        f"(run testbench/bin/sync_glm52_tasks.py)")
    try:
        meta = json.loads((task / "task.json").read_text())
    except Exception as e:
        return problems + [f"task.json does not parse: {e}"]
    for key in TASK_JSON_KEYS:
        if key not in meta:
            problems.append(f"task.json missing key {key!r}")
    for forbidden in ("diff_tol", "rel_tol", "abs_tol_factor", "sweep", "K", "N",
                      "correctness", "performance", "contract"):
        if forbidden in meta:
            problems.append(f"task.json restates {forbidden!r}, which glm52_ops owns")
    try:
        for i, line in enumerate((task / "workload.jsonl").read_text().splitlines(), 1):
            if line.strip():
                wl = json.loads(line)
                if "M" not in wl.get("axes", {}):
                    problems.append(f"workload.jsonl line {i}: no axes.M")
    except Exception as e:
        problems.append(f"workload.jsonl does not parse: {e}")
    try:
        src = (task / "candidate.py").read_text()
        if "run" not in _defines(src, "candidate.py", {"run"}):
            problems.append("candidate.py defines no top-level run()")
        compile(src, "candidate.py", "exec")
    except SyntaxError as e:
        problems.append(f"candidate.py does not compile: {e}")
    return problems

def main() -> int:
    if len(sys.argv) > 1:
        tasks = [Path(sys.argv[1]).resolve()]
    else:
        root = Path(__file__).resolve().parent.parent / "tasks"
        tasks = sorted(p for p in root.glob("*/*") if p.is_dir())

    total_problems = 0
    for task in tasks:
        for p in check_task(task):
            print(f"{task.parent.name}/{task.name}: {p}")
            total_problems += 1
    print(f"selftest: {len(tasks)} tasks, {total_problems} problems")
    return 1 if total_problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
