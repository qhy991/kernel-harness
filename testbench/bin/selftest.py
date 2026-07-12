#!/usr/bin/env python3
"""GPU-free structural pre-flight over the task directories.

Validates the on-disk task contract (docs/HARNESS_DESIGN.md §2) with the standard
library only — no torch, no GPU, no venv — so it runs anywhere: a laptop before
pushing, CI, or the GPU node before an optimization session.

    python3 testbench/bin/selftest.py             # every task
    python3 testbench/bin/selftest.py <task_dir>  # one task

Per task it checks that: the six contract files exist; definition.json / task.json /
workload.jsonl parse; definition.json's embedded `reference` is byte-identical to
reference.py (the driver executes the embedded copy as the oracle while the baseline
is timed from the file — they must never drift); reference.py defines `run` and the
declared `custom_inputs_entrypoint`, solution.py defines `run`, and both compile;
every workload line carries the swept var axes and a full tolerance block.

Exit 0 = all clean, 1 = problems found.
"""
from __future__ import annotations

import ast
import json
import sys
from pathlib import Path

CONTRACT_FILES = ("definition.json", "reference.py", "solution.py",
                  "task.json", "workload.jsonl", "run.sh")
TOL_KEYS = ("max_atol", "max_rtol", "required_matched_ratio")
TASK_JSON_KEYS = ("name", "op", "phase", "family", "sweep", "tolerance", "baseline")


def _defines(source: str, filename: str, names: set[str]) -> set[str]:
    """Top-level function names from `names` defined in the source."""
    tree = ast.parse(source, filename)
    return {n.name for n in tree.body
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name in names}


def check_task(task: Path) -> list[str]:
    problems = []
    missing = [f for f in CONTRACT_FILES if not (task / f).is_file()]
    if missing:
        return [f"missing contract files: {missing}"]

    try:
        defn = json.loads((task / "definition.json").read_text())
    except Exception as e:
        return [f"definition.json does not parse: {e}"]
    for key in ("axes", "inputs", "outputs", "reference", "custom_inputs_entrypoint"):
        if key not in defn:
            problems.append(f"definition.json missing key {key!r}")
    if problems:
        return problems

    ref_src = (task / "reference.py").read_text()
    if defn["reference"] != ref_src:
        problems.append("definition.json embedded reference != reference.py "
                        "(regenerate the task; the oracle and the timed baseline would differ)")

    entry = defn["custom_inputs_entrypoint"]
    try:
        found = _defines(ref_src, "reference.py", {"run", entry})
        if "run" not in found:
            problems.append("reference.py does not define run()")
        if entry not in found:
            problems.append(f"reference.py does not define {entry}() "
                            "(definition.custom_inputs_entrypoint)")
    except SyntaxError as e:
        problems.append(f"reference.py does not compile: {e}")
    try:
        if "run" not in _defines((task / "solution.py").read_text(), "solution.py", {"run"}):
            problems.append("solution.py does not define run()")
    except SyntaxError as e:
        problems.append(f"solution.py does not compile: {e}")

    try:
        meta = json.loads((task / "task.json").read_text())
        for key in TASK_JSON_KEYS:
            if key not in meta:
                problems.append(f"task.json missing key {key!r}")
    except Exception as e:
        problems.append(f"task.json does not parse: {e}")

    var_axes = {n for n, a in defn["axes"].items() if a.get("type") == "var"}
    lines = [l for l in (task / "workload.jsonl").read_text().splitlines() if l.strip()]
    if not lines:
        problems.append("workload.jsonl is empty")
    for i, line in enumerate(lines, 1):
        try:
            wl = json.loads(line)
        except Exception as e:
            problems.append(f"workload.jsonl line {i} does not parse: {e}")
            continue
        axes = wl.get("axes", {})
        if not var_axes <= set(axes):
            problems.append(f"workload.jsonl line {i} missing var axes {var_axes - set(axes)}")
        tol = wl.get("tolerance", {})
        bad = [k for k in TOL_KEYS if k not in tol]
        if bad:
            problems.append(f"workload.jsonl line {i} tolerance missing {bad}")
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
