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

Also validates the single shared SGLANG_DIR wiring: no task may pin a per-task
`sglang_dir`.
"""
from __future__ import annotations

import ast
import json
import sys
from pathlib import Path

CONTRACT_FILES = ("definition.json", "reference.py", "solution.py",
                  "task.json", "workload.jsonl", "run.sh")

# GLM-5.2 does not use the contract above. Its operators are defined once, in
# testbench/harness/glm52_ops.py, and the task directory only names which problem
# it is — there is no per-task definition.json/reference.py/solution.py to
# validate, because there is no per-task definition. See sync_glm52_tasks.py.
GLM52_CONTRACT_FILES = ("task.json", "problem.json", "workload.jsonl", "candidate.py",
                        "run.sh", "README.md")
GLM52_OBSOLETE = ("definition.json", "reference.py", "solution.py",
                  "impl.py", "verify.py")
GLM52_TASK_JSON_KEYS = ("name", "model", "operator", "phase", "family",
                        "performance_gate")


def check_glm52_task(task: Path) -> list[str]:
    """Structural check for the glm52 contract (stdlib only, like the rest of this
    file). Semantic checks — operator exists, family matches, sweep matches — need
    glm52_ops and therefore torch, so evaluate_task does them at run time and
    exits 3 on a mismatch; `sync_glm52_tasks.py --check` covers them in CI."""
    problems = []
    missing = [f for f in GLM52_CONTRACT_FILES if not (task / f).is_file()]
    if missing:
        return [f"missing contract files: {missing}"]
    left = [f for f in GLM52_OBSOLETE if (task / f).exists()]
    if left:
        problems.append(f"obsolete files from a superseded stack: {left} "
                        f"(run testbench/bin/sync_glm52_tasks.py)")
    try:
        meta = json.loads((task / "task.json").read_text())
    except Exception as e:
        return problems + [f"task.json does not parse: {e}"]
    for key in GLM52_TASK_JSON_KEYS:
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
TOL_KEYS = ("max_atol", "max_rtol", "required_matched_ratio")
TASK_JSON_KEYS = ("name", "op", "phase", "family", "sweep", "tolerance", "baseline")


def _defines(source: str, filename: str, names: set[str]) -> set[str]:
    """Top-level function names from `names` defined in the source."""
    tree = ast.parse(source, filename)
    return {n.name for n in tree.body
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name in names}


def check_task(task: Path) -> list[str]:
    if task.parent.name == "glm52":
        return check_glm52_task(task)
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
        meta = {}

    if "sglang_dir" in meta:
        problems.append("task.json must not set sglang_dir "
                        "(all tasks share the single SGLANG_DIR)")

    # Optional performance_model / workload_metrics (advisory; must be well-formed if present).
    pm = defn.get("performance_model", meta.get("performance_model"))
    if pm is not None:
        if not isinstance(pm, dict):
            problems.append("performance_model must be an object")
        elif "kind" not in pm and "family" not in pm:
            problems.append("performance_model needs 'kind' or 'family'")
    wmetrics = defn.get("workload_metrics", meta.get("workload_metrics"))
    if wmetrics is not None:
        if not isinstance(wmetrics, list) or not all(isinstance(x, str) for x in wmetrics):
            problems.append("workload_metrics must be a list of strings")
    fexpr = defn.get("flops_expr")
    if fexpr is not None:
        if not isinstance(fexpr, str) or not fexpr.strip():
            problems.append("flops_expr must be a non-empty string")
        else:
            try:
                tree = ast.parse(fexpr, mode="eval")
                names = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
                axes = set(defn.get("axes", {}))
                unknown = names - axes
                if unknown:
                    problems.append(f"flops_expr references unknown axes {sorted(unknown)}")
            except SyntaxError as e:
                problems.append(f"flops_expr does not parse: {e}")

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


def check_sglang_wiring() -> list[str]:
    """Validate the single SGLANG_DIR configuration (GPU-free)."""
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from config import SGLANG_DIR, has_m3_kernels, is_usable_sglang_checkout, resolve

    problems = []
    root = Path(SGLANG_DIR)
    if not (root / "python" / "sglang").is_dir():
        problems.append(f"SGLANG_DIR={root} is not a usable sglang checkout")
        return problems

    try:
        resolve("MM_M3_SGLANG_DIR")
        problems.append("MM_M3_SGLANG_DIR is still resolvable; "
                        "it must be removed so only SGLANG_DIR remains")
    except KeyError:
        pass

    if not has_m3_kernels(root) and not is_usable_sglang_checkout(root):
        problems.append(
            f"SGLANG_DIR={root} lacks MiniMax-M3 DSA kernel markers; "
            "DSA tasks will require them from the installed sglang package"
        )
    return problems


def main() -> int:
    if len(sys.argv) > 1:
        tasks = [Path(sys.argv[1]).resolve()]
        wiring_problems = []
    else:
        root = Path(__file__).resolve().parent.parent / "tasks"
        tasks = sorted(p for p in root.glob("*/*") if p.is_dir())
        wiring_problems = check_sglang_wiring()

    total_problems = 0
    for msg in wiring_problems:
        print(f"sglang-wiring: {msg}")
        total_problems += 1

    for task in tasks:
        for p in check_task(task):
            print(f"{task.parent.name}/{task.name}: {p}")
            total_problems += 1
    print(f"selftest: {len(tasks)} tasks, {total_problems} problems")
    return 1 if total_problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
