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
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
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


def check_harness_utilities() -> list[str]:
    """Cheap regression checks for stdlib-only harness helpers."""
    problems = []
    root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(root))
    try:
        from testbench.harness import gpu_lease, result_store
        audit_path = root / "testbench" / "bin" / "audit_result.py"
        spec = importlib.util.spec_from_file_location("audit_result", audit_path)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"cannot load {audit_path}")
        audit_result = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(audit_result)
        verify_path = root / "testbench" / "bin" / "verify_harness.py"
        spec = importlib.util.spec_from_file_location("verify_harness", verify_path)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"cannot load {verify_path}")
        verify_harness = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(verify_harness)

        for stdout, expected in (
            (json.dumps({"official": True, "provisional": False, "errors": []}), "official"),
            (json.dumps({"official": False, "provisional": True, "errors": []}), "provisional"),
            (json.dumps({"official": False, "provisional": False, "errors": ["bad"]}), "invalid"),
            ("not json", "invalid"),
        ):
            label, _verdict, _parse_error = verify_harness._audit_classification(stdout)
            if label != expected:
                problems.append(f"verify_harness audit classification {stdout!r}: {label} != {expected}")

        review_status = verify_harness._parse_status_rows([
            " M AGENTS.md",
            "A  testbench/bin/verify_harness.py",
            "R  old.md -> testbench/README.md",
            "?? testbench/knowledge/distilled.json",
            "bad",
        ])
        expected_status = [
            {"status": " M", "path": "AGENTS.md"},
            {"status": "A ", "path": "testbench/bin/verify_harness.py"},
            {"status": "R ", "path": "testbench/README.md", "old_path": "old.md"},
            {"status": "??", "path": "testbench/knowledge/distilled.json"},
        ]
        if review_status != expected_status:
            problems.append(f"review status parser regression: {review_status!r}")

        required_review_paths = {"AGENTS.md", "testbench/README.md", "testbench/VERIFY.md"}
        if not required_review_paths.issubset(set(verify_harness.REVIEW_PATHS)):
            problems.append("review pathspecs omit a top-level harness handoff doc")
        if not required_review_paths.issubset(set(verify_harness.REVIEW_DIFF_PATHS)):
            problems.append("review diff pathspecs omit a top-level harness handoff doc")
        if "testbench/VERIFY.md" not in verify_harness.DIFF_CHECK_PATHS:
            problems.append("diff hygiene pathspecs omit testbench/VERIFY.md")
        if ":(exclude)testbench/tasks/glm52/*/candidate.py" not in verify_harness.REVIEW_DIFF_PATHS:
            problems.append("review diff pathspecs must exclude task candidate.py WIP")
        if any(path.startswith("archive/") or path == "archive" for path in verify_harness.REVIEW_DIFF_PATHS):
            problems.append("review diff pathspecs must not include archive WIP")

        old_verify_root = verify_harness.ROOT
        try:
            with tempfile.TemporaryDirectory() as td:
                root_tmp = Path(td)
                verify_harness.ROOT = root_tmp
                run_dir = root_tmp / "runs" / "glm52" / "synthetic" / "run-ok"
                run_dir.mkdir(parents=True)
                result = {
                    "run": {"run_id": "run-ok"},
                    "task": {"name": "synthetic"},
                    "verdict": {"status": "CORRECT", "exit_code": 0},
                    "aggregate": {"shapes_won": 1},
                    "candidate": {"sha256": "abc123"},
                }
                (run_dir / "result.json").write_text(json.dumps(result))
                index_row = {
                    "run_id": "run-ok",
                    "task": "synthetic",
                    "status": "CORRECT",
                    "exit_code": 0,
                    "candidate_sha256": "abc123",
                    "result": "runs/glm52/synthetic/run-ok/result.json",
                }
                latest_dir = root_tmp / "runs" / "glm52" / "synthetic"
                (latest_dir / "latest.json").write_text(json.dumps({
                    "run_id": "run-ok",
                    "result": "runs/glm52/synthetic/run-ok/result.json",
                    "verdict": {"status": "CORRECT", "exit_code": 0},
                    "aggregate": {"shapes_won": 1},
                }))
                (root_tmp / "runs" / "index.jsonl").write_text(json.dumps(index_row) + "\n")
                summary = verify_harness._pointer_audit(quiet=True, collect_problems=True)
                if summary.get("returncode") or summary.get("stale_index") or summary.get("stale_latest"):
                    problems.append(f"clean pointer audit should pass: {summary!r}")
                with open(root_tmp / "runs" / "index.jsonl", "a") as fh:
                    fh.write(json.dumps({"result": "runs/glm52/synthetic/missing/result.json"}) + "\n")
                summary = verify_harness._pointer_audit(quiet=True, collect_problems=True)
                if summary.get("returncode") or summary.get("stale_index") != 1:
                    problems.append(f"advisory pointer audit should report stale index only: {summary!r}")
                if summary.get("problem_count") != 1 or len(summary.get("_problems", [])) != 1:
                    problems.append(f"pointer audit should expose complete advisory problems: {summary!r}")
                summary = verify_harness._pointer_audit(quiet=True, strict=True)
                if summary.get("returncode") != 1 or summary.get("stale_index") != 1:
                    problems.append(f"strict pointer audit should fail on stale index: {summary!r}")
                (latest_dir / "latest.json").write_text(json.dumps({
                    "run_id": "run-ok",
                    "result": "runs/glm52/synthetic/missing-latest/result.json",
                    "verdict": {"status": "CORRECT", "exit_code": 0},
                    "aggregate": {"shapes_won": 1},
                }))
                summary = verify_harness._pointer_audit(quiet=True, collect_problems=True)
                if summary.get("problem_count") != 2 or len(summary.get("_problems", [])) != 2:
                    problems.append(f"pointer audit should retain all problems: {summary!r}")
                (latest_dir / "latest.json").write_text(json.dumps({
                    "run_id": "wrong",
                    "result": "runs/glm52/synthetic/run-ok/result.json",
                    "verdict": {"status": "CORRECT", "exit_code": 0},
                    "aggregate": {"shapes_won": 1},
                }))
                summary = verify_harness._pointer_audit(quiet=True, strict=True)
                if summary.get("returncode") != 1 or summary.get("mismatched") != 1:
                    problems.append(f"strict pointer audit should catch latest mismatch: {summary!r}")
                (latest_dir / "latest.json").write_text(json.dumps({
                    "run_id": "run-ok",
                    "result": "runs/glm52/synthetic/run-ok/result.json",
                    "verdict": {"status": "INCORRECT", "exit_code": 2},
                    "aggregate": {"shapes_won": 1},
                }))
                summary = verify_harness._pointer_audit(quiet=True, strict=True)
                if summary.get("returncode") != 1 or summary.get("mismatched") != 1:
                    problems.append(f"strict pointer audit should catch latest summary mismatch: {summary!r}")
        finally:
            verify_harness.ROOT = old_verify_root

        old_verify_root = verify_harness.ROOT
        try:
            with tempfile.TemporaryDirectory() as td:
                root_tmp = Path(td)
                verify_harness.ROOT = root_tmp
                for run_id, schema_version in (("official", "1.3"), ("old", "1.1")):
                    run_dir = root_tmp / "runs" / "glm52" / "synthetic" / run_id
                    run_dir.mkdir(parents=True)
                    cand = run_dir / "candidate.py"
                    cand.write_text("def run(inputs):\n    return inputs\n")
                    cand_sha = hashlib.sha256(cand.read_bytes()).hexdigest()
                    result = {
                        "schema_version": schema_version,
                        "task": {"name": "synthetic"},
                        "run": {"result_dir": str(run_dir)},
                        "candidate": {
                            "sha256": cand_sha,
                            "is_reference_fallback": False,
                            "git": {"in_repo": True, "path": "candidate.py", "dirty": False, "status": []},
                        },
                        "environment": {"git_dirty": False, "git_status": []},
                        "per_shape": [{"uuid": f"synthetic-{run_id}"}],
                        "aggregate": {"complete_sweep": True, "shapes_won": 1, "shapes_regressed": 0},
                        "verdict": {
                            "correct": True,
                            "performance_ok": True,
                            "status": "CORRECT",
                            "exit_code": 0,
                            "terminal_state": "COMPLETE_WIN",
                        },
                    }
                    (run_dir / "result.json").write_text(json.dumps(result))
                summary = verify_harness._audit_sweep(quiet=True, collect_findings=True)
                if summary.get("returncode") or summary.get("audited") != 2:
                    problems.append(f"synthetic audit sweep should pass advisory mode: {summary!r}")
                if summary.get("official") != 1 or summary.get("provisional") != 1:
                    problems.append(f"synthetic audit sweep classification regression: {summary!r}")
                findings = summary.get("_findings", [])
                if len(findings) != 1 or findings[0].get("classification") != "provisional":
                    problems.append(f"audit report should retain all non-official findings: {summary!r}")
                summary = verify_harness._audit_sweep(quiet=True, strict=True,
                                                      collect_findings=True)
                if summary.get("returncode") != 1 or len(summary.get("_findings", [])) != 1:
                    problems.append(f"strict audit report should fail on provisional findings: {summary!r}")
        finally:
            verify_harness.ROOT = old_verify_root

        rows = result_store._parse_git_status_porcelain(
            "M  AGENTS.md\n"
            " M testbench/harness/result_store.py\n"
            "R  old.py -> new.py\n"
            "?? testbench/bin/audit_result.py\n")
        expected = [
            {"status": "M ", "path": "AGENTS.md"},
            {"status": " M", "path": "testbench/harness/result_store.py"},
            {"status": "R ", "path": "new.py", "old_path": "old.py"},
            {"status": "??", "path": "testbench/bin/audit_result.py"},
        ]
        if rows != expected:
            problems.append(f"git porcelain parser regression: {rows!r}")

        old_cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
        try:
            os.environ["CUDA_VISIBLE_DEVICES"] = "3"
            if gpu_lease._visible_physical_to_logical() != {3: 0}:
                problems.append("CUDA_VISIBLE_DEVICES=3 did not map physical 3 -> logical 0")
            if gpu_lease._lock_index("cuda:0") != 3:
                problems.append("cuda:0 did not resolve to physical lock index 3 under CUDA_VISIBLE_DEVICES=3")
            os.environ["CUDA_VISIBLE_DEVICES"] = "GPU-deadbeef"
            if gpu_lease._visible_physical_to_logical() is not None:
                problems.append("UUID CUDA_VISIBLE_DEVICES should degrade to unknown mapping")
        finally:
            if old_cvd is None:
                os.environ.pop("CUDA_VISIBLE_DEVICES", None)
            else:
                os.environ["CUDA_VISIBLE_DEVICES"] = old_cvd

        with tempfile.TemporaryDirectory() as td:
            run_dir = Path(td)
            cand = run_dir / "candidate.py"
            cand.write_text("def run(inputs):\n    return inputs\n")
            cand_sha = hashlib.sha256(cand.read_bytes()).hexdigest()
            result = {
                "schema_version": "1.3",
                "task": {"name": "synthetic"},
                "run": {"result_dir": str(run_dir)},
                "candidate": {
                    "sha256": cand_sha,
                    "is_reference_fallback": False,
                    "git": {"in_repo": True, "path": "candidate.py", "dirty": False, "status": []},
                },
                "environment": {"git_dirty": False, "git_status": []},
                "per_shape": [{"uuid": "synthetic-M1"}],
                "aggregate": {"complete_sweep": True, "shapes_won": 1, "shapes_regressed": 0},
                "verdict": {
                    "correct": True,
                    "performance_ok": True,
                    "status": "CORRECT",
                    "exit_code": 0,
                    "terminal_state": "COMPLETE_WIN",
                },
            }
            result_path = run_dir / "result.json"
            result_path.write_text(json.dumps(result))
            verdict = audit_result.audit(result_path)
            if not verdict.get("official") or verdict.get("warnings") or verdict.get("errors"):
                problems.append(f"clean audit should be official: {verdict!r}")
            strict = subprocess.run([sys.executable, str(audit_path), str(result_path), "--strict"],
                                    capture_output=True, text=True)
            if strict.returncode:
                problems.append(f"clean audit --strict should pass: rc={strict.returncode} {strict.stdout} {strict.stderr}")

            result["environment"] = {"git_dirty": True, "git_status": [{"status": " M", "path": "x.py"}]}
            result_path.write_text(json.dumps(result))
            verdict = audit_result.audit(result_path)
            if verdict.get("official") or not verdict.get("provisional"):
                problems.append(f"dirty audit should be provisional: {verdict!r}")
            strict = subprocess.run([sys.executable, str(audit_path), str(result_path), "--strict"],
                                    capture_output=True, text=True)
            if strict.returncode != 1:
                problems.append(f"dirty audit --strict should exit 1: rc={strict.returncode} {strict.stdout} {strict.stderr}")

            result["environment"] = {"git_dirty": False, "git_status": []}
            result["schema_version"] = "1.1"
            result_path.write_text(json.dumps(result))
            verdict = audit_result.audit(result_path)
            if verdict.get("official") or not verdict.get("provisional"):
                problems.append(f"old-schema audit should be provisional: {verdict!r}")
            result["schema_version"] = "1.3"

            result["run"]["result_dir"] = str(run_dir / "wrong")
            result_path.write_text(json.dumps(result))
            verdict = audit_result.audit(result_path)
            if not verdict.get("errors"):
                problems.append(f"bad result_dir should be invalid: {verdict!r}")
            result["run"]["result_dir"] = str(run_dir)

            del result["run"]["result_dir"]
            result_path.write_text(json.dumps(result))
            verdict = audit_result.audit(result_path)
            if verdict.get("official") or not verdict.get("provisional"):
                problems.append(f"missing result_dir should be provisional: {verdict!r}")
            result["run"]["result_dir"] = str(run_dir)

            result["verdict"]["terminal_state"] = "NO_WIN_WITH_EVIDENCE"
            result_path.write_text(json.dumps(result))
            verdict = audit_result.audit(result_path)
            if not verdict.get("errors"):
                problems.append(f"bad terminal_state should be invalid: {verdict!r}")
            strict = subprocess.run([sys.executable, str(audit_path), str(result_path), "--strict"],
                                    capture_output=True, text=True)
            if strict.returncode != 2:
                problems.append(f"invalid audit --strict should exit 2: rc={strict.returncode} {strict.stdout} {strict.stderr}")

        old_runs_root = result_store.RUNS_ROOT
        try:
            with tempfile.TemporaryDirectory() as td:
                td_path = Path(td)
                result_store.RUNS_ROOT = td_path / "runs"
                cand = td_path / "candidate.py"
                cand_bytes = b"# coding: latin-1\ndef run(inputs):\n    return inputs\n# exact byte: \xff\n"
                cand.write_bytes(cand_bytes)
                result = {
                    "run": {"finished_utc": "2026-07-21T00:00:00Z"},
                    "verdict": {"status": "CORRECT", "exit_code": 0},
                    "aggregate": {},
                    "candidate": {"sha256": hashlib.sha256(cand_bytes).hexdigest()},
                    "environment": {},
                }
                out_dir = result_store.persist(
                    result, model="glm52", task="synthetic", run_id="run-byte-copy",
                    stdout_text="stdout\n", candidate_path=cand)
                copied = out_dir / "candidate.py"
                if copied.read_bytes() != cand_bytes:
                    problems.append("persist() did not preserve candidate.py bytes exactly")
                persisted = json.loads((out_dir / "result.json").read_text())
                expected_dir = str(out_dir)
                expected_result = str(out_dir / "result.json")
                if persisted.get("run", {}).get("result_dir") != expected_dir:
                    problems.append("persisted result.json is missing the external result_dir")
                latest = json.loads((result_store.RUNS_ROOT / "glm52" / "synthetic" / "latest.json").read_text())
                if latest.get("result") != expected_result:
                    problems.append(f"latest.json result pointer mismatch: {latest!r}")
                index_row = json.loads((result_store.RUNS_ROOT / "index.jsonl").read_text().splitlines()[-1])
                if index_row.get("result") != expected_result:
                    problems.append(f"index.jsonl result pointer mismatch: {index_row!r}")
        finally:
            result_store.RUNS_ROOT = old_runs_root
    except Exception as exc:
        problems.append(f"harness utility selftest failed: {type(exc).__name__}: {exc}")
    finally:
        try:
            sys.path.remove(str(root))
        except ValueError:
            pass
    return problems


def main() -> int:
    if len(sys.argv) > 1:
        tasks = [Path(sys.argv[1]).resolve()]
    else:
        root = Path(__file__).resolve().parent.parent / "tasks"
        tasks = sorted(p for p in root.glob("*/*") if p.is_dir())

    total_problems = 0
    for p in check_harness_utilities():
        print(f"harness: {p}")
        total_problems += 1
    for task in tasks:
        for p in check_task(task):
            print(f"{task.parent.name}/{task.name}: {p}")
            total_problems += 1
    print(f"selftest: {len(tasks)} tasks, {total_problems} problems")
    return 1 if total_problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
