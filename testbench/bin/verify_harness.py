#!/usr/bin/env python3
"""One-command harness verification for review and CI.

Runs the GPU-free consistency checks that prove the generated task mirrors,
knowledge bank, reviewer audit schema, and touched Python files are coherent. It
does not run CUDA kernels; use a task's run.sh separately for GPU gate checks.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TESTBENCH = ROOT / "testbench"
AUDIT_RESULT = Path(__file__).resolve().parent / "audit_result.py"

PY_FILES = [
    "testbench/bin/check_env.py",
    "testbench/bin/audit_result.py",
    "testbench/bin/selftest.py",
    "testbench/bin/brief.py",
    "testbench/bin/bw_ceiling.py",
    "testbench/bin/knowledge.py",
    "testbench/bin/kwiki_bridge.py",
    "testbench/bin/sync_glm52_tasks.py",
    "testbench/bin/verify_harness.py",
    "testbench/harness/evaluate_task.py",
    "testbench/harness/glm52_ops.py",
    "testbench/harness/gpu_lease.py",
    "testbench/harness/result_store.py",
]

DIFF_CHECK_PATHS = [
    "AGENTS.md",
    "testbench/README.md",
    "testbench/VERIFY.md",
    "testbench/setup_env.sh",
    "testbench/bin",
    "testbench/harness",
    "testbench/knowledge",
    "testbench/tasks/glm52",
    ":(exclude)testbench/tasks/glm52/*/candidate.py",
]

REVIEW_PATHS = [
    "AGENTS.md",
    "testbench/README.md",
    "testbench/VERIFY.md",
    "testbench/setup_env.sh",
    "testbench/bin",
    "testbench/harness",
    "testbench/knowledge",
    "testbench/tasks/glm52/*/README.md",
    "testbench/tasks/glm52/*/problem.json",
    "testbench/tasks/glm52/*/task.json",
]

REVIEW_DIFF_PATHS = [
    "AGENTS.md",
    "testbench/README.md",
    "testbench/VERIFY.md",
    "testbench/setup_env.sh",
    "testbench/bin",
    "testbench/harness",
    "testbench/knowledge",
    "testbench/tasks/glm52",
    ":(exclude)testbench/tasks/glm52/*/candidate.py",
]


def _run(cmd: list[str], *, capture: bool = False) -> dict:
    if not capture:
        print("$ " + " ".join(cmd), flush=True)
    r = subprocess.run(cmd, cwd=ROOT, text=True, capture_output=capture)
    return {
        "cmd": cmd,
        "returncode": r.returncode,
        **({"stdout": r.stdout, "stderr": r.stderr} if capture else {}),
    }


def _env_file() -> dict[str, str]:
    out: dict[str, str] = {}
    for p in (TESTBENCH / "harness.env", TESTBENCH / "harness.env.example"):
        if not p.is_file():
            continue
        for line in p.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            out.setdefault(k.strip(), v.strip().strip('"').strip("'"))
    return out


def _path(value: str | None, default: Path) -> Path:
    if not value:
        return default.resolve()
    p = Path(value).expanduser()
    return p.resolve() if p.is_absolute() else (ROOT / p).resolve()


def _venv_python() -> str:
    cfg = _env_file()
    venv = _path(os.environ.get("VENV") or cfg.get("VENV"), ROOT / ".venv")
    py = venv / "bin" / "python"
    return str(py if py.exists() else Path(sys.executable))


def _review_files() -> list[str]:
    tracked = subprocess.run(
        ["git", "diff", "--name-only", "--", *REVIEW_DIFF_PATHS],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    ).stdout.splitlines()
    untracked = subprocess.run(
        ["git", "ls-files", "--others", "--exclude-standard", "--", *REVIEW_DIFF_PATHS],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    ).stdout.splitlines()
    return sorted(set(tracked + untracked))


def _parse_status_rows(rows: list[str]) -> list[dict[str, str]]:
    out = []
    for row in rows:
        if len(row) < 4:
            continue
        status = row[:2]
        path = row[3:]
        if " -> " in path:
            old_path, path = path.split(" -> ", 1)
            out.append({"status": status, "path": path, "old_path": old_path})
        else:
            out.append({"status": status, "path": path})
    return out


def _review_file_status() -> list[dict[str, str]]:
    rows = subprocess.run(
        ["git", "status", "--short", "--", *REVIEW_DIFF_PATHS],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    ).stdout.splitlines()
    return _parse_status_rows(rows)


def _audit_classification(stdout: str) -> tuple[str, dict | None, str | None]:
    """Classify audit_result.py --json output without relying on formatting."""
    try:
        verdict = json.loads(stdout)
    except json.JSONDecodeError as exc:
        return "invalid", None, f"audit_result.py emitted non-JSON output: {exc}"
    if verdict.get("errors"):
        return "invalid", verdict, None
    if verdict.get("official") is True:
        return "official", verdict, None
    if verdict.get("provisional") is True:
        return "provisional", verdict, None
    return "invalid", verdict, "audit_result.py emitted an unclassified verdict"


def _compact_audit_example(path: Path, verdict: dict | None, parse_error: str | None = None) -> dict:
    out = {"path": str(path)}
    if parse_error:
        out["error"] = parse_error
        return out
    if verdict is None:
        return out
    for key in ("task", "gate", "terminal_state", "schema_version"):
        if key in verdict:
            out[key] = verdict.get(key)
    for key in ("errors", "warnings"):
        items = verdict.get(key) or []
        if items:
            out[key] = items[:3]
    return out


def _audit_sweep(*, quiet: bool = False, strict: bool = False,
                 collect_findings: bool = False) -> dict:
    results = list((ROOT / "runs" / "glm52").glob("*/*/result.json"))
    invalid = 0
    official = 0
    provisional = 0
    first_invalid = None
    first_provisional = None
    audit_items = []
    for path in results:
        r = subprocess.run(
            [sys.executable, str(AUDIT_RESULT), str(path), "--json"],
            cwd=ROOT,
            capture_output=True,
            text=True,
        )
        label, verdict, parse_error = _audit_classification(r.stdout)
        if r.returncode or label == "invalid":
            invalid += 1
            if invalid == 1:
                first_invalid = {
                    **_compact_audit_example(path, verdict, parse_error),
                    "stdout": r.stdout,
                    "stderr": r.stderr,
                    "returncode": r.returncode,
                }
            if invalid == 1 and not quiet:
                print(f"first invalid audit: {path}")
                print(r.stdout, end="")
                print(r.stderr, end="", file=sys.stderr)
            if collect_findings:
                audit_items.append({
                    "classification": "invalid",
                    **_compact_audit_example(path, verdict, parse_error),
                })
        elif label == "official":
            official += 1
        else:
            provisional += 1
            if first_provisional is None:
                first_provisional = _compact_audit_example(path, verdict)
            if collect_findings:
                audit_items.append({
                    "classification": "provisional",
                    **_compact_audit_example(path, verdict),
                })
    summary = {"audited": len(results), "invalid": invalid,
               "official": official, "provisional": provisional}
    rc = 1 if invalid or (strict and provisional) else 0
    if not quiet:
        print("audit sweep: " + " ".join(f"{k}={v}" for k, v in summary.items()))
        if strict and provisional:
            print("audit sweep strict mode: provisional results are not accepted", file=sys.stderr)
            if first_provisional:
                print(f"first provisional audit: {first_provisional['path']}", file=sys.stderr)
    return {
        "cmd": ["audit_result.py", "runs/glm52/*/*/result.json"],
        "returncode": rc,
        "strict": strict,
        **summary,
        **({"_invalid": first_invalid} if first_invalid else {}),
        **({"_provisional": first_provisional} if first_provisional else {}),
        **({"_findings": audit_items} if audit_items else {}),
    }


def _print_audit_report(report: dict) -> None:
    fields = ("audited", "invalid", "official", "provisional")
    print("audit sweep: " + " ".join(f"{k}={report.get(k, 0)}" for k in fields))
    for item in report.get("_findings", []):
        print(f"{item['classification']} {item['path']}")
        for key in ("errors", "warnings"):
            for detail in item.get(key, []):
                print(f"  {key[:-1]}: {detail}")


def _resolve_run_path(value: str) -> Path:
    p = Path(value).expanduser()
    return p.resolve() if p.is_absolute() else (ROOT / p).resolve()


def _result_metadata(result: dict) -> dict:
    return {
        "run_id": result.get("run", {}).get("run_id"),
        "task": result.get("task", {}).get("name"),
        "status": result.get("verdict", {}).get("status"),
        "exit_code": result.get("verdict", {}).get("exit_code"),
        "candidate_sha256": result.get("candidate", {}).get("sha256"),
    }


def _compare_pointer_metadata(row: dict, result: dict, keys: tuple[str, ...]) -> str | None:
    meta = _result_metadata(result)
    for key in keys:
        if row.get(key) != meta.get(key):
            return f"{key} disagrees with pointed result ({row.get(key)!r} != {meta.get(key)!r})"
    return None


def _compare_latest_summary(row: dict, result: dict) -> str | None:
    for key in ("verdict", "aggregate"):
        if row.get(key) != result.get(key, {}):
            return f"{key} summary disagrees with pointed result"
    return None


def _pointer_audit(*, quiet: bool = False, strict: bool = False,
                   collect_problems: bool = False) -> dict:
    index_path = ROOT / "runs" / "index.jsonl"
    latest_paths = list((ROOT / "runs" / "glm52").glob("*/latest.json"))
    index_rows = 0
    stale_index = 0
    stale_latest = 0
    malformed = 0
    mismatched = 0
    problem_items = []

    def remember(kind: str, path: Path, detail: str) -> None:
        problem_items.append({"kind": kind, "path": str(path), "detail": detail})

    if index_path.is_file():
        for lineno, line in enumerate(index_path.read_text().splitlines(), 1):
            if not line.strip():
                continue
            index_rows += 1
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                malformed += 1
                remember("index", index_path, f"line {lineno}: invalid JSON: {exc}")
                continue
            result_path = row.get("result")
            if not isinstance(result_path, str) or not result_path:
                malformed += 1
                remember("index", index_path, f"line {lineno}: missing result path")
                continue
            if not _resolve_run_path(result_path).is_file():
                stale_index += 1
                remember("index", index_path, f"line {lineno}: stale result pointer {result_path}")
                continue
            try:
                result = json.loads(_resolve_run_path(result_path).read_text())
            except json.JSONDecodeError as exc:
                malformed += 1
                remember("index", index_path, f"line {lineno}: result target is invalid JSON: {exc}")
                continue
            mismatch = _compare_pointer_metadata(
                row, result, ("run_id", "task", "status", "exit_code", "candidate_sha256"))
            if mismatch:
                mismatched += 1
                remember("index", index_path, f"line {lineno}: {mismatch}")
    else:
        malformed += 1
        remember("index", index_path, "runs/index.jsonl is missing")

    for latest_path in latest_paths:
        try:
            row = json.loads(latest_path.read_text())
        except json.JSONDecodeError as exc:
            malformed += 1
            remember("latest", latest_path, f"invalid JSON: {exc}")
            continue
        result_path = row.get("result")
        if not isinstance(result_path, str) or not result_path:
            malformed += 1
            remember("latest", latest_path, "missing result path")
            continue
        resolved = _resolve_run_path(result_path)
        if not resolved.is_file():
            stale_latest += 1
            remember("latest", latest_path, f"stale result pointer {result_path}")
            continue
        try:
            result = json.loads(resolved.read_text())
        except json.JSONDecodeError as exc:
            malformed += 1
            remember("latest", latest_path, f"result target is invalid JSON: {exc}")
            continue
        mismatch = _compare_pointer_metadata(row, result, ("run_id",))
        if mismatch is None:
            mismatch = _compare_latest_summary(row, result)
        if mismatch:
            mismatched += 1
            remember("latest", latest_path, mismatch)

    summary = {
        "index_rows": index_rows,
        "latest_files": len(latest_paths),
        "stale_index": stale_index,
        "stale_latest": stale_latest,
        "malformed": malformed,
        "mismatched": mismatched,
    }
    problem_count = stale_index + stale_latest + malformed + mismatched
    rc = 1 if strict and problem_count else 0
    if not quiet:
        print("pointer audit: " + " ".join(f"{k}={v}" for k, v in summary.items()))
        if problem_items:
            print(f"pointer audit first problem: {problem_items[0]['detail']}", file=sys.stderr)
    return {
        "cmd": ["verify_harness.py", "pointer-audit"],
        "returncode": rc,
        "strict": strict,
        **summary,
        "problem_count": problem_count,
        **({"_problem": problem_items[0]} if problem_items else {}),
        **({"_problems": problem_items} if collect_problems and problem_items else {}),
    }


def _print_pointer_report(report: dict) -> None:
    fields = ("index_rows", "latest_files", "stale_index", "stale_latest",
              "malformed", "mismatched", "problem_count")
    print("pointer audit: " + " ".join(f"{k}={report.get(k, 0)}" for k in fields))
    for item in report.get("_problems", []):
        print(f"{item['kind']} {item['path']}: {item['detail']}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--skip-audit-sweep", action="store_true",
                    help="skip historical runs/glm52 audit_result.py sweep")
    ap.add_argument("--skip-pointer-audit", action="store_true",
                    help="skip runs/index.jsonl and latest.json pointer audit")
    ap.add_argument("--strict-audit-sweep", action="store_true",
                    help="fail the audit sweep when any historical result is provisional")
    ap.add_argument("--strict-pointer-audit", action="store_true",
                    help="fail when runs/index.jsonl or latest.json has stale, malformed, or mismatched pointers")
    ap.add_argument("--audit-report", action="store_true",
                    help="print every non-official historical result audit finding and exit")
    ap.add_argument("--pointer-report", action="store_true",
                    help="print every runs/index.jsonl/latest.json pointer problem and exit")
    ap.add_argument("--json", action="store_true", help="emit a machine-readable summary")
    ap.add_argument("--print-review-paths", action="store_true",
                    help="print the harness-reviewable pathspecs and exit")
    ap.add_argument("--print-review-files", action="store_true",
                    help="print changed harness-reviewable files and exit")
    ap.add_argument("--with-status", action="store_true",
                    help="with --print-review-files, include git status codes")
    ap.add_argument("-0", "--null", action="store_true",
                    help="with --print-review-files, separate paths with NUL bytes")
    args = ap.parse_args()
    if args.print_review_paths:
        print("\n".join(REVIEW_PATHS))
        return 0
    if args.print_review_files:
        sep = "\0" if args.null else "\n"
        if args.with_status:
            rows = _review_file_status()
            lines = [f"{r['status']} {r['path']}" for r in rows]
        else:
            lines = _review_files()
        sys.stdout.write(sep.join(lines) + (sep if lines else ""))
        return 0
    if args.pointer_report:
        report = _pointer_audit(quiet=True, strict=args.strict_pointer_audit,
                                collect_problems=True)
        if args.json:
            print(json.dumps(report, indent=2))
        else:
            _print_pointer_report(report)
        return report["returncode"]
    if args.audit_report:
        report = _audit_sweep(quiet=True, strict=args.strict_audit_sweep,
                              collect_findings=True)
        if args.json:
            print(json.dumps(report, indent=2))
        else:
            _print_audit_report(report)
        return report["returncode"]
    harness_py = _venv_python()

    checks = [
        [sys.executable, "-m", "py_compile", *PY_FILES],
        [sys.executable, "testbench/bin/selftest.py"],
        [sys.executable, "testbench/bin/knowledge.py", "lint"],
        [sys.executable, "testbench/bin/knowledge.py", "index", "--check"],
        [sys.executable, "testbench/bin/knowledge.py", "distill", "--check"],
        [harness_py, "testbench/bin/sync_glm52_tasks.py", "--check"],
        ["git", "diff", "--check", "--", *DIFF_CHECK_PATHS],
    ]
    results = []
    for cmd in checks:
        result = _run(cmd, capture=args.json)
        results.append(result)
        if args.json:
            continue
        if result["returncode"]:
            print(f"failed: {' '.join(cmd)}", file=sys.stderr)
    if not args.skip_audit_sweep:
        results.append(_audit_sweep(quiet=args.json, strict=args.strict_audit_sweep))
    if not args.skip_pointer_audit:
        results.append(_pointer_audit(quiet=args.json, strict=args.strict_pointer_audit))
    failures = sum(1 for r in results if r["returncode"])
    if args.json:
        print(json.dumps({"ok": failures == 0, "checks": results}, indent=2))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
