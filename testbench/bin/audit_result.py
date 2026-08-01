#!/usr/bin/env python3
"""Structured auto-check for a GLM-5.2 harness result.json.

This is the reviewer-side cheap pass: it does not re-run CUDA, but it does verify
that the persisted result is internally consistent, names the exact candidate
bytes, applies the current gate semantics, and classifies dirty-tree provenance.

    python3 testbench/bin/audit_result.py runs/glm52/o_proj_decode/<run>/result.json
    python3 testbench/bin/audit_result.py ... --json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
CURRENT_SCHEMA_VERSION = "1.4"
MIN_GATE_REPEAT = 10
MIN_GATE_ITERATIONS = 2
MIN_GATE_WARMUP = 8
PHYSICAL_REWARD_LIMIT = 1.0
ALLOWED_STATUSES = {"CORRECT", "INCORRECT", "INVALID"}
TERMINAL_STATES = {
    "COMPLETE_WIN",
    "NO_WIN_WITH_EVIDENCE",
    "PARTIAL_OR_REGRESSED_WITH_EVIDENCE",
    "INCORRECT_OR_INCOMPLETE",
    "PROBE_ONLY_NO_VERDICT",
    "UNSTABLE_NO_VERDICT",
    "INVALID_MEASUREMENT",
}


def _load(path: Path) -> tuple[dict | None, list[str]]:
    try:
        return json.loads(path.read_text()), []
    except Exception as exc:
        return None, [f"result.json is not readable JSON: {type(exc).__name__}: {exc}"]


def _sha256(path: Path) -> str | None:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        return None


def _candidate_copy(path: Path) -> Path:
    return path.parent / "candidate.py"


def _resolve_recorded_path(value: str) -> Path:
    p = Path(value).expanduser()
    return p.resolve() if p.is_absolute() else (REPO / p).resolve()


def audit(path: Path) -> dict:
    result, errors = _load(path)
    warnings: list[str] = []
    info: list[str] = []
    if result is None:
        return {"official": False, "provisional": False, "errors": errors,
                "warnings": warnings, "info": info}

    verdict = result.get("verdict") or {}
    aggregate = result.get("aggregate") or {}
    candidate = result.get("candidate") or {}
    environment = result.get("environment") or {}
    run = result.get("run") or {}
    per_shape = result.get("per_shape") or []
    schema_version = result.get("schema_version")

    if schema_version != CURRENT_SCHEMA_VERSION:
        warnings.append(f"schema_version={schema_version!r}; "
                        f"current reviewer schema is {CURRENT_SCHEMA_VERSION!r}")

    current_schema = schema_version == CURRENT_SCHEMA_VERSION
    correct = bool(verdict.get("correct"))
    perf_ok = bool(verdict.get("performance_ok"))
    measurement_valid = verdict.get("measurement_valid", True) is True
    exit_code = verdict.get("exit_code")
    status = verdict.get("status")
    wins = aggregate.get("shapes_won")
    regressions = aggregate.get("shapes_regressed")
    complete = aggregate.get("complete_sweep")
    terminal_state = verdict.get("terminal_state")

    if status not in ALLOWED_STATUSES:
        errors.append(f"verdict.status must be one of {sorted(ALLOWED_STATUSES)}, got {status!r}")
    if status == "CORRECT" and not correct:
        errors.append("verdict.status is CORRECT but verdict.correct is false")
    if status == "INCORRECT" and correct:
        errors.append("verdict.status is INCORRECT but verdict.correct is true")
    if status == "INVALID" and measurement_valid:
        errors.append("verdict.status is INVALID but verdict.measurement_valid is true")
    if current_schema and not measurement_valid and status != "INVALID":
        errors.append("schema 1.4 invalid measurement must use verdict.status=INVALID")

    repeat = run.get("repeat")
    iterations = run.get("iterations")
    warmup = run.get("warmup")
    unstable_shapes = aggregate.get("timing_unstable_shapes") or []
    gate_eligible = bool(run.get("gate_eligible")) if current_schema else True
    expected_eligible = bool(
        complete is True and isinstance(repeat, int) and repeat >= MIN_GATE_REPEAT and
        isinstance(iterations, int) and iterations >= MIN_GATE_ITERATIONS and
        isinstance(warmup, int) and warmup >= MIN_GATE_WARMUP and
        not unstable_shapes and measurement_valid
    )
    if current_schema and gate_eligible != expected_eligible:
        errors.append(
            "run.gate_eligible disagrees with full-sweep/repeat/stability/physical gates "
            f"({gate_eligible!r} != {expected_eligible!r})")
    expected_perf = bool(
        correct and measurement_valid and wins is not None and regressions is not None and
        wins >= 1 and regressions == 0 and (gate_eligible if current_schema else True)
    )
    expected_gate = (
        "invalid" if not measurement_valid else
        "win" if expected_perf else
        "correct-no-verdict" if correct and current_schema and not gate_eligible else
        "correct-no-win" if correct else "incorrect"
    )
    reported_gate = "win" if perf_ok else ("correct-no-win" if correct else "incorrect")
    if perf_ok != expected_perf:
        errors.append("verdict.performance_ok disagrees with aggregate gate "
                      f"(correct={correct}, shapes_won={wins}, shapes_regressed={regressions})")
    expected_exit = 2 if not measurement_valid else (0 if perf_ok else (1 if correct else 2))
    if exit_code != expected_exit:
        errors.append(f"verdict.exit_code={exit_code!r} but expected {expected_exit}")
    if complete is not True:
        if perf_ok:
            errors.append(f"aggregate.complete_sweep={complete!r} but verdict.performance_ok is true")
        elif current_schema and correct:
            info.append("restricted sweep is correctly classified as probe-only")
        else:
            warnings.append(f"sweep incomplete ({complete!r}); result is an incorrect/partial run")
    if not isinstance(per_shape, list) or not per_shape:
        errors.append("per_shape must be a non-empty list")

    if current_schema:
        if run.get("pairing") != "adjacent balanced R/C,C/R":
            errors.append("schema 1.4 run.pairing must declare adjacent balanced R/C,C/R")
        if run.get("first_recorded_iteration_discarded") is not True:
            errors.append("schema 1.4 must record first_recorded_iteration_discarded=true")
        if run.get("post_timing_seed_differs") is not True:
            errors.append("schema 1.4 must record post_timing_seed_differs=true")
        if gate_eligible and (not isinstance(warmup, int) or warmup < MIN_GATE_WARMUP):
            errors.append(f"gate-eligible run warmup must be >= {MIN_GATE_WARMUP}, got {warmup!r}")
        if gate_eligible and (not isinstance(repeat, int) or repeat < MIN_GATE_REPEAT):
            errors.append(f"gate-eligible run repeat must be >= {MIN_GATE_REPEAT}, got {repeat!r}")
        if gate_eligible and (
                not isinstance(iterations, int) or iterations < MIN_GATE_ITERATIONS):
            errors.append(
                f"gate-eligible run iterations must be >= {MIN_GATE_ITERATIONS}, "
                f"got {iterations!r}")
        probe_reasons = run.get("probe_reasons")
        if not isinstance(probe_reasons, list):
            errors.append("schema 1.4 run.probe_reasons must be a list")
        else:
            should_be_probe = bool(
                complete is not True or not isinstance(repeat, int) or
                repeat < MIN_GATE_REPEAT or not isinstance(warmup, int) or
                warmup < MIN_GATE_WARMUP or not isinstance(iterations, int) or
                iterations < MIN_GATE_ITERATIONS)
            if bool(probe_reasons) != should_be_probe:
                errors.append(
                    "run.probe_reasons disagrees with full-sweep/repeat probe status")
    else:
        if perf_ok and isinstance(repeat, int) and repeat < MIN_GATE_REPEAT:
            warnings.append(
                f"legacy win used repeat={repeat} below the current minimum {MIN_GATE_REPEAT}")
        if perf_ok and unstable_shapes:
            warnings.append(
                "legacy win includes unstable timing; current schema would return no verdict")

    physical_violations = []
    timed_rows = 0
    for index, row in enumerate(per_shape if isinstance(per_shape, list) else []):
        if not isinstance(row, dict):
            errors.append(f"per_shape[{index}] must be an object")
            continue
        for key in ("reward", "reference_reward"):
            value = row.get(key)
            if isinstance(value, (int, float)) and value > PHYSICAL_REWARD_LIMIT:
                physical_violations.append(f"per_shape[{index}].{key}={value}")
        if current_schema and "candidate_us" in row:
            timed_rows += 1
            pairs = row.get("timing_pairs")
            samples = row.get("samples")
            if not isinstance(pairs, list) or not pairs:
                errors.append(f"per_shape[{index}] has timing but no timing_pairs")
            elif not isinstance(samples, int) or len(pairs) != samples:
                errors.append(
                    f"per_shape[{index}] timing_pairs length does not match samples={samples!r}")
            else:
                expected_orders = ["reference,candidate" if n % 2 == 0
                                   else "candidate,reference" for n in range(len(pairs))]
                orders = [pair.get("order") if isinstance(pair, dict) else None
                          for pair in pairs]
                if orders != expected_orders:
                    errors.append(f"per_shape[{index}] timing pair order is not balanced R/C,C/R")
            post_seed = row.get("post_timing_seed")
            if not isinstance(post_seed, int):
                errors.append(f"per_shape[{index}] is missing integer post_timing_seed")
            elif post_seed == (result.get("task") or {}).get("seed"):
                errors.append(f"per_shape[{index}] post-timing seed reuses the pre-timing seed")
    if physical_violations:
        errors.append("physical reward > 1.0: " + ", ".join(physical_violations[:4]))
        if current_schema and measurement_valid:
            errors.append("physical reward violation was not marked measurement_valid=false")
    if current_schema and correct and timed_rows != len(per_shape):
        errors.append("correct schema 1.4 result must have paired timing for every shape")
    if current_schema:
        observed_unstable = [row.get("uuid") for row in per_shape
                             if isinstance(row, dict) and row.get("timing_unstable")]
        if sorted(str(value) for value in unstable_shapes) != sorted(
                str(value) for value in observed_unstable):
            errors.append("aggregate.timing_unstable_shapes disagrees with per_shape rows")

    result_dir = run.get("result_dir")
    if result_dir is None:
        warnings.append("run.result_dir is missing; result predates schema 1.3 persisted run-dir provenance")
    elif not isinstance(result_dir, str) or not result_dir:
        errors.append(f"run.result_dir must be a non-empty string, got {result_dir!r}")
    elif _resolve_recorded_path(result_dir) != path.parent.resolve():
        errors.append(f"run.result_dir={result_dir!r} does not match result.json parent")

    cand_sha = candidate.get("sha256")
    cand_copy = _candidate_copy(path)
    copy_sha = _sha256(cand_copy)
    if candidate.get("is_reference_fallback"):
        warnings.append("candidate is the reference fallback; this can validate the gate but is not an optimization")
    elif not cand_sha:
        errors.append("candidate.sha256 is missing")
    elif copy_sha is None:
        warnings.append("run directory has no candidate.py copy to re-hash")
    elif copy_sha != cand_sha:
        errors.append("candidate.py copy sha256 does not match result candidate.sha256")

    git_dirty = bool(environment.get("git_dirty"))
    git_status = environment.get("git_status")
    if git_dirty and not isinstance(git_status, list):
        warnings.append("environment.git_dirty is true but no structured git_status list is present")
    elif git_dirty:
        warnings.append("environment git tree was dirty at evaluation time")
        info.append(f"dirty_tree_files={len(git_status)}")

    cand_git = candidate.get("git")
    if isinstance(cand_git, dict):
        if cand_git.get("dirty"):
            warnings.append("candidate file was dirty in git at evaluation time")
    else:
        warnings.append("candidate.git provenance is missing; result predates schema 1.1")

    if current_schema:
        current_probe_reasons = run.get("probe_reasons") or []
        expected_terminal = (
            "INVALID_MEASUREMENT" if not measurement_valid else
            "INCORRECT_OR_INCOMPLETE" if not correct else
            "PROBE_ONLY_NO_VERDICT" if current_probe_reasons else
            "UNSTABLE_NO_VERDICT" if unstable_shapes else
            "COMPLETE_WIN" if expected_perf else
            "NO_WIN_WITH_EVIDENCE" if wins == 0 and regressions == 0 else
            "PARTIAL_OR_REGRESSED_WITH_EVIDENCE"
        )
    else:
        expected_terminal = (
            "COMPLETE_WIN" if expected_perf else
            "NO_WIN_WITH_EVIDENCE" if correct and wins == 0 and regressions == 0 else
            "PARTIAL_OR_REGRESSED_WITH_EVIDENCE" if correct else
            "INCORRECT_OR_INCOMPLETE"
        )
    if terminal_state is None:
        warnings.append("verdict.terminal_state is missing; result predates schema 1.2")
    elif terminal_state not in TERMINAL_STATES:
        errors.append(f"verdict.terminal_state must be one of {sorted(TERMINAL_STATES)}, got {terminal_state!r}")
    elif terminal_state != expected_terminal:
        errors.append(f"verdict.terminal_state={terminal_state!r} but expected {expected_terminal!r}")

    official = not errors and perf_ok and not warnings
    provisional = not errors and (not official)
    return {
        "official": official,
        "provisional": provisional,
        "gate": expected_gate,
        "reported_gate": reported_gate,
        "terminal_state": terminal_state,
        "schema_version": schema_version,
        "task": result.get("task", {}).get("name"),
        "candidate_sha256": cand_sha,
        "errors": errors,
        "warnings": warnings,
        "info": info,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("result_json", type=Path)
    ap.add_argument("--json", action="store_true", help="emit the structured verdict only")
    ap.add_argument("--strict", action="store_true",
                    help="exit nonzero for PROVISIONAL as well as INVALID")
    args = ap.parse_args()

    verdict = audit(args.result_json.resolve())
    if args.json:
        print(json.dumps(verdict, indent=2))
    else:
        label = "OFFICIAL" if verdict["official"] else "PROVISIONAL" if verdict["provisional"] else "INVALID"
        print(f"{label}: {args.result_json}")
        print(f"  gate={verdict.get('gate')} task={verdict.get('task')} schema={verdict.get('schema_version')}")
        for key in ("errors", "warnings", "info"):
            for item in verdict.get(key, []):
                print(f"  {key[:-1]}: {item}")
    if verdict["errors"]:
        return 2
    if args.strict and not verdict["official"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
