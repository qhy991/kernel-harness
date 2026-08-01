#!/usr/bin/env python3
"""Run the DSA decode leaf across a traced KV-context profile.

This deliberately does not emulate incremental prefill.  SGLang chooses its
prefill backend from ForwardBatch prefix/extend metadata, so that phase must be
measured at a real containing-region or end-to-end boundary.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
RUNNER = HERE / "run.sh"


def validate_profile(path: Path) -> dict[str, Any]:
    profile = json.loads(path.read_text())
    if profile.get("schema_version") != 1:
        raise ValueError("context profile schema_version must be 1")
    if profile.get("id") != path.stem:
        raise ValueError("context profile id must equal its filename stem")
    if not isinstance(profile.get("production_evidence"), bool):
        raise ValueError("production_evidence must be a boolean")
    deployment = profile.get("deployment")
    if not isinstance(deployment, dict):
        raise ValueError("deployment must be an object")
    for key in ("model", "sglang_commit", "gpu", "cache_dtype", "page_size"):
        if not deployment.get(key):
            raise ValueError(f"deployment.{key} is required")
    if deployment["cache_dtype"] != "float8_e4m3fn" or deployment["page_size"] != 64:
        raise ValueError(
            "this DSA decode lane is valid only for FP8 E4M3 paged KV with page_size=64"
        )

    scenarios = profile.get("scenarios")
    if not isinstance(scenarios, list) or not scenarios:
        raise ValueError("scenarios must be a non-empty list")
    ids: set[str] = set()
    phases: set[str] = set()
    decode_context_signatures: set[tuple[int, ...]] = set()
    phase_weights: dict[str, float] = {"decode": 0.0, "incremental_prefill": 0.0}
    for scenario in scenarios:
        if not isinstance(scenario, dict):
            raise ValueError("each scenario must be an object")
        scenario_id = scenario.get("id")
        if not isinstance(scenario_id, str) or not scenario_id or scenario_id in ids:
            raise ValueError(f"scenario id must be unique and non-empty: {scenario_id!r}")
        ids.add(scenario_id)
        phase = scenario.get("phase")
        if phase not in ("decode", "incremental_prefill"):
            raise ValueError(f"{scenario_id}: unsupported phase {phase!r}")
        phases.add(phase)
        weight = scenario.get("weight")
        if isinstance(weight, bool) or not isinstance(weight, (int, float)) or weight <= 0:
            raise ValueError(f"{scenario_id}: weight must be a positive number")
        phase_weights[phase] += float(weight)

        if phase == "decode":
            workload = scenario.get("workload")
            if workload not in ("dsa_trtllm_decode_m16", "dsa_trtllm_decode_m32"):
                raise ValueError(f"{scenario_id}: decode workload must be a DSA M16/M32 task")
            has_scalar = "kv_context" in scenario
            has_vector = "kv_contexts" in scenario
            if has_scalar == has_vector:
                raise ValueError(
                    f"{scenario_id}: set exactly one of kv_context or kv_contexts"
                )
            values = ([scenario["kv_context"]] if has_scalar
                      else scenario["kv_contexts"])
            if (not isinstance(values, list) or not values or
                    any(isinstance(value, bool) or not isinstance(value, int) or value <= 0
                        for value in values)):
                raise ValueError(f"{scenario_id}: KV lengths must be positive integers")
            expected_batch = 16 if workload.endswith("m16") else 32
            if has_vector and len(values) != expected_batch:
                raise ValueError(
                    f"{scenario_id}: {workload} requires exactly {expected_batch} ragged lengths"
                )
            decode_context_signatures.add(tuple(values))
        else:
            for key in ("prefix_tokens", "extend_tokens", "batch"):
                value = scenario.get(key)
                if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                    raise ValueError(f"{scenario_id}: {key} must be a positive integer")
            if scenario.get("execution_scope") not in (
                "sglang-containing-region", "sglang-end-to-end"
            ):
                raise ValueError(
                    f"{scenario_id}: incremental prefill requires a real SGLang execution_scope"
                )
            if not scenario.get("expected_backend"):
                raise ValueError(f"{scenario_id}: expected_backend is required")

    if profile["production_evidence"]:
        source = str(profile.get("evidence_source") or "").strip()
        if not source or "example" in source.lower() or "replace" in source.lower():
            raise ValueError("a production profile needs a concrete evidence_source")
        if phases != {"decode", "incremental_prefill"}:
            raise ValueError(
                "a production profile must cover decode and incremental_prefill"
            )
        if len(decode_context_signatures) < 2:
            raise ValueError("a production decode context matrix needs at least two scenarios")
        for phase, total in phase_weights.items():
            if abs(total - 1.0) > 1e-6:
                raise ValueError(
                    f"production {phase} scenario weights must sum to 1.0, got {total}"
                )
        for scenario in scenarios:
            if (scenario["phase"] == "incremental_prefill" and
                    "assume" in str(scenario["expected_backend"]).lower()):
                raise ValueError(
                    f"{scenario['id']}: production expected_backend must come from a trace"
                )
    return profile


def capture_runtime_facts(profile: dict[str, Any]) -> dict[str, Any]:
    sglang_root = Path(
        os.environ.get("SGLANG_ROOT", "/home/qinhaiyan/sglang")
    ).expanduser().resolve()
    facts: dict[str, Any] = {"sglang_root": str(sglang_root)}
    try:
        facts["sglang_commit"] = subprocess.check_output(
            ["git", "-C", str(sglang_root), "rev-parse", "HEAD"], text=True
        ).strip()
        facts["sglang_dirty"] = bool(subprocess.check_output(
            ["git", "-C", str(sglang_root), "status", "--porcelain"], text=True
        ).strip())
    except Exception as exc:
        facts["sglang_error"] = f"{type(exc).__name__}: {exc}"

    if profile["production_evidence"]:
        actual = facts.get("sglang_commit")
        expected = str(profile["deployment"]["sglang_commit"])
        if not actual or not actual.startswith(expected):
            raise ValueError(
                f"profile pins SGLang {expected}, runtime is {actual or 'unavailable'}"
            )
    return facts


def _scenario_command(
    scenario: dict[str, Any], args: argparse.Namespace, result_path: Path
) -> list[str]:
    command = [
        str(RUNNER),
        scenario["workload"],
        "--warmup", str(args.warmup),
        "--repeat", str(args.repeat),
        "--execution-mode", args.execution_mode,
        "--output", str(result_path),
    ]
    if args.candidate:
        command.extend(["--candidate", str(Path(args.candidate).expanduser().resolve())])
    if "kv_context" in scenario:
        command.extend(["--kv-context", str(scenario["kv_context"])])
    else:
        command.extend([
            "--kv-contexts",
            ",".join(str(value) for value in scenario["kv_contexts"]),
        ])
    return command


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--profile", required=True)
    parser.add_argument("--candidate")
    parser.add_argument("--warmup", type=int, default=8)
    parser.add_argument("--repeat", type=int, default=10)
    parser.add_argument(
        "--execution-mode", choices=("eager", "graph", "both"), default="both"
    )
    parser.add_argument("--output-dir")
    parser.add_argument("--scenario-timeout", type=int, default=1800)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.warmup < 0 or args.repeat < 1 or args.scenario_timeout < 1:
        parser.error("warmup must be >=0; repeat and scenario-timeout must be positive")

    profile_path = Path(args.profile).expanduser().resolve()
    try:
        profile = validate_profile(profile_path)
        runtime_facts = capture_runtime_facts(profile)
    except Exception as exc:
        parser.error(str(exc))
    decode_scenarios = [
        scenario for scenario in profile["scenarios"] if scenario["phase"] == "decode"
    ]
    if not decode_scenarios:
        parser.error("profile has no decode scenarios")

    if args.output_dir:
        output_dir = Path(args.output_dir).expanduser().resolve()
    else:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        output_dir = (
            REPO_ROOT / "runs" / "serving_native" / "context_matrix" /
            f"{stamp}-{profile['id']}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    environment = dict(os.environ)
    environment["KERNEL_HARNESS_TIMEOUT_SECONDS"] = str(args.scenario_timeout)
    records: list[dict[str, Any]] = []
    for scenario in decode_scenarios:
        result_path = output_dir / f"{scenario['id']}.json"
        log_path = output_dir / f"{scenario['id']}.log"
        command = _scenario_command(scenario, args, result_path)
        if args.dry_run:
            print(" ".join(command))
            records.append({"id": scenario["id"], "command": command, "dry_run": True})
            continue
        with log_path.open("w") as log:
            completed = subprocess.run(
                command,
                cwd=REPO_ROOT,
                env=environment,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=False,
            )
        result = None
        error = None
        try:
            result = json.loads(result_path.read_text())
        except Exception as exc:
            error = f"result unavailable: {type(exc).__name__}: {exc}"
        records.append({
            "id": scenario["id"],
            "phase": "decode",
            "workload": scenario["workload"],
            "weight": scenario["weight"],
            "kv_context": scenario.get("kv_context"),
            "kv_contexts": scenario.get("kv_contexts"),
            "command_exit_code": completed.returncode,
            "result_path": str(result_path),
            "log_path": str(log_path),
            "terminal_state": ((result or {}).get("verdict") or {}).get("terminal_state"),
            "leaf_gate_ok": bool(((result or {}).get("verdict") or {}).get("leaf_gate_ok")),
            "correct": bool(
                ((result or {}).get("correctness") or {}).get("pre_timing") and
                ((result or {}).get("correctness") or {}).get("post_timing_different_seed")
            ) if result else False,
            "error": error,
        })

    if args.dry_run:
        return 0

    all_commands_completed = all(record["command_exit_code"] == 0 for record in records)
    all_correct = all(record["correct"] for record in records)
    protocol_eligible = (
        args.execution_mode == "both" and args.warmup >= 8 and args.repeat >= 10
    )
    incremental = [
        scenario["id"] for scenario in profile["scenarios"]
        if scenario["phase"] == "incremental_prefill"
    ]

    if args.candidate is None:
        exit_code = 0 if all_commands_completed else 1
        terminal_state = (
            "REFERENCE_DECODE_CONTEXT_MATRIX" if exit_code == 0
            else "REFERENCE_CONTEXT_MATRIX_INCOMPLETE"
        )
    elif not all_correct:
        exit_code, terminal_state = 2, "INCORRECT_IN_CONTEXT_MATRIX"
    elif not protocol_eligible or not profile["production_evidence"]:
        exit_code, terminal_state = 1, "PROBE_ONLY_NO_VERDICT"
    elif all_commands_completed:
        exit_code, terminal_state = 0, "DECODE_SPARSE_ATTN_CONTEXT_MATRIX_LEAF_WIN"
    else:
        exit_code, terminal_state = 1, "NO_DECODE_CONTEXT_MATRIX_WIN_WITH_EVIDENCE"

    aggregate = {
        "schema_version": 1,
        "profile_id": profile["id"],
        "profile_path": str(profile_path),
        "profile_production_evidence": profile["production_evidence"],
        "runtime_facts": runtime_facts,
        "candidate": str(Path(args.candidate).expanduser().resolve()) if args.candidate else None,
        "execution_mode": args.execution_mode,
        "warmup": args.warmup,
        "repeat": args.repeat,
        "decode_scenarios": records,
        "incremental_prefill_scenarios_not_run_here": incremental,
        "verdict": {
            "exit_code": exit_code,
            "terminal_state": terminal_state,
            "production_ready": False,
            "next_required": (
                "run the decode containing region including full-context indexer/top-k, "
                "run every incremental-prefill profile scenario at the real SGLang "
                "containing-region boundary, then run the complete end-to-end trace"
            ),
        },
    }
    matrix_path = output_dir / "matrix.json"
    matrix_path.write_text(json.dumps(aggregate, indent=2, sort_keys=True) + "\n")
    print(json.dumps(aggregate, indent=2, sort_keys=True))
    print(f"context matrix: {matrix_path}", file=sys.stderr)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
