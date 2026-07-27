#!/usr/bin/env python3
"""Run final checks inside a flexible-GPU lease and persist their raw output."""

from __future__ import annotations

import argparse
import datetime
import json
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SGLANG = ROOT.parent / "sglang"


def run(label: str, command: list[str], cwd: Path) -> dict:
    completed = subprocess.run(
        command,
        cwd=cwd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    record = {
        "label": label,
        "command": command,
        "cwd": str(cwd),
        "returncode": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
    }
    if label == "verify_harness" and completed.returncode == 0:
        record["parsed_stdout"] = json.loads(completed.stdout)
    return record


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite {args.output}")
    if not os.environ.get("CUDA_VISIBLE_DEVICES"):
        raise SystemExit("CUDA_VISIBLE_DEVICES is unset; use with_flexible_gpu.sh")

    checks = [
        run(
            "check_env",
            [sys.executable, "testbench/bin/check_env.py"],
            ROOT,
        ),
        run(
            "serving_native_selftest",
            [sys.executable, "serving_native/selftest.py"],
            ROOT,
        ),
        run(
            "testbench_selftest",
            ["python3", "testbench/bin/selftest.py"],
            ROOT,
        ),
        run(
            "verify_harness",
            ["python3", "testbench/bin/verify_harness.py", "--json"],
            ROOT,
        ),
        run(
            "kernel_harness_diff_check",
            ["git", "diff", "--check"],
            ROOT,
        ),
        run(
            "sglang_status",
            ["git", "status", "--porcelain"],
            SGLANG,
        ),
    ]
    record = {
        "schema_version": 1,
        "timestamp_utc": datetime.datetime.now(
            datetime.timezone.utc
        ).isoformat(),
        "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
        "python": sys.executable,
        "checks": checks,
        "ok": all(
            check["returncode"] == 0
            and (check["label"] != "sglang_status" or not check["stdout"])
            for check in checks
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    print(json.dumps(record, indent=2, sort_keys=True))
    return 0 if record["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
