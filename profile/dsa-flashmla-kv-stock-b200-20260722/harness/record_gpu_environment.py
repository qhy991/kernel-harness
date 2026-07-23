#!/usr/bin/env python3
"""Record the selected physical GPU, logical CUDA device, clocks, and pins."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from gpu_lease_env import require_flexible_gpu


KH = Path(
    "/home/qinhaiyan/glm52-goal-runs/22-dsa_flashmla_kv_production/kernel-harness"
).resolve()
SG = Path(
    "/home/qinhaiyan/glm52-goal-runs/22-dsa_flashmla_kv_production/sglang"
).resolve()
FLASHMLA = (SG / "third_party/FlashMLA-goal22").resolve()
FLEXIBLE_WRAPPER = Path(
    "/home/qinhaiyan/glm52-goal-runs/with_flexible_gpu.sh"
).resolve()
QUERY_FIELDS = (
    "timestamp",
    "index",
    "uuid",
    "pci.bus_id",
    "name",
    "driver_version",
    "pstate",
    "clocks.current.graphics",
    "clocks.current.sm",
    "clocks.current.memory",
    "clocks.max.graphics",
    "clocks.max.sm",
    "clocks.max.memory",
    "power.draw",
    "power.limit",
    "temperature.gpu",
)


def command(*args: str, cwd: Path | None = None) -> str:
    return subprocess.run(
        args,
        cwd=cwd,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    ).stdout.strip()


def git_status(cwd: Path, *, include_untracked: bool) -> list[str]:
    # Directory-level untracked disclosure is enough to identify campaign and
    # build-artifact dirtiness without serializing every generated wheel file.
    untracked = "normal" if include_untracked else "no"
    output = command(
        "git",
        "status",
        "--short",
        f"--untracked-files={untracked}",
        cwd=cwd,
    )
    return output.splitlines() if output else []


def sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    physical_gpu = require_flexible_gpu()
    output = Path(args.output).expanduser().resolve()
    if output.exists():
        raise RuntimeError(f"refusing to overwrite GPU environment evidence: {output}")

    import torch

    if torch.cuda.device_count() != 1 or torch.cuda.current_device() != 0:
        raise RuntimeError(
            "flexible wrapper must expose exactly one device as logical cuda:0; "
            f"count={torch.cuda.device_count()} current={torch.cuda.current_device()}"
        )
    smi_line = command(
        "nvidia-smi",
        "-i",
        str(physical_gpu),
        f"--query-gpu={','.join(QUERY_FIELDS)}",
        "--format=csv,noheader,nounits",
    )
    values = [value.strip() for value in smi_line.split(",")]
    if len(values) != len(QUERY_FIELDS):
        raise RuntimeError(f"unexpected nvidia-smi record: {smi_line!r}")
    smi = dict(zip(QUERY_FIELDS, values))
    expected_uuid = os.environ.get("GOAL22_GPU_UUID")
    if expected_uuid and smi["uuid"] != expected_uuid:
        raise RuntimeError(
            f"wrapper/campaign UUID mismatch: {expected_uuid} != {smi['uuid']}"
        )
    props = torch.cuda.get_device_properties(0)
    result = {
        "schema_version": 1,
        "campaign_id": os.environ.get("GOAL22_CAMPAIGN_ID"),
        "stage": args.stage,
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "physical_gpu_index": physical_gpu,
        "logical_gpu_index": 0,
        "gpu_uuid": smi["uuid"],
        "nvidia_smi": smi,
        "torch_device": {
            "name": props.name,
            "capability": list(torch.cuda.get_device_capability(0)),
            "multiprocessor_count": props.multi_processor_count,
            "total_memory_bytes": props.total_memory,
        },
        "environment": {
            key: os.environ.get(key)
            for key in (
                "CUDA_VISIBLE_DEVICES",
                "SGLANG_ROOT",
                "SGLANG_GLM52_OPT",
                "KERNEL_HARNESS_PYTHON",
                "GOAL22_CAMPAIGN_ID",
                "GOAL22_PHYSICAL_GPU",
                "GOAL22_GPU_UUID",
                "GOAL22_STOCK_OVERLAY",
                "GOAL22_STOCK_MANIFEST",
                "GOAL22_CANDIDATE_OVERLAY",
                "GOAL22_CANDIDATE_MANIFEST",
            )
        },
        "stack": {
            "python": command(str(KH / ".venv/bin/python"), "--version"),
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "nvcc": command("nvcc", "--version"),
            "nsight_compute": command("ncu", "--version"),
            "nsight_systems": command("nsys", "--version"),
            "kernel_harness_commit": command("git", "rev-parse", "HEAD", cwd=KH),
            "sglang_commit": command("git", "rev-parse", "HEAD", cwd=SG),
            "flashmla_commit": command("git", "rev-parse", "HEAD", cwd=FLASHMLA),
            "scheduler_wrapper": str(FLEXIBLE_WRAPPER),
            "scheduler_wrapper_sha256": sha256(FLEXIBLE_WRAPPER),
        },
        "repository_status": {
            "kernel_harness": {
                "tracked": git_status(KH, include_untracked=False),
                "all": git_status(KH, include_untracked=True),
            },
            "sglang": {
                "tracked": git_status(SG, include_untracked=False),
                "all": git_status(SG, include_untracked=True),
            },
            "flashmla": {
                "tracked": git_status(FLASHMLA, include_untracked=False),
                "all": git_status(FLASHMLA, include_untracked=True),
            },
        },
    }
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(rendered)
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
