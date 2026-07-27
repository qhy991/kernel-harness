#!/usr/bin/env python3
"""Capture the exact locked-GPU software and repository environment."""

from __future__ import annotations

import importlib.metadata
import importlib.util
import json
import os
import subprocess
from pathlib import Path

import torch


KH_ROOT = Path(__file__).resolve().parents[2]
SGLANG_ROOT = Path(os.environ["SGLANG_ROOT"]).resolve()


def command(*args: str) -> str:
    return subprocess.run(
        args, check=False, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT
    ).stdout.strip()


def package(name: str) -> dict[str, str]:
    try:
        version = importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        version = "not-installed"
    spec = importlib.util.find_spec(name.replace("-", "_"))
    return {"version": version, "origin": str(spec.origin) if spec else "not-found"}


def repo(path: Path) -> dict[str, object]:
    return {
        "path": str(path),
        "head": command("git", "-C", str(path), "rev-parse", "HEAD"),
        "branch": command("git", "-C", str(path), "branch", "--show-current"),
        "status": command("git", "-C", str(path), "status", "--short").splitlines(),
    }


def main() -> int:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    props = torch.cuda.get_device_properties(0)
    result = {
        "cuda_visible_devices": visible,
        "physical_gpu_query": command(
            "nvidia-smi",
            "-i",
            visible,
            "--query-gpu=index,uuid,name,clocks.current.sm,clocks.current.memory,pstate,temperature.gpu,power.draw",
            "--format=csv,noheader",
        ),
        "nvidia_smi": command("nvidia-smi"),
        "torch": {
            "version": torch.__version__,
            "cuda": torch.version.cuda,
            "device_name": props.name,
            "compute_capability": list(torch.cuda.get_device_capability(0)),
            "total_memory": props.total_memory,
        },
        "packages": {
            name: package(name)
            for name in ("flashinfer", "flashinfer-python", "sglang", "sglang-kernel")
        },
        "tools": {
            "ncu": command("ncu", "--version"),
            "nsys": command("nsys", "--version"),
        },
        "repositories": {
            "kernel_harness": repo(KH_ROOT),
            "sglang": repo(SGLANG_ROOT),
        },
        "relevant_environment": {
            key: os.environ.get(key)
            for key in (
                "CUDA_VISIBLE_DEVICES",
                "SGLANG_ROOT",
                "SGLANG_GLM52_OPT",
                "SGLANG_DISABLE_DSA_INDEXER_FUSION",
                "INDEXER_WK_FLASHINFER_BACKEND",
            )
        },
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
