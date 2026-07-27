#!/usr/bin/env python3
"""Collect reproducible source, package, and physical-GPU provenance."""

from __future__ import annotations

import importlib.metadata
import inspect
import json
import os
import subprocess
from pathlib import Path


KH_ROOT = Path(__file__).resolve().parents[2]
SGLANG_ROOT = Path(
    "/home/qinhaiyan/glm52-goal-runs/03-dsa_prefill_attention/sglang"
)


def command(*args: str, cwd: Path | None = None) -> str:
    return subprocess.check_output(args, cwd=cwd, text=True).strip()


def package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "not-installed"


def main() -> None:
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "3":
        raise RuntimeError("collect with physical GPU 3 as the only visible CUDA device")

    import deep_gemm
    import flashinfer
    import flashinfer.decode
    import sgl_kernel
    import sglang
    import torch

    device = torch.cuda.get_device_properties(0)
    result = {
        "collected_utc": command("date", "-u", "+%Y-%m-%dT%H:%M:%SZ"),
        "visibility": {
            "CUDA_VISIBLE_DEVICES": os.environ["CUDA_VISIBLE_DEVICES"],
            "logical_device": 0,
            "physical_device": 3,
            "visible_count": torch.cuda.device_count(),
        },
        "gpu": {
            "name": device.name,
            "compute_capability": list(torch.cuda.get_device_capability(0)),
            "total_memory_bytes": device.total_memory,
            "nvidia_smi_physical_3": command(
                "nvidia-smi",
                "-i",
                "3",
                "--query-gpu=index,uuid,name,pstate,temperature.gpu,memory.total,memory.used,clocks.max.sm,power.draw,power.limit,driver_version",
                "--format=csv,noheader,nounits",
            ),
            "host_topology_read_only": command("nvidia-smi", "topo", "-m"),
        },
        "python": {
            "executable": os.sys.executable,
            "version": os.sys.version,
        },
        "packages": {
            "torch": {"version": torch.__version__, "path": inspect.getfile(torch)},
            "torch_cuda": torch.version.cuda,
            "sglang": {"version": package_version("sglang"), "path": inspect.getfile(sglang)},
            "flashinfer_python": {
                "version": package_version("flashinfer-python"),
                "path": inspect.getfile(flashinfer),
                "decode_path": inspect.getfile(flashinfer.decode),
            },
            "sgl_kernel": {
                "version": package_version("sglang-kernel"),
                "path": inspect.getfile(sgl_kernel),
            },
            "deep_gemm": {
                "version": package_version("sgl-deep-gemm"),
                "path": inspect.getfile(deep_gemm),
            },
        },
        "tools": {
            "nsys": command("nsys", "--version"),
            "ncu": command("ncu", "--version"),
        },
        "repositories": {
            "kernel_harness": {
                "root": str(KH_ROOT),
                "head": command("git", "rev-parse", "HEAD", cwd=KH_ROOT),
                "status": command("git", "status", "--short", cwd=KH_ROOT),
            },
            "sglang": {
                "root": str(SGLANG_ROOT),
                "base": "f93f8867b4bc124c9809c9110ec7361ed11b6b4a",
                "trial": "b03db3f648f9db5b9264638716d20adacc510d6e",
                "stock_revert_head": command("git", "rev-parse", "HEAD", cwd=SGLANG_ROOT),
                "status": command("git", "status", "--short", cwd=SGLANG_ROOT),
            },
        },
        "build_and_import": {
            "sglang": "no build; isolated source imported through SGLANG_ROOT/python",
            "reached_kernel": "FlashInfer packaged TRT-LLM AOT binary; no package modified",
            "candidate": str(KH_ROOT / "serving_native/candidates/dsa_prefill_pdl_off.py"),
        },
    }
    destination = Path(__file__).with_name("environment.json")
    destination.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(destination)


if __name__ == "__main__":
    main()
