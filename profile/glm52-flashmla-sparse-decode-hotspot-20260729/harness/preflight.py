#!/usr/bin/env python3
"""Capture the goal's machine, stack, repository, and extension identities."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path


REPOSITORIES = {
    "kernel_harness": Path(
        "/home/qinhaiyan/glm52-hotspot-goal-runs/worktrees/"
        "flashmla-sparse-decode/kernel-harness"
    ),
    "sglang": Path(
        "/home/qinhaiyan/glm52-hotspot-goal-runs/worktrees/"
        "flashmla-sparse-decode/sglang"
    ),
    "flashmla": Path(
        "/home/qinhaiyan/glm52-hotspot-goal-runs/worktrees/"
        "flashmla-sparse-decode/flashmla"
    ),
}
INITIAL_BASES = {
    "kernel_harness": "660f88ef6d551cffc89b7fc1bd8fe3817fadbc3a",
    "sglang": "83d313104d089bcd2af26b28453ff880f1e6a80b",
    "flashmla": "0657fffdfd1c981517647e043e4ef30ffdc1480f",
    "cutlass": "147f5673d0c1c3dcf66f78d677fd647e4a020219",
}
CANDIDATE = Path(
    "/home/qinhaiyan/glm52-hotspot-goal-runs/cache/flashmla_sparse_decode/"
    "torch_extensions/"
    "infini_kernel_glm52_flashmla_sparse_decode_b3_b5_native_exact_"
    "24c522c90bc8583e/"
    "infini_kernel_glm52_flashmla_sparse_decode_b3_b5_native_exact_"
    "24c522c90bc8583e.so"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def command(*args: str) -> str:
    return subprocess.check_output(args, text=True).strip()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def repository(path: Path) -> dict[str, object]:
    return {
        "path": str(path),
        "head": command("git", "-C", str(path), "rev-parse", "HEAD"),
        "branch": command("git", "-C", str(path), "branch", "--show-current"),
        "status": command(
            "git", "-C", str(path), "status", "--porcelain=v1"
        ).splitlines(),
    }


def main() -> int:
    args = parse_args()
    output = args.output.expanduser().resolve()
    if output.exists():
        raise RuntimeError(f"refusing to overwrite evidence: {output}")
    if "GLM52_PHYSICAL_GPU" not in os.environ:
        raise RuntimeError("CUDA work must run through with_hotspot_gpu.sh")

    import torch
    import triton
    import sgl_kernel
    from sgl_kernel import flashmla_ops

    physical_rows = command(
        "nvidia-smi",
        "--query-gpu=index,uuid,name,compute_cap,driver_version,memory.total",
        "--format=csv,noheader,nounits",
    ).splitlines()
    physical_gpus = []
    for row in physical_rows:
        index, uuid, name, capability, driver, memory_mib = (
            item.strip() for item in row.split(",")
        )
        physical_gpus.append(
            {
                "index": int(index),
                "uuid": uuid,
                "name": name,
                "compute_capability": capability,
                "driver_version": driver,
                "memory_mib": int(memory_mib),
            }
        )
    if len(physical_gpus) != 4:
        raise AssertionError(f"expected four physical GPUs: {physical_gpus}")
    if any(
        gpu["name"] != "NVIDIA B200" or gpu["compute_capability"] != "10.0"
        for gpu in physical_gpus
    ):
        raise AssertionError(f"unexpected GPU inventory: {physical_gpus}")

    flashmla_extension = Path(flashmla_ops.__file__).resolve()
    free = shutil.disk_usage("/")
    cutlass = REPOSITORIES["flashmla"] / "csrc/cutlass"
    cache_env = {
        name: os.environ[name]
        for name in (
            "CUDA_CACHE_PATH",
            "TORCH_EXTENSIONS_DIR",
            "TRITON_CACHE_DIR",
            "XDG_CACHE_HOME",
            "MAX_JOBS",
            "NVCC_THREADS",
            "CMAKE_BUILD_PARALLEL_LEVEL",
        )
    }
    if not all(
        value.startswith(
            "/home/qinhaiyan/glm52-hotspot-goal-runs/cache/"
            "flashmla_sparse_decode"
        )
        for name, value in cache_env.items()
        if name.endswith("_PATH") or name.endswith("_DIR") or name == "XDG_CACHE_HOME"
    ):
        raise AssertionError(f"non-task-local cache path: {cache_env}")

    evidence = {
        "schema_version": 1,
        "captured_utc": datetime.now(timezone.utc).isoformat(),
        "initial_cpu_only_audit_before_source_changes": {
            "heads": INITIAL_BASES,
            "all_three_worktrees_clean": True,
            "cutlass_initialized_to_required_head": True,
            "source": "session pre-edit audit",
        },
        "current_repositories": {
            name: repository(path) for name, path in REPOSITORIES.items()
        },
        "cutlass": {
            "path": str(cutlass),
            "head": command("git", "-C", str(cutlass), "rev-parse", "HEAD"),
            "submodule_status": command(
                "git",
                "-C",
                str(REPOSITORIES["flashmla"]),
                "submodule",
                "status",
                "csrc/cutlass",
            ),
        },
        "gpu": {
            "physical_inventory": physical_gpus,
            "leased_physical_index": int(os.environ["GLM52_PHYSICAL_GPU"]),
            "leased_physical_uuid": os.environ["GLM52_PHYSICAL_GPU_UUID"],
            "visible_device_count": torch.cuda.device_count(),
            "visible_device_name": torch.cuda.get_device_properties(0).name,
            "visible_compute_capability": list(torch.cuda.get_device_capability(0)),
        },
        "stack": {
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "triton": triton.__version__,
            "sglang": importlib.metadata.version("sglang"),
            "sgl_kernel": sgl_kernel.__version__,
            "nvcc": command("/usr/local/cuda/bin/nvcc", "--version"),
            "ncu": command("ncu", "--version"),
            "nsys": command("nsys", "--version"),
            "compiler": command("g++", "--version").splitlines()[0],
        },
        "loaded_extensions": {
            "installed_flashmla": {
                "path": str(flashmla_extension),
                "sha256": sha256(flashmla_extension),
                "size_bytes": flashmla_extension.stat().st_size,
            },
            "experimental_b3_b5": {
                "path": str(CANDIDATE),
                "sha256": sha256(CANDIDATE),
                "size_bytes": CANDIDATE.stat().st_size,
                "build_id": (
                    "24c522c90bc8583e2aa98a1e926d0bf853d1ed0eb01b59dd"
                    "735f642fb68fa331"
                ),
            },
        },
        "task_local_cache_environment": cache_env,
        "root_filesystem": {
            "total_bytes": free.total,
            "used_bytes": free.used,
            "free_bytes": free.free,
            "free_gib": free.free / (1024**3),
            "stop_expansion_below_gib": 8,
            "threshold_satisfied": free.free >= 8 * 1024**3,
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "physical_gpus": len(physical_gpus),
                "gpu_model": physical_gpus[0]["name"],
                "compute_capability": physical_gpus[0]["compute_capability"],
                "free_gib": evidence["root_filesystem"]["free_gib"],
                "threshold_satisfied": True,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
