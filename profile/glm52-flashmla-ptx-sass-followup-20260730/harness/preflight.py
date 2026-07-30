#!/usr/bin/env python3
"""Fresh machine-readable preflight for the PTX/SASS follow-up goal.

Adapted from the prior campaign's preflight. Differences: this task's own
task-local cache prefix is required, and the prior campaign's rejected
experimental shared object is recorded as read-only prior evidence rather than
as a build input.
"""

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
REQUIRED_BASES = {
    "kernel_harness": "c93b342",
    "sglang": "c52f23b56",
    "flashmla": "65293ac",
}
TASK_CACHE_PREFIX = (
    "/home/qinhaiyan/glm52-hotspot-goal-runs/cache/flashmla_ptx_sass_followup"
)
PRIOR_REJECTED_CANDIDATE = Path(
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
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def repository(name: str, path: Path) -> dict[str, object]:
    head = command("git", "-C", str(path), "rev-parse", "HEAD")
    required = REQUIRED_BASES[name]
    contains = (
        subprocess.run(
            ["git", "-C", str(path), "merge-base", "--is-ancestor", required, head],
            check=False,
        ).returncode
        == 0
    )
    return {
        "path": str(path),
        "head": head,
        "branch": command("git", "-C", str(path), "branch", "--show-current"),
        "status": command("git", "-C", str(path), "status", "--porcelain=v1").splitlines(),
        "required_base": required,
        "head_is_at_or_after_required_base": contains,
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
        name: os.environ.get(name)
        for name in (
            "CUDA_CACHE_PATH",
            "TORCH_EXTENSIONS_DIR",
            "TRITON_CACHE_DIR",
            "XDG_CACHE_HOME",
            "DG_JIT_CACHE_DIR",
            "SGLANG_DG_CACHE_DIR",
            "GLM52_TASK_BUILD_DIR",
            "MAX_JOBS",
            "NVCC_THREADS",
            "CMAKE_BUILD_PARALLEL_LEVEL",
        )
    }
    path_like = {
        name: value
        for name, value in cache_env.items()
        if value and ("DIR" in name or "PATH" in name or name == "XDG_CACHE_HOME")
    }
    if not all(value.startswith(TASK_CACHE_PREFIX) for value in path_like.values()):
        raise AssertionError(f"non-task-local cache path: {path_like}")

    device_props = torch.cuda.get_device_properties(0)
    evidence = {
        "schema_version": 1,
        "stage": "P0_gpu_preflight",
        "task": "flashmla_ptx_sass_followup",
        "captured_utc": datetime.now(timezone.utc).isoformat(),
        "repositories": {
            name: repository(name, path) for name, path in REPOSITORIES.items()
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
            "visible_device_name": device_props.name,
            "visible_compute_capability": list(torch.cuda.get_device_capability(0)),
            "multi_processor_count": device_props.multi_processor_count,
            "shared_memory_per_block_optin": getattr(
                device_props, "shared_memory_per_block_optin", None
            ),
            "l2_cache_size": getattr(device_props, "L2_cache_size", None),
            "clock_snapshot": command(
                "nvidia-smi",
                "-i",
                os.environ["GLM52_PHYSICAL_GPU"],
                "--query-gpu=timestamp,uuid,pstate,clocks.sm,clocks.max.sm,"
                "clocks.mem,temperature.gpu,power.draw,power.limit",
                "--format=csv,noheader,nounits",
            ),
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
            "installed_flashmla_control": {
                "path": str(flashmla_extension),
                "sha256": sha256(flashmla_extension),
                "size_bytes": flashmla_extension.stat().st_size,
            },
            "prior_campaign_rejected_candidate": (
                {
                    "path": str(PRIOR_REJECTED_CANDIDATE),
                    "sha256": sha256(PRIOR_REJECTED_CANDIDATE),
                    "size_bytes": PRIOR_REJECTED_CANDIDATE.stat().st_size,
                    "role": "read-only prior evidence; rejected no-replacement",
                }
                if PRIOR_REJECTED_CANDIDATE.is_file()
                else {"path": str(PRIOR_REJECTED_CANDIDATE), "present": False}
            ),
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
                "leased": evidence["gpu"]["leased_physical_index"],
                "sm_count": device_props.multi_processor_count,
                "torch": torch.__version__,
                "free_gib": round(evidence["root_filesystem"]["free_gib"], 2),
                "threshold_satisfied": evidence["root_filesystem"][
                    "threshold_satisfied"
                ],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
