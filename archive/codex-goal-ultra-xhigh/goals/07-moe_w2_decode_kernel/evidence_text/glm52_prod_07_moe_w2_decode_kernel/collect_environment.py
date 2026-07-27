#!/usr/bin/env python3
"""Capture the exact locked GPU, package, import, and worktree provenance."""

from __future__ import annotations

import hashlib
import importlib.metadata
import inspect
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SGLANG_ROOT = Path(
    os.environ.get(
        "SGLANG_ROOT",
        "/home/qinhaiyan/glm52-goal-runs/07-moe_w2_decode_kernel/sglang",
    )
).resolve()
EXPECTED_DEEP_GEMM_ROOT = (
    Path(os.environ["DEEP_GEMM_ROOT"]).resolve()
    if os.environ.get("DEEP_GEMM_ROOT")
    else None
)
EXPECTED_DEEP_GEMM_VERSION = os.environ.get("EXPECTED_DEEP_GEMM_VERSION")
EXPECTED_DEEP_GEMM_EXTENSION_SHA256 = os.environ.get(
    "EXPECTED_DEEP_GEMM_EXTENSION_SHA256"
)
EXPECTED_DEEP_GEMM_DEVICE_SOURCE_SHA256 = os.environ.get(
    "EXPECTED_DEEP_GEMM_DEVICE_SOURCE_SHA256"
)
for path in (SGLANG_ROOT / "python", ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import deep_gemm  # noqa: E402
import sglang  # noqa: E402
import torch  # noqa: E402
from deep_gemm.utils.layout import (  # noqa: E402
    get_mk_alignment_for_contiguous_layout,
    get_theoretical_mk_alignment_for_contiguous_layout,
)


def command(*args: str) -> str:
    return subprocess.run(
        args, check=True, text=True, capture_output=True
    ).stdout.strip()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_state(path: Path) -> dict:
    return {
        "root": str(path),
        "branch": command("git", "-C", str(path), "branch", "--show-current"),
        "head": command("git", "-C", str(path), "rev-parse", "HEAD"),
        "status": command("git", "-C", str(path), "status", "--short").splitlines(),
    }


def atomic_write(path: Path, text: str) -> None:
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(text)
    os.replace(temporary, path)


def main() -> int:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; invoke under with_all_gpus_lock.sh")
    torch.cuda.set_device(0)
    original_pdl = bool(deep_gemm.get_pdl())
    deep_gemm.set_pdl(True)
    if not bool(deep_gemm.get_pdl()):
        raise RuntimeError("failed to apply SGLang's production DeepGEMM PDL policy")
    properties = torch.cuda.get_device_properties(0)
    package_path = Path(deep_gemm.__file__).resolve()
    distribution_version = importlib.metadata.version("sgl-deep-gemm")
    if (
        EXPECTED_DEEP_GEMM_ROOT is not None
        and EXPECTED_DEEP_GEMM_ROOT not in package_path.parents
    ):
        raise RuntimeError(
            f"wrong DeepGEMM import: {package_path}; "
            f"expected under {EXPECTED_DEEP_GEMM_ROOT}"
        )
    if (
        EXPECTED_DEEP_GEMM_VERSION is not None
        and distribution_version != EXPECTED_DEEP_GEMM_VERSION
    ):
        raise RuntimeError(
            f"wrong DeepGEMM distribution: {distribution_version}; "
            f"expected {EXPECTED_DEEP_GEMM_VERSION}"
        )
    extension_path = package_path.parent / "_C.so"
    if not extension_path.is_file():
        raise FileNotFoundError(f"DeepGEMM extension missing: {extension_path}")
    extension_sha256 = sha256(extension_path)
    device_source_path = (
        package_path.parent
        / "include/deep_gemm/impls/sm100_fp8_fp4_gemm_1d1d.cuh"
    )
    if not device_source_path.is_file():
        raise FileNotFoundError(f"DeepGEMM device source missing: {device_source_path}")
    device_source_sha256 = sha256(device_source_path)
    if (
        EXPECTED_DEEP_GEMM_EXTENSION_SHA256 is not None
        and extension_sha256 != EXPECTED_DEEP_GEMM_EXTENSION_SHA256
    ):
        raise RuntimeError(
            f"wrong DeepGEMM extension SHA256: {extension_sha256}; "
            f"expected {EXPECTED_DEEP_GEMM_EXTENSION_SHA256}"
        )
    if (
        EXPECTED_DEEP_GEMM_DEVICE_SOURCE_SHA256 is not None
        and device_source_sha256 != EXPECTED_DEEP_GEMM_DEVICE_SOURCE_SHA256
    ):
        raise RuntimeError(
            f"wrong DeepGEMM device-source SHA256: {device_source_sha256}; "
            f"expected {EXPECTED_DEEP_GEMM_DEVICE_SOURCE_SHA256}"
        )
    sglang_path = Path(sglang.__file__).resolve()
    if SGLANG_ROOT not in sglang_path.parents:
        raise RuntimeError(f"wrong SGLang import: {sglang_path}")

    payload = {
        "schema_version": 1,
        "captured_utc": datetime.now(timezone.utc).isoformat(),
        "platform": platform.platform(),
        "python": sys.version,
        "python_executable": sys.executable,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "nvidia_smi_list": command("nvidia-smi", "-L").splitlines(),
        "nvidia_smi_topology": command("nvidia-smi", "topo", "-m").splitlines(),
        "nvidia_smi_query": command(
            "nvidia-smi",
            "--query-gpu=index,uuid,name,driver_version,clocks.sm,clocks.mem,temperature.gpu,power.draw",
            "--format=csv,noheader,nounits",
        ).splitlines(),
        "nvcc": command("nvcc", "--version").splitlines(),
        "ncu": command("ncu", "--version").splitlines(),
        "nsys": command("nsys", "--version").splitlines(),
        "torch": {
            "version": torch.__version__,
            "cuda": torch.version.cuda,
            "device_count": torch.cuda.device_count(),
            "device_name": properties.name,
            "compute_capability": [properties.major, properties.minor],
            "total_memory": properties.total_memory,
            "multi_processor_count": properties.multi_processor_count,
        },
        "deep_gemm": {
            "distribution_version": distribution_version,
            "python_path": str(package_path),
            "python_sha256": sha256(package_path),
            "extension_path": str(extension_path),
            "extension_sha256": extension_sha256,
            "device_source_path": str(device_source_path),
            "device_source_sha256": device_source_sha256,
            "grouped_masked_signature": str(
                inspect.signature(deep_gemm.fp8_m_grouped_gemm_nt_masked)
            ),
            "num_sms": int(deep_gemm.get_num_sms()),
            "pdl": bool(deep_gemm.get_pdl()),
            "pdl_before_production_setup": original_pdl,
            "pdl_policy": "SGLang default SGLANG_DEEPGEMM_PDL=true",
            "alignment": int(get_mk_alignment_for_contiguous_layout()),
            "theoretical_alignment_expected_m4": int(
                get_theoretical_mk_alignment_for_contiguous_layout(4)
            ),
            "theoretical_alignment_expected_m5": int(
                get_theoretical_mk_alignment_for_contiguous_layout(5)
            ),
            "theoretical_alignment_expected_m8": int(
                get_theoretical_mk_alignment_for_contiguous_layout(8)
            ),
            "theoretical_alignment_expected_m9": int(
                get_theoretical_mk_alignment_for_contiguous_layout(9)
            ),
            "jit_cache": os.environ.get("SGLANG_DG_CACHE_DIR"),
        },
        "sglang_import": str(sglang_path),
        "git": {
            "kernel_harness": git_state(ROOT),
            "sglang": git_state(SGLANG_ROOT),
        },
        "contract_environment": {
            key: os.environ.get(key)
            for key in (
                "SGLANG_ROOT",
                "SGLANG_GLM52_OPT",
                "SGLANG_DEEPGEMM_PDL",
                "SGLANG_JIT_DEEPGEMM_PRECOMPILE",
                "SGLANG_JIT_DEEPGEMM_FAST_WARMUP",
                "SGL_DG_USE_NVRTC",
                "DG_JIT_USE_NVRTC",
                "SGLANG_DEEPGEMM_SANITY_CHECK",
                "SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK",
                "SGLANG_DG_CACHE_DIR",
                "DG_JIT_CACHE_DIR",
                "DEEP_GEMM_ROOT",
                "EXPECTED_DEEP_GEMM_VERSION",
                "EXPECTED_DEEP_GEMM_EXTENSION_SHA256",
                "EXPECTED_DEEP_GEMM_DEVICE_SOURCE_SHA256",
                "DG_JIT_WITH_LINEINFO",
                "DG_JIT_PTXAS_VERBOSE",
                "DG_JIT_DUMP_ASM",
                "DG_PRINT_CONFIGS",
                "DG_USE_NVIDIA_TOOLS",
            )
        },
    }
    deep_gemm.set_pdl(original_pdl)
    restored_pdl = bool(deep_gemm.get_pdl())
    if restored_pdl != original_pdl:
        raise RuntimeError(
            f"DeepGEMM PDL restore failed: {restored_pdl} != {original_pdl}"
        )
    payload["deep_gemm"]["pdl_restore"] = {
        "restored": True,
        "restored_value": restored_pdl,
    }
    rendered = json.dumps(payload, indent=2, sort_keys=True)
    print(rendered)
    atomic_write(Path(__file__).resolve().parent / "environment.json", rendered + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
