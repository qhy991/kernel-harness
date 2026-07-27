#!/usr/bin/env python3
"""Capture and enforce the pinned stock TP4 diagnostic runtime."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
EVIDENCE_DIR = Path(__file__).resolve().parent
PINNED_SGLANG_ROOT = Path(
    "/home/qinhaiyan/glm52-goal-runs/07-moe_w2_decode_kernel/sglang"
).resolve()
PINNED_DEEPEP_ROOT = (PINNED_SGLANG_ROOT / "build/deepep-overlay").resolve()
PINNED_DEEP_GEMM_ROOT = (
    PINNED_SGLANG_ROOT / "build/deep-gemm-stock-0.1.4.post1"
).resolve()


def expect(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text())
    expect(isinstance(data, dict), f"{path}: expected a JSON object")
    return data


def git_state(path: Path) -> dict[str, Any]:
    def command(*args: str) -> str:
        return subprocess.run(
            args, cwd=path, check=True, text=True, capture_output=True
        ).stdout.strip()

    return {
        "root": str(path),
        "branch": command("git", "branch", "--show-current"),
        "head": command("git", "rev-parse", "HEAD"),
        "status": command("git", "status", "--short").splitlines(),
    }


def atomic_write_new(path: Path, text: str) -> None:
    """Publish a complete file atomically, without replacing any prior evidence."""
    expect(path.parent.is_dir(), f"output parent does not exist: {path.parent}")
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"refusing to overwrite runtime probe: {path}")
    fd, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.tmp."
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "w") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        # link(2) is an atomic create-if-absent publication on this filesystem.
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> int:
    if len(sys.argv) != 2:
        raise SystemExit(f"usage: {Path(sys.argv[0]).name} OUTPUT.json")
    output = Path(sys.argv[1]).resolve()
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"refusing to overwrite runtime probe: {output}")

    attempt_dir_value = os.environ.get("TP4_ATTEMPT_DIR")
    expect(attempt_dir_value is not None, "TP4_ATTEMPT_DIR is required")
    attempt_dir = Path(attempt_dir_value).resolve()
    expected_cache = (attempt_dir / "deep_gemm_cache").resolve()

    exact_environment = {
        "CUDA_VISIBLE_DEVICES": "0,1,2,3",
        "SGLANG_ROOT": str(PINNED_SGLANG_ROOT),
        "DEEP_EP_ROOT": str(PINNED_DEEPEP_ROOT),
        "DEEP_GEMM_ROOT": str(PINNED_DEEP_GEMM_ROOT),
        "SGLANG_GLM52_OPT": "0",
        "SGLANG_DEEPGEMM_PDL": "true",
        "SGLANG_JIT_DEEPGEMM_PRECOMPILE": "0",
        "SGLANG_JIT_DEEPGEMM_FAST_WARMUP": "0",
        "SGL_DG_USE_NVRTC": "0",
        "DG_JIT_USE_NVRTC": "0",
        "SGLANG_DEEPGEMM_SANITY_CHECK": "0",
        "SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK": "128",
        "SGLANG_DG_CACHE_DIR": str(expected_cache),
        "DG_JIT_CACHE_DIR": str(expected_cache),
        "DG_JIT_WITH_LINEINFO": "1",
        "DG_JIT_PTXAS_VERBOSE": "1",
        "DG_JIT_DUMP_ASM": "1",
        "DG_PRINT_CONFIGS": "1",
        "DG_USE_NVIDIA_TOOLS": "1",
        "PYTHONNOUSERSITE": "1",
    }
    for key, expected in exact_environment.items():
        expect(
            os.environ.get(key) == expected,
            f"{key}={os.environ.get(key)!r}; expected {expected!r}",
        )
    expected_pythonpath = os.pathsep.join(
        (
            str(PINNED_DEEPEP_ROOT),
            str(PINNED_DEEP_GEMM_ROOT),
            str(PINNED_SGLANG_ROOT / "python"),
            str(ROOT),
        )
    )
    expect(
        os.environ.get("PYTHONPATH") == expected_pythonpath,
        "PYTHONPATH is not the isolated DeepEP/DeepGEMM/SGLang path",
    )

    deepep_provenance_path = EVIDENCE_DIR / "deepep_overlay_provenance.json"
    deep_gemm_provenance_path = EVIDENCE_DIR / "stock_deep_gemm_provenance.json"
    deepep_provenance = read_json(deepep_provenance_path)
    deep_gemm_provenance = read_json(deep_gemm_provenance_path)
    expect(deepep_provenance.get("schema_version") == 1, "wrong DeepEP provenance schema")
    expect(
        deepep_provenance.get("installed_in_place") is False
        and deepep_provenance.get("overlay_role") == "stock diagnostic dependency",
        "DeepEP provenance is not the isolated stock diagnostic overlay",
    )
    expect(deep_gemm_provenance.get("schema_version") == 1, "wrong DeepGEMM provenance schema")
    expect(
        Path(deepep_provenance["extension_path"]).resolve()
        == PINNED_DEEPEP_ROOT
        / "deep_ep_cpp.cpython-312-x86_64-linux-gnu.so",
        "DeepEP provenance extension path drifted",
    )
    expect(
        Path(deep_gemm_provenance["isolated_overlay"]["path"]).resolve()
        == PINNED_DEEP_GEMM_ROOT,
        "DeepGEMM provenance overlay path drifted",
    )

    # Imports happen only after every non-CUDA precondition above has passed.
    import deep_ep  # noqa: PLC0415
    import deep_ep_cpp  # noqa: PLC0415
    import deep_gemm  # noqa: PLC0415
    import sglang  # noqa: PLC0415
    import torch  # noqa: PLC0415

    deep_ep_module = Path(deep_ep.__file__).resolve()
    deep_ep_extension = Path(deep_ep_cpp.__file__).resolve()
    deep_gemm_python = Path(deep_gemm.__file__).resolve()
    deep_gemm_extension = deep_gemm_python.parent / "_C.so"
    deep_gemm_device_source = (
        deep_gemm_python.parent
        / "include/deep_gemm/impls/sm100_fp8_fp4_gemm_1d1d.cuh"
    )
    sglang_module = Path(sglang.__file__).resolve()

    expect(
        deep_ep_module == PINNED_DEEPEP_ROOT / "deep_ep/__init__.py",
        f"wrong DeepEP import: {deep_ep_module}",
    )
    expect(
        deep_ep_extension == Path(deepep_provenance["extension_path"]).resolve(),
        f"wrong DeepEP extension import: {deep_ep_extension}",
    )
    deep_ep_version = importlib.metadata.version("deep-ep")
    expect(
        deep_ep_version == deepep_provenance["deep_ep_version"],
        f"wrong DeepEP version: {deep_ep_version}",
    )
    expect(
        sha256(deep_ep_extension) == deepep_provenance["extension_sha256"],
        "wrong DeepEP extension SHA256",
    )

    stock_overlay = deep_gemm_provenance["isolated_overlay"]
    target_source = deep_gemm_provenance["target_device_source"]
    deep_gemm_version = importlib.metadata.version("sgl-deep-gemm")
    expect(
        deep_gemm_python == PINNED_DEEP_GEMM_ROOT / "deep_gemm/__init__.py",
        f"wrong DeepGEMM import: {deep_gemm_python}",
    )
    expect(deep_gemm_version == deep_gemm_provenance["version"], "wrong DeepGEMM version")
    expect(
        sha256(deep_gemm_python) == stock_overlay["python_sha256"],
        "wrong DeepGEMM Python SHA256",
    )
    expect(
        sha256(deep_gemm_extension) == stock_overlay["extension_sha256"],
        "wrong DeepGEMM extension SHA256",
    )
    expect(
        sha256(deep_gemm_device_source) == target_source["sha256"],
        "wrong DeepGEMM device-source SHA256",
    )
    expect(
        (deep_gemm_python.parent / "VERSION").read_text().strip()
        == deep_gemm_provenance["version"],
        "DeepGEMM VERSION file drifted",
    )
    expect(
        sglang_module == PINNED_SGLANG_ROOT / "python/sglang/__init__.py",
        f"wrong SGLang import: {sglang_module}",
    )

    # Exercise SGLang's derivation, rather than merely trusting a shell export.
    declared_dg_jit_use_nvrtc = os.environ.pop("DG_JIT_USE_NVRTC")
    from sglang.srt.layers.deep_gemm_wrapper import compile_utils  # noqa: F401,PLC0415

    derived_dg_jit_use_nvrtc = os.environ.get("DG_JIT_USE_NVRTC")
    expect(declared_dg_jit_use_nvrtc == "0", "declared DG_JIT_USE_NVRTC drifted")
    expect(derived_dg_jit_use_nvrtc == "0", "derived DG_JIT_USE_NVRTC drifted")
    exact_environment["DG_JIT_USE_NVRTC"] = "0"

    expect(torch.cuda.is_available(), "CUDA is required; use with_all_gpus_lock.sh")
    visible_device_count = torch.cuda.device_count()
    expect(visible_device_count == 4, f"expected four visible GPUs, found {visible_device_count}")
    devices = []
    for index in range(visible_device_count):
        properties = torch.cuda.get_device_properties(index)
        capability = list(torch.cuda.get_device_capability(index))
        expect("B200" in properties.name, f"logical GPU {index} is not B200: {properties.name}")
        expect(capability == [10, 0], f"logical GPU {index} is not SM100: {capability}")
        devices.append(
            {
                "logical_index": index,
                "name": properties.name,
                "capability": capability,
                "multi_processor_count": properties.multi_processor_count,
                "total_memory": properties.total_memory,
            }
        )

    configs = {
        "ep4_dispatch": str(deep_ep.Buffer.get_dispatch_config(4)),
        "ep4_combine": str(deep_ep.Buffer.get_combine_config(4)),
        "ep8_dispatch": str(deep_ep.Buffer.get_dispatch_config(8)),
        "ep8_combine": str(deep_ep.Buffer.get_combine_config(8)),
    }
    expect(all(value and value != "None" for value in configs.values()), "empty DeepEP config")

    result = {
        "schema_version": 2,
        "evidence_scope": "diagnostic_tp4_not_production_tp8",
        "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
        "python": sys.version,
        "python_executable": sys.executable,
        "python_path": list(sys.path),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "visible_device_count": visible_device_count,
        "devices": devices,
        "paths": {
            "kernel_harness": str(ROOT),
            "sglang_root": str(PINNED_SGLANG_ROOT),
            "deepep_overlay": str(PINNED_DEEPEP_ROOT),
            "deep_gemm_overlay": str(PINNED_DEEP_GEMM_ROOT),
            "attempt": str(attempt_dir),
            "deep_gemm_cache": str(expected_cache),
        },
        "deep_ep": {
            "distribution_version": deep_ep_version,
            "module": str(deep_ep_module),
            "module_sha256": sha256(deep_ep_module),
            "extension": str(deep_ep_extension),
            "extension_sha256": sha256(deep_ep_extension),
            "source_commit": deepep_provenance["deep_ep_source_commit"],
            "provenance": str(deepep_provenance_path),
            "provenance_sha256": sha256(deepep_provenance_path),
        },
        "deep_gemm": {
            "distribution_version": deep_gemm_version,
            "python": str(deep_gemm_python),
            "python_sha256": sha256(deep_gemm_python),
            "extension": str(deep_gemm_extension),
            "extension_sha256": sha256(deep_gemm_extension),
            "device_source": str(deep_gemm_device_source),
            "device_source_sha256": sha256(deep_gemm_device_source),
            "pdl_before_worker_runtime_setup": bool(deep_gemm.get_pdl()),
            "provenance": str(deep_gemm_provenance_path),
            "provenance_sha256": sha256(deep_gemm_provenance_path),
        },
        "sglang": {
            "module": str(sglang_module),
            "module_sha256": sha256(sglang_module),
        },
        "contract_environment": {
            key: os.environ.get(key) for key in exact_environment
        },
        "nvrtc_derivation": {
            "source_environment_key": "SGL_DG_USE_NVRTC",
            "declared_dg_jit_use_nvrtc": declared_dg_jit_use_nvrtc,
            "derived_dg_jit_use_nvrtc": derived_dg_jit_use_nvrtc,
        },
        "git": {
            "kernel_harness": git_state(ROOT),
            "sglang": git_state(PINNED_SGLANG_ROOT),
        },
        "configs": configs,
    }
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    atomic_write_new(output, rendered)
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
