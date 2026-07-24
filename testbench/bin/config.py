#!/usr/bin/env python3
"""Portable environment and backend configuration.

Resolution order: environment variable → ``testbench/harness.env`` → default.
Paths resolve from the repository root.
"""
from __future__ import annotations

import os
import shlex
from pathlib import Path

_BIN = Path(__file__).resolve().parent
_TESTBENCH = _BIN.parent
_REPO = _TESTBENCH.parent

_cuda_default = next(
    (p for p in (Path("/usr/local/cuda"), Path("/usr/local/cuda-13.0")) if p.exists()),
    Path("/usr/local/cuda"),
)

_sglang_default = next(
    (
        p
        for p in (
            _REPO.parent / "SGLang-DGMK",
            _REPO.parent / "sglang",
        )
        if p.is_dir()
    ),
    _REPO.parent / "sglang",
)

_DEFAULTS = {
    "VENV": str(_REPO / ".venv"),
    "SGLANG_DIR": str(_sglang_default),
    "CUDA_HOME": str(_cuda_default),
    "AITER_PATH": str(_REPO.parent / "aiter"),
}

SGLANG_CAPABILITY_MARKERS = (
    "python/sglang/srt/layers/quantization/fp8_kernel.py",
)
M3_KERNEL_MARKERS = (
    "python/sglang/jit_kernel/minimax_decode_topk.py",
    "python/sglang/jit_kernel/minimax_store_kv_index.py",
    "python/sglang/jit_kernel/minimax_qknorm_rope.py",
    "python/sglang/srt/layers/attention/minimax_sparse_ops/decode/topk_sparse.py",
)


def _load_env_file() -> dict[str, str]:
    path = _TESTBENCH / "harness.env"
    if not path.exists():
        return {}
    values = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip()
    return values


_FILE_ENV = _load_env_file()


def resolve(name: str) -> str:
    raw = os.environ.get(name) or _FILE_ENV.get(name) or _DEFAULTS[name]
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = _REPO / path
    return str(path.absolute())


def setting(name: str, default: str) -> str:
    return os.environ.get(name) or _FILE_ENV.get(name) or default


def checkout_has_files(root: str | Path, relative_files: tuple[str, ...]) -> bool:
    root = Path(root)
    return all((root / rel).is_file() for rel in relative_files)


def is_usable_sglang_checkout(root: str | Path) -> bool:
    return checkout_has_files(root, SGLANG_CAPABILITY_MARKERS)


def has_m3_kernels(root: str | Path) -> bool:
    return checkout_has_files(root, M3_KERNEL_MARKERS)


def resolve_sglang_dir(pinned: str | None = None) -> str:
    del pinned
    return resolve("SGLANG_DIR")


def sglang_python_root(sglang_dir: str | Path | None = None) -> str | None:
    root = Path(sglang_dir or resolve("SGLANG_DIR"))
    return str((root / "python").resolve()) if is_usable_sglang_checkout(root) else None


VENV = Path(resolve("VENV"))
SGLANG_DIR = resolve("SGLANG_DIR")
CUDA_HOME = resolve("CUDA_HOME")
AITER_PATH = resolve("AITER_PATH")
KERNEL_HARNESS_PLATFORM = setting("KERNEL_HARNESS_PLATFORM", "cuda")
KERNEL_HARNESS_PROFILE = setting("KERNEL_HARNESS_PROFILE", "cuda-b200")
KERNEL_HARNESS_PROVIDER = setting(
    "KERNEL_HARNESS_PROVIDER", "deep-gemm-sgl-kernel"
)
KERNEL_HARNESS_TIMER = setting("KERNEL_HARNESS_TIMER", "auto")
TESTBENCH_ROOT = _TESTBENCH


if __name__ == "__main__":
    for key, value in (
        ("VENV", str(VENV)),
        ("SGLANG_DIR", SGLANG_DIR),
        ("CUDA_HOME", CUDA_HOME),
        ("AITER_PATH", AITER_PATH),
        ("KERNEL_HARNESS_PLATFORM", KERNEL_HARNESS_PLATFORM),
        ("KERNEL_HARNESS_PROFILE", KERNEL_HARNESS_PROFILE),
        ("KERNEL_HARNESS_PROVIDER", KERNEL_HARNESS_PROVIDER),
        ("KERNEL_HARNESS_TIMER", KERNEL_HARNESS_TIMER),
    ):
        print(f"{key}={shlex.quote(value)}")
