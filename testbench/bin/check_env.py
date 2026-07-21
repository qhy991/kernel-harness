#!/usr/bin/env python3
"""Fail-fast environment check for agents and humans."""
from __future__ import annotations

import importlib
import subprocess
import sys
from pathlib import Path

from config import (
    AITER_PATH,
    CUDA_HOME,
    KERNEL_HARNESS_PLATFORM,
    KERNEL_HARNESS_PROFILE,
    KERNEL_HARNESS_PROVIDER,
    KERNEL_HARNESS_TIMER,
    SGLANG_DIR,
    VENV,
    has_m3_kernels,
    is_usable_sglang_checkout,
    sglang_python_root,
)

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from testbench.harness.backends import get_backend  # noqa: E402


def _commit(path: Path) -> str:
    result = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "--short", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout.strip() or "unknown"


def main() -> int:
    errors = []
    expected_python = VENV / "bin" / "python"
    if Path(sys.executable).resolve() != expected_python.resolve():
        errors.append(f"use {expected_python}, got {sys.executable}")

    try:
        bundle = get_backend()
    except Exception as exc:
        print(f"backend selection failed: {exc}", file=sys.stderr)
        return 2

    modules = ("torch", "triton", "sglang") + tuple(bundle.provider.required_modules)
    if bundle.profile.platform == "cuda":
        modules = modules + ("flashinfer", "cupti")
    loaded = {}
    for name in modules:
        try:
            loaded[name] = importlib.import_module(name)
        except Exception as exc:
            errors.append(f"cannot import {name}: {type(exc).__name__}: {exc}")

    sglang_path = Path(SGLANG_DIR)
    if not (sglang_path / "python" / "sglang").is_dir():
        errors.append(f"invalid SGLANG_DIR: {sglang_path}")

    torch = loaded.get("torch")
    if torch is not None and not torch.cuda.is_available():
        errors.append("torch.cuda.is_available() is false; run evaluation on a GPU node")

    print(f"python:      {sys.executable}")
    print(f"venv:        {VENV}")
    print(
        f"backend:     {bundle.profile.platform}/{bundle.profile.id}/"
        f"{bundle.provider.id}/{bundle.timer.id}"
    )
    print(
        f"env select:  PLATFORM={KERNEL_HARNESS_PLATFORM} "
        f"PROFILE={KERNEL_HARNESS_PROFILE} "
        f"PROVIDER={KERNEL_HARNESS_PROVIDER} "
        f"TIMER={KERNEL_HARNESS_TIMER}"
    )
    print(f"sglang:      {sglang_path} ({_commit(sglang_path)})")
    print(f"cuda home:   {CUDA_HOME}")
    print(f"aiter path:  {AITER_PATH}")
    if torch is not None and torch.cuda.is_available():
        index = torch.cuda.current_device()
        name = torch.cuda.get_device_name(index)
        if getattr(torch.version, "hip", None):
            props = torch.cuda.get_device_properties(index)
            arch = getattr(props, "gcnArchName", "unknown")
            print(f"gpu:         {name} {arch}")
            print(f"torch/hip:   {torch.__version__} / {torch.version.hip}")
        else:
            print(
                f"gpu:         {name} sm_"
                f"{''.join(map(str, torch.cuda.get_device_capability(index)))}"
            )
            print(f"torch/cuda:  {torch.__version__} / {torch.version.cuda}")

    usable = sglang_python_root(sglang_path)
    if usable:
        print(f"sglang src:  checkout (PYTHONPATH prepends {usable})")
    else:
        print("sglang src:  installed package (checkout incomplete — missing fp8_kernel)")
    if has_m3_kernels(sglang_path):
        print("M3 kernels:  present in SGLANG_DIR")
    else:
        print("M3 kernels:  not found in SGLANG_DIR; DSA tasks need them from the installed sglang")

    if errors:
        print("\nEnvironment check failed:", file=sys.stderr)
        for error in errors:
            print(f"- {error}", file=sys.stderr)
        return 2
    print("\nEnvironment check passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
