#!/usr/bin/env python3
"""Fail-fast environment check for the GLM-5.2 harness."""
from __future__ import annotations

import importlib
import inspect
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
TESTBENCH = REPO / "testbench"


def _env_file() -> dict[str, str]:
    out: dict[str, str] = {}
    for p in (TESTBENCH / "harness.env", TESTBENCH / "harness.env.example"):
        if not p.is_file():
            continue
        for line in p.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            out.setdefault(k.strip(), v.strip().strip('"').strip("'"))
    return out


def _path(value: str | None, default: Path) -> Path:
    if not value:
        return default
    p = Path(value).expanduser()
    return p if p.is_absolute() else (REPO / p).resolve()


def _commit(path: Path) -> str:
    result = subprocess.run(["git", "-C", str(path), "rev-parse", "--short", "HEAD"],
                            capture_output=True, text=True, check=False)
    return result.stdout.strip() or "unknown"


def _module_path(mod) -> str:
    try:
        return str(Path(inspect.getfile(mod)).resolve())
    except Exception:
        return "unknown"


def _has_m3_kernels(sglang_dir: Path) -> bool:
    return any((sglang_dir / rel).exists() for rel in (
        "python/sglang/srt/layers/attention/triton_ops/decode_attention.py",
        "python/sglang/srt/layers/attention/triton_ops/prefill_attention.py",
        "python/sglang/srt/layers/attention/flashinfer_backend.py",
    ))


def main() -> int:
    cfg = _env_file()
    venv = _path(os.environ.get("VENV") or cfg.get("VENV"), REPO / ".venv")
    cuda_home = _path(os.environ.get("CUDA_HOME") or cfg.get("CUDA_HOME"), Path("/usr/local/cuda"))
    sglang_dir = _path(os.environ.get("SGLANG_DIR") or cfg.get("SGLANG_DIR"), REPO.parent / "sglang")

    errors: list[str] = []
    expected_python = (venv / "bin" / "python").resolve()
    if Path(sys.executable).resolve() != expected_python:
        errors.append(f"use {expected_python}, got {sys.executable}")

    modules = ("torch", "triton", "sglang", "sgl_kernel", "deep_gemm", "flashinfer", "cupti")
    loaded = {}
    for name in modules:
        try:
            loaded[name] = importlib.import_module(name)
        except Exception as exc:
            errors.append(f"cannot import {name}: {type(exc).__name__}: {exc}")

    checkout_ok = (sglang_dir / "python" / "sglang").is_dir()
    installed_sglang = loaded.get("sglang")
    torch = loaded.get("torch")
    if torch is not None and not torch.cuda.is_available():
        errors.append("torch.cuda.is_available() is false; run evaluation on a GPU node")

    print(f"python:      {sys.executable}")
    print(f"venv:        {venv}")
    print(f"cuda home:   {cuda_home}")
    if checkout_ok:
        print(f"sglang:      checkout {sglang_dir} ({_commit(sglang_dir)})")
    elif installed_sglang is not None:
        print(f"sglang:      installed package at {_module_path(installed_sglang)}")
    else:
        print(f"sglang:      missing (configured checkout {sglang_dir})")
    if torch is not None and torch.cuda.is_available():
        index = torch.cuda.current_device()
        cap = ''.join(map(str, torch.cuda.get_device_capability(index)))
        print(f"gpu:         {torch.cuda.get_device_name(index)} sm_{cap}")
        print(f"visible gpus:{torch.cuda.device_count()}  CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '')}")
        print(f"torch/cuda:  {torch.__version__} / {torch.version.cuda}")

    if checkout_ok:
        print(f"M3 kernels:  {'present' if _has_m3_kernels(sglang_dir) else 'not found'} in SGLANG_DIR")
    elif installed_sglang is not None:
        print("M3 kernels:  using installed sglang package")

    if errors:
        print("\nEnvironment check failed:", file=sys.stderr)
        for error in errors:
            print(f"- {error}", file=sys.stderr)
        return 2
    print("\nEnvironment check passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
