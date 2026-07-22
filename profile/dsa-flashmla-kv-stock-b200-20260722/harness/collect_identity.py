#!/usr/bin/env python3
"""Collect dependency identities without importing or initializing CUDA ops."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import re
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
SGLANG_ROOT = Path(
    "/home/qinhaiyan/glm52-goal-runs/22-dsa_flashmla_kv_production/sglang"
).resolve()
FLASHMLA_OVERLAY_ROOT = (SGLANG_ROOT / "third_party/FlashMLA-goal22").resolve()


def command(*args: str, cwd: Path | None = None) -> str:
    return subprocess.run(
        args,
        cwd=cwd,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    ).stdout.strip()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def match(text: str, pattern: str) -> str:
    found = re.search(pattern, text)
    if found is None:
        raise RuntimeError(f"identity pattern not found: {pattern}")
    return found.group(1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", help="also persist the rendered JSON here")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    expected_prefix = (REPO_ROOT / ".venv").resolve()
    if Path(sys.prefix).resolve() != expected_prefix:
        raise RuntimeError(
            f"run with the repository venv: {expected_prefix}/bin/python"
        )
    cmake_text = (SGLANG_ROOT / "sgl-kernel/cmake/flashmla.cmake").read_text()
    site_packages = REPO_ROOT / ".venv/lib/python3.12/site-packages"
    extension = site_packages / "sgl_kernel/flashmla_ops.abi3.so"
    torch_version_text = (site_packages / "torch/version.py").read_text()
    notes = command("readelf", "-n", str(extension))
    build_id = match(notes, r"Build ID: ([0-9a-f]+)")
    nvcc = command("/usr/local/cuda/bin/nvcc", "--version")

    torch_version = match(torch_version_text, r"__version__ = '([^']+)'")
    identity = {
        "kernel_harness": {
            "root": str(REPO_ROOT),
            "git_head": command("git", "rev-parse", "HEAD", cwd=REPO_ROOT),
            "git_branch": command("git", "branch", "--show-current", cwd=REPO_ROOT),
            "git_status": command("git", "status", "--short", cwd=REPO_ROOT).splitlines(),
        },
        "sglang": {
            "root": str(SGLANG_ROOT),
            "git_head": command("git", "rev-parse", "HEAD", cwd=SGLANG_ROOT),
            "git_branch": command("git", "branch", "--show-current", cwd=SGLANG_ROOT),
            "git_status": command("git", "status", "--short", cwd=SGLANG_ROOT).splitlines(),
            "python_distribution": importlib.metadata.version("sglang"),
            "source_python_root": str(SGLANG_ROOT / "python"),
        },
        "sgl_kernel": {
            "source": "SGLang monorepo sgl-kernel",
            "python_distribution": importlib.metadata.version("sglang-kernel"),
            "python_module": str(site_packages / "sgl_kernel/flash_mla.py"),
        },
        "flashmla": {
            "git_revision": match(
                cmake_text, r"sgl-project/FlashMLA/archive/([0-9a-f]{40})\.tar\.gz"
            ),
            "archive_sha256": match(
                cmake_text,
                r"FlashMLA/archive/[0-9a-f]{40}\.tar\.gz\s+URL_HASH SHA256=([0-9a-f]{64})",
            ),
        },
        "flashmla_overlay": {
            "root": str(FLASHMLA_OVERLAY_ROOT),
            "git_head": command(
                "git", "rev-parse", "HEAD", cwd=FLASHMLA_OVERLAY_ROOT
            ),
            "git_branch": command(
                "git", "branch", "--show-current", cwd=FLASHMLA_OVERLAY_ROOT
            ),
            "git_status": command(
                "git", "status", "--short", cwd=FLASHMLA_OVERLAY_ROOT
            ).splitlines(),
        },
        "flashmla_cutlass": {
            "git_revision": match(
                cmake_text, r"NVIDIA/cutlass/archive/([0-9a-f]{40})\.tar\.gz"
            ),
            "archive_sha256": match(
                cmake_text,
                r"cutlass/archive/[0-9a-f]{40}\.tar\.gz\s+URL_HASH SHA256=([0-9a-f]{64})",
            ),
        },
        "torch": {
            "version": torch_version,
            "wheel_cuda": match(torch_version_text, r"cuda: Optional\[str\] = '([^']+)'"),
            "git_revision": match(torch_version_text, r"git_version = '([0-9a-f]+)'"),
        },
        "cuda_toolkit": {
            "nvcc": nvcc,
        },
        "built_extension": {
            "path": str(extension),
            "size_bytes": extension.stat().st_size,
            "sha256": sha256(extension),
            "elf_build_id": build_id,
        },
        "sm100_dispatch": {
            "instantiation": "csrc/sm100/decode/head64/instantiations/v32.cu",
            "main_kernel": "flash_fwd_splitkv_mla_fp8_sparse_kernel*",
            "combine_kernel": "flash_fwd_mla_combine_kernel*",
        },
    }
    rendered = json.dumps(identity, indent=2, sort_keys=True) + "\n"
    if args.output:
        output = Path(args.output).expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered)
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
