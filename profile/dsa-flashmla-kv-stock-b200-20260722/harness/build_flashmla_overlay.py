#!/usr/bin/env python3
"""Build and extract a pinned upstream FlashMLA wheel with a raw build log."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import time
import zipfile
from pathlib import Path


PIN = "05e26647fe840b8baedae486c2d86d5ce4efeb7c"
CUTLASS_PIN = "147f5673d0c1c3dcf66f78d677fd647e4a020219"


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--expected-source-commit", default=PIN)
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--nvcc-threads", type=int, default=2)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    source = Path(args.source).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    manifest_path = Path(args.manifest).expanduser().resolve()
    expected_artifact_root = (source / "build-artifacts").resolve()
    if output_root.parent != expected_artifact_root or output_root.name != args.label:
        raise RuntimeError(
            "output root must be the explicit per-label directory "
            f"{expected_artifact_root / args.label}"
        )
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    expected_artifact_root.mkdir(parents=True, exist_ok=True)
    if output_root.exists():
        raise RuntimeError(f"refusing to overwrite immutable build label: {output_root}")
    if manifest_path.exists():
        raise RuntimeError(f"refusing to overwrite build manifest: {manifest_path}")

    source_commit = command("git", "rev-parse", "HEAD", cwd=source)
    cutlass_commit = command("git", "rev-parse", "HEAD", cwd=source / "csrc/cutlass")
    if source_commit != args.expected_source_commit or cutlass_commit != CUTLASS_PIN:
        raise RuntimeError(
            "pin mismatch: "
            f"FlashMLA={source_commit} (expected {args.expected_source_commit}), "
            f"CUTLASS={cutlass_commit} (expected {CUTLASS_PIN})"
        )
    command("git", "merge-base", "--is-ancestor", PIN, source_commit, cwd=source)
    tracked_status = command(
        "git", "status", "--short", "--untracked-files=no", cwd=source
    ).splitlines()
    cutlass_status = command(
        "git", "status", "--short", "--untracked-files=no", cwd=source / "csrc/cutlass"
    ).splitlines()
    if tracked_status or cutlass_status:
        raise RuntimeError(
            "build source must have no uncommitted tracked changes: "
            f"FlashMLA={tracked_status}, CUTLASS={cutlass_status}"
        )
    source_untracked = command(
        "git", "ls-files", "--others", "--exclude-standard", cwd=source
    ).splitlines()
    unexpected_untracked = [
        path for path in source_untracked if not path.startswith("build-artifacts/")
    ]
    cutlass_untracked = command(
        "git", "ls-files", "--others", "--exclude-standard", cwd=source / "csrc/cutlass"
    ).splitlines()
    if unexpected_untracked or cutlass_untracked:
        raise RuntimeError(
            "untracked source inputs are forbidden outside build-artifacts/: "
            f"FlashMLA={unexpected_untracked}, CUTLASS={cutlass_untracked}"
        )

    staging_root = Path(
        tempfile.mkdtemp(prefix=f".{args.label}.staging-", dir=expected_artifact_root)
    )
    wheels = staging_root / "wheels"
    overlay = staging_root / "overlay"
    build_base = staging_root / "build"
    bdist_dir = staging_root / "bdist"
    wheels.mkdir(parents=True, exist_ok=True)
    overlay.mkdir(parents=True)

    env = os.environ.copy()
    cuda_home = Path(os.environ.get("CUDA_HOME", "/usr/local/cuda")).resolve()
    cuda_cccl_include = cuda_home / "targets" / "x86_64-linux" / "include" / "cccl"
    if not (cuda_cccl_include / "cuda" / "std" / "utility").is_file():
        raise RuntimeError(f"CUDA CCCL headers not found under {cuda_cccl_include}")
    prior_cplus_include = env.get("CPLUS_INCLUDE_PATH", "")
    cplus_include = str(cuda_cccl_include)
    if prior_cplus_include:
        cplus_include = f"{cplus_include}{os.pathsep}{prior_cplus_include}"
    env.update(
        {
            "FLASH_MLA_DISABLE_SM90": "1",
            "MAX_JOBS": str(args.jobs),
            "NVCC_THREADS": str(args.nvcc_threads),
            "CPLUS_INCLUDE_PATH": cplus_include,
        }
    )
    build_command = [
        sys.executable,
        "setup.py",
        "build",
        "--build-base",
        str(build_base),
        "bdist_wheel",
        "--skip-build",
        "--bdist-dir",
        str(bdist_dir),
        "--dist-dir",
        str(wheels),
    ]
    started = time.time()
    completed = subprocess.run(
        build_command,
        cwd=source,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    elapsed = time.time() - started
    (staging_root / "build.log").write_text(completed.stdout)
    if completed.returncode != 0:
        raise RuntimeError(
            f"FlashMLA build failed with {completed.returncode}; "
            f"preserved under {staging_root}"
        )

    wheel_paths = sorted(wheels.glob("*.whl"))
    if len(wheel_paths) != 1:
        raise RuntimeError(f"expected one wheel, found {wheel_paths}")
    wheel = wheel_paths[0]
    with zipfile.ZipFile(wheel) as archive:
        archive.extractall(overlay)
    extensions = sorted((overlay / "flash_mla").glob("cuda*.so"))
    if len(extensions) != 1:
        raise RuntimeError(f"expected one extension, found {extensions}")
    extension = extensions[0]

    source_patch = staging_root / "source.patch"
    source_patch.write_text(
        subprocess.run(
            ["git", "diff", "--binary", f"{PIN}..{source_commit}"],
            cwd=source,
            check=True,
            text=True,
            stdout=subprocess.PIPE,
        ).stdout
    )

    # Publish only after the wheel, extracted extension, and committed source
    # delta are all complete.  A failed build cannot destroy prior evidence.
    staging_root.replace(output_root)
    wheel = output_root / "wheels" / wheel.name
    overlay = output_root / "overlay"
    extension = overlay / "flash_mla" / extension.name
    source_patch = output_root / "source.patch"

    manifest = {
        "label": args.label,
        "source": str(source),
        "source_commit": source_commit,
        "source_status": tracked_status,
        "source_untracked_status": command(
            "git", "status", "--short", "--untracked-files=normal", cwd=source
        ).splitlines(),
        "allowed_prebuild_untracked_count": len(source_untracked),
        "source_patch_base": PIN,
        "source_patch": str(source_patch),
        "source_patch_sha256": sha256(source_patch),
        "cutlass_commit": cutlass_commit,
        "cutlass_status": cutlass_status,
        "python": sys.executable,
        "build_command": build_command,
        "environment": {
            "FLASH_MLA_DISABLE_SM90": env["FLASH_MLA_DISABLE_SM90"],
            "MAX_JOBS": env["MAX_JOBS"],
            "NVCC_THREADS": env["NVCC_THREADS"],
            "CPLUS_INCLUDE_PATH": env["CPLUS_INCLUDE_PATH"],
        },
        "elapsed_seconds": elapsed,
        "build_log": str(output_root / "build.log"),
        "wheel": str(wheel),
        "wheel_sha256": sha256(wheel),
        "overlay": str(overlay),
        "extension": str(extension),
        "extension_size_bytes": extension.stat().st_size,
        "extension_sha256": sha256(extension),
        "overlay_python_sha256": {
            str(path.relative_to(overlay)): sha256(path)
            for path in sorted((overlay / "flash_mla").glob("*.py"))
        },
    }
    rendered = json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    manifest_path.write_text(rendered)
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
