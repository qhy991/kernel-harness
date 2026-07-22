"""Run identity, environment capture, and append-only result persistence.

Layout:
    runs/<model>/<task>/<run_id>/result.json      stable schema, one run
                                 /stdout.log       full terminal output
                                 /environment.json environment block alone
                                 /candidate.py     byte-exact copy of what ran
    runs/<model>/<task>/latest.json               atomically replaced pointer
    runs/index.jsonl                              append-only, one line per run

Every write goes through a temp file + os.replace, so a concurrent reader never
observes a partial file and two concurrent runs cannot interleave a write. Run
directories are never overwritten: run_id carries a random suffix so two runs
starting in the same second still get distinct directories.
"""
from __future__ import annotations

import datetime as _dt
import hashlib
import json
import os
import platform
import secrets
import subprocess
import sys
from pathlib import Path

SCHEMA_VERSION = "1.1"

_REPO_ROOT = Path(__file__).resolve().parents[2]
RUNS_ROOT = _REPO_ROOT / "runs"


def new_run_id() -> str:
    """UTC second + 6 random hex. The suffix is what makes concurrent runs safe."""
    stamp = _dt.datetime.now(_dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{stamp}-{secrets.token_hex(3)}"


def utc_now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def sha256_file(path: Path) -> str | None:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        return None


def _cmd(args: list[str]) -> str | None:
    try:
        out = subprocess.run(args, capture_output=True, text=True, timeout=10,
                             cwd=str(_REPO_ROOT))
        return out.stdout.strip() or None if out.returncode == 0 else None
    except Exception:
        return None


def _pkg_version(name: str) -> str | None:
    try:
        return getattr(__import__(name), "__version__", None)
    except Exception:
        return None


def capture_environment() -> dict:
    """Everything needed to decide whether two results are comparable."""
    env: dict = {
        "hostname": platform.node(),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "git_sha": _cmd(["git", "rev-parse", "HEAD"]),
        "git_branch": _cmd(["git", "rev-parse", "--abbrev-ref", "HEAD"]),
        "git_dirty": bool(_cmd(["git", "status", "--porcelain"])),
        "torch": _pkg_version("torch"),
        "deep_gemm": _pkg_version("deep_gemm"),
        "sgl_kernel": _pkg_version("sgl_kernel"),
        "aiter": _pkg_version("aiter"),
        "kernel_harness_platform": os.environ.get("KERNEL_HARNESS_PLATFORM", "rocm"),
        "kernel_harness_profile": os.environ.get("KERNEL_HARNESS_PROFILE", "amd-mi300x"),
        "kernel_harness_provider": os.environ.get(
            "KERNEL_HARNESS_PROVIDER", "aiter-torch-reference"
        ),
        "kernel_harness_timer": os.environ.get("KERNEL_HARNESS_TIMER", "event"),
    }
    try:
        import torch
        env["cuda"] = torch.version.cuda
        env["hip"] = getattr(torch.version, "hip", None)
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            env["gpu"] = props.name
            if env["hip"]:
                env["gpu_arch"] = getattr(props, "gcnArchName", None)
            else:
                env["gpu_capability"] = f"{props.major}.{props.minor}"
            env["gpu_memory_gb"] = round(props.total_memory / 1e9, 1)
            env["gpu_count"] = torch.cuda.device_count()
            env["driver"] = (
                _cmd(["rocm-smi", "--showdriverversion"]) if env["hip"] else
                _cmd(["nvidia-smi", "--query-gpu=driver_version",
                      "--format=csv,noheader"])
            )
    except Exception as exc:
        env["environment_error"] = str(exc)[:200]
    return env


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.parent / f".{path.name}.{os.getpid()}.{secrets.token_hex(3)}.tmp"
    tmp.write_text(text)
    os.replace(tmp, path)


def _append_line(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # O_APPEND writes under PIPE_BUF are atomic on Linux, so concurrent runs
    # appending one short line each cannot interleave.
    with open(path, "a") as fh:
        fh.write(text.rstrip("\n") + "\n")


def run_dir(model: str, task: str, run_id: str) -> Path:
    return RUNS_ROOT / model / task / run_id


def persist(result: dict, *, model: str, task: str, run_id: str,
            stdout_text: str, candidate_path: Path | None) -> Path:
    """Write the run directory, refresh latest.json, append to the index."""
    d = run_dir(model, task, run_id)
    d.mkdir(parents=True, exist_ok=True)

    _atomic_write(d / "result.json", json.dumps(result, indent=2) + "\n")
    _atomic_write(d / "environment.json",
                  json.dumps(result.get("environment", {}), indent=2) + "\n")
    _atomic_write(d / "stdout.log", stdout_text)
    if candidate_path is not None and candidate_path.is_file():
        _atomic_write(d / "candidate.py", candidate_path.read_text())

    # latest.json is a pointer, not a copy: the history directory is the record.
    _atomic_write(RUNS_ROOT / model / task / "latest.json", json.dumps({
        "run_id": run_id,
        "result": str((d / "result.json").relative_to(_REPO_ROOT)),
        "finished_utc": result.get("run", {}).get("finished_utc"),
        "verdict": result.get("verdict", {}),
        "aggregate": result.get("aggregate", {}),
    }, indent=2) + "\n")

    _append_line(RUNS_ROOT / "index.jsonl", json.dumps({
        "run_id": run_id,
        "model": model,
        "task": task,
        "finished_utc": result.get("run", {}).get("finished_utc"),
        "status": result.get("verdict", {}).get("status"),
        "exit_code": result.get("verdict", {}).get("exit_code"),
        "candidate_sha256": result.get("candidate", {}).get("sha256"),
        "git_sha": result.get("environment", {}).get("git_sha"),
        "result": str((d / "result.json").relative_to(_REPO_ROOT)),
    }, separators=(",", ":")))
    return d
