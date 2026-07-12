#!/usr/bin/env python3
"""Portable path resolver shared by all testbench drivers.

Every driver (`evaluate.py`, `report.py`, `gen_tasks.py`, `run.sh` via `harness.env`)
resolves the same locations through here, so a checkout on another machine needs no
source edits.

Resolution order per var:  explicit env var  →  testbench/harness.env  →  built-in
portable default. Relative paths are resolved from the repository root.
"""
import os
import shlex
from pathlib import Path

# testbench/bin/config.py -> testbench/ -> repo root
_BIN = Path(__file__).resolve().parent
_TESTBENCH = _BIN.parent
_REPO = _TESTBENCH.parent

_cuda_default = next(
    (p for p in (Path("/usr/local/cuda"), Path("/usr/local/cuda-13.0")) if p.exists()),
    Path("/usr/local/cuda"),
)

_DEFAULTS = {
    "SGLANG_DIR": str(_REPO.parent / "sglang"),
    "CUDA_HOME": str(_cuda_default),
    "MM_M3_SGLANG_DIR": str(_REPO.parent / "sglang-m3"),
}


def _load_env_file():
    """Parse testbench/harness.env (KEY=VALUE lines, # comments) if present."""
    f = _TESTBENCH / "harness.env"
    if not f.exists():
        return {}
    out = {}
    for line in f.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        out[k.strip()] = v.strip()
    return out


_FILE_ENV = _load_env_file()


def resolve(name: str) -> str:
    raw = os.environ.get(name) or _FILE_ENV.get(name) or _DEFAULTS[name]
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = _REPO / path
    return str(path.absolute())


def resolve_sglang_dir(pinned: str | None = None) -> str:
    """Resolve a task's symbolic checkout selector without embedding machine paths."""
    if not pinned:
        return resolve("SGLANG_DIR")
    if pinned in {"SGLANG_DIR", "MM_M3_SGLANG_DIR"}:
        return resolve(pinned)
    path = Path(pinned).expanduser()
    if not path.is_absolute():
        path = _REPO / path
    return str(path.resolve())


VENV = (_REPO / ".venv").absolute()
SGLANG_DIR = resolve("SGLANG_DIR")
CUDA_HOME = resolve("CUDA_HOME")
TESTBENCH_ROOT = _TESTBENCH


if __name__ == "__main__":
    # `python bin/config.py` prints resolved values (also consumed by run.sh via eval).
    print(f"VENV={shlex.quote(str(VENV))}")
    for k in ("SGLANG_DIR", "MM_M3_SGLANG_DIR", "CUDA_HOME"):
        print(f"{k}={shlex.quote(resolve(k))}")
