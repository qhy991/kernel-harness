#!/usr/bin/env python3
"""Central path resolver for the testbench drivers.

Every driver (`evaluate.py`, `report.py`, `gen_tasks.py`, `run.sh` via `harness.env`)
resolves the same four external locations through here, so a checkout on a different
machine only needs env vars (or a `testbench/harness.env`) — no source edits.

Resolution order per var:  explicit env var  →  testbench/harness.env  →  built-in
default (the original hardcoded path, kept so nothing breaks in place).
"""
import os
from pathlib import Path

# testbench/bin/config.py -> testbench/ -> repo root
_BIN = Path(__file__).resolve().parent
_TESTBENCH = _BIN.parent
_REPO = _TESTBENCH.parent

# Built-in defaults = the paths the harness originally hardcoded.
_DEFAULTS = {
    "VENV": str(_REPO / ".venv"),
    "SOLEXEC": "/home/qinhaiyan/sol-execbench",
    "SGLANG_DIR": "/home/qinhaiyan/sglang",
    "CUDA_HOME": "/usr/local/cuda-13.0",
    # Separate M3-DSA sglang checkout used only by the (currently blocked) DSA tasks.
    "MM_M3_SGLANG_DIR": "/home/qinhaiyan/sglang-m3",
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
    return os.environ.get(name) or _FILE_ENV.get(name) or _DEFAULTS[name]


VENV = Path(resolve("VENV"))
SOLEXEC = Path(resolve("SOLEXEC"))
SGLANG_DIR = resolve("SGLANG_DIR")
CUDA_HOME = resolve("CUDA_HOME")
TESTBENCH_ROOT = _TESTBENCH


if __name__ == "__main__":
    # `python bin/config.py` prints resolved values (also consumed by run.sh via eval).
    for k in ("VENV", "SOLEXEC", "SGLANG_DIR", "CUDA_HOME"):
        print(f"{k}={resolve(k)}")
