#!/usr/bin/env python3
"""Portable path resolver shared by all testbench drivers.

Every driver (`evaluate.py`, `report.py`, `gen_tasks.py`, `run.sh` via `harness.env`)
resolves the same locations through here, so a checkout on another machine needs no
source edits.

Resolution order per var:  explicit env var  →  testbench/harness.env  →  built-in
portable default. Relative paths are resolved from the repository root.

There is a single SGLang source path: ``SGLANG_DIR``. MiniMax-M3 DSA kernels live in
the same mainline tree (or the installed ``sglang`` package when the checkout is too
incomplete to prepend safely).
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
}

# Prefer a local checkout on PYTHONPATH only when it can host the modules recipes
# import. A shallow/incomplete tree must not shadow the installed package.
SGLANG_CAPABILITY_MARKERS = (
    "python/sglang/srt/layers/quantization/fp8_kernel.py",
)
# Soft check used by env/selftest to confirm mainline vendors MiniMax-M3 DSA.
M3_KERNEL_MARKERS = (
    "python/sglang/jit_kernel/minimax_decode_topk.py",
    "python/sglang/jit_kernel/minimax_store_kv_index.py",
    "python/sglang/jit_kernel/minimax_qknorm_rope.py",
    "python/sglang/srt/layers/attention/minimax_sparse_ops/decode/topk_sparse.py",
)


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


def checkout_has_files(root: str | Path, relative_files: tuple[str, ...]) -> bool:
    root = Path(root)
    return all((root / rel).is_file() for rel in relative_files)


def is_usable_sglang_checkout(root: str | Path) -> bool:
    """True when the checkout is complete enough to safely prepend to PYTHONPATH."""
    return checkout_has_files(root, SGLANG_CAPABILITY_MARKERS)


def has_m3_kernels(root: str | Path) -> bool:
    """True when the tree vendors the MiniMax-M3 DSA kernels used by DSA tasks."""
    return checkout_has_files(root, M3_KERNEL_MARKERS)


def resolve_sglang_dir(pinned: str | None = None) -> str:
    """Always resolve to the single configured SGLANG_DIR.

    ``pinned`` is accepted for backward compatibility with older callers/task.json
    fields that mentioned a symbolic checkout selector; it is ignored.
    """
    del pinned  # single source of truth
    return resolve("SGLANG_DIR")


def sglang_python_root(sglang_dir: str | Path | None = None) -> str | None:
    """Return ``<checkout>/python`` when safe to prepend, else None (use site-packages)."""
    root = Path(sglang_dir or resolve("SGLANG_DIR"))
    if is_usable_sglang_checkout(root):
        return str((root / "python").resolve())
    return None


VENV = (_REPO / ".venv").absolute()
SGLANG_DIR = resolve("SGLANG_DIR")
CUDA_HOME = resolve("CUDA_HOME")
TESTBENCH_ROOT = _TESTBENCH


if __name__ == "__main__":
    # `python bin/config.py` prints resolved values (also consumed by run.sh via eval).
    print(f"VENV={shlex.quote(str(VENV))}")
    for k in ("SGLANG_DIR", "CUDA_HOME"):
        print(f"{k}={shlex.quote(resolve(k))}")
