"""Small timing helpers used by every benchmark script.

- bench(): median wall-clock latency in microseconds
- gemm_tflops / gemm_gbs: convenience formulas
- device_info(): pretty header line
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Callable

import torch


def sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def bench(fn: Callable[[], object], warmup: int = 5, iters: int = 20) -> float:
    """Return median wall-clock latency of `fn` in microseconds."""
    for _ in range(warmup):
        fn()
    sync()
    ts: list[float] = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        sync()
        ts.append((time.perf_counter() - t0) * 1e6)
    ts.sort()
    return ts[len(ts) // 2]


def gemm_tflops(m: int, n: int, k: int, us: float) -> float:
    return (2 * m * n * k) / (us * 1e-6) / 1e12


def gemm_gbs(m: int, n: int, k: int, us: float, bytes_per_elem: int = 2) -> float:
    """Approx traffic: A + B + C in bytes (bf16=2)."""
    return (m * k + k * n + m * n) * bytes_per_elem / (us * 1e-6) / 1e9


def device_info() -> str:
    if not torch.cuda.is_available():
        return f"# no CUDA; torch={torch.__version__}"
    return (
        f"# GPU: {torch.cuda.get_device_name(0)}  "
        f"Cap: {torch.cuda.get_device_capability()}  "
        f"torch={torch.__version__}  cuda={torch.version.cuda}"
    )


def print_header() -> None:
    print(device_info(), flush=True)
    print("#" + "-" * 140, flush=True)


def print_row(op_id: int, name: str, phase: str, backend: str,
              shape: str, us: float, extra: dict | None = None) -> None:
    parts = [
        f"{op_id:>4}",
        f"{name[:34]:<34}",
        f"{phase:<8}",
        f"{backend[:32]:<32}",
        f"{shape[:44]:<44}",
        f"{us:>10.2f} us",
    ]
    if extra:
        parts.append(" ".join(f"{k}={v}" for k, v in extra.items()))
    print(" | ".join(parts), flush=True)
