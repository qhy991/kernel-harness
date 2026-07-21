"""GPU kernel timing — self-contained extraction of the kernel-harness method.

Primary: CUPTI activity-trace device-kernel time (excludes CPU launch overhead), with
the L2 cache flushed and args cloned between iterations, median over `rep` reps.
Fallback: torch.cuda.Event timing if the `cupti` package is unavailable.

The implementation is maintained in this repository; its only runtime dependencies are
torch and the optional standalone cupti package.
"""
from __future__ import annotations

import bisect
import statistics
from dataclasses import dataclass
from functools import partial
from typing import Any, Callable, Optional

import torch

try:
    from cupti import cupti  # standalone NVIDIA package (CUDA 13+)
    _HAVE_CUPTI = True
except Exception:
    _HAVE_CUPTI = False


def clone_args(args: list) -> list:
    """Fresh copies of tensor args (prevents cross-iteration contamination);
    non-tensors pass through unchanged."""
    return [a.clone() if isinstance(a, torch.Tensor) else a for a in args]


def _l2_size(device) -> int:
    return torch.cuda.get_device_properties(device).L2_cache_size


def _empty_cache(device) -> torch.Tensor:
    return torch.empty(int(_l2_size(device) * 2), dtype=torch.int8, device=device)


def _clear_cache(buf: torch.Tensor) -> None:
    buf.zero_()


@dataclass
class _Kernel:
    name: str
    start: int
    end: int
    correlation_id: int
    kind: Any


def _demangle(name):
    return name


def bench_gpu_time_with_cupti(fn: Callable, warmup=10, rep=100, setup=None,
                              cold_l2_cache=True, device="cuda") -> list[float]:
    """Per-iteration device-side GPU kernel time (ms) via CUPTI activity tracing.

    fn(args) runs the kernel; setup() returns the args for each iteration (its cost is
    NOT timed). L2 is flushed before each iteration for cold-cache measurements.
    """
    def buffer_requested():
        return 8 * 1024 * 1024, 0

    def buffer_completed(launches, kernels, activities):
        for a in activities:
            if a.kind in (cupti.ActivityKind.CONCURRENT_KERNEL, cupti.ActivityKind.MEMCPY,
                          cupti.ActivityKind.MEMSET):
                nm = _demangle(a.name) if a.kind == cupti.ActivityKind.CONCURRENT_KERNEL else str(a.kind)
                kernels.append(_Kernel(nm, a.start, a.end, a.correlation_id, a.kind))
            elif a.kind in (cupti.ActivityKind.RUNTIME, cupti.ActivityKind.DRIVER):
                launches.append((a.start, a.end, a.correlation_id, a.cbid, a.kind))

    if setup is None:
        _fn = fn
        def fn(_):  # noqa: E731
            return _fn()
        def setup():  # noqa: E731
            return None

    buf = _empty_cache(device) if cold_l2_cache else None

    torch.cuda.synchronize()
    for _ in range(warmup):
        args = setup()
        if cold_l2_cache:
            _clear_cache(buf)
        fn(args)
    torch.cuda.synchronize()

    launches: list = []
    kernels: list = []
    iters: list = []
    for k in (cupti.ActivityKind.RUNTIME, cupti.ActivityKind.CONCURRENT_KERNEL,
              cupti.ActivityKind.DRIVER, cupti.ActivityKind.MEMCPY, cupti.ActivityKind.MEMSET):
        cupti.activity_enable(k)
    cupti.activity_register_callbacks(buffer_requested, partial(buffer_completed, launches, kernels))
    torch.cuda.synchronize()
    for _ in range(rep):
        args = setup()
        if cold_l2_cache:
            _clear_cache(buf)
        s = cupti.get_timestamp()
        fn(args)
        e = cupti.get_timestamp()
        torch.cuda.synchronize()
        iters.append((s, e))
    torch.cuda.synchronize()
    cupti.activity_flush_all(0)
    for k in (cupti.ActivityKind.RUNTIME, cupti.ActivityKind.CONCURRENT_KERNEL,
              cupti.ActivityKind.DRIVER, cupti.ActivityKind.MEMCPY, cupti.ActivityKind.MEMSET):
        cupti.activity_disable(k)
    cupti.finalize()

    sorted_launches = sorted(launches, key=lambda la: la[0])
    launch_starts = [la[0] for la in sorted_launches]
    corr_to_kernels: dict[int, list] = {}
    for k in kernels:
        corr_to_kernels.setdefault(k.correlation_id, []).append(k)

    times = []
    for idx, (s, e) in enumerate(iters):
        lo = bisect.bisect_left(launch_starts, s)
        hi = bisect.bisect_right(launch_starts, e)
        corr_ids = {sorted_launches[i][2] for i in range(lo, hi)}
        iter_kernels = [k for cid in corr_ids for k in corr_to_kernels.get(cid, [])]
        if not iter_kernels:
            raise ValueError(f"no kernel activity recorded for iteration {idx}")
        span_ms = (max(k.end for k in iter_kernels) - min(k.start for k in iter_kernels)) / 1e6
        times.append(span_ms)
    return times


def bench_time_with_cuda_events(fn: Callable, warmup=10, rep=100, setup=None,
                                device="cuda") -> list[float]:
    """Fallback: torch.cuda.Event timing (includes a little launch overhead), L2 flushed,
    explicit sync before each start event."""
    if setup is None:
        _fn = fn
        def fn(_):  # noqa: E731
            return _fn()
        def setup():  # noqa: E731
            return None
    buf = _empty_cache(device)
    torch.cuda.synchronize()
    for _ in range(warmup):
        _clear_cache(buf)
        fn(setup())
    torch.cuda.synchronize()
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(rep)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(rep)]
    times = []
    for i in range(rep):
        args = setup()
        _clear_cache(buf)
        torch.cuda.synchronize()
        starts[i].record()
        fn(args)
        ends[i].record()
    torch.cuda.synchronize()
    return [starts[i].elapsed_time(ends[i]) for i in range(rep)]


def time_runnable(fn: Callable, setup=None, warmup=10, rep=100, device="cuda") -> float:
    """Median latency through the selected backend's timing protocol."""
    from testbench.harness.backends import get_backend

    samples = get_backend().timer.measure(
        fn, setup=setup, warmup=warmup, rep=rep, device=device
    )
    return statistics.median(samples)
