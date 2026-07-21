"""Backend bundle registry and explicit selection.

Two bundles are registered: the CUDA/B200 + DeepGEMM bundle, and the
ROCm/MI300X + torch-triton bundle (testbench/harness/backends/rocm_mi300x.py).
Any other combination fails loudly.
"""
from __future__ import annotations

import os
from functools import lru_cache

from .base import BackendBundle
from .cuda_b200 import PROFILE as CUDA_B200_PROFILE
from .cuda_b200 import PROVIDER as DEEP_GEMM_PROVIDER
from .cuda_b200 import TIMER as CUDA_TIMER
from .rocm_mi300x import PROFILE as ROCM_MI300X_PROFILE
from .rocm_mi300x import PROVIDER as ROCM_PROVIDER
from .rocm_mi300x import TIMER as ROCM_TIMER


def _cuda_bundle(timer_key: str) -> BackendBundle:
    del timer_key  # CUDA timer auto-selects CUPTI vs Event
    return BackendBundle(
        profile=CUDA_B200_PROFILE,
        provider=DEEP_GEMM_PROVIDER,
        timer=CUDA_TIMER,
    )


def _rocm_bundle(timer_key: str) -> BackendBundle:
    del timer_key  # ROCm has one timer: HIP-event cold-L2
    return BackendBundle(
        profile=ROCM_MI300X_PROFILE,
        provider=ROCM_PROVIDER,
        timer=ROCM_TIMER,
    )


_BUNDLES = {
    ("cuda", "cuda-b200", "deep-gemm-sgl-kernel", "auto"): _cuda_bundle("auto"),
    ("cuda", "cuda-b200", "deep-gemm-sgl-kernel", "cupti"): _cuda_bundle("cupti"),
    ("rocm", "rocm-mi300x", "torch-triton-rocm", "auto"): _rocm_bundle("auto"),
    ("rocm", "rocm-mi300x", "torch-triton-rocm", "event"): _rocm_bundle("event"),
}


def registered() -> tuple[tuple[str, str, str, str], ...]:
    return tuple(_BUNDLES)


@lru_cache(maxsize=1)
def _harness_env() -> dict[str, str]:
    """testbench/harness.env, parsed once — the same file bin/config.py reads, so
    backend selection matches whether or not the shell sourced activate_env.sh."""
    from pathlib import Path

    path = Path(__file__).resolve().parents[2] / "harness.env"
    values: dict[str, str] = {}
    if path.exists():
        for line in path.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                values[k.strip()] = v.strip()
    return values


def _select(name: str, default: str) -> str:
    return (os.environ.get(name) or _harness_env().get(name) or default).lower()


@lru_cache(maxsize=1)
def get_backend() -> BackendBundle:
    platform = _select("KERNEL_HARNESS_PLATFORM", "cuda")
    profile = _select("KERNEL_HARNESS_PROFILE", "cuda-b200")
    provider = _select("KERNEL_HARNESS_PROVIDER", "deep-gemm-sgl-kernel")
    timer = _select("KERNEL_HARNESS_TIMER", "auto")
    key = (platform, profile, provider, timer)
    try:
        bundle = _BUNDLES[key]
    except KeyError as exc:
        choices = ", ".join("/".join(item) for item in registered())
        raise RuntimeError(
            "unsupported kernel-harness backend combination "
            f"{'/'.join(key)}; registered: {choices}."
        ) from exc
    bundle.validate()
    return bundle


def reset_backend_cache() -> None:
    get_backend.cache_clear()
