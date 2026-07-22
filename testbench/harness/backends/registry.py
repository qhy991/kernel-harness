"""Backend bundle registry and explicit selection."""
from __future__ import annotations

import os
from functools import lru_cache

from .base import BackendBundle
from .cuda_b200 import PROFILE as CUDA_B200_PROFILE
from .cuda_b200 import PROVIDER as DEEP_GEMM_PROVIDER
from .cuda_b200 import TIMER as CUDA_TIMER
from .rocm_amd import PROFILE as ROCM_AMD_PROFILE
from .rocm_amd import PROVIDER as AITER_TORCH_PROVIDER
from .rocm_amd import TIMER as ROCM_EVENT_TIMER


def _cuda_bundle(timer_key: str) -> BackendBundle:
    del timer_key  # CUDA timer auto-selects CUPTI vs Event
    return BackendBundle(
        profile=CUDA_B200_PROFILE,
        provider=DEEP_GEMM_PROVIDER,
        timer=CUDA_TIMER,
    )


def _rocm_bundle(timer_key: str) -> BackendBundle:
    del timer_key
    return BackendBundle(
        profile=ROCM_AMD_PROFILE,
        provider=AITER_TORCH_PROVIDER,
        timer=ROCM_EVENT_TIMER,
    )


_BUNDLES = {
    ("cuda", "cuda-b200", "deep-gemm-sgl-kernel", "auto"): _cuda_bundle("auto"),
    ("cuda", "cuda-b200", "deep-gemm-sgl-kernel", "cupti"): _cuda_bundle("cupti"),
    ("rocm", "amd-mi300x", "aiter-torch-reference", "event"): _rocm_bundle("event"),
    ("rocm", "amd-mi300x", "aiter-torch-reference", "auto"): _rocm_bundle("auto"),
}


def registered() -> tuple[tuple[str, str, str, str], ...]:
    return tuple(_BUNDLES)


@lru_cache(maxsize=1)
def get_backend() -> BackendBundle:
    platform = os.environ.get("KERNEL_HARNESS_PLATFORM", "rocm").lower()
    profile = os.environ.get("KERNEL_HARNESS_PROFILE", "amd-mi300x").lower()
    provider = os.environ.get(
        "KERNEL_HARNESS_PROVIDER", "aiter-torch-reference"
    ).lower()
    timer = os.environ.get("KERNEL_HARNESS_TIMER", "event").lower()
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
