"""Backend bundle registry and explicit selection.

Only the existing CUDA/B200 + DeepGEMM bundle is registered today. ROCm/AITER
bundles are intentionally absent — requesting them fails loudly.
"""
from __future__ import annotations

import os
from functools import lru_cache

from .base import BackendBundle
from .cuda_b200 import PROFILE as CUDA_B200_PROFILE
from .cuda_b200 import PROVIDER as DEEP_GEMM_PROVIDER
from .cuda_b200 import TIMER as CUDA_TIMER


def _cuda_bundle(timer_key: str) -> BackendBundle:
    del timer_key  # CUDA timer auto-selects CUPTI vs Event
    return BackendBundle(
        profile=CUDA_B200_PROFILE,
        provider=DEEP_GEMM_PROVIDER,
        timer=CUDA_TIMER,
    )


_BUNDLES = {
    ("cuda", "cuda-b200", "deep-gemm-sgl-kernel", "auto"): _cuda_bundle("auto"),
    ("cuda", "cuda-b200", "deep-gemm-sgl-kernel", "cupti"): _cuda_bundle("cupti"),
}


def registered() -> tuple[tuple[str, str, str, str], ...]:
    return tuple(_BUNDLES)


@lru_cache(maxsize=1)
def get_backend() -> BackendBundle:
    platform = os.environ.get("KERNEL_HARNESS_PLATFORM", "cuda").lower()
    profile = os.environ.get("KERNEL_HARNESS_PROFILE", "cuda-b200").lower()
    provider = os.environ.get(
        "KERNEL_HARNESS_PROVIDER", "deep-gemm-sgl-kernel"
    ).lower()
    timer = os.environ.get("KERNEL_HARNESS_TIMER", "auto").lower()
    key = (platform, profile, provider, timer)
    try:
        bundle = _BUNDLES[key]
    except KeyError as exc:
        choices = ", ".join("/".join(item) for item in registered())
        raise RuntimeError(
            "unsupported kernel-harness backend combination "
            f"{'/'.join(key)}; registered: {choices}. "
            "ROCm/AMD providers have not been implemented yet."
        ) from exc
    bundle.validate()
    return bundle


def reset_backend_cache() -> None:
    get_backend.cache_clear()
