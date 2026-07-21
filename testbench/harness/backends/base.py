"""Platform / provider / timer contracts.

Keep the three concepts separate so ``rocm`` is never synonymous with AITER
(and ``cuda`` is never synonymous with DeepGEMM).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping, Protocol


@dataclass(frozen=True)
class DeviceProfile:
    id: str
    platform: str
    accelerator: str
    deployment: str
    fp8_dtype_name: str
    peaks: Mapping[str, float]
    peaks_source: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "platform": self.platform,
            "gpu": self.accelerator,
            "deployment": self.deployment,
            "fp8_dtype": self.fp8_dtype_name,
            "hbm_bytes_per_s": self.peaks["hbm_bytes_per_s"],
            "peak_flops_fp8": self.peaks["fp8"],
            "peak_flops_bf16": self.peaks["bf16"],
            "source": self.peaks_source,
        }


class OperatorProvider(Protocol):
    id: str
    platform: str
    capabilities: frozenset[str]
    required_modules: tuple[str, ...]
    baseline_caveat: str
    accepted_candidate_forms: tuple[str, ...]

    def supports(self, op: str, phase: str) -> bool: ...

    def baseline_name(self, family: str, phase: str) -> str: ...

    def per_token_cast(self, tensor, *, use_ue8m0: bool): ...

    def per_block_cast(self, tensor, *, use_ue8m0: bool): ...

    def align_scale(self, scale): ...

    def paged_mqa_metadata(self, seqlens, block_size: int): ...

    def reference(self, op: str, phase: str, family: str, inputs: dict): ...

    def version_info(self) -> Mapping[str, str | None]: ...


class Timer(Protocol):
    id: str
    platform: str
    description: str

    def available(self) -> bool: ...

    def measure(
        self,
        fn: Callable,
        *,
        setup: Callable | None,
        warmup: int,
        rep: int,
        device,
    ) -> list[float]: ...


@dataclass(frozen=True)
class BackendBundle:
    profile: DeviceProfile
    provider: OperatorProvider
    timer: Timer

    def identity(self) -> dict[str, Any]:
        return {
            "platform": self.profile.platform,
            "profile": self.profile.id,
            "provider": self.provider.id,
            "timer": self.timer.id,
            "accelerator": self.profile.accelerator,
            "capabilities": sorted(self.provider.capabilities),
            "provider_versions": dict(self.provider.version_info()),
        }

    def validate(self) -> None:
        if self.provider.platform != self.profile.platform:
            raise RuntimeError(
                f"provider {self.provider.id!r} targets {self.provider.platform!r}, "
                f"not profile platform {self.profile.platform!r}"
            )
        if self.timer.platform != self.profile.platform:
            raise RuntimeError(
                f"timer {self.timer.id!r} targets {self.timer.platform!r}, "
                f"not profile platform {self.profile.platform!r}"
            )
        if not self.timer.available():
            raise RuntimeError(f"timer {self.timer.id!r} is unavailable")
