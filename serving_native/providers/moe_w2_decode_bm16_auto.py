"""API-v1 provider for the exact GLM-5.2 MoE W2 decode hotspot.

This module is loaded by SGLang's existing ``hotspot_candidates`` provider
surface.  All filesystem and warmup work is confined to ``initialize``.  The
selected callback performs one DeepGEMM launch into the caller-owned output
and returns the stock ``None`` value.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Any, Callable

INFINI_KERNEL_API_VERSION = 1
KERNEL_SYMBOL = "infini_kernel_glm52_moe_w2_decode_bm16_auto"
BUILD_ID = (
    "glm52-moe-w2-decode-bm16-auto:"
    "deepgemm-edcf77b276965de8:sm100:e32:m1024:k2048:n6144:"
    "bm16:stages12:vector-valid-store:pdl1:sms148"
)
PROVIDER_INFO: dict[str, Any] = {
    "name": "glm52_moe_w2_decode_bm16_auto",
    "git_commit": "83d313104d089bcd2af26b28453ff880f1e6a80b",
    "build_id": BUILD_ID,
    "kernel_symbol": KERNEL_SYMBOL,
    "production_default": False,
}

_LAUNCH: Callable[..., Any] | None = None
_DEEP_GEMM: Any = None
_INITIALIZED = False
_ATTEMPTED_CALLS = 0
_COMPLETED_CALLS = 0
_WARMUP_CALLS = 0
_INITIALIZATION_EVIDENCE: dict[str, Any] = {}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _cache_snapshot(root: Path) -> dict[str, str]:
    return {
        str(path.relative_to(root)): _sha256(path)
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def _packed_scale(torch, groups: int, mn: int):
    # [G, 4, MN] -> [G, MN, 4] has the exact packed-UE8M0 strides required
    # by the production ABI: [MN*4, 1, MN].
    return torch.zeros(
        (groups, 4, mn),
        device="cuda",
        dtype=torch.int32,
    ).permute(0, 2, 1)


def initialize(*, gpu_id: int | None) -> None:
    """Verify the frozen runtime identity and load both cache entries."""

    global _DEEP_GEMM, _INITIALIZED, _LAUNCH, _WARMUP_CALLS
    if _INITIALIZED:
        raise RuntimeError("MoE W2 hotspot provider initialized twice")

    import deep_gemm
    import torch

    package_root = Path(
        os.environ["GLM52_MOE_W2_RUNTIME_PACKAGE"]
    ).resolve()
    cache_root = Path(os.environ["DG_JIT_CACHE_DIR"]).resolve()
    expected_dso_sha256 = os.environ["GLM52_MOE_W2_DSO_SHA256"]
    loaded_package = Path(deep_gemm.__file__).resolve().parent
    loaded_dso = loaded_package / "_C.so"
    if loaded_package != package_root:
        raise RuntimeError(
            f"DeepGEMM package drift: {loaded_package} != {package_root}"
        )
    if not loaded_dso.is_file() or _sha256(loaded_dso) != expected_dso_sha256:
        raise RuntimeError("DeepGEMM runtime DSO does not match READY identity")
    if not cache_root.is_dir():
        raise RuntimeError(f"task-local DeepGEMM cache is missing: {cache_root}")
    if gpu_id is not None and torch.cuda.current_device() != int(gpu_id):
        raise RuntimeError(
            f"provider device mismatch: {torch.cuda.current_device()} != {gpu_id}"
        )
    capability = tuple(torch.cuda.get_device_capability())
    if capability != (10, 0):
        raise RuntimeError(f"MoE W2 provider requires SM100, got {capability}")

    deep_gemm.set_num_sms(148)
    deep_gemm.set_pdl(True)
    if int(deep_gemm.get_num_sms()) != 148 or not bool(deep_gemm.get_pdl()):
        raise RuntimeError("DeepGEMM PDL/SM runtime state did not freeze")

    launch = deep_gemm.fp8_m_grouped_gemm_nt_masked
    before = _cache_snapshot(cache_root)
    groups, slab, k, n = 32, 1024, 2048, 6144
    a = torch.zeros(
        (groups, slab, k),
        device="cuda",
        dtype=torch.float8_e4m3fn,
    )
    a_scale = _packed_scale(torch, groups, slab)
    b = torch.zeros(
        (groups, n, k),
        device="cuda",
        dtype=torch.float8_e4m3fn,
    )
    b_scale = _packed_scale(torch, groups, n)
    out = torch.empty(
        (groups, slab, n),
        device="cuda",
        dtype=torch.bfloat16,
    )
    for expected_m, forward_m in ((4, 16), (5, 16), (8, 32), (9, 32)):
        counts = [0] * groups
        for index in range(forward_m * 8):
            counts[index % groups] += 1
        masked_m = torch.tensor(counts, device="cuda", dtype=torch.int32)
        returned = launch(
            (a, a_scale),
            (b, b_scale),
            out,
            masked_m,
            expected_m,
            disable_ue8m0_cast=True,
            glm52_moe_w2_decode_bm16_auto=True,
        )
        if returned is not None:
            raise RuntimeError("candidate warmup violated the None return ABI")
        _WARMUP_CALLS += 1
    torch.cuda.synchronize()
    after = _cache_snapshot(cache_root)
    if before != after:
        raise RuntimeError("candidate warmup changed the frozen JIT cache")

    _DEEP_GEMM = deep_gemm
    _LAUNCH = launch
    _INITIALIZED = True
    _INITIALIZATION_EVIDENCE.update(
        {
            "initialized": True,
            "gpu_id": torch.cuda.current_device(),
            "compute_capability": list(capability),
            "loaded_package": str(loaded_package),
            "loaded_dso": str(loaded_dso),
            "loaded_dso_sha256": expected_dso_sha256,
            "dg_jit_cache_dir": str(cache_root),
            "cache_snapshot_before": before,
            "cache_snapshot_after": after,
            "cache_unchanged_during_warmup": True,
            "num_sms": int(deep_gemm.get_num_sms()),
            "pdl": bool(deep_gemm.get_pdl()),
            "warmup_expected_m": [4, 5, 8, 9],
        }
    )
    PROVIDER_INFO.update(
        {
            "compute_capability": list(capability),
            "num_sms": 148,
            "pdl": True,
            "cache_unchanged_during_warmup": True,
        }
    )
    del a, a_scale, b, b_scale, out
    torch.cuda.empty_cache()


def moe_w2(*, lhs, rhs, out, masked_m, expected_m):
    """Launch exactly one selected W2 kernel and never invoke stock."""

    global _ATTEMPTED_CALLS, _COMPLETED_CALLS
    if not _INITIALIZED or _LAUNCH is None:
        raise RuntimeError("MoE W2 provider callback used before initialization")
    if expected_m not in (4, 5, 8, 9):
        raise RuntimeError(f"unsupported selected expected_m={expected_m}")
    _ATTEMPTED_CALLS += 1
    returned = _LAUNCH(
        lhs,
        rhs,
        out,
        masked_m,
        expected_m,
        disable_ue8m0_cast=True,
        glm52_moe_w2_decode_bm16_auto=True,
    )
    if returned is not None:
        raise RuntimeError("selected MoE W2 kernel violated the None return ABI")
    _COMPLETED_CALLS += 1
    return None


def call_counts() -> dict[str, int]:
    return {
        "attempted": _ATTEMPTED_CALLS,
        "completed": _COMPLETED_CALLS,
        "warmup": _WARMUP_CALLS,
    }


def runtime_evidence() -> dict[str, Any]:
    if not _INITIALIZED:
        raise RuntimeError("provider evidence requested before initialization")
    return {
        **_INITIALIZATION_EVIDENCE,
        "kernel_symbol": KERNEL_SYMBOL,
        "build_id": BUILD_ID,
        "production_default": False,
        "call_counts": call_counts(),
    }
