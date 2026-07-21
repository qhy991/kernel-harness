"""Existing B200 / DeepGEMM backend expressed through backend contracts."""
from __future__ import annotations

import importlib
import importlib.metadata
from functools import lru_cache

from .base import DeviceProfile
from ._load import load_harness_module


PROFILE = DeviceProfile(
    id="cuda-b200",
    platform="cuda",
    accelerator="NVIDIA B200",
    deployment="B200 DP1/TP1/EP32",
    fp8_dtype_name="float8_e4m3fn",
    peaks={
        "hbm_bytes_per_s": 8.0e12,
        "fp8": 4.5e15,
        "bf16": 2.25e15,
    },
    peaks_source=(
        "rewardbench (PR2) + testbench/harness/profile.py; "
        "opbench/mfu.py's 7.7e12 not used"
    ),
)


BASELINE_CAVEAT = (
    "This is deep_gemm's f32-blockwise-scale path, NOT SGLang's production dispatch "
    "(deepgemm_w8a8_block_fp8_linear_with_fallback, which passes int32-packed ue8m0 "
    "scales to w8a8_block_fp8_matmul_deepgemm). Both land in deep_gemm; only the scale "
    "representation differs. Measured on B200 under this exact timing protocol, o_proj "
    "decode: production 33.1us vs this reference 53.3us at M=16 — production is ~1.6x "
    "FASTER. So reproducing SGLang's production call scores a ~1.6x 'win' here having "
    "improved nothing: a sub-1.6x speedup does not mean beating production. Kept "
    "deliberately — opbench (PR1) and rewardbench (PR2) agree on this definition and it "
    "is the frozen, verified standard."
)


ACCEPTED_CANDIDATE_FORMS = (
    "Python / PyTorch — a .py defining run(inputs)",
    "Triton — @triton.jit / @triton.autotune live in that same .py; nothing special needed",
    "CUDA .cu — pass a directory holding candidate.py + the .cu, and let candidate.py "
    "torch.utils.cpp_extension.load() it at import time (compilation happens outside the "
    "timed window). A bare .cu cannot be passed: nothing in it says which __global__ to "
    "launch, with what grid, or how the inputs dict maps to its arguments. run(inputs) is "
    "that missing statement, and it is the whole ABI.",
)


@lru_cache(maxsize=1)
def _modules():
    deep_gemm = importlib.import_module("deep_gemm")
    sgl_kernel = importlib.import_module("sgl_kernel")
    flash_mla = importlib.import_module("sgl_kernel.flash_mla")
    layout = importlib.import_module("deep_gemm.utils.layout")
    math_utils = importlib.import_module("deep_gemm.utils.math")
    return deep_gemm, sgl_kernel, flash_mla, layout, math_utils


def _pkg_version(distribution: str) -> str | None:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return None


class DeepGemmProvider:
    id = "deep-gemm-sgl-kernel"
    platform = "cuda"
    capabilities = frozenset({"gemm", "bmm", "moe", "mla", "score"})
    required_modules = ("deep_gemm", "sgl_kernel")
    baseline_caveat = BASELINE_CAVEAT
    accepted_candidate_forms = ACCEPTED_CANDIDATE_FORMS

    def supports(self, op: str, phase: str) -> bool:
        del op, phase
        return True

    def baseline_name(self, family: str, phase: str) -> str:
        if family == "gemm":
            return "deep_gemm.fp8_gemm_nt"
        if family == "bmm":
            return "sgl_kernel.bmm_fp8"
        if family == "moe":
            return "deep_gemm.fp8_m_grouped_gemm_nt_masked"
        if family == "mla":
            return "sgl_kernel.flash_mla.flash_mla_sparse_fwd"
        if phase == "prefill":
            return "deep_gemm.fp8_mqa_logits"
        return "deep_gemm.fp8_paged_mqa_logits"

    def per_token_cast(self, tensor, *, use_ue8m0: bool):
        return _modules()[4].per_token_cast_to_fp8(tensor, use_ue8m0=use_ue8m0)

    def per_block_cast(self, tensor, *, use_ue8m0: bool):
        return _modules()[4].per_block_cast_to_fp8(tensor, use_ue8m0=use_ue8m0)

    def align_scale(self, scale):
        return _modules()[3].get_mn_major_tma_aligned_tensor(scale)

    def paged_mqa_metadata(self, seqlens, block_size: int):
        deep_gemm = _modules()[0]
        return deep_gemm.get_paged_mqa_logits_metadata(
            seqlens, block_size, deep_gemm.get_num_sms()
        )

    def reference(self, op: str, phase: str, family: str, inputs: dict):
        del op
        deep_gemm, sgl_kernel, flash_mla, _, _ = _modules()
        if family == "gemm":
            out = inputs["out"]
            deep_gemm.fp8_gemm_nt(
                (inputs["x_fp8"], inputs["x_scale"]),
                (inputs["w_fp8"], inputs["w_scale"]),
                out,
            )
            return out
        if family == "bmm":
            import torch

            return sgl_kernel.bmm_fp8(
                inputs["A_fp8"],
                inputs["B_fp8"],
                inputs["A_scale"],
                inputs["B_scale"],
                torch.bfloat16,
            )
        if family == "moe":
            out = inputs["out"]
            deep_gemm.fp8_m_grouped_gemm_nt_masked(
                (inputs["x_fp8"], inputs["x_scale"]),
                (inputs["w_fp8"], inputs["w_scale"]),
                out,
                inputs["masked_m"],
                inputs["expected_m"],
            )
            return out
        if family == "mla":
            return flash_mla.flash_mla_sparse_fwd(
                inputs["q"],
                inputs["kv"],
                inputs["indices"],
                inputs["sm_scale"],
                inputs["d_v"],
            )
        if phase == "prefill":
            return deep_gemm.fp8_mqa_logits(
                inputs["q_fp8"],
                (inputs["k_fp8"], inputs["k_scale"]),
                inputs["weights"],
                inputs["ks"],
                inputs["ke"],
                clean_logits=False,
            )
        return deep_gemm.fp8_paged_mqa_logits(
            inputs["q_fp8"],
            inputs["kv_cache_fp8"],
            inputs["weights"],
            inputs["seqlens"],
            inputs["block_tables"],
            inputs["schedule_metadata"],
            inputs["max_seq_len"],
            clean_logits=False,
        )

    def version_info(self):
        return {
            "deep_gemm": _pkg_version("deep-gemm") or _pkg_version("deep_gemm"),
            "sgl_kernel": _pkg_version("sglang-kernel") or _pkg_version("sgl_kernel"),
        }


class CudaKernelTimer:
    """CUPTI device spans when available; otherwise the historical CUDA Event path."""

    platform = "cuda"
    # Contract text stays CUPTI-primary so generated problem.json does not depend
    # on whether the sync host has the cupti package installed.
    contract_id = "cupti-cold-l2-device-kernel-median"
    contract_description = (
        "CUPTI cold-L2 device-kernel median: inputs cloned per iteration and "
        "L2 flushed before each, both outside the measured window"
    )

    @property
    def id(self) -> str:
        timing = load_harness_module("timing")
        return (
            self.contract_id
            if timing._HAVE_CUPTI
            else "event-cold-l2-median-NO-CUPTI"
        )

    @property
    def description(self) -> str:
        return self.contract_description

    def available(self) -> bool:
        return True

    def measure(self, fn, *, setup, warmup, rep, device):
        timing = load_harness_module("timing")
        bench = (
            timing.bench_gpu_time_with_cupti
            if timing._HAVE_CUPTI
            else timing.bench_time_with_cuda_events
        )
        return bench(fn, warmup=warmup, rep=rep, setup=setup, device=device)


PROVIDER = DeepGemmProvider()
TIMER = CudaKernelTimer()
