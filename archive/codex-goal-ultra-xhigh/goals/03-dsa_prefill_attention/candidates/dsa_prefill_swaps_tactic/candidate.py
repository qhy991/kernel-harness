"""Goal-scoped FlashInfer sparse-MLA tactic oracle for GLM-5.2 prefill.

The repo-local headers change only the exact FP8 M=4096, H=64, D=576,
top-k=2048 selector from the shipped Q64/Keeps tactic to a shipped Q16 or
Q32/Swaps tactic.  FlashInfer JIT compilation and cubin-loader setup happen at
module import, outside the runner's timed region.
"""

from __future__ import annotations

import functools
import os
import types
from pathlib import Path


_HERE = Path(__file__).resolve().parent
_TILE_Q = int(os.environ.get("GLM52_DSA_SWAPS_TILE_Q", "32"))
if _TILE_Q not in (16, 32):
    raise ValueError("GLM52_DSA_SWAPS_TILE_Q must be 16 or 32")
if os.environ.get("FLASHINFER_LOGLEVEL", "0") != "0":
    raise RuntimeError("the isolated tactic candidate requires FLASHINFER_LOGLEVEL=0")


def _build_custom_call():
    import flashinfer.mla._core as core
    from flashinfer.artifacts import ArtifactPath, CheckSumHash
    from flashinfer.jit import env as jit_env
    from flashinfer.jit import gen_jit_spec, setup_cubin_loader
    from flashinfer.jit.cubin_loader import get_artifact, get_meta_hash

    artifact = ArtifactPath.TRTLLM_GEN_FMHA
    artifact_include = f"{artifact}/include"
    checksums = get_artifact(
        f"{artifact}/checksums.txt",
        CheckSumHash.TRTLLM_GEN_FMHA,
    )
    meta_hash = get_meta_hash(checksums)
    if not get_artifact(f"{artifact_include}/flashInferMetaInfo.h", meta_hash):
        raise RuntimeError("FlashInfer TRTLLM-gen metadata header is unavailable")

    spec = gen_jit_spec(
        f"fmha_gen_glm52_dsa_swaps_q{_TILE_Q}_v2",
        [
            jit_env.FLASHINFER_CSRC_DIR / "trtllm_fmha_kernel_launcher.cu",
            jit_env.FLASHINFER_CSRC_DIR / "fmhaReduction.cu",
        ],
        extra_include_paths=[
            _HERE / "include",
            jit_env.FLASHINFER_CUBIN_DIR / artifact_include,
        ],
        extra_cuda_cflags=[
            f"-DGLM52_DSA_PREFILL_FORCE_SWAPS_Q={_TILE_Q}",
            f'-DTLLM_GEN_FMHA_CUBIN_PATH=\\"{artifact}\\"',
            f'-DTLLM_GEN_FMHA_METAINFO_HASH=\\"{meta_hash}\\"',
            "-lineinfo",
        ],
    )
    custom_op = spec.build_and_load()
    setup_cubin_loader(str(spec.get_library_path()))

    @functools.cache
    def custom_getter():
        return custom_op

    stock_call = core.trtllm_batch_decode_with_kv_cache_mla
    # FlashInfer 0.6.12 keeps the API-logging decorator even at log level 0,
    # but exposes the zero-overhead implementation through ``__wrapped__``.
    # Clone that inner function so the reference's global module getter is not
    # mutated when this candidate is imported before the runner constructs it.
    while (
        "get_trtllm_gen_fmha_module" not in stock_call.__globals__
        and hasattr(stock_call, "__wrapped__")
    ):
        stock_call = stock_call.__wrapped__
    if "get_trtllm_gen_fmha_module" not in stock_call.__globals__:
        raise RuntimeError("FlashInfer MLA wrapper layout changed; refusing an unverified clone")
    isolated_globals = dict(stock_call.__globals__)
    isolated_globals["get_trtllm_gen_fmha_module"] = custom_getter
    custom_call = types.FunctionType(
        stock_call.__code__,
        isolated_globals,
        stock_call.__name__,
        stock_call.__defaults__,
        stock_call.__closure__,
    )
    custom_call.__kwdefaults__ = dict(stock_call.__kwdefaults__ or {})
    return custom_call, str(spec.get_library_path())


_CUSTOM_CALL, CUSTOM_LIBRARY_PATH = _build_custom_call()
CUSTOM_TILE_Q = _TILE_Q


def _matches_exact_abi(inputs, runtime) -> bool:
    torch = runtime.torch
    return (
        runtime.workload.family == "dsa_trtllm_prefill"
        and tuple(inputs["query"].shape) == (4096, 1, 64, 576)
        and tuple(inputs["kv_cache"].shape) in (
            (512, 1, 64, 576),
            (513, 1, 64, 576),
        )
        and tuple(inputs["block_tables"].shape) == (4096, 1, 2048)
        and tuple(inputs["seq_lens"].shape) == (4096,)
        and inputs["max_seq_len"] == 32768
        and inputs["sparse_topk"] == 2048
        and inputs["query"].dtype == torch.float8_e4m3fn
        and inputs["kv_cache"].dtype == torch.float8_e4m3fn
    )


def run(inputs, runtime):
    if not _matches_exact_abi(inputs, runtime):
        return runtime.reference(inputs)

    p = runtime.workload.params
    out = _CUSTOM_CALL(
        query=inputs["query"],
        kv_cache=inputs["kv_cache"],
        workspace_buffer=inputs["workspace"],
        qk_nope_head_dim=p["qk_nope_head_dim"],
        kv_lora_rank=p["kv_lora_rank"],
        qk_rope_head_dim=p["qk_rope_head_dim"],
        block_tables=inputs["block_tables"],
        seq_lens=inputs["seq_lens"],
        max_seq_len=inputs["max_seq_len"],
        sparse_mla_top_k=inputs["sparse_topk"],
        bmm1_scale=inputs["bmm1_scale"],
        backend="trtllm-gen",
    )
    return out.squeeze(1) if out.ndim == 4 and out.shape[1] == 1 else out
