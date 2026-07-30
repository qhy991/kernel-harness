"""API-v1 provider for exact GLM-5.2 FlashMLA sparse-decode combine variants.

Separate from ``flashmla_sparse_decode_provider.py`` on purpose. That provider
owns the main-kernel variant lineage and is being edited concurrently by the
round-3 main-kernel goal; this one owns the BF16 split-combine lineage, builds
``api_combine.cpp`` instead of ``api.cpp``, and never compiles or links the
stock ``smxx/decode/combine/combine.cu`` translation unit. Distinct source sets
give distinct build ids, so neither campaign can silently inherit the other's
binary.

The main kernel is pinned to the round-2 survivor ``p1_consumer_scale``
(FlashMLA ``b5af443``) exactly as the round-1 combine plan requires, so the only
difference between this provider's variants is the combine kernel.

Compilation and fixed-buffer allocation happen before any measured callback. The
callback performs one extension call, launching the prefixed V32 main and the
prefixed combine variant on the current PyTorch CUDA stream.
"""

from __future__ import annotations

import hashlib
import os
import subprocess
from pathlib import Path

import torch
from torch.utils.cpp_extension import load


INFINI_KERNEL_API_VERSION = 1
_SOURCE = Path(
    os.environ.get(
        "GLM52_FLASHMLA_SOURCE",
        (
            "/home/qinhaiyan/glm52-hotspot-goal-runs/worktrees/"
            "flashmla-sparse-decode/flashmla"
        ),
    )
).resolve()
_VARIANT = os.environ.get("GLM52_FLASHMLA_COMBINE_VARIANT", "").strip()
_VARIANT_SOURCES = {
    # Upstream b5af443 combine algorithm, renamed only. This is the
    # "P1 + stock combine" denominator arm, measured through the identical
    # provider mechanism as the candidates.
    "combine_identity": "v32_combine_identity.cu",
    # Eight-stage register-batched o_accum gather.
    "combine_c1_stage8": "v32_combine_c1_stage8.cu",
    # Bucket-specialized stage depth: 8 at M16, 4 plus a two-CTA-per-SM launch
    # bound at M32, where 256 CTAs make register pressure cost a whole wave.
    "combine_c2_bucket_stages": "v32_combine_c2_bucket_stages.cu",
}
if _VARIANT not in _VARIANT_SOURCES:
    raise RuntimeError(
        f"unsupported GLM52_FLASHMLA_COMBINE_VARIANT={_VARIANT!r}; "
        f"expected one of {sorted(_VARIANT_SOURCES)}"
    )

# The combine campaign keeps the main kernel frozen at the round-2 survivor.
_MAIN_VARIANT = "p1_consumer_scale"
_MAIN_SOURCE = _SOURCE / "csrc/glm52_hotspot/v32_p1_consumer_scale.cu"
_COMBINE_SOURCE = _SOURCE / "csrc/glm52_hotspot" / _VARIANT_SOURCES[_VARIANT]
_CPP_SOURCE = _SOURCE / "csrc/glm52_hotspot/api_combine.cpp"
_BUILD_INPUTS = (_CPP_SOURCE, _MAIN_SOURCE, _COMBINE_SOURCE)
_HASH_INPUTS = _BUILD_INPUTS + (
    _SOURCE / "csrc/glm52_hotspot/combine.cuh",
    _SOURCE / "csrc/glm52_hotspot/combine.h",
    _SOURCE / "csrc/sm100/decode/head64/config.h",
    _SOURCE / "csrc/sm100/decode/head64/kernel.cuh",
    _SOURCE / "csrc/sm100/decode/head64/kernel.h",
    _SOURCE / "csrc/sm100/helpers.h",
    _SOURCE / "csrc/kerutils/include/kerutils/device/sm100/intrinsics.cuh",
    _SOURCE / "csrc/params.h",
)
for _path in _HASH_INPUTS:
    if not _path.is_file():
        raise RuntimeError(f"missing FlashMLA hotspot build input: {_path}")

_digest = hashlib.sha256()
for _path in _HASH_INPUTS:
    _digest.update(str(_path.relative_to(_SOURCE)).encode())
    _digest.update(_path.read_bytes())
_BUILD_ID = _digest.hexdigest()
_MODULE_NAME = (
    f"infini_kernel_glm52_flashmla_sparse_decode_{_VARIANT}_" + _BUILD_ID[:16]
)
_INCLUDE_DIRS = [
    _SOURCE / "csrc",
    _SOURCE / "csrc/kerutils/include",
    _SOURCE / "csrc/sm90",
    _SOURCE / "csrc/cutlass/include",
    _SOURCE / "csrc/cutlass/tools/util/include",
    Path("/usr/local/cuda/targets/x86_64-linux/include/cccl"),
]
_EXTENSION = load(
    name=_MODULE_NAME,
    sources=[str(path) for path in _BUILD_INPUTS],
    extra_cflags=[
        "-O3",
        "-std=c++20",
        "-DNDEBUG",
        "-Wno-deprecated-declarations",
    ],
    extra_cuda_cflags=[
        "-O3",
        "-std=c++20",
        "-DNDEBUG",
        "-D_USE_MATH_DEFINES",
        "-Wno-deprecated-declarations",
        "-U__CUDA_NO_HALF_OPERATORS__",
        "-U__CUDA_NO_HALF_CONVERSIONS__",
        "-U__CUDA_NO_HALF2_OPERATORS__",
        "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
        "--expt-relaxed-constexpr",
        "--expt-extended-lambda",
        "--use_fast_math",
        (
            "--ptxas-options=-v,--register-usage-level=10,"
            "--warn-on-spills,--warn-on-local-memory-usage,"
            "--warn-on-double-precision-use"
        ),
        "-lineinfo",
        "--source-in-ptx",
        "-gencode",
        "arch=compute_100f,code=sm_100f",
        "--threads",
        os.environ.get("NVCC_THREADS", "2"),
    ],
    extra_include_paths=[str(path) for path in _INCLUDE_DIRS],
    with_cuda=True,
    verbose=True,
)

_git_commit = subprocess.check_output(
    ["git", "-C", str(_SOURCE), "rev-parse", "HEAD"],
    text=True,
).strip()
_git_dirty = bool(
    subprocess.check_output(
        ["git", "-C", str(_SOURCE), "status", "--porcelain"],
        text=True,
    ).strip()
)
PROVIDER_INFO = {
    "name": (
        "glm52_flashmla_combine_control_identity"
        if _VARIANT == "combine_identity"
        else f"glm52_flashmla_{_VARIANT}"
    ),
    "role": "control" if _VARIANT == "combine_identity" else "experimental",
    "variant": _VARIANT,
    "main_variant": _MAIN_VARIANT,
    "git_commit": _git_commit + ("-dirty" if _git_dirty else ""),
    "build_id": _BUILD_ID,
    "module_name": _MODULE_NAME,
    "main_symbol_prefix": "infini_kernel_glm52_flashmla_sparse_decode",
    "combine_symbol_prefix": (
        "infini_kernel_glm52_flashmla_sparse_decode_combine"
    ),
    "stock_combine_translation_unit_linked": False,
    "source_sha256": {
        str(path.relative_to(_SOURCE)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in _HASH_INPUTS
    },
}

_WORKSPACES: dict[int, tuple[torch.Tensor, ...]] = {}


def initialize(*, gpu_id: int | None) -> None:
    device = torch.device("cuda" if gpu_id is None else f"cuda:{gpu_id}")
    for m in (16, 32):
        out = torch.empty((m, 1, 64, 512), dtype=torch.bfloat16, device=device)
        lse_base = torch.empty((m, 1, 64), dtype=torch.float32, device=device)
        lse = lse_base.transpose(1, 2)
        lse_accum = torch.empty(
            (m + 148, 1, 64), dtype=torch.float32, device=device
        )
        o_accum = torch.empty(
            (m + 148, 1, 64, 512), dtype=torch.float32, device=device
        )
        _WORKSPACES[m] = (out, lse_base, lse, lse_accum, o_accum)
    _EXTENSION.reset_launch_count()


def flashmla_sparse_decode(
    *,
    q,
    k_cache,
    cache_seqlens,
    head_dim_v,
    tile_scheduler_metadata,
    num_splits,
    softmax_scale,
    indices,
    block_table,
    is_fp8_kvcache,
):
    del cache_seqlens, block_table
    if head_dim_v != 512 or softmax_scale != 0.0625 or not is_fp8_kvcache:
        raise RuntimeError("selected provider received a non-promotional ABI")
    workspace = _WORKSPACES.get(q.shape[0])
    if workspace is None:
        raise RuntimeError("selected provider received an unsupported M")
    out, lse_base, lse, lse_accum, o_accum = workspace
    _EXTENSION.launch(
        q,
        k_cache,
        indices,
        tile_scheduler_metadata,
        num_splits,
        out,
        lse_base,
        lse_accum,
        o_accum,
    )
    return out, lse


def candidate_evidence() -> dict[str, object]:
    return {
        **PROVIDER_INFO,
        "extension_file": str(Path(_EXTENSION.__file__).resolve()),
        "launch_count": int(_EXTENSION.launch_count()),
        "workspace_pointers": {
            str(m): {
                "out": int(workspace[0].data_ptr()),
                "lse_base": int(workspace[1].data_ptr()),
                "lse_view": int(workspace[2].data_ptr()),
                "lse_stride": list(workspace[2].stride()),
                "lse_accum": int(workspace[3].data_ptr()),
                "o_accum": int(workspace[4].data_ptr()),
            }
            for m, workspace in sorted(_WORKSPACES.items())
        },
    }
