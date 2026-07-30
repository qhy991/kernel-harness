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
_PREBUILT_DIR = (
    Path(__file__).resolve().parents[1]
    / "prebuilt"
    / "flashmla_sparse_decode"
)


def _truthy(value: str) -> bool:
    return value.strip().lower() in ("1", "true", "yes", "on")


def _prebuilt_manifest() -> dict:
    import json

    path = _PREBUILT_DIR / "MANIFEST.json"
    if not path.is_file():
        return {}
    data = json.loads(path.read_text())
    return data if isinstance(data, dict) else {}


def _resolve_prebuilt_so() -> Path | None:
    explicit = os.environ.get("GLM52_FLASHMLA_PREBUILT_SO", "").strip()
    if explicit:
        path = Path(explicit).expanduser().resolve()
        if path.is_dir():
            matches = sorted(
                path.glob(
                    f"infini_kernel_glm52_flashmla_sparse_decode_{_VARIANT}_*.so"
                )
            )
            if not matches:
                raise RuntimeError(
                    f"GLM52_FLASHMLA_PREBUILT_SO={explicit!r} has no .so for {_VARIANT}"
                )
            return matches[0]
        if not path.is_file():
            raise RuntimeError(
                f"GLM52_FLASHMLA_PREBUILT_SO={explicit!r} is not a .so file"
            )
        return path
    if not _truthy(os.environ.get("GLM52_FLASHMLA_USE_PREBUILT", "")):
        return None
    for entry in _prebuilt_manifest().get("binaries", []):
        if entry.get("variant") == _VARIANT and entry.get("so_file"):
            candidate = _PREBUILT_DIR / str(entry["so_file"])
            if candidate.is_file():
                return candidate
    matches = sorted(
        _PREBUILT_DIR.glob(
            f"infini_kernel_glm52_flashmla_sparse_decode_{_VARIANT}_*.so"
        )
    )
    if not matches:
        raise RuntimeError(
            "GLM52_FLASHMLA_USE_PREBUILT=1 but no prebuilt .so for "
            f"{_VARIANT} under {_PREBUILT_DIR}"
        )
    return matches[0]


def _load_prebuilt_extension(so_path: Path):
    import importlib.util
    import sys

    module_name = so_path.stem
    digest = hashlib.sha256(so_path.read_bytes()).hexdigest()
    for entry in _prebuilt_manifest().get("binaries", []):
        if entry.get("so_file") == so_path.name:
            expected = entry.get("sha256")
            if expected and expected != digest:
                raise RuntimeError(
                    f"prebuilt .so sha256 mismatch: got {digest}, expected {expected}"
                )
            break
    spec = importlib.util.spec_from_file_location(module_name, so_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to create import spec for {so_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


_PREBUILT_SO = _resolve_prebuilt_so()
if _PREBUILT_SO is None:
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
    _LOADED_MODULE_NAME = _MODULE_NAME
    _SOURCE_SHA256 = {
        str(path.relative_to(_SOURCE)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in _HASH_INPUTS
    }
else:
    _EXTENSION = _load_prebuilt_extension(_PREBUILT_SO)
    _LOADED_MODULE_NAME = _PREBUILT_SO.stem
    _MODULE_NAME = _LOADED_MODULE_NAME
    _BUILD_ID = _LOADED_MODULE_NAME.rsplit("_", 1)[-1]
    _SOURCE_SHA256 = {}

if _PREBUILT_SO is None and _SOURCE.is_dir():
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
else:
    _git_commit = "prebuilt"
    _git_dirty = False
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
    "module_name": _LOADED_MODULE_NAME,
    "source_module_name": _MODULE_NAME,
    "main_symbol_prefix": "infini_kernel_glm52_flashmla_sparse_decode",
    "combine_symbol_prefix": (
        "infini_kernel_glm52_flashmla_sparse_decode_combine"
    ),
    "stock_combine_translation_unit_linked": False,
    "prebuilt_so": str(_PREBUILT_SO) if _PREBUILT_SO is not None else None,
    "source_sha256": _SOURCE_SHA256,
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
