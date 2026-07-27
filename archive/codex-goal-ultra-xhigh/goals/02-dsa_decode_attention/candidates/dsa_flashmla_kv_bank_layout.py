"""Isolated FlashMLA raw-NoPE shared-layout experiment for GLM-5.2.

The custom extension registers under ``sgl_kernel_goal02`` and therefore
coexists with the installed stock ``sgl_kernel`` extension.  Loading and
manifest verification happen at import time, outside measured ``run`` calls.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import torch


_EXPERIMENT = (
    Path(__file__).resolve().parents[2]
    / "profile"
    / "dsa_flashmla_kv_bank_conflict_20260722"
)
_LIBRARY = _EXPERIMENT / "artifacts" / "flashmla_goal02_ops.so"
_MANIFEST_PATH = _EXPERIMENT / "build_manifest.json"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if not _LIBRARY.is_file() or not _MANIFEST_PATH.is_file():
    raise FileNotFoundError(
        "build the isolated Goal 02 FlashMLA artifact before loading this candidate: "
        f"{_LIBRARY}"
    )
_MANIFEST = json.loads(_MANIFEST_PATH.read_text())
if Path(_MANIFEST["artifact"]).resolve() != _LIBRARY:
    raise RuntimeError("candidate artifact path disagrees with build manifest")
if _sha256(_LIBRARY) != _MANIFEST["artifact_sha256"]:
    raise RuntimeError("candidate artifact hash disagrees with build manifest")
torch.ops.load_library(str(_LIBRARY))


def candidate_evidence() -> dict[str, Any]:
    return {
        "candidate": "dsa_flashmla_kv_bank_layout",
        "operator_namespace": "sgl_kernel_goal02",
        "artifact": str(_LIBRARY),
        "artifact_sha256": _sha256(_LIBRARY),
        "manifest": str(_MANIFEST_PATH),
        "manifest_sha256": _sha256(_MANIFEST_PATH),
        "flashmla_base": _MANIFEST["flashmla_base"],
        "flashmla_source_commit": _MANIFEST["flashmla_source_commit"],
        "cutlass_commit": _MANIFEST["cutlass_commit"],
        "sglang_source_commit": _MANIFEST["sglang_source_commit"],
        "source_patch_sha256": _MANIFEST["source_patch_sha256"],
    }


def run(inputs: dict[str, Any], runtime: Any) -> torch.Tensor:
    """Mirror ``_forward_flashmla_kv`` for the fixed no-padding ABI."""

    del runtime
    q_all = inputs["query"]
    layer = inputs["layer_stub"]
    backend = inputs["backend_stub"]
    q_input = q_all.view(-1, 1, layer.tp_q_head_num, layer.head_dim)
    if backend.flashmla_kv_num_q_heads != q_input.shape[2]:
        raise RuntimeError("Goal 02 candidate only covers the 64-head no-padding path")
    kv_cache = inputs["kv_cache"].view(
        -1, backend.real_page_size, 1, backend.kv_cache_dim
    )
    indices = inputs["page_table_1"].unsqueeze(1)
    block_table = torch.empty(
        (q_input.shape[0], 0), dtype=torch.int32, device=q_input.device
    )
    out, _lse = (
        torch.ops.sgl_kernel_goal02.fwd_kvcache_mla.default(
            q_input,
            kv_cache,
            inputs["head_dim_v"],
            inputs["cache_seqlens"],
            block_table,
            inputs["softmax_scale"],
            False,
            inputs["tile_scheduler_metadata"],
            inputs["num_splits"],
            True,
            indices,
            None,
            None,
            None,
            None,
            None,
        )
    )
    return out
