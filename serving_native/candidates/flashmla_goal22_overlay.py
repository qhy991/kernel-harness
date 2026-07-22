"""Pinned upstream FlashMLA overlay used for Goal 22 source experiments.

Set ``GOAL22_FLASHMLA_OVERLAY`` to an extracted wheel directory before import.
The upstream ``flash_mla.cuda`` pybind module uses a namespace distinct from
the installed production ``sgl_kernel::*`` extension, so A/B calls can coexist.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path


_OVERLAY = Path(os.environ["GOAL22_FLASHMLA_OVERLAY"]).expanduser().resolve()
_MANIFEST_PATH = Path(os.environ["GOAL22_FLASHMLA_MANIFEST"]).expanduser().resolve()
if not (_OVERLAY / "flash_mla").is_dir():
    raise RuntimeError(f"invalid GOAL22_FLASHMLA_OVERLAY: {_OVERLAY}")
_MANIFEST = json.loads(_MANIFEST_PATH.read_text())
if Path(_MANIFEST["overlay"]).resolve() != _OVERLAY:
    raise RuntimeError(
        f"manifest overlay {_MANIFEST['overlay']} does not match {_OVERLAY}"
    )
sys.path.insert(0, str(_OVERLAY))

import flash_mla  # noqa: E402
import flash_mla.cuda as flash_mla_cuda  # noqa: E402


_MODULE = Path(flash_mla.__file__).resolve()
_CUDA_MODULE = Path(flash_mla_cuda.__file__).resolve()
if not _MODULE.is_relative_to(_OVERLAY):
    raise RuntimeError(f"flash_mla resolved outside the requested overlay: {_MODULE}")
if not _CUDA_MODULE.is_relative_to(_OVERLAY):
    raise RuntimeError(
        f"flash_mla.cuda resolved outside the requested overlay: {_CUDA_MODULE}"
    )

from flash_mla import flash_mla_with_kvcache, get_mla_metadata  # noqa: E402


_SCHEDULERS = {}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _extension() -> Path:
    matches = sorted((_OVERLAY / "flash_mla").glob("cuda*.so"))
    if len(matches) != 1:
        raise RuntimeError(f"expected one overlay extension, found {matches}")
    return matches[0].resolve()


def candidate_evidence() -> dict:
    extension = _extension()
    expected_extension = Path(_MANIFEST["extension"]).resolve()
    if extension != _CUDA_MODULE or extension != expected_extension:
        raise RuntimeError(
            "loaded extension/overlay/manifest mismatch: "
            f"loaded={_CUDA_MODULE}, overlay={extension}, manifest={expected_extension}"
        )
    extension_sha256 = _sha256(extension)
    if extension_sha256 != _MANIFEST["extension_sha256"]:
        raise RuntimeError(
            "loaded extension hash does not match manifest: "
            f"{extension_sha256} != {_MANIFEST['extension_sha256']}"
        )
    for relative, expected_sha256 in _MANIFEST.get(
        "overlay_python_sha256", {}
    ).items():
        python_source = _OVERLAY / relative
        actual_sha256 = _sha256(python_source)
        if actual_sha256 != expected_sha256:
            raise RuntimeError(
                f"overlay Python hash mismatch for {python_source}: "
                f"{actual_sha256} != {expected_sha256}"
            )
    source_patch_sha256 = _MANIFEST.get("source_patch_sha256")
    if source_patch_sha256 is not None:
        source_patch = Path(_MANIFEST["source_patch"]).resolve()
        actual_patch_sha256 = _sha256(source_patch)
        if actual_patch_sha256 != source_patch_sha256:
            raise RuntimeError(
                f"source patch hash mismatch for {source_patch}: "
                f"{actual_patch_sha256} != {source_patch_sha256}"
            )
    schedulers = []
    for key, scheduler in _SCHEDULERS.items():
        schedulers.append(
            {
                "key": list(key),
                "have_initialized": scheduler.have_initialized,
                "config": (
                    None if scheduler.config is None else dict(vars(scheduler.config))
                ),
                "tile_scheduler_metadata_shape": (
                    None
                    if scheduler.tile_scheduler_metadata is None
                    else list(scheduler.tile_scheduler_metadata.shape)
                ),
                "tile_scheduler_metadata": (
                    None
                    if scheduler.tile_scheduler_metadata is None
                    else scheduler.tile_scheduler_metadata.detach().cpu().tolist()
                ),
                "num_splits": (
                    None
                    if scheduler.num_splits is None
                    else scheduler.num_splits.detach().cpu().tolist()
                ),
            }
        )
    return {
        "label": _MANIFEST["label"],
        "overlay": str(_OVERLAY),
        "python_module": str(_MODULE),
        "loaded_cuda_module": str(_CUDA_MODULE),
        "extension": str(extension),
        "extension_sha256": extension_sha256,
        "manifest": str(_MANIFEST_PATH),
        "manifest_sha256": _sha256(_MANIFEST_PATH),
        "source_commit": _MANIFEST["source_commit"],
        "source_patch_sha256": source_patch_sha256,
        "cutlass_commit": _MANIFEST["cutlass_commit"],
        "schedulers": schedulers,
    }


def run(inputs, runtime):
    q = inputs["q"]
    kv_cache = inputs["kv_cache"]
    indices = inputs["indices"]
    scheduler_key = (
        str(q.device),
        *q.shape,
        *kv_cache.shape,
        *indices.shape,
        inputs["head_dim_v"],
        inputs["softmax_scale"],
    )
    scheduler = _SCHEDULERS.get(scheduler_key)
    if scheduler is None:
        scheduler, _ = get_mla_metadata()
        _SCHEDULERS[scheduler_key] = scheduler
    del runtime
    out, _ = flash_mla_with_kvcache(
        q=q,
        k_cache=kv_cache,
        block_table=None,
        cache_seqlens=None,
        head_dim_v=inputs["head_dim_v"],
        tile_scheduler_metadata=scheduler,
        num_splits=None,
        softmax_scale=inputs["softmax_scale"],
        causal=False,
        is_fp8_kvcache=True,
        indices=indices,
    )
    return out
