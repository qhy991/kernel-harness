"""FlashMLA sparse-decode experiment with a smaller useful SM partition set.

The installed FlashMLA operator requires a 148-row ``DecodingSchedMeta`` tensor
on B200.  This candidate reproduces the pinned FlashMLA scheduler on the CPU
for a configurable number of useful partitions, then pads the remaining rows
with explicit no-ops.  CUDA tensors are materialized on the first candidate
call, outside the harness's timed repetitions.

Set ``GLM52_FLASHMLA_NUM_SM_PARTS`` to select the useful partition count.  The
default is 128 so an accidentally omitted setting still names a reproducible
experiment rather than silently reverting to the stock 148-part schedule.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any


OVERRIDE_ENV = "GLM52_FLASHMLA_NUM_SM_PARTS"
DEFAULT_OVERRIDE_PARTS = 128
STOCK_METADATA_ROWS = 148
TOPK = 2048
BLOCK_SIZE_N = 64
FIXED_OVERHEAD_BLOCKS = 5
SUPPORTED_BATCHES = (16, 32)


@dataclass(frozen=True)
class SchedulerResult:
    """Host representation of FlashMLA's two decode-scheduler outputs."""

    batch: int
    useful_partitions: int
    payload_blocks: int
    useful_rows: int
    metadata: tuple[tuple[int, ...], ...]
    num_splits: tuple[int, ...]


_CPU_SCHEDULES: dict[tuple[int, int], SchedulerResult] = {}
_CUDA_SCHEDULES: dict[tuple[str, int | None, int, int, int], Any] = {}


def _ceil_div(value: int, divisor: int) -> int:
    return (value + divisor - 1) // divisor


def _selected_partitions() -> int:
    raw = os.environ.get(OVERRIDE_ENV)
    if raw is None:
        return DEFAULT_OVERRIDE_PARTS
    try:
        value = int(raw)
    except ValueError as exc:
        raise RuntimeError(f"{OVERRIDE_ENV} must be an integer, got {raw!r}") from exc
    if not 1 <= value <= STOCK_METADATA_ROWS:
        raise RuntimeError(
            f"{OVERRIDE_ENV} must be in [1, {STOCK_METADATA_ROWS}], got {value}"
        )
    return value


def generate_scheduler_metadata(
    batch: int, useful_partitions: int
) -> SchedulerResult:
    """Mirror pinned FlashMLA's equal-topk decode scheduler exactly.

    This is the host equivalent of
    ``csrc/smxx/decode/get_decoding_sched_meta/get_decoding_sched_meta.cu`` at
    FlashMLA commit ``05e26647fe840b8baedae486c2d86d5ce4efeb7c`` for the
    production GLM-5.2 sparse-decode constants: top-k 2048, block size 64,
    fixed per-request overhead 5, and no extra KV cache.
    """

    if batch <= 0:
        raise ValueError(f"batch must be positive, got {batch}")
    if not 1 <= useful_partitions <= STOCK_METADATA_ROWS:
        raise ValueError(
            "useful_partitions must be in "
            f"[1, {STOCK_METADATA_ROWS}], got {useful_partitions}"
        )

    cache_key = (batch, useful_partitions)
    cached = _CPU_SCHEDULES.get(cache_key)
    if cached is not None:
        return cached

    num_blocks = _ceil_div(TOPK, BLOCK_SIZE_N)
    total_num_blocks = batch * (num_blocks + FIXED_OVERHEAD_BLOCKS)
    payload = (
        _ceil_div(total_num_blocks, useful_partitions) + FIXED_OVERHEAD_BLOCKS
    )

    now_req_idx = 0
    now_block = 0
    now_split_idx = 0
    cumulative_splits = 0
    num_splits = [0] * (batch + 1)
    metadata: list[tuple[int, ...]] = []

    for _part in range(useful_partitions):
        # The CUDA scheduler may reach batch before exhausting the physical
        # partition envelope (stock 148 produces 20 such rows here).  Its main
        # kernel treats begin_req_idx >= batch as a no-op.  Populate every
        # other field deterministically instead of preserving undefined shared
        # memory reads from the device implementation.
        if now_req_idx >= batch:
            metadata.append((batch, batch - 1, 0, 0, 0, 0, 0, 0))
            continue

        begin_req_idx = now_req_idx
        begin_block_idx = now_block
        begin_split_idx = now_split_idx
        is_first_req_splitted = int(now_block != 0)
        remain_payload = payload

        while now_req_idx < batch:
            now_remain_blocks = num_blocks - now_block
            if (
                remain_payload
                >= now_remain_blocks + FIXED_OVERHEAD_BLOCKS
            ):
                cumulative_splits += now_split_idx + 1
                num_splits[now_req_idx + 1] = cumulative_splits
                remain_payload -= now_remain_blocks + FIXED_OVERHEAD_BLOCKS
                now_req_idx += 1
                now_block = 0
                now_split_idx = 0
            else:
                if remain_payload - FIXED_OVERHEAD_BLOCKS > 0:
                    now_block += remain_payload - FIXED_OVERHEAD_BLOCKS
                    now_split_idx += 1
                    remain_payload = 0
                break

        end_req_idx = now_req_idx if now_block > 0 else now_req_idx - 1
        end_block_idx = now_block if now_block > 0 else num_blocks
        is_last_req_splitted = int(end_block_idx != num_blocks)
        if begin_req_idx == end_req_idx:
            same_req_is_split = int(
                bool(is_first_req_splitted or is_last_req_splitted)
            )
            is_first_req_splitted = same_req_is_split
            is_last_req_splitted = same_req_is_split

        metadata.append(
            (
                begin_req_idx,
                end_req_idx,
                begin_block_idx,
                end_block_idx,
                begin_split_idx,
                is_first_req_splitted,
                is_last_req_splitted,
                0,
            )
        )

    if now_req_idx != batch or now_block != 0 or now_split_idx != 0:
        raise RuntimeError(
            "partition override cannot schedule the complete batch: "
            f"batch={batch}, useful_partitions={useful_partitions}, "
            f"stopped_at=({now_req_idx}, {now_block}, {now_split_idx})"
        )

    useful_rows = sum(row[0] < batch for row in metadata)
    metadata.extend(
        (batch, batch - 1, 0, 0, 0, 0, 0, 0)
        for _ in range(STOCK_METADATA_ROWS - len(metadata))
    )
    result = SchedulerResult(
        batch=batch,
        useful_partitions=useful_partitions,
        payload_blocks=payload,
        useful_rows=useful_rows,
        metadata=tuple(metadata),
        num_splits=tuple(num_splits),
    )
    _CPU_SCHEDULES[cache_key] = result
    return result


def _schedule_digest(schedule: SchedulerResult) -> str:
    payload = {
        "metadata": schedule.metadata,
        "num_splits": schedule.num_splits,
    }
    serialized = json.dumps(payload, separators=(",", ":"), sort_keys=True)
    return hashlib.sha256(serialized.encode()).hexdigest()


def _cuda_metadata(inputs: dict[str, Any]) -> Any:
    """Materialize and cache the candidate's CUDA metadata on first use."""

    import torch
    from sglang.srt.layers.attention.dsa_backend import DSAFlashMLAMetadata

    query = inputs["query"]
    cache_seqlens = inputs["cache_seqlens"]
    batch = int(query.shape[0])
    if batch not in SUPPORTED_BATCHES:
        raise RuntimeError(
            f"scheduler override only covers production buckets {SUPPORTED_BATCHES}, "
            f"got batch={batch}"
        )
    useful_partitions = _selected_partitions()
    schedule = generate_scheduler_metadata(batch, useful_partitions)
    cache_key = (
        query.device.type,
        query.device.index,
        batch,
        useful_partitions,
        cache_seqlens.data_ptr(),
    )
    cached = _CUDA_SCHEDULES.get(cache_key)
    if cached is not None:
        return cached

    tile_scheduler_metadata = torch.tensor(
        schedule.metadata, dtype=torch.int32, device=query.device
    )
    num_splits = torch.tensor(
        schedule.num_splits, dtype=torch.int32, device=query.device
    )
    flashmla_metadata = DSAFlashMLAMetadata(
        flashmla_metadata=tile_scheduler_metadata,
        num_splits=num_splits,
    )
    metadata_stub = SimpleNamespace(
        dsa_cache_seqlens_int32=cache_seqlens,
        flashmla_metadata=flashmla_metadata,
    )
    _CUDA_SCHEDULES[cache_key] = metadata_stub
    return metadata_stub


def candidate_evidence() -> dict[str, Any]:
    """Return JSON-serializable configuration and exact generated metadata."""

    useful_partitions = _selected_partitions()
    generated: dict[str, Any] = {}
    for batch in SUPPORTED_BATCHES:
        schedule = generate_scheduler_metadata(batch, useful_partitions)
        generated[f"m{batch}"] = {
            "batch": batch,
            "payload_blocks": schedule.payload_blocks,
            "useful_rows": schedule.useful_rows,
            "total_splits": schedule.num_splits[-1],
            "metadata_sha256": _schedule_digest(schedule),
            "raw_tile_scheduler_metadata": [list(row) for row in schedule.metadata],
            "raw_num_splits": list(schedule.num_splits),
        }
    return {
        "candidate": "dsa_flashmla_kv_sched_override",
        "flashmla_commit": "05e26647fe840b8baedae486c2d86d5ce4efeb7c",
        "override_env": OVERRIDE_ENV,
        "override_env_raw": os.environ.get(OVERRIDE_ENV),
        "default_override_parts": DEFAULT_OVERRIDE_PARTS,
        "selected_useful_partitions": useful_partitions,
        "stock_metadata_rows": STOCK_METADATA_ROWS,
        "scheduler_constants": {
            "topk": TOPK,
            "block_size_n": BLOCK_SIZE_N,
            "fixed_overhead_num_blocks": FIXED_OVERHEAD_BLOCKS,
        },
        "generated": generated,
    }


def run(inputs: dict[str, Any], runtime: Any) -> Any:
    """Invoke the exact production backend method with candidate metadata."""

    del runtime
    from sglang.srt.layers.attention.dsa_backend import DeepseekSparseAttnBackend

    metadata_stub = _cuda_metadata(inputs)
    return DeepseekSparseAttnBackend._forward_flashmla_kv(
        inputs["backend_stub"],
        inputs["query"],
        inputs["kv_cache"],
        inputs.get("head_dim_v", 512),
        inputs.get("softmax_scale", 0.0625),
        inputs["layer_stub"],
        metadata_stub,
        inputs["page_table_1"],
    )
