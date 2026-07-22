#!/usr/bin/env python3
"""Reproduce pinned FlashMLA's equal-length sparse scheduler on the CPU."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--topk", type=int, default=2048)
    parser.add_argument("--block-size", type=int, default=64)
    parser.add_argument("--overhead", type=int, default=5)
    parser.add_argument("--parts", type=int, nargs="+", default=[96, 112, 128, 144, 148])
    parser.add_argument("--batches", type=int, nargs="+", default=[16, 32])
    parser.add_argument("--output")
    return parser.parse_args()


def simulate(batch: int, topk: int, block_size: int, overhead: int, parts: int):
    blocks_per_request = (topk + block_size - 1) // block_size
    total_weighted_blocks = batch * (blocks_per_request + overhead)
    payload = (total_weighted_blocks + parts - 1) // parts + overhead
    request = 0
    request_block = 0
    splits = [0] * batch
    useful_parts = 0
    part_payloads = []

    for part in range(parts):
        remaining = payload
        assigned = []
        while request < batch:
            request_remaining = blocks_per_request - request_block
            if remaining >= request_remaining + overhead:
                assigned.append(
                    {"request": request, "blocks": request_remaining}
                )
                if request_remaining:
                    splits[request] += 1
                remaining -= request_remaining + overhead
                request += 1
                request_block = 0
            else:
                data_blocks = max(remaining - overhead, 0)
                if data_blocks:
                    assigned.append({"request": request, "blocks": data_blocks})
                    splits[request] += 1
                    request_block += data_blocks
                break
        if assigned:
            useful_parts += 1
        part_payloads.append({"part": part, "assignments": assigned})

    if request != batch or request_block != 0:
        raise RuntimeError(
            f"scheduler did not finish: request={request}, block={request_block}"
        )
    if parts <= 32:
        combine_template = 32
    elif parts <= 64:
        combine_template = 64
    elif parts <= 96:
        combine_template = 96
    elif parts <= 128:
        combine_template = 128
    elif parts <= 160:
        combine_template = 160
    else:
        raise ValueError("pinned combine supports at most 160 parts")

    return {
        "batch": batch,
        "topk": topk,
        "block_size": block_size,
        "fixed_overhead_blocks": overhead,
        "num_sm_parts": parts,
        "total_weighted_blocks": total_weighted_blocks,
        "payload_including_overhead": payload,
        "data_blocks_per_partial_part": payload - overhead,
        "useful_parts": useful_parts,
        "empty_parts": parts - useful_parts,
        "splits_per_request": splits,
        "splits_unique": sorted(set(splits)),
        "combine_max_splits_template": combine_template,
        "part_payloads": part_payloads,
    }


def main() -> int:
    args = parse_args()
    results = [
        simulate(batch, args.topk, args.block_size, args.overhead, parts)
        for batch in args.batches
        for parts in args.parts
    ]
    rendered = json.dumps(
        {
            "source": (
                "FlashMLA 05e26647 csrc/smxx/decode/get_decoding_sched_meta/"
                "get_decoding_sched_meta.cu"
            ),
            "results": results,
        },
        indent=2,
        sort_keys=True,
    ) + "\n"
    if args.output:
        output = Path(args.output).expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered)
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
