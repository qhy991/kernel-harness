#!/usr/bin/env python3
"""Extract the final FlashMLA main/combine chain from an Nsight Systems CSV."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


MAIN = "flash_fwd_splitkv_mla_fp8_sparse_kernel"
COMBINE = "flash_fwd_mla_combine_kernel"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", required=True)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def kernel_record(row: dict[str, str]) -> dict:
    return {
        "start_ns": int(row["Start (ns)"]),
        "duration_ns": int(row["Duration (ns)"]),
        "grid": [int(row["GrdX"]), int(row["GrdY"]), int(row["GrdZ"])],
        "block": [int(row["BlkX"]), int(row["BlkY"]), int(row["BlkZ"])],
        "registers_per_thread": int(row["Reg/Trd"]),
        "static_shared_mb": float(row["StcSMem (MB)"]),
        "dynamic_shared_mb": float(row["DymSMem (MB)"]),
        "stream": int(row["Strm"]),
        "name": row["Name"],
    }


def main() -> int:
    args = parse_args()
    csv_path = Path(args.csv).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    if output.exists():
        raise RuntimeError(f"refusing to overwrite Nsight analysis: {output}")
    with csv_path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    mains = [row for row in rows if MAIN in row["Name"]]
    combines = [row for row in rows if COMBINE in row["Name"]]
    if not mains or not combines:
        raise RuntimeError(
            f"missing FlashMLA kernels: mains={len(mains)}, combines={len(combines)}"
        )
    main_row = mains[-1]
    main_start = int(main_row["Start (ns)"])
    combine_row = min(
        (row for row in combines if int(row["Start (ns)"]) >= main_start),
        key=lambda row: int(row["Start (ns)"]),
    )
    main_kernel = kernel_record(main_row)
    combine_kernel = kernel_record(combine_row)
    main_end = main_kernel["start_ns"] + main_kernel["duration_ns"]
    combine_end = combine_kernel["start_ns"] + combine_kernel["duration_ns"]
    overlap = max(
        0,
        min(main_end, combine_end)
        - max(main_kernel["start_ns"], combine_kernel["start_ns"]),
    )
    gap = max(0, combine_kernel["start_ns"] - main_end)
    result = {
        "tag": args.tag,
        "source_csv": str(csv_path),
        "main_launch_count": len(mains),
        "combine_launch_count": len(combines),
        "selected_pair": "last main launch and its first following combine launch",
        "main": main_kernel,
        "combine": combine_kernel,
        "combine_start_after_main_start_ns": (
            combine_kernel["start_ns"] - main_kernel["start_ns"]
        ),
        "main_combine_overlap_ns": overlap,
        "main_to_combine_gap_ns": gap,
        "combined_chain_span_ns": max(main_end, combine_end) - main_start,
        "same_stream": main_kernel["stream"] == combine_kernel["stream"],
    }
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(rendered)
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
