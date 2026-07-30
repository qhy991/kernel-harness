#!/usr/bin/env python3
"""Audit and summarize the two-kernel chains in a retained Nsys report."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--m", type=int, choices=(16, 32), required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--candidate-main-symbol",
        required=True,
        help=(
            "exact candidate main symbol; must carry the goal's "
            "infini_kernel_glm52_flashmla_sparse_decode prefix"
        ),
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    args = parse_args()
    report = args.report.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if output.exists():
        raise RuntimeError(f"refusing to overwrite evidence: {output}")
    if not report.is_file():
        raise RuntimeError(f"missing Nsys report: {report}")

    completed = subprocess.run(
        [
            "nsys",
            "stats",
            "--report",
            "cuda_gpu_trace:nvtx-name:base",
            "--format",
            "csv",
            str(report),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    lines = completed.stdout.splitlines()
    header_index = next(
        index for index, line in enumerate(lines) if line.startswith("Start (ns),")
    )
    rows = list(csv.DictReader(lines[header_index:]))
    prefix = f"glm52_m{args.m}_"
    rows = [row for row in rows if row["Name"].startswith(prefix)]

    if not args.candidate_main_symbol.startswith(
        "infini_kernel_glm52_flashmla_sparse_decode"
    ):  # PREFIX_REQUIRED
        raise AssertionError(
            "candidate main symbol must carry the goal prefix: "
            f"{args.candidate_main_symbol}"
        )

    lanes: dict[str, list[dict[str, object]]] = {}
    for lane in (
        "stock_eager",
        "candidate_eager",
        "stock_graph",
        "candidate_graph",
    ):
        lane_prefix = f"{prefix}{lane}/"
        lane_rows = [row for row in rows if row["Name"].startswith(lane_prefix)]
        expected_replays = 5 if lane.endswith("_graph") else 1
        if len(lane_rows) != expected_replays * 2:
            raise AssertionError(
                f"{lane}: expected {expected_replays * 2} kernels, "
                f"found {len(lane_rows)}"
            )
        expected_main = (
            args.candidate_main_symbol
            if lane.startswith("candidate_")
            else "flash_fwd_splitkv_mla_fp8_sparse_kernel"
        )
        expected_names = [
            expected_main if index % 2 == 0 else "flash_fwd_mla_combine_kernel"
            for index in range(expected_replays * 2)
        ]
        observed_names = [row["Name"].removeprefix(lane_prefix) for row in lane_rows]
        if observed_names != expected_names:
            raise AssertionError(
                f"{lane}: unexpected device chain {observed_names!r}"
            )
        lanes[lane] = [
            {
                "name": name,
                "start_ns": int(row["Start (ns)"]),
                "duration_ns": int(row["Duration (ns)"]),
                "grid": [
                    int(row["GrdX"]),
                    int(row["GrdY"]),
                    int(row["GrdZ"]),
                ],
                "block": [
                    int(row["BlkX"]),
                    int(row["BlkY"]),
                    int(row["BlkZ"]),
                ],
                "registers_per_thread": int(row["Reg/Trd"]),
                "static_smem_mb": float(row["StcSMem (MB)"]),
                "dynamic_smem_mb": float(row["DymSMem (MB)"]),
            }
            for row, name in zip(lane_rows, observed_names, strict=True)
        ]

    evidence = {
        "schema_version": 1,
        "m": args.m,
        "report": {
            "path": str(report),
            "sha256": sha256(report),
            "size_bytes": report.stat().st_size,
            "nsys_version": subprocess.check_output(
                ["nsys", "--version"], text=True
            ).strip(),
        },
        "audit": {
            "candidate_main_prefix_correct": True,
            "stock_main_is_control_only": True,
            "combine_is_unchanged_stock_symbol": True,
            "one_main_then_one_combine_per_replay": True,
            "unexpected_device_kernels_in_ranges": 0,
            "graph_replays_per_lane": 5,
        },
        "lanes": lanes,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n")
    print(json.dumps(evidence["audit"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
