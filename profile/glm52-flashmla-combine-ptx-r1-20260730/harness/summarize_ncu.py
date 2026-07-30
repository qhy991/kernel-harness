#!/usr/bin/env python3
"""Reduce one NCU report to the machine-readable critical-path mechanism test.

Plan section 6 permits NCU only to test a stated critical-path mechanism. The
stated mechanism (plan section 0.3) is "dominant long-scoreboard / barrier
stalls, low L2 reuse". This script extracts exactly the metrics that confirm or
refute it, plus the shared-memory headroom that bounds any pipeline-deepening
alternative.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import subprocess
from pathlib import Path


STALL_PREFIX = "smsp__average_warps_issue_stalled_"
STALL_SUFFIX = "_per_issue_active.ratio"

SCALARS = {
    "duration_us": "gpu__time_duration.sum",
    "elapsed_cycles": "gpc__cycles_elapsed.max",
    "sm_active_cycles_avg": "sm__cycles_active.avg",
    "dram_throughput_pct": "gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed",
    "l2_hit_rate_pct": "lts__t_sector_hit_rate.pct",
    "l1tex_hit_rate_pct": "l1tex__t_sector_hit_rate.pct",
    "registers_per_thread": "launch__registers_per_thread",
    "barrier_count": "launch__barrier_count",
    "grid_size": "launch__grid_size",
    "block_size": "launch__block_size",
    # NCU reports these launch shared-memory metrics in kilobytes of 1000 bytes.
    "dynamic_shared_kbytes": "launch__shared_mem_per_block_dynamic",
    "device_shared_optin_bytes": (
        "device__attribute_max_shared_memory_per_block_optin"
    ),
    "waves_per_sm": "launch__waves_per_multiprocessor",
    "sm_active_cycles_max": "sm__cycles_active.max",
    "sm_active_cycles_min": "sm__cycles_active.min",
    "active_warps_per_scheduler": "smsp__warps_active.avg.per_cycle_active",
    "eligible_warps_per_scheduler": (
        "smsp__warps_eligible.avg.per_cycle_active"
    ),
    "warp_cycles_per_issued_inst": "smsp__average_warps_active_per_inst_executed.ratio",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--shared-limit-bytes",
        type=int,
        required=True,
        help="device shared_memory_per_block_optin from the preflight",
    )
    parser.add_argument("--label", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.output.exists():
        raise RuntimeError(f"refusing to overwrite evidence: {args.output}")
    raw = subprocess.check_output(
        ["ncu", "--import", str(args.report), "--page", "raw", "--csv"],
        text=True,
        stderr=subprocess.DEVNULL,
    )
    rows = list(csv.reader(io.StringIO(raw)))
    header, values = rows[0], rows[-1]
    metrics = dict(zip(header, values))

    def number(key: str):
        value = metrics.get(key)
        if value in (None, "", "n/a"):
            return None
        try:
            return float(value.replace(",", ""))
        except ValueError:
            return value

    stalls = {}
    for key, value in metrics.items():
        if key.startswith(STALL_PREFIX) and key.endswith(STALL_SUFFIX):
            reason = key[len(STALL_PREFIX) : -len(STALL_SUFFIX)]
            stalls[reason] = float(value) if value not in ("", "n/a") else 0.0
    total_stall = sum(stalls.values())
    stall_table = {
        reason: {
            "cycles_per_issued_instruction": cycles,
            "share_of_total_pct": (100.0 * cycles / total_stall) if total_stall else None,
        }
        for reason, cycles in sorted(stalls.items(), key=lambda kv: -kv[1])
    }

    scalars = {name: number(key) for name, key in SCALARS.items()}
    shared_kb = scalars.get("dynamic_shared_kbytes")
    shared_used = int(round(shared_kb * 1000)) if isinstance(shared_kb, float) else None
    device_limit = scalars.get("device_shared_optin_bytes")
    if device_limit is not None and int(device_limit) != args.shared_limit_bytes:
        raise AssertionError(
            f"device optin limit {device_limit} disagrees with preflight "
            f"{args.shared_limit_bytes}"
        )
    headroom = (
        args.shared_limit_bytes - shared_used if shared_used is not None else None
    )
    elapsed = scalars.get("elapsed_cycles")
    active = scalars.get("sm_active_cycles_avg")
    evidence = {
        "schema_version": 1,
        "label": args.label,
        "report": str(args.report),
        "purpose": (
            "test the plan's stated critical-path mechanism: dominant "
            "long-scoreboard / barrier stalls and low L2 reuse"
        ),
        "scalars": scalars,
        "stall_cycles_per_issued_instruction": stall_table,
        "total_stall_cycles_per_issued_instruction": total_stall,
        "mechanism_test": {
            "long_scoreboard_share_pct": stall_table.get("long_scoreboard", {}).get(
                "share_of_total_pct"
            ),
            "barrier_share_pct": stall_table.get("barrier", {}).get(
                "share_of_total_pct"
            ),
            "long_scoreboard_plus_barrier_share_pct": (
                (stall_table.get("long_scoreboard", {}).get("share_of_total_pct") or 0)
                + (stall_table.get("barrier", {}).get("share_of_total_pct") or 0)
            ),
            "shared_memory_stall_share_pct": (
                (stall_table.get("short_scoreboard", {}).get("share_of_total_pct") or 0)
                + (stall_table.get("mio_throttle", {}).get("share_of_total_pct") or 0)
            ),
            "stated_mechanism_confirmed": (
                (stall_table.get("long_scoreboard", {}).get("share_of_total_pct") or 0)
                + (stall_table.get("barrier", {}).get("share_of_total_pct") or 0)
                > 50.0
            ),
        },
        "shared_memory": {
            "device_limit_bytes": args.shared_limit_bytes,
            "launch_configured_bytes": shared_used,
            "headroom_bytes": headroom,
            "one_extra_kv_buffer_bytes": 32768 + 73728,
            "one_extra_index_buffer_bytes": 8 + 256 + 512 + 16,
            "extra_kv_buffer_fits": (
                headroom is not None and headroom >= 32768 + 73728
            ),
            "extra_index_buffer_fits": (
                headroom is not None and headroom >= 8 + 256 + 512 + 16
            ),
        },
        "sm_utilisation": {
            "elapsed_cycles": elapsed,
            "average_sm_active_cycles": active,
            "average_sm_active_fraction_of_elapsed": (
                active / elapsed if elapsed and active else None
            ),
            "note": (
                "the gap is tail imbalance across the 148 fixed scheduler "
                "partitions; tile_scheduler_metadata and num_splits are frozen "
                "caller inputs, so it is not addressable inside this boundary"
            ),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "duration_us": scalars.get("duration_us"),
                "top_stalls": {
                    reason: round(entry["share_of_total_pct"], 2)
                    for reason, entry in list(stall_table.items())[:5]
                },
                "mechanism": evidence["mechanism_test"],
                "shared_headroom_bytes": headroom,
                "avg_sm_active_fraction": evidence["sm_utilisation"][
                    "average_sm_active_fraction_of_elapsed"
                ],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
