#!/usr/bin/env python3
"""Map source-counter stall samples to exact SASS PCs."""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

from ncu_utils import load_report, metric_value_at

STALLS = (
    "long_scoreboard",
    "short_scoreboard",
    "barrier",
    "wait",
    "sleeping",
    "branch_resolving",
    "selected",
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    _report, action = load_report(args.report)
    by_pc: dict[int, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for short_name in STALLS:
        name = f"smsp__pcsamp_warps_issue_stalled_{short_name}"
        metric = action[name]
        if not metric.has_correlation_ids():
            continue
        correlations = metric.correlation_ids()
        for index in range(metric.num_instances()):
            value = metric_value_at(metric, index)
            if value:
                by_pc[correlations.as_uint64(index)][short_name] += int(value)

    rows = sorted(
        (
            (sum(stalls.values()), pc, dict(stalls), action.sass_by_pc(pc))
            for pc, stalls in by_pc.items()
        ),
        reverse=True,
    )
    lines = [
        f"SASS stall hotspots: {args.tag}",
        f"kernel: {action.name()}",
        "",
        "rank total address stalls SASS",
    ]
    for rank, (total, pc, stalls, sass) in enumerate(rows[:40], start=1):
        breakdown = ",".join(
            f"{name}={value}"
            for name, value in sorted(stalls.items(), key=lambda item: -item[1])
        )
        lines.append(
            f"{rank:02d} {total:6d} 0x{pc:x} {breakdown} {sass.strip() or '<none>'}"
        )
    args.output.write_text("\n".join(lines) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
