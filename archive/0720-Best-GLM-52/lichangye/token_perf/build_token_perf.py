#!/usr/bin/env python3
"""Build token/performance plots for the lichangye GLM-5.2 archive."""

from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt


REPO = Path(__file__).resolve().parents[4]
OUT = REPO / "archive/0720-Best-GLM-52/lichangye/token_perf"
SESSION = Path(
    "/home/lichangye/.claude/projects/-home-lichangye-kernel-harness-amd/"
    "ab5d0783-1275-46e7-bc91-2abf03b1bfd7.jsonl"
)
LOOP_START = "2026-07-22T05:18:49Z"
LOOP_COMPLETE = "2026-07-22T15:04:13Z"


def parse_ts(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(
        timezone.utc
    )


def gmean(values: list[float]) -> float:
    return math.exp(sum(math.log(v) for v in values) / len(values))


@dataclass
class TokenPoint:
    timestamp: datetime
    fresh_tokens: int
    effective_tokens: int


def token_series() -> list[TokenPoint]:
    seen: dict[str, tuple[datetime, int, int]] = {}
    with SESSION.open(errors="ignore") as fh:
        for line in fh:
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            message = event.get("message")
            if not isinstance(message, dict):
                continue
            usage = message.get("usage")
            message_id = message.get("id")
            if not usage or not message_id or message_id in seen:
                continue
            ts = event.get("timestamp")
            if not ts:
                continue
            input_tokens = int(usage.get("input_tokens") or 0)
            cache_create = int(usage.get("cache_creation_input_tokens") or 0)
            cache_read = int(usage.get("cache_read_input_tokens") or 0)
            output_tokens = int(usage.get("output_tokens") or 0)
            fresh = input_tokens + cache_create + output_tokens
            effective = fresh + cache_read
            seen[message_id] = (parse_ts(ts), fresh, effective)

    running_fresh = 0
    running_effective = 0
    series: list[TokenPoint] = []
    for ts, fresh, effective in sorted(seen.values(), key=lambda item: item[0]):
        running_fresh += fresh
        running_effective += effective
        series.append(TokenPoint(ts, running_fresh, running_effective))
    return series


TOKENS = token_series()
START_TS = parse_ts(LOOP_START)


def cumulative_tokens_at(timestamp: str) -> tuple[int, int]:
    target = parse_ts(timestamp)
    fresh = 0
    effective = 0
    for point in TOKENS:
        if point.timestamp <= target:
            fresh = point.fresh_tokens
            effective = point.effective_tokens
        else:
            break
    start_fresh = 0
    start_effective = 0
    for point in TOKENS:
        if point.timestamp <= START_TS:
            start_fresh = point.fresh_tokens
            start_effective = point.effective_tokens
        else:
            break
    return max(0, fresh - start_fresh), max(0, effective - start_effective)


def final_result(task: str, run_id: str) -> dict:
    path = REPO / f"runs/glm52/{task}/{run_id}/result.json"
    data = json.loads(path.read_text())
    finished = data["run"]["finished_utc"].replace("+00:00", "Z")
    return {
        "timestamp": finished,
        "perf_ratio": float(data["aggregate"]["geomean_primary_util_ratio"]),
        "geomean_mfu": float(data["aggregate"]["geomean_mfu"]),
        "geomean_bw_util": float(data["aggregate"]["geomean_bw_util"]),
        "source": str(path.relative_to(REPO)),
        "notes": (
            f"accepted result; wins={data['aggregate']['shapes_won']}; "
            f"regressions={data['aggregate']['shapes_regressed']}"
        ),
    }


ROUND_POINTS = {
    "moe_total_decode": [
        {
            "label": "round-0 gate snapshot",
            "timestamp": "2026-07-22T07:00:20Z",
            "perf_ratio": gmean([1.0683, 1.0700, 1.0757, 1.0757, 1.0566, 1.0]),
            "geomean_mfu": "",
            "geomean_bw_util": gmean([0.4928, 0.4877, 0.4845, 0.3462, 0.3308, 0.2895]),
            "source": ".humanize/rlcr/2026-07-22_13-18-49/round-0-summary.md",
            "notes": "5 wins + one neutral/reference-fallback shape in the round summary",
        },
        {
            "label": "accepted result.json",
            **final_result("moe_total_decode", "20260722T083714Z-126708"),
        },
    ],
    "moe_total_prefill": [
        {
            "label": "round-1 gate snapshot",
            "timestamp": "2026-07-22T07:18:26Z",
            "perf_ratio": gmean([1.1474, 1.0553, 1.0460]),
            "geomean_mfu": gmean([0.2463, 0.2704, 0.2787]),
            "geomean_bw_util": "",
            "source": ".humanize/rlcr/2026-07-22_13-18-49/round-1-summary.md",
            "notes": "3/3 passed in the round summary",
        },
        {
            "label": "accepted result.json",
            **final_result("moe_total_prefill", "20260722T083730Z-959e52"),
        },
    ],
    "dsa_prefill_attn": [
        {
            "label": "round-2 gate snapshot",
            "timestamp": "2026-07-22T07:49:08Z",
            "perf_ratio": gmean([1.6467, 1.5619, 1.3998]),
            "geomean_mfu": gmean([0.03297, 0.03256, 0.03308]),
            "geomean_bw_util": "",
            "source": ".humanize/rlcr/2026-07-22_13-18-49/round-2-summary.md",
            "notes": "3/3 passed in the round summary; later result.json is the authority",
        },
        {
            "label": "accepted result.json",
            **final_result("dsa_prefill_attn", "20260722T083802Z-1b233d"),
        },
    ],
    "index_score_prefill": [
        {
            "label": "search probe",
            "timestamp": "2026-07-22T08:00:15Z",
            "perf_ratio": gmean([1.38, 3.82, 3.60]),
            "geomean_mfu": "",
            "geomean_bw_util": "",
            "source": "Claude transcript search-probe note",
            "notes": "pre-candidate probe that selected BLOCK_KV=256",
        },
        {
            "label": "round-2 addendum gate snapshot",
            "timestamp": "2026-07-22T08:25:43Z",
            "perf_ratio": gmean([1.5573, 3.8931, 3.7618]),
            "geomean_mfu": gmean([0.1598, 0.1068, 0.1077]),
            "geomean_bw_util": "",
            "source": ".humanize/rlcr/2026-07-22_13-18-49/round-2-summary.md",
            "notes": "3/3 passed in the round summary",
        },
        {
            "label": "accepted result.json",
            **final_result("index_score_prefill", "20260722T084041Z-7a3d33"),
        },
    ],
}


TASK_LABELS = {
    "moe_total_decode": "moe_total_decode",
    "moe_total_prefill": "moe_total_prefill",
    "dsa_prefill_attn": "dsa_prefill_attn",
    "index_score_prefill": "index_score_prefill",
}


def build_rows() -> list[dict]:
    rows: list[dict] = []
    complete_fresh, complete_effective = cumulative_tokens_at(LOOP_COMPLETE)
    for task, points in ROUND_POINTS.items():
        rows.append(
            {
                "task": task,
                "label": "loop start / reference",
                "timestamp_utc": LOOP_START,
                "fresh_tokens_since_loop_start": 0,
                "effective_tokens_since_loop_start": 0,
                "fresh_tokens_millions": 0.0,
                "perf_ratio": 1.0,
                "geomean_mfu": "",
                "geomean_bw_util": "",
                "source": "loop baseline",
                "notes": "reference baseline",
            }
        )
        last_perf = 1.0
        last_mfu: float | str = ""
        last_bw: float | str = ""
        for point in points:
            fresh, effective = cumulative_tokens_at(point["timestamp"])
            last_perf = float(point["perf_ratio"])
            last_mfu = point.get("geomean_mfu", "")
            last_bw = point.get("geomean_bw_util", "")
            rows.append(
                {
                    "task": task,
                    "label": point["label"],
                    "timestamp_utc": point["timestamp"],
                    "fresh_tokens_since_loop_start": fresh,
                    "effective_tokens_since_loop_start": effective,
                    "fresh_tokens_millions": round(fresh / 1_000_000, 6),
                    "perf_ratio": round(last_perf, 6),
                    "geomean_mfu": last_mfu,
                    "geomean_bw_util": last_bw,
                    "source": point["source"],
                    "notes": point["notes"],
                }
            )
        rows.append(
            {
                "task": task,
                "label": "loop complete",
                "timestamp_utc": LOOP_COMPLETE,
                "fresh_tokens_since_loop_start": complete_fresh,
                "effective_tokens_since_loop_start": complete_effective,
                "fresh_tokens_millions": round(complete_fresh / 1_000_000, 6),
                "perf_ratio": round(last_perf, 6),
                "geomean_mfu": last_mfu,
                "geomean_bw_util": last_bw,
                "source": "Claude transcript finalization point",
                "notes": "no further perf change after accepted result",
            }
        )
    return rows


def write_csv(rows: list[dict]) -> None:
    path = OUT / "token_perf_points.csv"
    fields = [
        "task",
        "label",
        "timestamp_utc",
        "fresh_tokens_since_loop_start",
        "effective_tokens_since_loop_start",
        "fresh_tokens_millions",
        "perf_ratio",
        "geomean_mfu",
        "geomean_bw_util",
        "source",
        "notes",
    ]
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def plot_task(rows: list[dict], task: str) -> None:
    task_rows = [row for row in rows if row["task"] == task]
    xs = [float(row["fresh_tokens_millions"]) for row in task_rows]
    ys = [float(row["perf_ratio"]) for row in task_rows]

    fig, ax = plt.subplots(figsize=(8, 4.8), dpi=160)
    ax.plot(xs, ys, color="#246b8f", linewidth=2.2)
    ax.scatter(xs, ys, s=58, color="#c44536", zorder=3)

    label_text = {
        "loop start / reference": "start",
        "round-0 gate snapshot": "round snapshot",
        "round-1 gate snapshot": "round snapshot",
        "round-2 gate snapshot": "round snapshot",
        "round-2 addendum gate snapshot": "round snapshot",
        "search probe": "search probe",
        "accepted result.json": "accepted",
        "loop complete": "complete",
    }
    label_offsets = {
        "loop start / reference": (5, 8),
        "round-0 gate snapshot": (5, 10),
        "round-1 gate snapshot": (5, 10),
        "round-2 gate snapshot": (5, 10),
        "round-2 addendum gate snapshot": (5, 12),
        "search probe": (5, 8),
        "accepted result.json": (5, -18),
        "loop complete": (-58, -18),
    }
    for row, x, y in zip(task_rows, xs, ys):
        label = row["label"]
        ax.annotate(
            label_text.get(label, label),
            (x, y),
            textcoords="offset points",
            xytext=label_offsets.get(label, (5, 8)),
            fontsize=8,
        )

    ax.set_title(f"{TASK_LABELS[task]}: token/perf timeline")
    ax.set_xlabel("Fresh Claude tokens since loop start (millions)")
    ax.set_ylabel("Geomean primary-util ratio vs reference")
    ax.grid(True, alpha=0.25)
    ax.axhline(1.0, color="#666666", linewidth=1.0, linestyle="--", alpha=0.7)
    ax.set_ylim(bottom=0.92, top=max(ys) * 1.12)
    fig.tight_layout()
    for ext in ("png", "svg"):
        fig.savefig(OUT / f"token_perf_{task}.{ext}")
    plt.close(fig)


def plot_all(rows: list[dict]) -> None:
    fig, ax = plt.subplots(figsize=(9, 5.2), dpi=160)
    colors = {
        "moe_total_decode": "#246b8f",
        "moe_total_prefill": "#2f7d32",
        "dsa_prefill_attn": "#8a4f9e",
        "index_score_prefill": "#c44536",
    }
    for task in TASK_LABELS:
        task_rows = [row for row in rows if row["task"] == task]
        xs = [float(row["fresh_tokens_millions"]) for row in task_rows]
        ys = [float(row["perf_ratio"]) for row in task_rows]
        ax.plot(xs, ys, marker="o", linewidth=2.0, label=task, color=colors[task])

    ax.set_title("GLM-5.2 ROCm KDA-Pilot: token/perf timeline")
    ax.set_xlabel("Fresh Claude tokens since loop start (millions)")
    ax.set_ylabel("Geomean primary-util ratio vs reference")
    ax.grid(True, alpha=0.25)
    ax.axhline(1.0, color="#666666", linewidth=1.0, linestyle="--", alpha=0.7)
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    for ext in ("png", "svg"):
        fig.savefig(OUT / f"token_perf_all_tasks.{ext}")
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = build_rows()
    write_csv(rows)
    for task in TASK_LABELS:
        plot_task(rows, task)
    plot_all(rows)


if __name__ == "__main__":
    main()
