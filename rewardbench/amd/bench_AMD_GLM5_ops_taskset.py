#!/usr/bin/env python3
"""Run AMD rewardbench for the shared GLM-5.2 ROCm taskset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

import amd_glm5_ops_common as C
from amd_bench_glm5_decode import build_ops as build_decode_ops
from amd_bench_glm5_prefill import build_ops as build_prefill_ops


REPO = Path(__file__).resolve().parents[2]
DEFAULT_TASKSET = REPO / "tasksets" / "glm52_rocm_local.json"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--taskset", type=Path, default=DEFAULT_TASKSET)
    ap.add_argument("--phase", choices=("all", "prefill", "decode"), default="all")
    ap.add_argument("--task", action="append", default=None,
                    help="task id, harness task, or reward operator; can be passed more than once")
    ap.add_argument("--smoke", action="store_true",
                    help="single shape per phase: prefill M=1024, decode M=16")
    ap.add_argument("--m", type=int, default=None,
                    help="single M for selected phases; overrides --smoke")
    ap.add_argument("--s", type=int, default=65536)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--csv-prefix", default="amd_glm5_ops_local_taskset")
    args = ap.parse_args()

    taskset = load_taskset(args.taskset)
    selected = select_tasks(taskset["tasks"], args.phase, args.task)

    device = torch.device(args.device)
    torch.cuda.set_device(device)
    C.print_env_banner()
    print(f"GLM-5.2 AMD rewardbench taskset={taskset['name']} selected={len(selected)}")

    exit_code = 0
    for phase in ("prefill", "decode"):
        phase_tasks = [task for task in selected if task["phase"] == phase]
        if not phase_tasks:
            continue
        reward_ops = {task["reward_operator"] for task in phase_tasks}
        ops = filter_ops(build_prefill_ops() if phase == "prefill" else build_decode_ops(), reward_ops)
        missing = reward_ops - {op[0] for op in ops}
        if missing:
            raise SystemExit(f"{phase} rewardbench missing operators: {sorted(missing)}")
        sweep = build_sweep(phase, args.m, args.s, args.smoke, taskset.get("defaults", {}))
        csv_path = f"{args.csv_prefix}_{phase}.csv"
        print(f"\nRunning {phase}: operators={sorted(reward_ops)} csv={csv_path}")
        rows = C.run_ops(
            ops,
            sweep,
            device,
            phase,
            csv_path,
            row_metadata=metadata_by_reward_operator(phase_tasks),
        )
        if any("error" in row for row in rows):
            exit_code = 1
    return exit_code


def load_taskset(path: Path) -> dict:
    data = json.loads(path.read_text())
    if not data.get("tasks"):
        raise SystemExit(f"taskset has no tasks: {path}")
    return data


def select_tasks(tasks: list[dict], phase: str, requested: list[str] | None) -> list[dict]:
    selected = [task for task in tasks if phase == "all" or task["phase"] == phase]
    if requested:
        wanted = set(requested)
        selected = [
            task for task in selected
            if task["id"] in wanted
            or task["harness_task"] in wanted
            or task["reward_operator"] in wanted
        ]
    if not selected:
        raise SystemExit("no tasks selected")
    return selected


def filter_ops(ops: list[tuple], reward_ops: set[str]) -> list[tuple]:
    return [op for op in ops if op[0] in reward_ops]


def metadata_by_reward_operator(tasks: list[dict]) -> dict[str, dict]:
    out = {}
    for task in tasks:
        out[task["reward_operator"]] = {
            "score_scope": task.get("score_scope", "task"),
            "metric_group": task.get("metric_group"),
            "metric_component": task.get("metric_component"),
            "production_equivalent": task.get("production_equivalent"),
        }
    return out


def build_sweep(
    phase: str,
    m_value: int | None,
    s_value: int,
    smoke: bool,
    defaults: dict,
) -> list[dict]:
    if m_value is not None:
        m_list = [m_value]
    elif smoke:
        smoke_defaults = defaults.get("smoke") or {}
        key = "prefill_M" if phase == "prefill" else "decode_M"
        m_list = [smoke_defaults.get(key, 1024 if phase == "prefill" else 16)]
    elif phase == "prefill":
        m_list = defaults.get("prefill_M", [1024, 2048, 4096])
    else:
        m_list = defaults.get("decode_M", [1, 4, 8, 16, 32, 64])
    return [{"M": m, "S": s_value} for m in m_list]


if __name__ == "__main__":
    raise SystemExit(main())
