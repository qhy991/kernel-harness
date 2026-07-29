#!/usr/bin/env python3
"""Profile one exact same-source W13 launch after deterministic cache warmup."""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
runner_module = importlib.import_module("serving_native.runner")
workloads_module = importlib.import_module("serving_native.workloads")
Runtime = runner_module.Runtime
_load_candidate = runner_module._load_candidate
get_workload = workloads_module.get_workload
WORKLOAD = "moe_w13_grouped_decode_m16_em4"
CANDIDATE = (
    ROOT
    / "serving_native"
    / "candidates"
    / "w13_bm32_2sm.py"
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--implementation", choices=("stock", "candidate"), required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    workload = get_workload(WORKLOAD)
    candidate_module = _load_candidate(str(CANDIDATE))
    runtime = Runtime(workload, candidate_module)
    try:
        inputs = runtime.build_inputs()
        runtime.prepare_inputs(inputs)
        torch.cuda.synchronize(runtime.device)

        launcher = (
            runtime.w13_runtime.stock_launcher
            if args.implementation == "stock"
            else runtime.w13_runtime.candidate_launcher
        )
        range_name = f"w13_profile_{args.implementation}"
        torch.cuda.nvtx.range_push(range_name)
        try:
            result = runtime.run_w13_leaf(inputs, launcher=launcher)
        finally:
            torch.cuda.nvtx.range_pop()
        torch.cuda.synchronize(runtime.device)

        output = Path(args.output).resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(
                {
                    "candidate": str(CANDIDATE),
                    "expected_m": inputs["expected_m"],
                    "implementation": args.implementation,
                    "manifest": runtime.w13_runtime.identity["manifest"],
                    "masked_m_cpu": list(inputs["masked_m_initial_cpu"]),
                    "output_pointer": int(result.observed.data_ptr()),
                    "runtime": runtime.w13_runtime.identity,
                    "stream": int(
                        torch.cuda.current_stream(runtime.device).cuda_stream
                    ),
                    "workload": WORKLOAD,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )
        return 0
    finally:
        runtime.close()


if __name__ == "__main__":
    raise SystemExit(main())
