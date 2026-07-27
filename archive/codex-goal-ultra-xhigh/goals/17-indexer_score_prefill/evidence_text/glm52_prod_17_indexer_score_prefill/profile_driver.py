#!/usr/bin/env python3
"""Profile exactly one initialized indexer score/region invocation."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
SGLANG_ROOT = Path(
    os.environ.get(
        "SGLANG_ROOT",
        "/home/qinhaiyan/glm52-goal-runs/17-indexer_score_prefill/sglang",
    )
).resolve()
sys.path.insert(0, str(SGLANG_ROOT / "python"))
sys.path.insert(0, str(REPO_ROOT))

from serving_native.indexer_score_prefill import balanced_budget_bytes
from serving_native.runner import Runtime, TaskResult, _clone_observed, _compare
from serving_native.workloads import get_workload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True)
    parser.add_argument("--variant", choices=("stock", "balanced"), default="stock")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument(
        "--no-profiler-api",
        action="store_true",
        help="Do not call cudaProfilerStart/Stop (for ordinary smoke runs).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    workload = get_workload(args.task)
    if workload.family not in {
        "indexer_score_prefill",
        "indexer_complete_prefill",
        "indexer_dsa_prefill",
        "indexer_graph_split_prefill",
    }:
        raise ValueError(f"unsupported profile family: {workload.family}")

    runtime = Runtime(workload)
    try:
        inputs = runtime.build_inputs()
        budget = (
            balanced_budget_bytes(inputs["fixture"])
            if args.variant == "balanced"
            else None
        )

        def invoke() -> TaskResult:
            if workload.family == "indexer_score_prefill":
                return TaskResult(
                    {
                        "topk_indices": runtime.run_indexer_score_prefill(
                            inputs, budget_override_bytes=budget
                        )
                    }
                )
            if workload.family == "indexer_graph_split_prefill":
                return TaskResult(
                    inputs["region"].run_graph_split(
                        budget_override_bytes=budget
                    )
                )
            return TaskResult(
                runtime.run_indexer_prefill_region(
                    inputs, budget_override_bytes=budget
                )
            )

        # Validate the experiment against stock before entering a profiler
        # range. The snapshot clone keeps mutable/cache outputs honest.
        runtime.prepare_inputs(inputs)
        reference = runtime.reference(inputs)
        runtime.torch.cuda.synchronize(runtime.device)
        snapshot = TaskResult(_clone_observed(reference.observed))
        runtime.prepare_inputs(inputs)
        candidate = invoke()
        runtime.torch.cuda.synchronize(runtime.device)
        _compare(snapshot, candidate)

        for _ in range(args.warmup):
            runtime.prepare_inputs(inputs)
            invoke()
            runtime.torch.cuda.synchronize(runtime.device)

        runtime.prepare_inputs(inputs)
        cudart = runtime.torch.cuda.cudart()
        if not args.no_profiler_api:
            cudart.cudaProfilerStart()
        range_name = f"indexer_score_prefill/{args.task}/{args.variant}"
        runtime.torch.cuda.nvtx.range_push(range_name)
        observed = invoke()
        runtime.torch.cuda.nvtx.range_pop()
        runtime.torch.cuda.synchronize(runtime.device)
        if not args.no_profiler_api:
            cudart.cudaProfilerStop()

        leaf = observed.observed
        if isinstance(leaf, dict):
            leaf = leaf["topk_indices"]
        payload = {
            "task": args.task,
            "variant": args.variant,
            "correctness": "PASS",
            "output_shape": list(leaf.shape),
            "output_dtype": str(leaf.dtype),
            "runtime_metadata": runtime.result_metadata(inputs),
        }
        print(json.dumps(payload, indent=2, sort_keys=True), flush=True)
        return 0
    finally:
        runtime.close()


if __name__ == "__main__":
    raise SystemExit(main())
