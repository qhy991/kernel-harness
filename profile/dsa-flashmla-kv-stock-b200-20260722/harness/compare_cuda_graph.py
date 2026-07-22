#!/usr/bin/env python3
"""Paired real-CUDA-Graph comparison of stock and an overlay candidate."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import statistics
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
EXPECTED_SGLANG_ROOT = Path(
    "/home/qinhaiyan/glm52-goal-runs/22-dsa_flashmla_kv_production/sglang"
).resolve()
if Path(os.environ.get("SGLANG_ROOT", "")).resolve() != EXPECTED_SGLANG_ROOT:
    raise RuntimeError(f"SGLANG_ROOT must be the isolated checkout: {EXPECTED_SGLANG_ROOT}")
if os.environ.get("CUDA_VISIBLE_DEVICES") != "3":
    raise RuntimeError(
        "run through /home/qinhaiyan/glm52-goal-runs/with_gpu_lock.sh 3"
    )
sys.path.insert(0, str(REPO_ROOT))

from serving_native.runner import Runtime, TaskResult, _compare  # noqa: E402
from serving_native.workloads import get_workload  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeat", type=int, default=50)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def load_candidate(path: str):
    resolved = Path(path).expanduser().resolve()
    spec = importlib.util.spec_from_file_location("goal22_graph_candidate", resolved)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {resolved}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def summary(samples):
    ordered = sorted(samples)
    return {
        "samples_ms": samples,
        "median_ms": statistics.median(samples),
        "min_ms": ordered[0],
        "max_ms": ordered[-1],
        "p95_ms": ordered[min(len(ordered) - 1, int(0.95 * len(ordered)))],
    }


def main() -> int:
    args = parse_args()
    if args.warmup < 0 or args.repeat < 1:
        raise ValueError("--warmup must be >= 0 and --repeat must be >= 1")
    output = Path(args.output).expanduser().resolve()
    if output.exists():
        raise RuntimeError(f"refusing to overwrite graph evidence: {output}")
    candidate = load_candidate(args.candidate)
    runtime = Runtime(get_workload(args.task))
    torch = runtime.torch
    try:
        inputs = runtime.build_inputs()
        reference_once = runtime.reference(inputs).observed
        candidate_once = candidate.run(inputs, runtime)
        torch.cuda.synchronize(runtime.device)
        _compare(TaskResult(reference_once), TaskResult(candidate_once))

        side_stream = torch.cuda.Stream(device=runtime.device)
        side_stream.wait_stream(torch.cuda.current_stream(runtime.device))
        with torch.cuda.stream(side_stream):
            for _ in range(args.warmup):
                runtime.reference(inputs)
                candidate.run(inputs, runtime)
        torch.cuda.current_stream(runtime.device).wait_stream(side_stream)
        torch.cuda.synchronize(runtime.device)

        reference_graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(reference_graph):
            reference_output = runtime.reference(inputs).observed
        candidate_graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(candidate_graph):
            candidate_output = candidate.run(inputs, runtime)
        if reference_output.data_ptr() == candidate_output.data_ptr():
            raise AssertionError("reference and candidate graph outputs alias")
        reference_graph.replay()
        candidate_graph.replay()
        torch.cuda.synchronize(runtime.device)
        _compare(TaskResult(reference_output), TaskResult(candidate_output))

        def one(graph):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            graph.replay()
            end.record()
            end.synchronize()
            return start.elapsed_time(end)

        reference_samples = []
        candidate_samples = []
        paired = []
        for index in range(args.repeat):
            if index % 2 == 0:
                order = ["reference", "candidate"]
                ref_ms = one(reference_graph)
                cand_ms = one(candidate_graph)
            else:
                order = ["candidate", "reference"]
                cand_ms = one(candidate_graph)
                ref_ms = one(reference_graph)
            reference_samples.append(ref_ms)
            candidate_samples.append(cand_ms)
            paired.append(
                {
                    "pair": index,
                    "order": order,
                    "reference_ms": ref_ms,
                    "candidate_ms": cand_ms,
                    "speedup": ref_ms / cand_ms,
                }
            )

        reference_before_mutation = reference_output.clone()
        candidate_before_mutation = candidate_output.clone()
        original_q = inputs["q"].clone()
        original_indices = inputs["indices"].clone()
        batch = runtime.workload.params["batch"]
        context = runtime.workload.params["context"]
        page_size = runtime.workload.params["page_size"]
        bases = page_size + torch.arange(
            batch, dtype=torch.int32, device=runtime.device
        ) * context
        inputs["q"].copy_(original_q.mul(-0.75))
        local_indices = original_indices - bases[:, None, None]
        inputs["indices"].copy_(
            bases[:, None, None] + (local_indices + 7919) % context
        )
        reference_graph.replay()
        candidate_graph.replay()
        torch.cuda.synchronize(runtime.device)
        _compare(TaskResult(reference_output), TaskResult(candidate_output))
        reference_mutation_change = float(
            (reference_output.float() - reference_before_mutation.float())
            .abs()
            .max()
            .item()
        )
        candidate_mutation_change = float(
            (candidate_output.float() - candidate_before_mutation.float())
            .abs()
            .max()
            .item()
        )
        if reference_mutation_change <= 1e-3:
            raise AssertionError("mutated reference graph replay left a stale output")
        if candidate_mutation_change <= 1e-3:
            raise AssertionError("mutated candidate graph replay left a stale output")

        paired_speedups = [item["speedup"] for item in paired]
        result = {
            "task": args.task,
            "mode": "real_cuda_graph_replay",
            "warmup": args.warmup,
            "repeat": args.repeat,
            "runtime_evidence": runtime.runtime_evidence(inputs),
            "candidate_evidence": candidate.candidate_evidence(),
            "reference": summary(reference_samples),
            "candidate": summary(candidate_samples),
            "paired_samples": paired,
            "paired_median_speedup": statistics.median(paired_speedups),
            "passes_3pct_gate": statistics.median(paired_speedups) >= 1.03,
            "correctness": {
                "initial_exact_dtype_and_tolerance": True,
                "mutated_inputs_match": True,
                "mutated_reference_change_max_abs": reference_mutation_change,
                "mutated_candidate_change_max_abs": candidate_mutation_change,
                "reference_output_data_ptr": reference_output.data_ptr(),
                "candidate_output_data_ptr": candidate_output.data_ptr(),
                "outputs_alias": False,
            },
        }
        rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered)
        print(rendered, end="")
        return 0
    finally:
        runtime.close()


if __name__ == "__main__":
    raise SystemExit(main())
