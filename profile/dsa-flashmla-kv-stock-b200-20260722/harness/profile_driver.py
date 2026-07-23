#!/usr/bin/env python3
"""Launch exactly one production FlashMLA-KV region after optional warmups."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path

from gpu_lease_env import require_flexible_gpu


REPO_ROOT = Path(__file__).resolve().parents[3]
EXPECTED_SGLANG_ROOT = Path(
    "/home/qinhaiyan/glm52-goal-runs/22-dsa_flashmla_kv_production/sglang"
).resolve()
if Path(os.environ.get("SGLANG_ROOT", "")).resolve() != EXPECTED_SGLANG_ROOT:
    raise RuntimeError(f"SGLANG_ROOT must be the isolated checkout: {EXPECTED_SGLANG_ROOT}")
require_flexible_gpu()

sys.path.insert(0, str(REPO_ROOT))

from serving_native.runner import Runtime  # noqa: E402
from serving_native.workloads import get_workload  # noqa: E402


TASKS = ("dsa_flashmla_kv_decode_m16", "dsa_flashmla_kv_decode_m32")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=TASKS, required=True)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--candidate")
    parser.add_argument("--output")
    args = parser.parse_args()
    if args.warmup < 0:
        parser.error("--warmup must be non-negative")
    return args


def load_candidate(path: str | None):
    if path is None:
        return None
    resolved = Path(path).expanduser().resolve()
    spec = importlib.util.spec_from_file_location("goal22_profile_candidate", resolved)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import candidate: {resolved}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> int:
    args = parse_args()
    output = None if args.output is None else Path(args.output).expanduser().resolve()
    if output is not None and output.exists():
        raise RuntimeError(f"refusing to overwrite profiler evidence: {output}")
    candidate = load_candidate(args.candidate)
    runtime = Runtime(get_workload(args.task))
    try:
        inputs = runtime.build_inputs()
        invoke = (
            (lambda: runtime.reference(inputs))
            if candidate is None
            else (lambda: candidate.run(inputs, runtime))
        )
        for _ in range(args.warmup):
            invoke()
        runtime.torch.cuda.synchronize(runtime.device)

        label = "stock" if candidate is None else "candidate"
        runtime.torch.cuda.nvtx.range_push(f"goal22:{args.task}:{label}_region")
        result = invoke()
        runtime.torch.cuda.nvtx.range_pop()
        runtime.torch.cuda.synchronize(runtime.device)

        observed = result.observed if hasattr(result, "observed") else result

        evidence = runtime.runtime_evidence(inputs)
        assert evidence is not None
        evidence["task"] = args.task
        evidence["profile_label"] = label
        evidence["warmup"] = args.warmup
        evidence["campaign"] = {
            "campaign_id": os.environ.get("GOAL22_CAMPAIGN_ID"),
            "physical_gpu": int(os.environ["CUDA_VISIBLE_DEVICES"]),
            "logical_gpu": 0,
            "gpu_uuid": os.environ.get("GOAL22_GPU_UUID"),
        }
        evidence["output_shape"] = list(observed.shape)
        evidence["output_dtype"] = str(observed.dtype)
        evidence["output_checksum"] = float(observed.float().sum().item())
        if candidate is not None and callable(
            getattr(candidate, "candidate_evidence", None)
        ):
            evidence["candidate_evidence"] = candidate.candidate_evidence()
        rendered = json.dumps(evidence, indent=2, sort_keys=True) + "\n"
        if output is not None:
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(rendered)
        print(rendered, end="")
        return 0
    finally:
        runtime.close()


if __name__ == "__main__":
    raise SystemExit(main())
