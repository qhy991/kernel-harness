#!/usr/bin/env python3
"""Launch one warmed complete-indexer region inside a named NVTX range."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = next(
    parent
    for parent in Path(__file__).resolve().parents
    if (parent / "serving_native").is_dir()
)
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from serving_native.runner import Runtime
from serving_native.workloads import get_workload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--task",
        choices=(
            "indexer_complete_decode_m16",
            "indexer_complete_decode_m32",
            "indexer_dsa_decode_m16",
            "indexer_dsa_decode_m32",
        ),
        required=True,
    )
    parser.add_argument(
        "--backend", choices=("deepgemm", "cutedsl"), required=True
    )
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--contract-output")
    args = parser.parse_args()

    runtime = Runtime(get_workload(args.task))
    try:
        inputs = runtime.build_inputs()
        for _ in range(args.warmup):
            runtime.prepare_inputs(inputs)
            runtime.run_indexer_containing_region(
                inputs, backend=args.backend
            )
        runtime.torch.cuda.synchronize(runtime.device)

        range_name = f"{args.task}_{args.backend}"
        runtime.prepare_inputs(inputs)
        runtime.torch.cuda.cudart().cudaProfilerStart()
        runtime.torch.cuda.nvtx.range_push(range_name)
        result = runtime.run_indexer_containing_region(
            inputs, backend=args.backend
        )
        runtime.torch.cuda.nvtx.range_pop()
        runtime.torch.cuda.cudart().cudaProfilerStop()
        runtime.torch.cuda.synchronize(runtime.device)

        if args.contract_output:
            output = Path(args.contract_output).expanduser().resolve()
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(
                json.dumps(
                    {
                        "range": range_name,
                        "backend": args.backend,
                        "task": args.task,
                        "runtime_contract": runtime.contract(inputs),
                        "observed": {
                            key: {
                                "shape": list(value.shape),
                                "dtype": str(value.dtype),
                                "stride": list(value.stride()),
                            }
                            for key, value in result.observed.items()
                        },
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
