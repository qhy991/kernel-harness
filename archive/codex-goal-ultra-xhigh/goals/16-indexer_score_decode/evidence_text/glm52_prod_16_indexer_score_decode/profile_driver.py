#!/usr/bin/env python3
"""Launch one warmed indexer score/top-k region inside a named NVTX range."""

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
    parser.add_argument("--batch", type=int, choices=(16, 32), required=True)
    parser.add_argument(
        "--backend", choices=("deepgemm", "cutedsl"), required=True
    )
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--contract-output")
    args = parser.parse_args()

    runtime = Runtime(get_workload(f"indexer_score_decode_m{args.batch}"))
    try:
        inputs = runtime.build_inputs()
        for _ in range(args.warmup):
            runtime.run_indexer_score_topk(inputs, backend=args.backend)
        runtime.torch.cuda.synchronize(runtime.device)

        range_name = f"indexer_score_topk_{args.backend}_m{args.batch}"
        runtime.torch.cuda.cudart().cudaProfilerStart()
        runtime.torch.cuda.nvtx.range_push(range_name)
        result = runtime.run_indexer_score_topk(inputs, backend=args.backend)
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
                        "batch": args.batch,
                        "runtime_contract": runtime.contract(inputs),
                        "logits": {
                            "shape": list(result.observed["logits"].shape),
                            "dtype": str(result.observed["logits"].dtype),
                            "stride": list(result.observed["logits"].stride()),
                        },
                        "topk_indices": {
                            "shape": list(result.observed["topk_indices"].shape),
                            "dtype": str(result.observed["topk_indices"].dtype),
                            "stride": list(
                                result.observed["topk_indices"].stride()
                            ),
                        },
                    },
                    indent=2,
                )
                + "\n"
            )
        return 0
    finally:
        runtime.close()


if __name__ == "__main__":
    raise SystemExit(main())
