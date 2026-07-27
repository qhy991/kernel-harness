#!/usr/bin/env python3
"""Untimed score/top-k correctness and CUDA-graph validation."""

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

from serving_native.runner import (
    Runtime,
    _capture_cuda_graph,
    _clone_observed,
)
from serving_native.workloads import get_workload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    rows = []
    all_passed = True
    for batch in (16, 32):
        runtime = Runtime(get_workload(f"indexer_score_decode_m{batch}"))
        try:
            inputs = runtime.build_inputs()
            reference = runtime.run_indexer_score_topk(inputs, backend="deepgemm")
            candidate = runtime.run_indexer_score_topk(inputs, backend="cutedsl")
            runtime.torch.cuda.synchronize(runtime.device)

            reference_graph, _ = _capture_cuda_graph(
                runtime,
                lambda: runtime.run_indexer_score_topk(
                    inputs, backend="deepgemm"
                ),
            )
            candidate_graph, _ = _capture_cuda_graph(
                runtime,
                lambda: runtime.run_indexer_score_topk(inputs, backend="cutedsl"),
            )
            reference_replay = reference_graph()
            candidate_replay = candidate_graph()
            runtime.torch.cuda.synchronize(runtime.device)
            reference_snapshot = _clone_observed(reference_replay.observed)
            candidate_snapshot = _clone_observed(candidate_replay.observed)

            ref_logits = reference_snapshot["logits"]
            cand_logits = candidate_snapshot["logits"]
            logits_close = bool(
                runtime.torch.allclose(
                    ref_logits.float(),
                    cand_logits.float(),
                    rtol=2e-2,
                    atol=2e-2,
                    equal_nan=False,
                )
            )
            ref_topk = reference_snapshot["topk_indices"]
            cand_topk = candidate_snapshot["topk_indices"]
            topk_exact = bool(runtime.torch.equal(ref_topk, cand_topk))
            topk_set_exact = bool(
                runtime.torch.equal(
                    runtime.torch.sort(ref_topk, dim=1).values,
                    runtime.torch.sort(cand_topk, dim=1).values,
                )
            )
            overlap = runtime.torch.stack(
                [
                    runtime.torch.isin(ref_topk[row], cand_topk[row]).sum()
                    for row in range(batch)
                ]
            )
            row_pass = logits_close and topk_set_exact
            all_passed = all_passed and row_pass
            rows.append(
                {
                    "workload": f"indexer_score_decode_m{batch}",
                    "cuda_graph_replay_correctness": (
                        "pass" if row_pass else "fail"
                    ),
                    "logits_allclose_rtol_2e-2_atol_2e-2": logits_close,
                    "max_abs_logit_diff": float(
                        (ref_logits.float() - cand_logits.float()).abs().max().item()
                    ),
                    "topk_exact_order": topk_exact,
                    "topk_set_exact": topk_set_exact,
                    "topk_min_overlap_of_2048": int(overlap.min().item()),
                    "topk_rows_with_nonidentical_set": int(
                        (overlap != inputs["topk"]).sum().item()
                    ),
                    "runtime_contract": runtime.contract(inputs),
                }
            )
        finally:
            runtime.close()

    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    status = "pass" if all_passed else "fail"
    output.write_text(json.dumps({"status": status, "rows": rows}, indent=2) + "\n")
    print(output)
    return 0 if all_passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
