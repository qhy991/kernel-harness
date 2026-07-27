#!/usr/bin/env python3
"""Untimed diagnostics for score/top-k determinism and gather variants."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SGLANG_ROOT = Path(
    os.environ.get(
        "SGLANG_ROOT",
        "/home/qinhaiyan/glm52-goal-runs/17-indexer_score_prefill/sglang",
    )
)
sys.path.insert(0, str(SGLANG_ROOT / "python"))
sys.path.insert(0, str(ROOT))

import torch

from serving_native.candidates.indexer_score_gather_tuned import _gather
from serving_native.indexer_score_prefill import (
    balanced_budget_bytes,
    run as run_score,
)
from serving_native.runner import Runtime
from serving_native.workloads import get_workload
from sglang.srt.layers.attention.dsa.index_buf_accessor import GetKAndS


def mismatch(a: torch.Tensor, b: torch.Tensor) -> dict:
    unequal = a != b
    return {
        "elements": a.numel(),
        "mismatches": int(unequal.sum().item()),
        "fraction": float(unequal.float().mean().item()),
    }


def topk_mismatch(a: torch.Tensor, b: torch.Tensor) -> dict:
    result = mismatch(a, b)
    result["rowwise_set"] = mismatch(
        torch.sort(a, dim=-1).values,
        torch.sort(b, dim=-1).values,
    )
    return result


def main() -> int:
    runtime = Runtime(get_workload("indexer_score_prefill_m4096_c256"))
    try:
        inputs = runtime.build_inputs()
        fixture = inputs["fixture"]
        pool = fixture.forward_context.attn_backend.token_to_kv_pool
        metadata = fixture.metadata
        block_tables = metadata.get_page_table_64()
        seq_lens = metadata.get_indexer_seq_len()
        seq_len_sum = sum(fixture.seq_lens)
        max_seq_len = max(fixture.seq_lens)
        stock_k, stock_s = GetKAndS.execute(
            pool,
            pool.get_index_k_with_scale_buffer(0),
            page_indices=block_tables,
            seq_len_tensor=seq_lens,
            seq_len_sum=seq_len_sum,
            max_seq_len=max_seq_len,
        )
        result = {"gather": {}, "score_topk": {}}
        for block_size, num_warps in ((64, 4), (128, 4), (128, 8)):
            tuned_k, tuned_s = _gather(
                pool,
                pool.get_index_k_with_scale_buffer(0),
                page_indices=block_tables,
                seq_len_tensor=seq_lens,
                seq_len_sum=seq_len_sum,
                max_seq_len=max_seq_len,
                block_size=block_size,
                num_warps=num_warps,
            )
            torch.cuda.synchronize()
            result["gather"][f"b{block_size}_w{num_warps}"] = {
                "k": mismatch(stock_k, tuned_k),
                "scale_bytes": mismatch(stock_s, tuned_s),
            }

        stock_a = run_score(inputs).clone()
        stock_b = run_score(inputs).clone()
        result["score_topk"]["stock_repeat"] = topk_mismatch(stock_a, stock_b)

        budget = balanced_budget_bytes(fixture)
        assert budget is not None
        balanced = run_score(inputs, budget_override_bytes=budget).clone()
        result["score_topk"]["balanced_vs_stock"] = topk_mismatch(
            stock_a, balanced
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    finally:
        runtime.close()


if __name__ == "__main__":
    raise SystemExit(main())
