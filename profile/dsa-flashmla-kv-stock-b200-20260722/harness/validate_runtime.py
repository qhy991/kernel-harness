#!/usr/bin/env python3
"""Validate exact FlashMLA-KV ABI, invalid indices, stream, and graph replay."""

from __future__ import annotations

import argparse
import json
import os
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

from serving_native.runner import Runtime  # noqa: E402
from serving_native.workloads import get_workload  # noqa: E402


TASKS = ("dsa_flashmla_kv_decode_m16", "dsa_flashmla_kv_decode_m32")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=TASKS, required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output = Path(args.output).expanduser().resolve()
    if output.exists():
        raise RuntimeError(f"refusing to overwrite runtime evidence: {output}")
    runtime = Runtime(get_workload(args.task))
    torch = runtime.torch
    try:
        inputs = runtime.build_inputs()
        eager = runtime.reference(inputs).observed
        torch.cuda.synchronize(runtime.device)
        eager_snapshot = eager.clone()
        expected_shape = (
            runtime.workload.params["batch"],
            1,
            runtime.workload.params["q_heads"],
            runtime.workload.params["v_head_dim"],
        )
        if tuple(eager.shape) != expected_shape or eager.dtype != torch.bfloat16:
            raise AssertionError(
                f"output ABI mismatch: {tuple(eager.shape)} {eager.dtype}, "
                f"expected {expected_shape} torch.bfloat16"
            )

        # SGLang emits -1 for invalid/padded sparse slots. Compare an
        # interspersed invalid layout with a second invocation containing only
        # the same valid physical indices.  Comparing two layouts that both
        # contain -1 would not prove masking: a broken kernel could map every
        # -1 to the same cache token and remain permutation invariant.
        # Do not manufacture positive OOB indices: the pinned SM100 source
        # masks -1 explicitly, while positive OOB TMA coordinates are not a
        # safe production input.
        invalid_mask = torch.zeros_like(inputs["indices"], dtype=torch.bool)
        invalid_mask[..., ::16] = True
        minus_one_inputs = dict(inputs)
        minus_one_inputs["indices"] = inputs["indices"].masked_fill(invalid_mask, -1)
        valid_per_row = (~invalid_mask).sum(dim=-1, keepdim=True)
        valid_topk = int(valid_per_row[0, 0].item())
        valid_only = torch.empty(
            (*inputs["indices"].shape[:-1], valid_topk),
            dtype=inputs["indices"].dtype,
            device=inputs["indices"].device,
        )
        for batch_idx in range(inputs["indices"].shape[0]):
            row = minus_one_inputs["indices"][batch_idx, 0]
            valid = row[row != -1]
            valid_only[batch_idx, 0].copy_(valid)
        valid_only_inputs = dict(inputs)
        valid_only_inputs["indices"] = valid_only.contiguous()
        valid_only_inputs["cache_seqlens"] = torch.full_like(
            inputs["cache_seqlens"], valid_topk
        )
        from sgl_kernel.flash_mla import get_mla_metadata

        valid_metadata, valid_num_splits = get_mla_metadata(
            cache_seqlens=valid_only_inputs["cache_seqlens"],
            num_q_tokens_per_head_k=runtime.workload.params["q_heads"],
            num_heads_k=1,
            num_heads_q=runtime.workload.params["q_heads"],
            is_fp8_kvcache=True,
            topk=valid_topk,
        )
        valid_only_inputs["tile_scheduler_metadata"] = valid_metadata
        valid_only_inputs["num_splits"] = valid_num_splits
        invalid_minus_one = runtime.reference(minus_one_inputs).observed
        valid_only_output = runtime.reference(valid_only_inputs).observed
        torch.cuda.synchronize(runtime.device)
        invalid_max_abs = float(
            (invalid_minus_one.float() - valid_only_output.float()).abs().max().item()
        )
        torch.testing.assert_close(
            invalid_minus_one, valid_only_output, rtol=2e-2, atol=2e-2
        )

        # The registered op must honor the current stream without a hidden
        # default-stream dependency or host synchronization.
        stream = torch.cuda.Stream(device=runtime.device)
        with torch.cuda.stream(stream):
            stream_output = runtime.reference(inputs).observed
            stream_done = torch.cuda.Event()
            stream_done.record(stream)
        torch.cuda.current_stream(runtime.device).wait_event(stream_done)
        torch.testing.assert_close(stream_output, eager_snapshot, rtol=0.0, atol=0.0)

        # Warm allocations on a side stream, then capture the complete main +
        # combine region. Both replays must reproduce eager output exactly.
        capture_stream = torch.cuda.Stream(device=runtime.device)
        capture_stream.wait_stream(torch.cuda.current_stream(runtime.device))
        with torch.cuda.stream(capture_stream):
            runtime.reference(inputs)
        torch.cuda.current_stream(runtime.device).wait_stream(capture_stream)
        torch.cuda.synchronize(runtime.device)

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            graph_output = runtime.reference(inputs).observed
        graph.replay()
        torch.cuda.synchronize(runtime.device)
        replay_one = graph_output.clone()
        graph.replay()
        torch.cuda.synchronize(runtime.device)
        replay_two = graph_output.clone()
        torch.testing.assert_close(replay_one, eager_snapshot, rtol=0.0, atol=0.0)
        torch.testing.assert_close(replay_two, eager_snapshot, rtol=0.0, atol=0.0)

        # Mutate both static inputs before a third replay, then compare with a
        # fresh eager invocation on the mutated tensors.  This rules out a
        # stale/no-op graph that merely leaves the first output in place.
        original_q = inputs["q"].clone()
        original_indices = inputs["indices"].clone()
        batch = runtime.workload.params["batch"]
        context = runtime.workload.params["context"]
        page_size = runtime.workload.params["page_size"]
        bases = page_size + torch.arange(
            batch,
            dtype=torch.int32,
            device=runtime.device,
        ) * context
        local_indices = original_indices - bases[:, None, None]
        alternate_indices = bases[:, None, None] + (local_indices + 7919) % context
        inputs["q"].copy_(original_q.mul(-0.75))
        inputs["indices"].copy_(alternate_indices)
        graph.replay()
        torch.cuda.synchronize(runtime.device)
        changed_replay = graph_output.clone()
        changed_eager = runtime.reference(inputs).observed
        torch.cuda.synchronize(runtime.device)
        torch.testing.assert_close(changed_replay, changed_eager, rtol=2e-2, atol=2e-2)
        changed_max_abs = float(
            (changed_replay.float() - eager_snapshot.float()).abs().max().item()
        )
        if changed_max_abs <= 1e-3:
            raise AssertionError(
                f"mutated graph replay did not change output: max_abs={changed_max_abs}"
            )

        evidence = runtime.runtime_evidence(inputs)
        assert evidence is not None
        evidence.update(
            {
                "task": args.task,
                "output_shape": list(eager.shape),
                "output_dtype": str(eager.dtype),
                "invalid_indices": {
                    "encoding": "-1",
                    "interspersed_matches_valid_only": True,
                    "masked_slots_per_request": int(invalid_mask[0].sum().item()),
                    "valid_slots_per_request": valid_topk,
                    "valid_only_topk": valid_topk,
                    "max_abs_diff": invalid_max_abs,
                    "rtol": 2e-2,
                    "atol": 2e-2,
                },
                "nondefault_stream": {"matches_eager_exactly": True},
                "cuda_graph": {
                    "capture_succeeded": True,
                    "replays": 3,
                    "unchanged_replays_match_eager_exactly": True,
                    "mutated_replay_matches_mutated_eager": True,
                    "mutated_replay_change_max_abs": changed_max_abs,
                    "mutated_rtol": 2e-2,
                    "mutated_atol": 2e-2,
                },
            }
        )
        rendered = json.dumps(evidence, indent=2, sort_keys=True) + "\n"
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered)
        print(rendered, end="")
        return 0
    finally:
        runtime.close()


if __name__ == "__main__":
    raise SystemExit(main())
