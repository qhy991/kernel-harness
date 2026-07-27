"""Balance two score chunks only for the measured mixed-context bucket.

The dispatch predicate uses host-resident request metadata and fails closed to
the stock production method. It does not read a CUDA tensor or synchronize.
"""

from serving_native.indexer_score_prefill import balanced_budget_bytes


_MIXED_EXTEND_LENS = (
    64,
    64,
    128,
    128,
    192,
    192,
    256,
    256,
    256,
    256,
    384,
    384,
    384,
    384,
    384,
    384,
)
_MIXED_SEQ_LENS = (
    2048,
    2048,
    4096,
    4096,
    8192,
    8192,
    8192,
    8192,
    16384,
    16384,
    16384,
    16384,
    32768,
    32768,
    32768,
    32768,
)


def _enabled(fixture) -> bool:
    return (
        fixture.q_offset == 4096
        and fixture.k_offset == 241664
        and fixture.batch_size == 16
        and tuple(fixture.extend_lens) == _MIXED_EXTEND_LENS
        and tuple(fixture.seq_lens) == _MIXED_SEQ_LENS
        and fixture.stock_chunk_rows == [3169, 927]
    )


def run(inputs, runtime):
    fixture = inputs["fixture"]
    if not _enabled(fixture):
        return runtime.reference(inputs)

    budget = balanced_budget_bytes(fixture)
    if budget is None:
        return runtime.reference(inputs)
    if runtime.workload.family in (
        "indexer_complete_prefill",
        "indexer_dsa_prefill",
    ):
        return runtime.run_indexer_prefill_region(
            inputs,
            budget_override_bytes=budget,
        )
    if runtime.workload.family == "indexer_graph_split_prefill":
        return inputs["region"].run_graph_split(
            budget_override_bytes=budget,
        )
    return {
        "topk_indices": runtime.run_indexer_score_prefill(
            inputs,
            budget_override_bytes=budget,
        )
    }


def describe():
    return {
        "policy": (
            "enable only local M4096, K241664, batch16, exact fixed mixed "
            "context/extend distribution, stock chunks [3169,927]"
        ),
        "enabled_chunk_rows": [2048, 2048],
        "fallback": "stock Indexer._get_topk_ragged for every other bucket",
        "dispatch_reads_device_state": False,
    }
