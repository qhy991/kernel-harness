"""Balance the production memory-safe chunk count across local M=4096 rows."""

from serving_native.indexer_score_prefill import balanced_budget_bytes


def run(inputs, runtime):
    budget = balanced_budget_bytes(inputs["fixture"])
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
