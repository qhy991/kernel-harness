"""Exact stock linears with the Indexer's supported single-stream schedule."""


CANDIDATE_METADATA = {
    "backend": "stock_bf16_single_stream_schedule",
    "shape_guard_intended": [4096, 160, 6144],
    "execution_mode": "eager_single_stream_trial",
    "delta": "Indexer._fused_q_prepare_and_store(enable_dual_stream=False)",
    "wk_backend": "stock ReplicatedLinear -> UnquantizedLinearMethod",
}


def run(inputs, runtime):
    return runtime.indexer_fused_prepare_store(
        inputs,
        enable_dual_stream=False,
    )
