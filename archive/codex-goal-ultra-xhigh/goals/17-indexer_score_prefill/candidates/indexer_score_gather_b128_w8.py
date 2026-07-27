"""Gather 128 sequence tokens per Triton program using eight warps."""

from serving_native.candidates.indexer_score_gather_tuned import run_with_config


def run(inputs, runtime):
    return run_with_config(inputs, runtime, block_size=128, num_warps=8)


def describe():
    return {
        "attempt": "bitwise-preserving GetKAndS Triton launch",
        "block_size_tokens": 128,
        "block_size_k": 128,
        "num_warps": 8,
    }
