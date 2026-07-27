"""Swap only indexer.wq_b inside the exact fused dual-stream region."""

from __future__ import annotations


def run(inputs: dict, runtime):
    return runtime.indexer_fused_prepare_store(inputs, candidate_wq=True)
