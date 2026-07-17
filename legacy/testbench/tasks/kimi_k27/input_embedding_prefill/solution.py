"""Vocab embedding gather — sglang production baseline (F.embedding).

Kimi-K2.7 input embedding: the operation sglang dispatches on the unquantized
VocabParallelEmbedding path (python/sglang/srt/layers/vocab_parallel_embedding.py:518
-> F.embedding). Gathers rows of the embedding table by token id. `reference.py` IS
the correctness oracle AND the latency baseline: an optimized solution.py must match
this output within tolerance and run faster.

  out[M, H] = weight[input_ids]   (pure gather).  Bandwidth-bound.

Only the gather is timed; the (untimed) get_inputs allocates the vocab table and ids.
"""

import torch
import torch.nn.functional as F


def get_inputs(axes_and_scalars: dict, device: torch.device) -> dict:
    M = axes_and_scalars["M"]
    V = axes_and_scalars["V"]
    H = axes_and_scalars["H"]
    input_ids = torch.randint(0, V, (M,), device=device, dtype=torch.long)
    weight = torch.randn(V, H, device=device, dtype=torch.bfloat16)
    return {"input_ids": input_ids, "weight": weight}


@torch.no_grad()
def run(input_ids, weight):
    return F.embedding(input_ids, weight)
