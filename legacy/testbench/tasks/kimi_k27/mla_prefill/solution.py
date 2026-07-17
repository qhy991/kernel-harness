"""MLA prefill attention — sglang baseline (flashinfer ragged prefill).

Kimi-K2.7 core Multi-head Latent Attention on the prefill/extend path. sglang's
FlashInfer MLA backend runs the "normal MHA" prefill over the full q/k/v via
flashinfer BatchPrefillWithRaggedKVCacheWrapper (FA3/flash_attn_varlen is not built for
B200 sm100, so the ragged flashinfer path is the production one here).

  o[total_q, num_heads, v_head_dim] = ragged_prefill(q, k, v, causal)

Kimi-K2.7: num_heads=64, qk_head_dim=192 (qk_nope; rope concatenated upstream),
v_head_dim=128. Single sequence; seqlen = sweep. The wrapper .plan() (index/schedule
setup) is built untimed in get_inputs; only .run() (the attention kernel) is timed.
reference.py IS the correctness oracle AND the latency baseline.
"""

import torch
import flashinfer

NUM_HEADS, HD_QK, HD_V = 64, 192, 128


def get_inputs(axes_and_scalars: dict, device: torch.device) -> dict:
    L = axes_and_scalars["M"]      # prefill sequence length (sweep), single seq
    q = torch.randn(L, NUM_HEADS, HD_QK, device=device, dtype=torch.bfloat16)
    k = torch.randn(L, NUM_HEADS, HD_QK, device=device, dtype=torch.bfloat16)
    v = torch.randn(L, NUM_HEADS, HD_V, device=device, dtype=torch.bfloat16)
    indptr = torch.tensor([0, L], device=device, dtype=torch.int32)
    # Build + plan the wrapper here (untimed schedule setup, exactly as sglang's backend
    # does before .run()); pass it through so only .run() (the kernel) is timed.
    fb = torch.empty(256 * 1024 * 1024, dtype=torch.uint8, device=device)
    wrapper = flashinfer.BatchPrefillWithRaggedKVCacheWrapper(fb, kv_layout="NHD", backend="fa2")
    wrapper.plan(indptr, indptr, NUM_HEADS, NUM_HEADS, HD_QK, head_dim_vo=HD_V,
                 causal=True, q_data_type=torch.bfloat16, kv_data_type=torch.bfloat16)
    return {"q": q, "k": k, "v": v, "wrapper": wrapper}


@torch.no_grad()
def run(q, k, v, wrapper):
    return wrapper.run(q, k, v)
