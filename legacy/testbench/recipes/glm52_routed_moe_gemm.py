"""GLM-5.2 routed-expert grouped GEMM — DeepGEMM baseline with realistic routing.

Used by the llm_flops-aligned separate MoE projections:
  moe_gate_proj_*: K=6144, N=2048
  moe_up_proj_*:   K=6144, N=2048
  moe_down_proj_*: K=2048, N=6144

llm_flops (and these tasks) use masked for BOTH prefill and decode:
  deep_gemm.fp8_m_grouped_gemm_nt_masked

Routing (fixed seed, EP-local filter):
  For each of M tokens draw top-8 experts uniformly from 256 global experts, keep
  only those belonging to this EP rank's local experts (E = n_routed/ep; EP32 → E=8).
  Local assignments ≈ M * (topk/n_global)*E on expectation, with natural empty
  experts and load jitter.
  Quant + scale layout prep is OFFLINE in get_inputs.

Axes:
  M = prefill tokens OR decode batch (sweep)
  E = local experts (8 under DP1/TP1/EP32)
  K, N = GEMM dims (Gate/Up: K=6144,N=2048; Down: K=2048,N=6144)
  layout = 0 contiguous / 1 masked   (const per task; MoE projs use 1)
"""

import torch
import deep_gemm
from deep_gemm.utils.math import align, per_block_cast_to_fp8, per_token_cast_to_fp8


def _sample_local_counts(M: int, E: int, n_global: int, topk: int,
                         device: torch.device, seed: int) -> torch.Tensor:
    """Return int32[E] local token counts from a top-k / n_global draw."""
    g = torch.Generator(device="cpu")
    g.manual_seed(seed)
    # Global expert ids for this EP rank: [0, E) locally == global [rank*E, (rank+1)*E).
    # We simulate rank 0 without loss of generality.
    counts = torch.zeros(E, dtype=torch.int32)
    # Vectorized: for each token, sample `topk` distinct global experts.
    # Rejection via randperm rows kept small for harness sizes (M<=4096, topk=8).
    for _ in range(M):
        choice = torch.randperm(n_global, generator=g)[:topk]
        local = choice[choice < E]
        for e in local.tolist():
            counts[e] += 1
    return counts.to(device)


def _build_masked(axes, device):
    E = axes["E"]
    M = axes["M"]          # decode batch / token count (routing population)
    K, N = axes["K"], axes["N"]
    n_global = axes.get("n_global", 256)
    topk = axes.get("topk", 8)
    Ka, Na = align(K, 128), align(N, 128)

    counts = _sample_local_counts(M, E, n_global, topk, device,
                                  seed=0x314159 + M * 17 + E)
    # Pad capacity per expert = max(max_count, 1) aligned up to 128 (DeepGEMM).
    max_c = int(counts.max().item()) if counts.numel() else 1
    Mp = align(max(max_c, 1), 128)

    a = torch.randn(E, Mp, Ka, device=device, dtype=torch.bfloat16)
    b = torch.randn(E, Na, Ka, device=device, dtype=torch.bfloat16) * (Ka ** -0.5)
    a_fp8, a_s = zip(*[per_token_cast_to_fp8(a[e], use_ue8m0=True) for e in range(E)])
    b_fp8, b_s = zip(*[per_block_cast_to_fp8(b[e], use_ue8m0=True) for e in range(E)])
    a_fp8, a_s = torch.stack(list(a_fp8)), torch.stack(list(a_s))
    b_fp8, b_s = torch.stack(list(b_fp8)), torch.stack(list(b_s))
    a_s = deep_gemm.transform_sf_into_required_layout(
        a_s, mn=Mp, k=Ka, recipe=(1, 128, 128), num_groups=E, is_sfa=True)
    b_s = deep_gemm.transform_sf_into_required_layout(
        b_s, mn=Na, k=Ka, recipe=(1, 128, 128), num_groups=E, is_sfa=False)
    out = torch.empty(E, Mp, Na, device=device, dtype=torch.bfloat16)
    # expected_m ≈ mean local load (deep_gemm hint); use max for safety
    expected_m = max(max_c, 1)
    return {
        "a_fp8": a_fp8, "a_s": a_s, "b_fp8": b_fp8, "b_s": b_s,
        "out": out, "masked_m": counts, "expected_m": expected_m,
        "m_indices": torch.empty(0, dtype=torch.int32, device=device),  # unused
        "layout": 1,
    }


def _build_contig(axes, device):
    E = axes["E"]
    M = axes["M"]
    K, N = axes["K"], axes["N"]
    n_global = axes.get("n_global", 256)
    topk = axes.get("topk", 8)
    Ka, Na = align(K, 128), align(N, 128)

    counts = _sample_local_counts(M, E, n_global, topk, device,
                                  seed=0x271828 + M * 13 + E)
    # Pin row count to exactly M so definition shapes stay stable while still
    # reflecting EP-local top-k imbalance (clip or top-up the sampled counts).
    total = int(counts.sum().item())
    if total < M:
        extra = M - total
        # Sprinkle leftovers onto the least-loaded experts.
        for i in range(extra):
            e = int(torch.argmin(counts).item())
            counts[e] += 1
    elif total > M:
        # Trim from the most-loaded experts.
        over = total - M
        for _ in range(over):
            e = int(torch.argmax(counts).item())
            if counts[e] > 0:
                counts[e] -= 1
    total = M

    a = torch.randn(total, Ka, device=device, dtype=torch.bfloat16)
    b = torch.randn(E, Na, Ka, device=device, dtype=torch.bfloat16) * (Ka ** -0.5)
    a_fp8, a_s = per_token_cast_to_fp8(a, use_ue8m0=True)
    b_fp8, b_s = zip(*[per_block_cast_to_fp8(b[e], use_ue8m0=True) for e in range(E)])
    b_fp8, b_s = torch.stack(list(b_fp8)), torch.stack(list(b_s))
    out = torch.empty(total, Na, device=device, dtype=torch.bfloat16)
    pieces = []
    for e in range(E):
        c = int(counts[e].item())
        if c:
            pieces.append(torch.full((c,), e, dtype=torch.int32, device=device))
    m_indices = torch.cat(pieces) if pieces else torch.zeros(
        M, dtype=torch.int32, device=device)
    assert m_indices.numel() == M
    return {
        "a_fp8": a_fp8, "a_s": a_s, "b_fp8": b_fp8, "b_s": b_s,
        "out": out,
        "masked_m": torch.empty(0, dtype=torch.int32, device=device),
        "expected_m": 0,
        "m_indices": m_indices,
        "layout": 0,
    }


def get_inputs(axes_and_scalars: dict, device: torch.device) -> dict:
    layout = int(axes_and_scalars.get("layout", 1))
    if layout == 0:
        d = _build_contig(axes_and_scalars, device)
    else:
        d = _build_masked(axes_and_scalars, device)
    # Scalar layout as 0-dim tensor for clone_args friendliness.
    d["layout"] = torch.tensor(layout, dtype=torch.int32, device=device)
    d["expected_m"] = torch.tensor(int(d["expected_m"]), dtype=torch.int32,
                                   device=device)
    return d


@torch.no_grad()
def run(a_fp8, a_s, b_fp8, b_s, out, masked_m, expected_m, m_indices, layout):
    lay = int(layout.item() if isinstance(layout, torch.Tensor) else layout)
    if lay == 0:
        deep_gemm.m_grouped_fp8_gemm_nt_contiguous(
            (a_fp8, a_s), (b_fp8, b_s), out, m_indices)
    else:
        em = int(expected_m.item() if isinstance(expected_m, torch.Tensor)
                 else expected_m)
        deep_gemm.fp8_m_grouped_gemm_nt_masked(
            (a_fp8, a_s), (b_fp8, b_s), out, masked_m, em)
    return out
