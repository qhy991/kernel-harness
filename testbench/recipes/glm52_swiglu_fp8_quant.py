"""GLM-5.2 routed-expert SwiGLU + FP8 post-quant — SGLang production baseline.

Prefill (contiguous): sglang.jit_kernel.dsv4.silu_and_mul_contig_post_quant
Decode  (masked):     sglang.jit_kernel.dsv4.silu_and_mul_masked_post_quant
                      (falls back to ep_moe_kernels.silu_and_mul_masked_post_quant_fwd)

This matches the DeepGEMM MoE runner's fused act+quant launch that precedes the
Down projection — NOT a bare torch.silu proxy. Returns (fp8_act, ue8m0_scale).

Axes:
  M = token population (routing), E = local experts (8 under EP32), I2 = 2*moe_inter = 4096
  layout = 0 contiguous / 1 masked
"""

import torch

GROUP = 128


def _sample_counts(M, E, n_global, topk, device, seed):
    g = torch.Generator(device="cpu")
    g.manual_seed(seed)
    counts = torch.zeros(E, dtype=torch.int32)
    for _ in range(M):
        choice = torch.randperm(n_global, generator=g)[:topk]
        for e in choice[choice < E].tolist():
            counts[e] += 1
    return counts.to(device)


def _align(x, a):
    return (x + a - 1) // a * a


def get_inputs(axes_and_scalars: dict, device: torch.device) -> dict:
    M = int(axes_and_scalars["M"])
    E = int(axes_and_scalars["E"])
    I2 = int(axes_and_scalars["I2"])
    layout = int(axes_and_scalars.get("layout", 1))
    n_global = int(axes_and_scalars.get("n_global", 256))
    topk = int(axes_and_scalars.get("topk", 8))
    I = I2 // 2

    counts = _sample_counts(M, E, n_global, topk, device,
                            seed=0xA5A5 + M * 31 + layout)
    if layout == 0:
        total = int(counts.sum().item())
        if total < M:
            for i in range(M - total):
                counts[int(torch.argmin(counts).item())] += 1
        elif total > M:
            for _ in range(total - M):
                e = int(torch.argmax(counts).item())
                if counts[e] > 0:
                    counts[e] -= 1
        total = M
        gate_up = torch.randn(total, I2, device=device, dtype=torch.bfloat16)
        pieces = []
        for e in range(E):
            c = int(counts[e].item())
            if c:
                pieces.append(torch.full((c,), e, dtype=torch.int32, device=device))
        m_indices = torch.cat(pieces) if pieces else torch.zeros(
            M, dtype=torch.int32, device=device)
        masked_m = torch.empty(0, dtype=torch.int32, device=device)
        out_fp8 = torch.empty(total, I, device=device, dtype=torch.float8_e4m3fn)
        n_scale = max(I // 512, 1)
        out_scale = torch.empty(total, n_scale, device=device, dtype=torch.int32)
    else:
        max_c = max(int(counts.max().item()), 1)
        Mp = _align(max_c, 128)
        gate_up = torch.randn(E, Mp, I2, device=device, dtype=torch.bfloat16)
        m_indices = torch.empty(0, dtype=torch.int32, device=device)
        masked_m = counts
        out_fp8 = torch.empty(E, Mp, I, device=device, dtype=torch.float8_e4m3fn)
        n_scale = max(I // 128, 1)
        out_scale = torch.empty(E, Mp, n_scale, device=device, dtype=torch.float32)

    return {
        "gate_up": gate_up,
        "out_fp8": out_fp8,
        "out_scale": out_scale,
        "masked_m": masked_m,
        "m_indices": m_indices,
        "layout": torch.tensor(layout, dtype=torch.int32, device=device),
        "group_size": torch.tensor(GROUP, dtype=torch.int32, device=device),
    }


@torch.no_grad()
def run(gate_up, out_fp8, out_scale, masked_m, m_indices, layout, group_size):
    lay = int(layout.item() if isinstance(layout, torch.Tensor) else layout)
    gs = int(group_size.item() if isinstance(group_size, torch.Tensor) else group_size)

    if lay == 0:
        try:
            from sglang.jit_kernel.dsv4 import silu_and_mul_contig_post_quant
            silu_and_mul_contig_post_quant(
                input=gate_up,
                output=out_fp8,
                output_scale=out_scale,
                quant_group_size=gs,
                scale_ue8m0=True,
                transposed=True,
                swiglu_limit=None,
            )
        except Exception:
            # Portable fallback: silu_and_mul then per-token group quant (same math).
            from sgl_kernel import silu_and_mul
            from sglang.srt.layers.quantization.fp8_kernel import (
                sglang_per_token_group_quant_fp8,
            )
            mid = torch.empty(gate_up.shape[0], gate_up.shape[-1] // 2,
                              device=gate_up.device, dtype=gate_up.dtype)
            silu_and_mul(gate_up, mid)
            q, s = sglang_per_token_group_quant_fp8(
                mid, gs, column_major_scales=True,
                scale_tma_aligned=True, scale_ue8m0=True)
            out_fp8.copy_(q)
            # scale layout may differ slightly; clone into buffer when shapes match
            if s.shape == out_scale.shape:
                out_scale.copy_(s)
            else:
                out_scale = s
        return out_fp8, out_scale

    # Masked decode path
    try:
        from sglang.jit_kernel.dsv4 import silu_and_mul_masked_post_quant
        silu_and_mul_masked_post_quant(
            gate_up, out_fp8, out_scale, gs, masked_m,
            scale_ue8m0=False, topk=1, transposed=False, swiglu_limit=None,
        )
    except Exception:
        from sglang.kernels.ops.moe.ep_moe_kernels import (
            silu_and_mul_masked_post_quant_fwd,
        )
        silu_and_mul_masked_post_quant_fwd(
            gate_up, out_fp8, out_scale, gs, masked_m, scale_ue8m0=False,
        )
    return out_fp8, out_scale
