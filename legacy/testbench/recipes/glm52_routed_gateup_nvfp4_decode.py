"""GLM-5.2 NVFP4 routed MoE decode through FlashInfer TRT-LLM.

This is the ModelOpt-FP4 production contract used by SGLang for
``nvidia/GLM-5.2-NVFP4 --quantization modelopt_fp4 --moe-runner-backend
flashinfer_trtllm``. Activations and expert weights are packed NVFP4 payloads
stored as uint8, with FP8 E4M3 block scales and FP32 scalar/global scales.

The exposed FlashInfer TRT-LLM primitive is fused MoE. The task targets the
Gate+Up decode stage by using the GLM Gate+Up dimensions and scale contract, but
the reference runs through the same routed FP4 MoE runner that SGLang calls.
"""

import torch


GROUP_SIZE = 16
ROUTING_METHOD_DEEPSEEK_V3 = 2
ACTIVATION_SWIGLU = 3


def _next_power_of_2(x: int) -> int:
    return 1 << (max(int(x), 1) - 1).bit_length()


def _sample_topk_ids(M: int, E: int, topk: int, device: torch.device) -> torch.Tensor:
    g = torch.Generator(device="cpu")
    g.manual_seed(0x524F5554 + M * 17 + E)
    rows = []
    for _ in range(M):
        rows.append(torch.randperm(E, generator=g)[:topk])
    return torch.stack(rows).to(device=device, dtype=torch.int32)


def _rand_uint8(shape, device, seed: int) -> torch.Tensor:
    g = torch.Generator(device=device)
    g.manual_seed(seed)
    return torch.randint(0, 256, shape, device=device, dtype=torch.uint8, generator=g)


def _ones_fp8(shape, device) -> torch.Tensor:
    return torch.ones(shape, device=device, dtype=torch.float8_e4m3fn)


def _prepare_trtllm_weights(E: int, H: int, I: int, device: torch.device):
    from sglang.srt.layers.quantization.utils import (
        prepare_static_weights_for_trtllm_fp4_moe,
        reorder_w1w3_to_w3w1,
    )

    w13 = _rand_uint8((E, 2 * I, H // 2), device, seed=0xA130 + H + I)
    w2 = _rand_uint8((E, H, I // 2), device, seed=0xA200 + H + I)
    w13_scale = _ones_fp8((E, 2 * I, H // GROUP_SIZE), device)
    w2_scale = _ones_fp8((E, H, I // GROUP_SIZE), device)

    w13, w13_scale = reorder_w1w3_to_w3w1(w13, w13_scale, dim=-2)
    return prepare_static_weights_for_trtllm_fp4_moe(
        w13, w2, w13_scale, w2_scale, H, I, E, is_gated=True
    )


def get_inputs(axes_and_scalars: dict, device: torch.device) -> dict:
    from sglang.srt.layers.quantization.fp4_utils import fp4_quantize

    M = int(axes_and_scalars["M"])
    E = int(axes_and_scalars["E"])
    H = int(axes_and_scalars["H"])
    I = int(axes_and_scalars["I"])
    topk = int(axes_and_scalars.get("topk", 8))
    n_global = int(axes_and_scalars.get("n_global", 256))

    g = torch.Generator(device=device)
    g.manual_seed(0x474C4D52 + M)
    hidden_bf16 = 0.1 * torch.randn(M, H, device=device, dtype=torch.bfloat16,
                                   generator=g)

    input_global_scale = torch.ones(1, dtype=torch.float32, device=device)
    hs_bytes, hs_sf_bytes = fp4_quantize(
        hidden_bf16,
        input_global_scale,
        GROUP_SIZE,
        False,
        False,
    )
    hidden_states = hs_bytes.reshape(M, H // 2)
    hidden_states_scale = hs_sf_bytes.view(torch.float8_e4m3fn).reshape(
        *hs_sf_bytes.shape[:-1], -1
    )

    w13, w13_scale, w2, w2_scale = _prepare_trtllm_weights(E, H, I, device)
    topk_ids = _sample_topk_ids(M, E, topk, device)
    topk_weights = torch.full(
        (M, topk), 1.0 / topk, dtype=torch.bfloat16, device=device
    )
    ones_e = torch.ones(E, dtype=torch.float32, device=device)

    return {
        "topk_ids": topk_ids,
        "topk_weights": topk_weights,
        "hidden_states": hidden_states,
        "hidden_states_scale": hidden_states_scale,
        "gemm1_weights": w13,
        "gemm1_weights_scale": w13_scale.view(torch.float8_e4m3fn),
        "gemm2_weights": w2,
        "gemm2_weights_scale": w2_scale.view(torch.float8_e4m3fn),
        "output": torch.empty(M, H, dtype=torch.bfloat16, device=device),
        "output1_scale_scalar": ones_e.clone(),
        "output1_scale_gate_scalar": ones_e.clone(),
        "output2_scale_scalar": ones_e.clone(),
        "num_experts": torch.tensor(n_global, dtype=torch.int32, device=device),
        "top_k": torch.tensor(topk, dtype=torch.int32, device=device),
        "local_expert_offset": torch.tensor(0, dtype=torch.int32, device=device),
        "local_num_experts": torch.tensor(E, dtype=torch.int32, device=device),
        "intermediate_size": torch.tensor(I, dtype=torch.int32, device=device),
        "routed_scaling_factor": torch.tensor(2.5, dtype=torch.float32, device=device),
    }


@torch.no_grad()
def run(topk_ids, topk_weights, hidden_states, hidden_states_scale,
        gemm1_weights, gemm1_weights_scale, gemm2_weights, gemm2_weights_scale,
        output, output1_scale_scalar, output1_scale_gate_scalar,
        output2_scale_scalar, num_experts, top_k, local_expert_offset,
        local_num_experts, intermediate_size, routed_scaling_factor):
    from flashinfer import trtllm_fp4_block_scale_routed_moe

    result = trtllm_fp4_block_scale_routed_moe(
        topk_ids=(topk_ids, topk_weights),
        routing_bias=None,
        hidden_states=hidden_states,
        hidden_states_scale=hidden_states_scale,
        gemm1_weights=gemm1_weights,
        gemm1_weights_scale=gemm1_weights_scale,
        gemm1_bias=None,
        gemm1_alpha=None,
        gemm1_beta=None,
        gemm1_clamp_limit=None,
        gemm2_weights=gemm2_weights,
        gemm2_weights_scale=gemm2_weights_scale,
        gemm2_bias=None,
        output1_scale_scalar=output1_scale_scalar,
        output1_scale_gate_scalar=output1_scale_gate_scalar,
        output2_scale_scalar=output2_scale_scalar,
        num_experts=int(num_experts.item()),
        top_k=int(top_k.item()),
        n_group=None,
        topk_group=None,
        intermediate_size=int(intermediate_size.item()),
        local_expert_offset=int(local_expert_offset.item()),
        local_num_experts=int(local_num_experts.item()),
        routed_scaling_factor=float(routed_scaling_factor.item()),
        routing_method_type=ROUTING_METHOD_DEEPSEEK_V3,
        do_finalize=True,
        activation_type=ACTIVATION_SWIGLU,
        output=output,
        tune_max_num_tokens=_next_power_of_2(hidden_states.shape[0]),
    )
    return result[0]
