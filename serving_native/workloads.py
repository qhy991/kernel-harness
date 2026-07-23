"""Fixed GLM-5.2/B200 serving workloads.

This suite intentionally does not reuse ``testbench/harness/glm52_ops.py``.
That file is the frozen synthetic oracle.  The workloads below are tied to the
verified B200 TP8/DP8/DeepEP deployment lane and name the production SGLang
callable that the reference invokes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Workload:
    name: str
    family: str
    phase: str
    world_size: int
    source_symbol: str
    params: dict[str, Any] = field(default_factory=dict)
    notes: str = ""

    @property
    def distributed(self) -> bool:
        return self.world_size > 1


_COMMON = dict(hidden=6144, experts=256, topk=8)
_DECODE_M_BUCKETS = (16, 32)


WORKLOADS: dict[str, Workload] = {
    "dp_allgather_prefill": Workload(
        "dp_allgather_prefill",
        "allgather",
        "prefill",
        8,
        "sglang.srt.distributed.parallel_state.GroupCoordinator.all_gather_into_tensor",
        dict(local_tokens=4096, hidden=_COMMON["hidden"], dtype="bfloat16"),
        "TP8/DP8 balanced chunked prefill: 4096 local rows -> 32768 global rows.",
    ),
    # DeepEP AUTO resolves extend/prefill to normal and decode to low latency.
    "deepep_normal_dispatch_prefill": Workload(
        "deepep_normal_dispatch_prefill",
        "deepep_normal_dispatch",
        "prefill",
        8,
        "deep_ep.Buffer.get_dispatch_layout + deep_ep.Buffer.dispatch",
        dict(
            local_tokens=4096,
            max_dispatch_tokens=128,
            hidden=_COMMON["hidden"],
            experts=_COMMON["experts"],
            topk=_COMMON["topk"],
        ),
        "Communication-stage ABI after SGLang FP8 group quantization.",
    ),
    "deepep_normal_combine_prefill": Workload(
        "deepep_normal_combine_prefill",
        "deepep_normal_combine",
        "prefill",
        8,
        "deep_ep.Buffer.combine",
        dict(
            local_tokens=4096,
            max_dispatch_tokens=128,
            hidden=_COMMON["hidden"],
            experts=_COMMON["experts"],
            topk=_COMMON["topk"],
        ),
        "Normal DeepEP combine with a handle produced by the exact dispatch path.",
    ),
}


def _add_decode_bucket(m: int) -> None:
    """Add the two production CUDA-graph decode buckets (M=16 and M=32)."""
    suffix = f"m{m}"
    packed_symbol = (
        "sglang.kernels.ops.quantization.fp8_kernel."
        "w8a8_block_fp8_matmul_deepgemm"
    )
    linear_specs = {
        "linear_fused_qkv_a_decode": (
            2624,
            6144,
            "Replicated fused_qkv_a_proj_with_mqa; packed int32 UE8M0 scales.",
        ),
        "linear_attn_q_b_decode": (
            16384,
            2048,
            "DP-attention Q-B uses all 64 heads on each rank.",
        ),
        "linear_attn_o_decode": (
            6144,
            16384,
            "DP-attention O projection with production packed-scale ABI.",
        ),
        "linear_indexer_wq_b_decode": (
            4096,
            2048,
            "Real DSA indexer.wq_b; distinct from self_attn.q_b_proj.",
        ),
    }
    for base, (n, k, notes) in linear_specs.items():
        name = f"{base}_{suffix}"
        WORKLOADS[name] = Workload(
            name,
            "packed_fp8_gemm",
            "decode",
            1,
            packed_symbol,
            dict(m=m, n=n, k=k),
            notes,
        )

    name = f"indexer_wk_weights_decode_{suffix}"
    WORKLOADS[name] = Workload(
        name,
        "bf16_linear",
        "decode",
        1,
        (
            "sglang.srt.layers.quantization.unquant."
            "UnquantizedLinearMethod.apply -> torch.nn.functional.linear"
        ),
        dict(m=m, n=160, k=6144),
        "Default fused BF16 wk_weights_proj: key[128] + weights[32].",
    )

    # DeepEP-LL allocates 128 tokens/rank * EP8 = 1024 rows per local expert.
    # With M tokens on every rank, this EP rank receives M*topk assignments in
    # expectation and SGLang selects expected_m=ceil(M*EP*topk/experts).
    moe = dict(
        decode_m=m,
        experts_per_rank=32,
        expert_slab=1024,
        expected_m=(m * 8 * _COMMON["topk"] + _COMMON["experts"] - 1)
        // _COMMON["experts"],
        valid_assignments=m * _COMMON["topk"],
        group_size=128,
        topk=_COMMON["topk"],
    )
    grouped_symbol = (
        "sglang.srt.layers.deep_gemm_wrapper.entrypoint."
        "grouped_gemm_nt_f8f8bf16_masked"
    )
    for base, family, params, symbol, notes in (
        (
            "moe_w13_grouped_decode",
            "moe_grouped_masked",
            dict(**moe, k=6144, n=4096),
            grouped_symbol,
            "Real fused W13 grouped GEMM after DeepEP low-latency dispatch.",
        ),
        (
            "moe_swiglu_quant_decode",
            "moe_swiglu_quant",
            dict(**moe, gate_up=4096),
            "sglang.srt.layers.moe.moe_runner.deep_gemm._varlen_deep_gemm_silu_mul_quant",
            "Real fused SwiGLU+quant stage between W13 and W2.",
        ),
        (
            "moe_w2_grouped_decode",
            "moe_grouped_masked",
            dict(**moe, k=2048, n=6144),
            grouped_symbol,
            "Real W2 grouped GEMM before DeepEP low-latency combine.",
        ),
    ):
        name = f"{base}_{suffix}"
        WORKLOADS[name] = Workload(
            name, family, "decode", 1, symbol, params, notes
        )

    name = f"dsa_trtllm_decode_{suffix}"
    WORKLOADS[name] = Workload(
        name,
        "dsa_trtllm",
        "decode",
        1,
        "flashinfer.decode.trtllm_batch_decode_with_kv_cache_mla",
        dict(
            batch=m,
            heads=64,
            head_dim=576,
            context=8192,
            sparse_topk=2048,
            page_size=64,
        ),
        "Blackwell FP8-KV production backend (trtllm-gen), not flash_mla_sparse_fwd.",
    )

    name = f"dp_allgather_decode_{suffix}"
    WORKLOADS[name] = Workload(
        name,
        "allgather",
        "decode",
        8,
        "sglang.srt.distributed.parallel_state.GroupCoordinator.all_gather_into_tensor",
        dict(local_tokens=m, hidden=_COMMON["hidden"], dtype="bfloat16"),
        f"TP8/DP8 decode bucket: {m} local rows -> {m * 8} gathered rows.",
    )

    deepep_params = dict(
        local_tokens=m,
        max_dispatch_tokens=128,
        hidden=_COMMON["hidden"],
        experts=_COMMON["experts"],
        topk=_COMMON["topk"],
    )
    for base, family, symbol, notes in (
        (
            "deepep_ll_dispatch_decode",
            "deepep_ll_dispatch",
            "deep_ep.Buffer.low_latency_dispatch",
            "DeepEP AUTO decode dispatch with FP8/packed UE8M0.",
        ),
        (
            "deepep_ll_combine_decode",
            "deepep_ll_combine",
            "deep_ep.Buffer.low_latency_combine",
            "DeepEP AUTO decode combine using the dispatch-produced handle.",
        ),
    ):
        name = f"{base}_{suffix}"
        WORKLOADS[name] = Workload(
            name, family, "decode", 8, symbol, dict(deepep_params), notes
        )


for _decode_m in _DECODE_M_BUCKETS:
    _add_decode_bucket(_decode_m)


def _add_four_gpu_lane() -> None:
    """Add a TP4/DP4/EP4 lane without changing the production TP8 lane."""
    world_size = 4
    prefill_local_tokens = 8192

    for collective, family, symbol in (
        (
            "allgather",
            "allgather",
            "sglang.srt.distributed.parallel_state.GroupCoordinator.all_gather_into_tensor",
        ),
        (
            "allreduce",
            "allreduce",
            "sglang.srt.distributed.parallel_state.GroupCoordinator.all_reduce",
        ),
    ):
        name = f"tp4_{collective}_prefill"
        WORKLOADS[name] = Workload(
            name,
            family,
            "prefill",
            world_size,
            symbol,
            dict(
                local_tokens=prefill_local_tokens,
                hidden=_COMMON["hidden"],
                dtype="bfloat16",
            ),
            "TP4/DP4 balanced 32768-token prefill lane.",
        )

    deepep_prefill = dict(
        local_tokens=prefill_local_tokens,
        max_dispatch_tokens=128,
        hidden=_COMMON["hidden"],
        experts=_COMMON["experts"],
        topk=_COMMON["topk"],
    )
    for base, family, symbol, notes in (
        (
            "ep4_deepep_normal_dispatch_prefill",
            "deepep_normal_dispatch",
            "deep_ep.Buffer.get_dispatch_layout + deep_ep.Buffer.dispatch",
            "EP4 normal dispatch; 64 local experts per rank.",
        ),
        (
            "ep4_deepep_normal_combine_prefill",
            "deepep_normal_combine",
            "deep_ep.Buffer.combine",
            "EP4 normal combine using the exact dispatch-produced handle.",
        ),
    ):
        WORKLOADS[base] = Workload(
            base,
            family,
            "prefill",
            world_size,
            symbol,
            dict(deepep_prefill),
            notes,
        )

    for m in _DECODE_M_BUCKETS:
        suffix = f"m{m}"
        for collective, family, symbol in (
            (
                "allgather",
                "allgather",
                "sglang.srt.distributed.parallel_state.GroupCoordinator.all_gather_into_tensor",
            ),
            (
                "allreduce",
                "allreduce",
                "sglang.srt.distributed.parallel_state.GroupCoordinator.all_reduce",
            ),
        ):
            name = f"tp4_{collective}_decode_{suffix}"
            detail = (
                f"gathered rows={m * world_size}."
                if family == "allgather"
                else "sum reduction."
            )
            WORKLOADS[name] = Workload(
                name,
                family,
                "decode",
                world_size,
                symbol,
                dict(local_tokens=m, hidden=_COMMON["hidden"], dtype="bfloat16"),
                f"TP4/DP4 decode bucket: local M={m}; {detail}",
            )

        deepep_decode = dict(
            local_tokens=m,
            max_dispatch_tokens=128,
            hidden=_COMMON["hidden"],
            experts=_COMMON["experts"],
            topk=_COMMON["topk"],
        )
        for base, family, symbol, notes in (
            (
                "ep4_deepep_ll_dispatch_decode",
                "deepep_ll_dispatch",
                "deep_ep.Buffer.low_latency_dispatch",
                "EP4 low-latency dispatch with FP8/packed UE8M0.",
            ),
            (
                "ep4_deepep_ll_combine_decode",
                "deepep_ll_combine",
                "deep_ep.Buffer.low_latency_combine",
                "EP4 low-latency combine using the dispatch-produced handle.",
            ),
        ):
            name = f"{base}_{suffix}"
            WORKLOADS[name] = Workload(
                name,
                family,
                "decode",
                world_size,
                symbol,
                dict(deepep_decode),
                notes,
            )


_add_four_gpu_lane()


def get_workload(name: str) -> Workload:
    try:
        return WORKLOADS[name]
    except KeyError as exc:
        raise KeyError(f"unknown workload {name!r}; choose from: {', '.join(WORKLOADS)}") from exc


def as_dict(workload: Workload) -> dict[str, Any]:
    return {
        "name": workload.name,
        "family": workload.family,
        "phase": workload.phase,
        "world_size": workload.world_size,
        "distributed": workload.distributed,
        "source_symbol": workload.source_symbol,
        "params": dict(workload.params),
        "notes": workload.notes,
    }
