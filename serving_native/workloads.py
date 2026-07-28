"""Fixed GLM-5.2/B200 serving workloads.

This suite intentionally does not reuse ``testbench/harness/glm52_ops.py``.
That file is the frozen synthetic oracle.  The workloads below are tied to the
verified B200 TP8/DP8/DeepEP deployment lane and name the production SGLang
callable that the reference invokes.
"""

# The registry consistently uses dict(...) for compact keyword-shaped tensor
# metadata; preserve that established generated-description style.
# ruff: noqa: C408

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
    execution_modes: tuple[str, ...] = ("eager",)

    @property
    def distributed(self) -> bool:
        return self.world_size > 1


_COMMON = dict(hidden=6144, experts=256, topk=8)
_DECODE_M_BUCKETS = (16, 32)
W2_EDGE_MASK_CASES: tuple[tuple[str, tuple[int, ...]], ...] = (
    ("zero_all_experts", (0,) * 32),
    ("single_expert_one_row", (1,) + (0,) * 31),
    (
        "full_boundary_sweep",
        (15, 16, 17, 31, 32, 33, 127, 1024) + (0,) * 24,
    ),
)


WORKLOADS: dict[str, Workload] = {
    "linear_attn_o_prefill_m4096": Workload(
        "linear_attn_o_prefill_m4096",
        "packed_fp8_gemm",
        "prefill",
        1,
        (
            "sglang.kernels.ops.quantization.fp8_kernel."
            "w8a8_block_fp8_matmul_deepgemm"
        ),
        dict(m=4096, n=6144, k=16384),
        (
            "DP8 attention O projection at the balanced local prefill bucket; "
            "packed int32 UE8M0 activation and weight scales."
        ),
        ("eager",),
    ),
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
            (
                ("eager", "cuda_graph")
                if base == "linear_attn_o_decode"
                else ("eager",)
            ),
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
            dict(
                **moe,
                k=2048,
                n=6144,
                candidate_jit_identity=(
                    "sm100_m_grouped_fp8_fp4_gemm_masked_1d1d_"
                    f"glm52_w2_bm16_v2_em{moe['expected_m']}"
                ),
            ),
            grouped_symbol,
            "Real W2 grouped GEMM before DeepEP low-latency combine.",
        ),
    ):
        name = f"{base}_{suffix}"
        WORKLOADS[name] = Workload(
            name,
            family,
            "decode",
            1,
            symbol,
            params,
            notes,
            (
                ("eager", "cuda_graph")
                if base == "moe_w2_grouped_decode"
                else ("eager",)
            ),
        )

    # Goal07 recorded a second host expected_m hint for the same exact decode
    # masks. Keep it as an independent identity because expected_m participates
    # in the candidate's generated-code/JIT key.
    current_expected_m = moe["expected_m"] + 1
    name = (
        f"moe_w2_grouped_decode_{suffix}_"
        f"current_source_m{current_expected_m}"
    )
    WORKLOADS[name] = Workload(
        name,
        "moe_grouped_masked",
        "decode",
        1,
        grouped_symbol,
        dict(
            moe,
            expected_m=current_expected_m,
            k=2048,
            n=6144,
            candidate_jit_identity=(
                "sm100_m_grouped_fp8_fp4_gemm_masked_1d1d_"
                f"glm52_w2_bm16_v2_em{current_expected_m}"
            ),
        ),
        (
            "Real W2 grouped GEMM with the Goal07 current-source host "
            "expected_m hint; same deterministic mask as its decode bucket."
        ),
        ("eager", "cuda_graph"),
    )

    region_name = (
        f"moe_w13_swiglu_w2_region_decode_{suffix}_"
        f"current_source_m{current_expected_m}"
    )
    WORKLOADS[region_name] = Workload(
        region_name,
        "moe_compute_region",
        "decode",
        1,
        (
            "sglang.srt.layers.deep_gemm_wrapper.entrypoint."
            "grouped_gemm_nt_f8f8bf16_masked + "
            "sglang.srt.layers.moe.moe_runner.deep_gemm."
            "_varlen_deep_gemm_silu_mul_quant"
        ),
        dict(
            moe,
            expected_m=current_expected_m,
            hidden=6144,
            gate_up=4096,
            w2_k=2048,
            candidate_jit_identity=(
                "sm100_m_grouped_fp8_fp4_gemm_masked_1d1d_"
                f"glm52_w2_bm16_v2_em{current_expected_m}"
            ),
        ),
        (
            "Single-B200 containing compute region: stock W13, fused "
            "SwiGLU+packed UE8M0 quant, then W2. This is the local "
            "no-overlap region gate, not DeepEP/TP8 acceptance."
        ),
        ("eager",),
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


_STAGE11_JIT_IDENTITY = (
    "sm100_m_grouped_fp8_fp4_gemm_masked_1d1d_"
    "glm52_w2_em8_bm16_stage11_v3"
)
_STAGE11_COMMON = dict(
    decode_m=32,
    experts_per_rank=32,
    expert_slab=1024,
    expected_m=8,
    valid_assignments=32 * _COMMON["topk"],
    group_size=128,
    topk=_COMMON["topk"],
    candidate_variant="em8_bm16_stage11",
    candidate_variant_version=3,
    candidate_jit_identity=_STAGE11_JIT_IDENTITY,
    masked_block_m_override=16,
    masked_num_stages_override=11,
    scale_abi="packed-int32-ue8m0",
    provenance_snapshot_contract="before_timed_series_warmup_v1",
    performance_estimator_contract="strict_four_estimator_every_series_1p03_v1",
    predeclared_fallback="em8_bm16_stage10",
    fallback_eligible=False,
    pipeline_smem_per_stage_bytes=18432,
    pipeline_fixed_bytes=9004,
    stock_pipeline_num_stages=12,
    stock_pipeline_smem_bytes=230188,
    candidate_pipeline_num_stages=11,
    candidate_pipeline_smem_bytes=211756,
    two_ctas_per_sm_enabled=False,
    performance_hypothesis="reduced-pipeline-pressure-falsifiable",
)
_STAGE11_GROUPED_SYMBOL = (
    "sglang.srt.layers.deep_gemm_wrapper.entrypoint."
    "grouped_gemm_nt_f8f8bf16_masked"
)
WORKLOADS["moe_w2_grouped_decode_m32_em8_bm16_stage11"] = Workload(
    "moe_w2_grouped_decode_m32_em8_bm16_stage11",
    "moe_grouped_masked",
    "decode",
    1,
    _STAGE11_GROUPED_SYMBOL,
    dict(_STAGE11_COMMON, k=2048, n=6144),
    (
        "Separately versioned Task26 stage11 leaf: exact M32/em8, BM16, "
        "packed-int32 UE8M0, PDL, and no overlap."
    ),
    ("eager", "cuda_graph"),
)
WORKLOADS[
    "moe_w13_swiglu_w2_region_decode_m32_em8_bm16_stage11"
] = Workload(
    "moe_w13_swiglu_w2_region_decode_m32_em8_bm16_stage11",
    "moe_compute_region",
    "decode",
    1,
    (
        f"{_STAGE11_GROUPED_SYMBOL} + "
        "sglang.srt.layers.moe.moe_runner.deep_gemm."
        "_varlen_deep_gemm_silu_mul_quant"
    ),
    dict(
        _STAGE11_COMMON,
        hidden=6144,
        gate_up=4096,
        w2_k=2048,
    ),
    (
        "Containing W13+SwiGLU/packed-UE8M0+W2 region for the exact "
        "em8/BM16/stage11 candidate; local no-overlap evidence only."
    ),
    ("eager",),
)

_STAGE11_V4_JIT_IDENTITY = (
    "sm100_m_grouped_fp8_fp4_gemm_masked_1d1d_"
    "glm52_w2_em8_bm16_stage11_v4"
)
_STAGE11_V4_COMMON = dict(
    _STAGE11_COMMON,
    candidate_variant_version=4,
    candidate_jit_identity=_STAGE11_V4_JIT_IDENTITY,
    readiness_contract="content-addressed-ready-v1",
    build_phase="cpu-only-before-gpu-lease",
    required_lane_portfolio=(
        "leaf_eager",
        "leaf_cuda_graph",
        "containing_region_eager",
        "containing_region_cuda_graph",
    ),
)
WORKLOADS["moe_w2_grouped_decode_m32_em8_bm16_stage11_v4"] = Workload(
    "moe_w2_grouped_decode_m32_em8_bm16_stage11_v4",
    "moe_grouped_masked",
    "decode",
    1,
    _STAGE11_GROUPED_SYMBOL,
    dict(_STAGE11_V4_COMMON, k=2048, n=6144),
    (
        "Task26 v4 READY-bound leaf: unchanged exact M32/em8 BM16/stage11 "
        "kernel semantics with a prebuilt content-addressed bundle."
    ),
    ("eager", "cuda_graph"),
)
WORKLOADS[
    "moe_w13_swiglu_w2_region_decode_m32_em8_bm16_stage11_v4"
] = Workload(
    "moe_w13_swiglu_w2_region_decode_m32_em8_bm16_stage11_v4",
    "moe_compute_region",
    "decode",
    1,
    (
        f"{_STAGE11_GROUPED_SYMBOL} + "
        "sglang.srt.layers.moe.moe_runner.deep_gemm."
        "_varlen_deep_gemm_silu_mul_quant"
    ),
    dict(
        _STAGE11_V4_COMMON,
        hidden=6144,
        gate_up=4096,
        w2_k=2048,
    ),
    (
        "Task26 v4 containing W13+SwiGLU/packed-UE8M0+W2 region; eager "
        "and CUDA Graph lanes must substitute only the W2 node."
    ),
    ("eager", "cuda_graph"),
)


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
        "execution_modes": list(workload.execution_modes),
    }
