"""GPU-free structural checks for the serving-native suite."""

from __future__ import annotations

import ast
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from serving_native.workloads import WORKLOADS


def main() -> int:
    assert len(WORKLOADS) == len(set(WORKLOADS))
    assert {workload.world_size for workload in WORKLOADS.values()} == {1, 4, 8}
    assert all(workload.params for workload in WORKLOADS.values())
    assert all(workload.source_symbol for workload in WORKLOADS.values())
    assert all(workload.execution_modes for workload in WORKLOADS.values())
    assert all(
        set(workload.execution_modes).issubset({"eager", "cuda_graph"})
        for workload in WORKLOADS.values()
    )

    names = set(WORKLOADS)
    assert len(names) == 44
    assert "linear_attn_o_prefill_m4096" in names
    assert "linear_indexer_wq_b_decode_m16" in names
    assert "linear_indexer_wq_b_decode_m32" in names
    assert "indexer_wk_weights_decode_m16" in names
    assert "dsa_trtllm_decode_m32" in names
    assert "moe_w13_grouped_decode_m16" in names
    assert "moe_w2_grouped_decode_m32" in names
    assert "moe_w2_grouped_decode_m16_current_source_m5" in names
    assert "moe_w2_grouped_decode_m32_current_source_m9" in names
    assert (
        "moe_w13_swiglu_w2_region_decode_m16_current_source_m5"
        in names
    )
    assert (
        "moe_w13_swiglu_w2_region_decode_m32_current_source_m9"
        in names
    )
    assert "deepep_normal_dispatch_prefill" in names
    assert "deepep_ll_combine_decode_m16" in names
    assert "deepep_ll_combine_decode_m32" in names
    assert "tp4_allreduce_decode_m16" in names
    assert "tp4_allreduce_decode_m32" in names
    assert "tp4_allgather_decode_m16" in names
    assert "ep4_deepep_ll_dispatch_decode_m16" in names
    assert "ep4_deepep_ll_combine_decode_m32" in names
    assert "ep4_deepep_normal_dispatch_prefill" in names
    assert not any("index_k_proj" in name for name in names)
    assert not any("moe_gate" in name or "moe_up" in name for name in names)

    decode_ms = {16, 32}
    assert {
        workload.params["m"]
        for workload in WORKLOADS.values()
        if workload.phase == "decode"
        and workload.family in ("packed_fp8_gemm", "bf16_linear")
    } == decode_ms
    assert {
        (
            workload.params["m"],
            workload.params["n"],
            workload.params["k"],
        )
        for workload in WORKLOADS.values()
        if workload.phase == "prefill" and workload.family == "packed_fp8_gemm"
    } == {(4096, 6144, 16384)}
    assert {
        workload.name
        for workload in WORKLOADS.values()
        if "cuda_graph" in workload.execution_modes
    } == {
        "linear_attn_o_decode_m16",
        "linear_attn_o_decode_m32",
        "moe_w2_grouped_decode_m16",
        "moe_w2_grouped_decode_m16_current_source_m5",
        "moe_w2_grouped_decode_m32",
        "moe_w2_grouped_decode_m32_current_source_m9",
    }
    assert {
        workload.params["batch"]
        for workload in WORKLOADS.values()
        if workload.family == "dsa_trtllm"
    } == decode_ms
    assert {
        workload.params["local_tokens"]
        for workload in WORKLOADS.values()
        if workload.family in ("deepep_ll_dispatch", "deepep_ll_combine")
    } == decode_ms
    for workload in WORKLOADS.values():
        if workload.family in (
            "moe_grouped_masked",
            "moe_swiglu_quant",
            "moe_compute_region",
        ):
            m = workload.params["decode_m"]
            assert m in decode_ms
            expected = m // 4
            if "current_source" in workload.name:
                expected += 1
            assert workload.params["expected_m"] == expected
            assert workload.params["valid_assignments"] == m * 8
    w2_identities = {
        (workload.params["decode_m"], workload.params["expected_m"]):
        workload.params["candidate_jit_identity"]
        for workload in WORKLOADS.values()
        if workload.name.startswith("moe_w2_grouped_decode_")
    }
    assert set(w2_identities) == {(16, 4), (16, 5), (32, 8), (32, 9)}
    for (_decode_m, expected_m), identity in w2_identities.items():
        assert identity.endswith(f"glm52_w2_bm16_v2_em{expected_m}")
    region_identities = {
        (workload.params["decode_m"], workload.params["expected_m"]):
        workload.params["candidate_jit_identity"]
        for workload in WORKLOADS.values()
        if workload.family == "moe_compute_region"
    }
    assert set(region_identities) == {(16, 5), (32, 9)}

    paired_sglang = REPO_ROOT.parent / "sglang"
    default_sglang = (
        paired_sglang if paired_sglang.is_dir() else Path("/home/qinhaiyan/sglang")
    )
    sglang_root = Path(os.environ.get("SGLANG_ROOT", default_sglang))
    source_checks = {
        "python/sglang/kernels/ops/quantization/fp8_kernel.py": (
            "def w8a8_block_fp8_matmul_deepgemm",
        ),
        "python/sglang/srt/layers/attention/dsa/dsa_indexer.py": (
            "self.wq_b = ReplicatedLinear",
            "self.wk_weights_proj = ReplicatedLinear",
        ),
        "python/sglang/srt/layers/attention/dsa_backend.py": (
            "trtllm_batch_decode_with_kv_cache_mla",
            'backend="trtllm-gen"',
        ),
        "python/sglang/srt/layers/moe/moe_runner/deep_gemm.py": (
            "def _varlen_deep_gemm_silu_mul_quant",
        ),
        "python/sglang/srt/layers/deep_gemm_wrapper/entrypoint.py": (
            "def grouped_gemm_nt_f8f8bf16_masked",
            "fp8_m_grouped_gemm_nt_masked",
        ),
        "python/sglang/srt/layers/moe/token_dispatcher/deepep.py": (
            "class DeepEPBuffer:",
            "DeepEPBuffer.get_deepep_buffer(",
            "buffer.dispatch(",
            "buffer.combine(",
            "buffer.low_latency_dispatch(",
            "buffer.low_latency_combine(",
        ),
        "python/sglang/srt/distributed/parallel_state.py": (
            "def all_gather_into_tensor",
            "def all_reduce",
        ),
    }
    for relative, needles in source_checks.items():
        source = (sglang_root / relative).read_text()
        for needle in needles:
            assert needle in source, f"production source contract drift: {relative}: {needle}"

    for path in HERE.glob("*.py"):
        ast.parse(path.read_text(), filename=str(path))
    for path in (HERE / "candidates").glob("*.py"):
        ast.parse(path.read_text(), filename=str(path))
    print(f"serving_native selftest OK: {len(WORKLOADS)} fixed workloads")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
