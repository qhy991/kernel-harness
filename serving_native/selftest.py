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

from serving_native.context_matrix import validate_profile
from serving_native.workloads import WORKLOADS, with_kv_contexts


def main() -> int:
    assert len(WORKLOADS) == len(set(WORKLOADS))
    assert {workload.world_size for workload in WORKLOADS.values()} == {1, 4, 8}
    assert all(workload.params for workload in WORKLOADS.values())
    assert all(workload.source_symbol for workload in WORKLOADS.values())

    names = set(WORKLOADS)
    assert "linear_indexer_wq_b_decode_m16" in names
    assert "linear_indexer_wq_b_decode_m32" in names
    assert "indexer_wk_weights_decode_m16" in names
    assert "dsa_trtllm_decode_m32" in names
    assert "moe_w13_grouped_decode_m16" in names
    assert "moe_w2_grouped_decode_m32" in names
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

    dsa_m16 = WORKLOADS["dsa_trtllm_decode_m16"]
    assert dsa_m16.kv_cache_dependency == "direct-paged-kv"
    assert dsa_m16.context_contract["runtime_override"] == (
        "--kv-context or --kv-contexts"
    )
    scalar_context = with_kv_contexts(dsa_m16, [32768])
    assert scalar_context.params["context"] == 32768
    assert scalar_context.params["context_lengths"] == [32768] * 16
    ragged_contexts = list(range(1024, 1024 * 17, 1024))
    ragged_context = with_kv_contexts(dsa_m16, ragged_contexts)
    assert ragged_context.params["context"] == 16384
    assert ragged_context.params["context_lengths"] == ragged_contexts
    try:
        with_kv_contexts(dsa_m16, [1024, 2048])
    except ValueError:
        pass
    else:
        raise AssertionError("short ragged KV vectors must not be silently cycled")

    profiles = sorted((HERE / "context_profiles").glob("*.json"))
    assert profiles, "at least one context profile example is required"
    for profile_path in profiles:
        profile = validate_profile(profile_path)
        assert profile["production_evidence"] is False, (
            "repository examples must never impersonate traced production evidence"
        )

    decode_ms = {16, 32}
    assert {
        workload.params["m"]
        for workload in WORKLOADS.values()
        if workload.family in ("packed_fp8_gemm", "bf16_linear")
    } == decode_ms
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
        if workload.family in ("moe_grouped_masked", "moe_swiglu_quant"):
            m = workload.params["decode_m"]
            assert m in decode_ms
            assert workload.params["expected_m"] == m // 4
            assert workload.params["valid_assignments"] == m * 8

    sglang_root = Path(os.environ.get("SGLANG_ROOT", "/home/qinhaiyan/sglang"))
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
            "total_kv_tokens < total_q_tokens * 512",
            "TopkTransformMethod.RAGGED",
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
    source_checked = sglang_root.is_dir()
    if source_checked:
        for relative, needles in source_checks.items():
            source = (sglang_root / relative).read_text()
            for needle in needles:
                assert needle in source, f"production source contract drift: {relative}: {needle}"

    for path in HERE.glob("*.py"):
        ast.parse(path.read_text(), filename=str(path))
    for path in (HERE / "candidates").glob("*.py"):
        ast.parse(path.read_text(), filename=str(path))
    print(f"serving_native selftest OK: {len(WORKLOADS)} fixed workloads; "
          f"production_source={'checked' if source_checked else 'skipped (SGLANG_ROOT absent)'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
