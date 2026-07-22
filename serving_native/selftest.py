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

    names = set(WORKLOADS)
    assert "linear_indexer_wq_b_decode_m16" in names
    assert "linear_indexer_wq_b_decode_m32" in names
    assert "indexer_wk_weights_decode_m16" in names
    assert "dsa_trtllm_decode_m32" in names
    assert "dsa_flashmla_kv_decode_m16" in names
    assert "dsa_flashmla_kv_decode_m32" in names
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
    flashmla_workloads = [
        workload
        for workload in WORKLOADS.values()
        if workload.family == "dsa_flashmla_kv"
    ]
    assert {workload.params["batch"] for workload in flashmla_workloads} == decode_ms
    for workload in flashmla_workloads:
        assert workload.params["q_heads"] == 64
        assert workload.params["q_head_dim"] == 576
        assert workload.params["v_head_dim"] == 512
        assert workload.params["qk_nope_head_dim"] == 192
        assert workload.params["qk_rope_head_dim"] == 64
        assert workload.params["softmax_scale"] == 0.0625
        assert workload.params["kv_cache_dim"] == 656
        assert workload.params["context"] == 8192
        assert workload.params["sparse_topk"] == 2048
        assert workload.params["page_size"] == 64
        assert "_forward_flashmla_kv" in workload.source_symbol
        assert "fwd_kvcache_mla" in workload.source_symbol
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

    # Worktrees are provisioned as sibling Kernel-Harness/sglang directories.
    # Honor an explicit override, otherwise audit the sibling paired with this
    # checkout instead of silently reaching into a global SGLang tree.
    sglang_root = Path(os.environ.get("SGLANG_ROOT", REPO_ROOT.parent / "sglang"))
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
            "def _forward_flashmla_kv(",
            "from sgl_kernel.flash_mla import flash_mla_with_kvcache",
            "target_q_heads = self.flashmla_kv_num_q_heads",
            "kv_cache.view(-1, self.real_page_size, 1, self.kv_cache_dim)",
            "indices = page_table_1.unsqueeze(1)",
            "is_fp8_kvcache=True",
        ),
        "sgl-kernel/python/sgl_kernel/flash_mla.py": (
            "def flash_mla_with_kvcache(",
            "torch.ops.sgl_kernel.fwd_kvcache_mla.default",
        ),
        "sgl-kernel/cmake/flashmla.cmake": (
            "05e26647fe840b8baedae486c2d86d5ce4efeb7c",
            "csrc/sm100/decode/head64/instantiations/v32.cu",
        ),
        "sgl-kernel/csrc/flashmla_extension.cc": (
            'm.impl("fwd_kvcache_mla", torch::kCUDA, &fwd_kvcache_mla)',
        ),
        "test/registered/attention/unittests/dsa/test_dsa.py": (
            "PRODUCTION_FLASHMLA_KV_DECODE_CASES",
            "for batch in (16, 32)",
            "index_topk=2048",
            'index_pattern="affine"',
            "require_fused_topk=True",
            "softmax_scale=(192 + 64) ** -0.5",
            "model_head_dim=192",
            "model_v_head_dim=256",
            "test_production_flashmla_kv_cuda_graph_metadata_lifecycle_cases",
            'dsa_decode_backend="flashmla_kv"',
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
