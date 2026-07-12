#!/usr/bin/env python3
"""Baseline microbench for the *untested* (未测 / "B200 待补测") Kimi-K2.7 kernels.

Covers every row in logs/.../kimi_k27.csv marked 未测, using the SAME sglang /
deep_gemm / sgl_kernel APIs that sglang dispatches at runtime on B200 (sm100):

  MLA projections (FP8 W8A8)  -> deep_gemm w8a8_block_fp8   (reuse bench_fp8_gemm)
  Q_a+KV_a fused decode (bf16)-> sgl_kernel.dsv3_fused_a_gemm
  MoE Router prefill (bf16)   -> torch.mm  (cuBLAS; M>16 -> F.linear path)
  MoE Router decode  (bf16)   -> sgl_kernel.dsv3_router_gemm
  MoE Down grouped prefill    -> deep_gemm.m_grouped_fp8_gemm_nt_contiguous
  MoE GateUp/Down grouped dec -> deep_gemm.fp8_m_grouped_gemm_nt_masked
  KV_b / V absorb BMM (bf16)  -> torch.bmm  (reuse bench_bmm)
  MLA attention prefill       -> flashinfer.single_prefill_with_kv_cache (ragged)
  MLA attention decode        -> sgl_kernel.flash_mla.flash_mla_with_kvcache (paged)

Shapes follow the CSV exactly for the GEMM/BMM/router/grouped ops. The two MLA
attention rows are under-specified in the CSV, so they use real Kimi-K2.7 MLA
dims (64 heads, kv_lora=512, qk_rope=64, qk_nope=128, v=128) at documented
context lengths 4096 and 8192 (see ATTN_SEQLENS).
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

# Reuse the validated helpers from the real-kernel harness.
from bench_kimi_real_kernels import (  # noqa: E402
    bench,
    bench_bmm,
    bench_fp8_gemm,
    bench_grouped_fp8,
    gemm_tflops,
    sync,
)

import deep_gemm  # noqa: E402
from deep_gemm.utils.math import align, per_block_cast_to_fp8, per_token_cast_to_fp8  # noqa: E402

ATTN_SEQLENS = (4096, 8192)


# --------------------------------------------------------------------------- #
# MoE grouped GEMM — decode masked (low-latency DeepEP path)
# --------------------------------------------------------------------------- #
def bench_grouped_fp8_masked(E, M, K, N, name, phase):
    """deep_gemm.fp8_m_grouped_gemm_nt_masked — the decode LOW_LATENCY MoE path.

    Scale prep mirrors sglang's moe_runner/deep_gemm.py: per-token (act) and
    per-block (weight) UE8M0 casts, then transform_sf_into_required_layout.
    """
    Ka, Na, Mp = align(K, 128), align(N, 128), align(M, 128)
    a = torch.randn(E, Mp, Ka, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(E, Na, Ka, device="cuda", dtype=torch.bfloat16)

    a_fp8, a_s = zip(*[per_token_cast_to_fp8(a[e], use_ue8m0=True) for e in range(E)])
    b_fp8, b_s = zip(*[per_block_cast_to_fp8(b[e], use_ue8m0=True) for e in range(E)])
    a_fp8, a_s = torch.stack(a_fp8), torch.stack(a_s)
    b_fp8, b_s = torch.stack(b_fp8), torch.stack(b_s)
    a_s = deep_gemm.transform_sf_into_required_layout(
        a_s, mn=Mp, k=Ka, recipe=(1, 128, 128), num_groups=E, is_sfa=True
    )
    b_s = deep_gemm.transform_sf_into_required_layout(
        b_s, mn=Na, k=Ka, recipe=(1, 128, 128), num_groups=E, is_sfa=False
    )
    out = torch.empty(E, Mp, Na, device="cuda", dtype=torch.bfloat16)
    masked_m = torch.full((E,), M, device="cuda", dtype=torch.int32)

    def run():
        deep_gemm.fp8_m_grouped_gemm_nt_masked((a_fp8, a_s), (b_fp8, b_s), out, masked_m, M)

    us = bench(run, warmup=10, iters=50)
    return {
        "name": name, "phase": phase,
        "backend": "deep_gemm.fp8_m_grouped_gemm_nt_masked",
        "shape": f"{E}x[{M},{K}]x[{K},{N}] (masked, padded M={Mp})",
        "us": round(us, 2),
        "TFLOPS": round(gemm_tflops(E * M, Ka, Na, us), 2),
    }


# --------------------------------------------------------------------------- #
# MoE Router
# --------------------------------------------------------------------------- #
def bench_router_prefill(M, K, N, name):
    """Prefill (M>16): sglang falls through to F.linear -> cuBLAS bf16 GEMM."""
    x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
    w = torch.randn(N, K, device="cuda", dtype=torch.bfloat16)

    def run():
        torch.nn.functional.linear(x, w)

    us = bench(run, warmup=10, iters=50)
    return {
        "name": name, "phase": "prefill",
        "backend": "F.linear / torch.mm (cuBLAS bf16)",
        "shape": f"[{M},{K}]x[{K},{N}]",
        "us": round(us, 2),
        "TFLOPS": round(gemm_tflops(M, K, N, us), 2),
    }


def bench_router_decode(M, K, N, name):
    """Decode (M<=16, hidden==7168, experts in {256,384}): sgl_kernel.dsv3_router_gemm."""
    from sgl_kernel import dsv3_router_gemm

    x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
    w = torch.randn(N, K, device="cuda", dtype=torch.bfloat16)

    def run():
        dsv3_router_gemm(x, w, out_dtype=torch.float32)

    us = bench(run, warmup=10, iters=50)
    return {
        "name": name, "phase": "decode",
        "backend": "sgl_kernel.dsv3_router_gemm (CUDA JIT, out fp32)",
        "shape": f"[{M},{K}]x[{K},{N}]",
        "us": round(us, 2),
        "TFLOPS": round(gemm_tflops(M, K, N, us), 2),
    }


# --------------------------------------------------------------------------- #
# Q_a+KV_a fused decode — dedicated min-latency bf16 kernel
# --------------------------------------------------------------------------- #
def bench_fused_a_decode(M, K, N, name):
    """sgl_kernel.dsv3_fused_a_gemm — min-latency fused Q_a/KV_a for M in [1,16] bf16.

    Weight must be column-major: pass a contiguous [N,K] weight's .T view.
    """
    from sgl_kernel import dsv3_fused_a_gemm

    x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
    w = torch.randn(N, K, device="cuda", dtype=torch.bfloat16)
    wt = w.T  # column-major view expected by the kernel

    def run():
        dsv3_fused_a_gemm(x, wt)

    us = bench(run, warmup=10, iters=50)
    return {
        "name": name, "phase": "decode",
        "backend": "sgl_kernel.dsv3_fused_a_gemm (CUDA, bf16)",
        "shape": f"[{M},{K}]x[{K},{N}]",
        "us": round(us, 2),
        "TFLOPS": round(gemm_tflops(M, K, N, us), 2),
    }


# --------------------------------------------------------------------------- #
# MLA attention
# --------------------------------------------------------------------------- #
def bench_mla_prefill(seqlen, name):
    """MLA core attention, prefill. flashinfer ragged single_prefill.

    Kimi-K2.7 MLA (post up-proj): 64 heads, qk head_dim=192 (128 nope+64 rope),
    v head_dim=128, causal. Runs per TP shard would be 8 heads; we bench the full
    64-head attention core (DP-attention style) to mirror the decode row.
    """
    import flashinfer

    H, D_qk, D_v = 64, 192, 128
    q = torch.randn(seqlen, H, D_qk, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(seqlen, H, D_qk, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(seqlen, H, D_v, device="cuda", dtype=torch.bfloat16)

    def run():
        flashinfer.single_prefill_with_kv_cache(q, k, v, causal=True)

    us = bench(run, warmup=3, iters=10)
    # causal MLA FLOPs: QK^T (D_qk) + PV (D_v), halved for causal
    flops = 2.0 * H * seqlen * seqlen * (D_qk + D_v) * 0.5
    return {
        "name": name, "phase": "prefill",
        "backend": "flashinfer.single_prefill_with_kv_cache (ragged, causal)",
        "shape": f"Q/K:[{seqlen},{H},{D_qk}] V:[{seqlen},{H},{D_v}] seqlen={seqlen}",
        "us": round(us, 2),
        "TFLOPS": round(flops / (us * 1e-6) / 1e12, 2),
    }


def bench_mla_decode(seqlen, batch, name):
    """MLA absorbed decode on B200 (sm100): flashinfer TRTLLM-gen MLA.

    This is the sglang B200 production decode path (trtllm_mla_backend). The
    Hopper sgl_kernel flash_mla_with_kvcache dense decode is SM90a-only, so on
    Blackwell sglang dispatches trtllm_batch_decode_with_kv_cache_mla instead.

    Absorbed decode: q head_dim = kv_lora(512)+qk_rope(64)=576, out v=512,
    MQA (single compressed KV head), num_q_heads=64, paged KV (page=64), bf16.
    """
    import flashinfer

    H, D_ckv, D_kpe, PAGE = 64, 512, 64, 64
    D = D_ckv + D_kpe  # 576 = head_dim_qk (concat q_nope[=ckv] + q_rope)
    q_len = 1
    pages_per_seq = (seqlen + PAGE - 1) // PAGE
    total_pages = pages_per_seq * batch

    q = torch.randn(batch, q_len, H, D, device="cuda", dtype=torch.bfloat16)
    kv_cache = torch.randn(total_pages, PAGE, D, device="cuda", dtype=torch.bfloat16)
    block_tables = torch.arange(total_pages, device="cuda", dtype=torch.int32).view(
        batch, pages_per_seq
    )
    seq_lens = torch.full((batch,), seqlen, device="cuda", dtype=torch.int32)
    workspace = torch.zeros(128 * 1024 * 1024, device="cuda", dtype=torch.int8)
    scale = D ** -0.5

    def run():
        flashinfer.decode.trtllm_batch_decode_with_kv_cache_mla(
            query=q, kv_cache=kv_cache, workspace_buffer=workspace,
            qk_nope_head_dim=128, kv_lora_rank=D_ckv, qk_rope_head_dim=D_kpe,
            block_tables=block_tables, seq_lens=seq_lens, max_seq_len=seqlen,
            bmm1_scale=scale,
        )

    us = bench(run, warmup=10, iters=50)
    flops = 2.0 * batch * H * seqlen * (D + D_ckv)
    return {
        "name": name, "phase": "decode",
        "backend": "flashinfer.trtllm_batch_decode_with_kv_cache_mla (sm100)",
        "shape": f"Q:[{batch},{q_len},{H},{D}] KV:paged({seqlen},{D}) v={D_ckv}",
        "us": round(us, 2),
        "TFLOPS": round(flops / (us * 1e-6) / 1e12, 2),
    }


# --------------------------------------------------------------------------- #
def main():
    print(f"# GPU: {torch.cuda.get_device_name(0)}  torch={torch.__version__}")
    results = []

    def add(fn, *a, **kw):
        label = kw.pop("label", "")
        try:
            r = fn(*a, **kw)
            results.append(r)
            print(f"  OK  | {r['name']:26s} | {r['phase']:7s} | {r['us']:10.2f} us | "
                  f"{r.get('TFLOPS','   -   '):>8} TFLOPS | {r['backend']}")
        except Exception as e:
            print(f"  FAIL| {label:26s} | {type(e).__name__}: {e}")
            results.append({"name": label, "error": f"{type(e).__name__}: {e}"})

    print("\n### MLA projections — FP8 W8A8 (deep_gemm w8a8_block_fp8) ###")
    #     (M, K, N, name, phase)
    for M, K, N, nm, ph in [
        (16384, 7168, 2112, "Q_a+KV_a fused", "prefill"),
        (16,    7168, 2112, "Q_a+KV_a fused", "decode"),
        (16384, 1536, 1536, "Q_b",            "prefill"),
        (16,    1536, 12288, "Q_b",           "decode"),
        (16384, 512,  2048, "KV_b",           "prefill"),
        (16384, 1024, 7168, "O_proj",         "prefill"),
        (16,    8192, 7168, "O_proj",         "decode"),
    ]:
        add(bench_fp8_gemm, M, K, N, nm, -1, ph, label=f"{nm} {ph}")

    print("\n### Q_a+KV_a fused decode — dedicated bf16 min-latency kernel ###")
    add(bench_fused_a_decode, 16, 7168, 2112, "Q_a+KV_a fused (dsv3_fused_a)",
        label="Q_a fused_a decode")

    print("\n### MoE Router ###")
    add(bench_router_prefill, 16384, 7168, 384, "MoE Router", label="MoE Router prefill")
    add(bench_router_decode, 16, 7168, 384, "MoE Router", label="MoE Router decode")

    print("\n### MoE grouped GEMM ###")
    add(bench_grouped_fp8, 384, 512, 256, 7168, "MoE Down GroupGEMM", -1, "prefill",
        label="MoE Down grouped prefill")
    add(bench_grouped_fp8_masked, 8, 16, 7168, 2048, "MoE GateUp GroupGEMM", "decode",
        label="MoE GateUp masked decode")
    add(bench_grouped_fp8_masked, 8, 16, 1024, 7168, "MoE Down GroupGEMM", "decode",
        label="MoE Down masked decode")

    print("\n### MLA absorb BMM — bf16 torch.bmm ###")
    add(bench_bmm, 16, 8, 128, 512, "KV_b absorb BMM", -1, "decode", label="KV_b absorb BMM")
    add(bench_bmm, 16, 8, 512, 128, "V absorb BMM", -1, "decode", label="V absorb BMM")

    print("\n### MLA attention core ###")
    for s in ATTN_SEQLENS:
        add(bench_mla_prefill, s, f"MLA Attention core (seq={s})",
            label=f"MLA prefill seq={s}")
        add(bench_mla_decode, s, 16, f"MLA Attention decode (seq={s})",
            label=f"MLA decode seq={s}")

    out = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "logs" / "kimi_untested.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    print(f"\n# wrote {out} ({len(results)} entries)")


if __name__ == "__main__":
    main()
