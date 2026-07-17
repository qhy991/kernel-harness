#!/usr/bin/env python3
"""Real-kernel microbench for Kimi K2.x DSA ops on B200.

Quantization path mirrors sglang's benchmark_deepgemm_fp8_gemm_blackwell.py.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Tuple

import torch
from deep_gemm import ceil_div

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from shapes import LINEAR_OPS, GROUPED_GEMM_OPS, BMM_OPS, INDEX_SCORE_OPS, ATTENTION_OPS  # noqa: E402

from sglang.srt.layers.quantization.fp8_kernel import (  # noqa: E402
    sglang_per_token_group_quant_fp8,
    w8a8_block_fp8_matmul_deepgemm,
)
from sglang.srt.layers.quantization.fp8_utils import requant_weight_ue8m0  # noqa: E402
import deep_gemm  # noqa: E402

BLOCK = 128


def sync():
    torch.cuda.synchronize()


def bench(fn, warmup=10, iters=50) -> float:
    for _ in range(warmup):
        fn()
    sync()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    sync()
    return (time.perf_counter() - t0) / iters * 1e6


def gemm_tflops(M, K, N, us):
    return 2.0 * M * K * N / (us * 1e-6) / 1e12


def per_block_cast_to_fp8(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    m, n = x.shape
    x_padded = torch.zeros(
        (ceil_div(m, 128) * 128, ceil_div(n, 128) * 128), dtype=x.dtype, device=x.device
    )
    x_padded[:m, :n] = x
    x_view = x_padded.view(-1, 128, x_padded.size(1) // 128, 128)
    x_amax = x_view.abs().float().amax(dim=(1, 3), keepdim=True).clamp(1e-4)
    x_scaled = (x_view * (448.0 / x_amax)).to(torch.float8_e4m3fn)
    return x_scaled.view_as(x_padded)[:m, :n].contiguous(), (x_amax / 448.0).view(
        x_view.size(0), x_view.size(2)
    )


def prep_fp8_gemm(M, K, N):
    K_a = (K + 127) // 128 * 128
    N_a = (N + 127) // 128 * 128
    x = torch.randn((M, K_a), device="cuda", dtype=torch.bfloat16)
    y = torch.randn((N_a, K_a), device="cuda", dtype=torch.bfloat16)
    y_fp8, y_scale = per_block_cast_to_fp8(y)
    dg_x_fp8, dg_x_scale = sglang_per_token_group_quant_fp8(
        x, BLOCK, column_major_scales=True, scale_tma_aligned=True, scale_ue8m0=True,
    )
    dg_y_fp8, dg_y_scale = requant_weight_ue8m0(y_fp8, y_scale, [BLOCK, BLOCK])
    return dg_x_fp8, dg_x_scale, dg_y_fp8, dg_y_scale, M, K_a, N_a


def bench_fp8_gemm(M, K, N, name, op_id, phase):
    x_fp8, x_s, y_fp8, y_s, M, K_a, N_a = prep_fp8_gemm(M, K, N)

    def run():
        w8a8_block_fp8_matmul_deepgemm(
            x_fp8, y_fp8, x_s, y_s, [BLOCK, BLOCK], output_dtype=torch.bfloat16
        )

    us = bench(run)
    return {
        "op_id": op_id, "name": name, "phase": phase,
        "backend": "deep_gemm w8a8_block_fp8 (blackwell path)",
        "shape": f"[{M},{K}]x[{K},{N}] aligned({K_a},{N_a})",
        "us": round(us, 2),
        "TFLOPS": round(gemm_tflops(M, K_a, N_a, us), 2),
    }


def bench_grouped_fp8(E, M, K, N, name, op_id, phase):
    K_a = (K + 127) // 128 * 128
    N_a = (N + 127) // 128 * 128
    m_sum = E * M
    a = torch.randn(m_sum, K_a, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(E, N_a, K_a, device="cuda", dtype=torch.bfloat16)
    a_fp8, a_s = deep_gemm.per_token_cast_to_fp8(a, False)
    b_fp8s, b_ss = zip(*[per_block_cast_to_fp8(b[e]) for e in range(E)])
    b_fp8 = torch.stack(list(b_fp8s))
    b_s = torch.stack(list(b_ss))
    out = torch.empty(m_sum, N_a, device="cuda", dtype=torch.bfloat16)
    m_indices = torch.arange(m_sum, device="cuda", dtype=torch.int32) // max(M, 1)

    def run():
        deep_gemm.m_grouped_fp8_gemm_nt_contiguous(
            (a_fp8, a_s), (b_fp8, b_s), out, m_indices
        )

    us = bench(run, warmup=5, iters=20)
    return {
        "op_id": op_id, "name": name, "phase": phase,
        "backend": "deep_gemm.m_grouped_fp8_gemm_nt_contiguous",
        "shape": f"{E}x[{M},{K}]x[{K},{N}]",
        "us": round(us, 2),
        "TFLOPS": round(gemm_tflops(m_sum, K_a, N_a, us), 2),
    }


def bench_bmm(M, H, IN, OUT, name, op_id, phase):
    a = torch.randn(H, M, IN, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(H, IN, OUT, device="cuda", dtype=torch.bfloat16)

    def run():
        torch.bmm(a, b)

    us = bench(run)
    return {
        "op_id": op_id, "name": name, "phase": phase,
        "backend": "torch.bmm bf16",
        "shape": f"[{M},{H},{IN}]x[{H},{IN},{OUT}]",
        "us": round(us, 2),
        "TFLOPS": round(gemm_tflops(M * H, IN, OUT, us), 2),
    }


def bench_mqa(M_q, M_k, H, D, topk, name, op_id, phase):
    q = torch.randn(M_q * H, D, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(M_k, D, device="cuda", dtype=torch.bfloat16)
    q_fp8, q_s = deep_gemm.per_token_cast_to_fp8(q, False)
    k_fp8, k_s = deep_gemm.per_token_cast_to_fp8(k, False)
    q_fp8 = q_fp8.view(M_q, H, D)
    # deep_gemm.fp8_mqa_logits expects k_scale as 1-D [M_k]
    k_s = k_s.squeeze(-1).contiguous()
    weights = torch.ones(M_q, H, device="cuda", dtype=torch.float32)
    # ragged: each query sees full K range [0, M_k)
    ks = torch.zeros(M_q, device="cuda", dtype=torch.int32)
    ke = torch.full((M_q,), M_k, device="cuda", dtype=torch.int32)

    def run():
        deep_gemm.fp8_mqa_logits(
            q_fp8, (k_fp8, k_s), weights, ks, ke, clean_logits=False
        )

    us = bench(run, warmup=5, iters=20)
    flops = 2.0 * M_q * H * M_k * D
    return {
        "op_id": op_id, "name": name, "phase": phase,
        "backend": "deep_gemm.fp8_mqa_logits",
        "shape": f"Q:[{M_q},{H},{D}] K:[{M_k},{D}] topk={topk}",
        "us": round(us, 2),
        "TFLOPS": round(flops / (us * 1e-6) / 1e12, 2),
    }


def main():
    print(f"# GPU: {torch.cuda.get_device_name(0)}  torch={torch.__version__}")
    results = []

    print("\n### linear fp8 ###")
    for op in LINEAR_OPS:
        try:
            r = bench_fp8_gemm(op.M, op.K, op.N, op.name, op.op_id, op.phase)
            results.append(r)
            print(f"  {op.op_id:2d} | {op.name:28s} | {op.phase:7s} | {r['us']:10.2f} us | {r['TFLOPS']:8.2f} TFLOPS")
        except Exception as e:
            print(f"  {op.op_id:2d} | {op.name:28s} | FAIL: {type(e).__name__}: {e}")
            results.append({"op_id": op.op_id, "name": op.name, "error": f"{type(e).__name__}: {e}"})

    print("\n### grouped fp8 ###")
    for op in GROUPED_GEMM_OPS:
        try:
            r = bench_grouped_fp8(op.E, op.M, op.K, op.N, op.name, op.op_id, op.phase)
            results.append(r)
            print(f"  {op.op_id:2d} | {op.name:28s} | {op.phase:7s} | {r['us']:10.2f} us | {r['TFLOPS']:8.2f} TFLOPS")
        except Exception as e:
            print(f"  {op.op_id:2d} | {op.name:28s} | FAIL: {type(e).__name__}: {e}")
            results.append({"op_id": op.op_id, "name": op.name, "error": f"{type(e).__name__}: {e}"})

    print("\n### bmm ###")
    for op in BMM_OPS:
        try:
            r = bench_bmm(op.M, op.H, op.IN, op.OUT, op.name, op.op_id, op.phase)
            results.append(r)
            print(f"  {op.op_id:2d} | {op.name:28s} | {op.phase:7s} | {r['us']:10.2f} us | {r['TFLOPS']:8.2f} TFLOPS")
        except Exception as e:
            print(f"  {op.op_id:2d} | {op.name:28s} | FAIL: {type(e).__name__}: {e}")
            results.append({"op_id": op.op_id, "name": op.name, "error": f"{type(e).__name__}: {e}"})

    print("\n### indexer ###")
    for op in INDEX_SCORE_OPS:
        try:
            r = bench_mqa(op.M_q, op.M_k, op.H_idx, op.D, op.topk, op.name, op.op_id, op.phase)
            results.append(r)
            print(f"  {op.op_id:2d} | {op.name:28s} | {op.phase:7s} | {r['us']:10.2f} us | {r['TFLOPS']:8.2f} TFLOPS")
        except Exception as e:
            print(f"  {op.op_id:2d} | {op.name:28s} | FAIL: {type(e).__name__}: {e}")
            results.append({"op_id": op.op_id, "name": op.name, "error": f"{type(e).__name__}: {e}"})

    print("\n### attention ###")
    for op in ATTENTION_OPS:
        try:
            if op.op_id == 26:
                from sgl_kernel.flash_mla import flash_mla_sparse_fwd
                s_q, h_q, d = op.T_q, op.H_q, op.D_qk
                s_kv, topk = op.T_kv_full, op.T_kv
                h_kv = 1
                q = torch.randn(s_q, h_q, d, device="cuda", dtype=torch.bfloat16)
                kv = torch.randn(s_kv, h_kv, d, device="cuda", dtype=torch.bfloat16)
                # API wants (s_q, h_kv, topk)
                indices = torch.randint(0, s_kv, (s_q, h_kv, topk), device="cuda", dtype=torch.int32)
                sm_scale = d ** -0.5

                def run():
                    flash_mla_sparse_fwd(q, kv, indices, sm_scale)

                us = bench(run, warmup=5, iters=20)
                r = {
                    "op_id": 26, "name": op.name, "phase": op.phase,
                    "backend": "sgl_kernel.flash_mla_sparse_fwd",
                    "shape": f"Q:[{s_q},{h_q},{d}] KV:[{s_kv},{h_kv},{d}] topk={topk}",
                    "us": round(us, 2),
                }
                results.append(r)
                print(f"  26 | {op.name:28s} | {us:10.2f} us")
            else:
                # op 27: prefer flash_attn; on B200 (sm100) sgl_kernel FA3 is N/A,
                # so fall back to flashinfer (DSA SM100+ production path).
                B, H, T, D = op.B, op.H_q, op.T_q, op.D_qk
                backend = None
                run = None
                try:
                    from flash_attn import flash_attn_func  # type: ignore

                    q = torch.randn(B, T, H, D, device="cuda", dtype=torch.bfloat16)
                    k = torch.randn(B, T, H, D, device="cuda", dtype=torch.bfloat16)
                    v = torch.randn(B, T, H, D, device="cuda", dtype=torch.bfloat16)

                    def run():
                        flash_attn_func(q, k, v, causal=op.causal)

                    backend = "flash_attn.flash_attn_func"
                except Exception:
                    import flashinfer

                    q = torch.randn(T, H, D, device="cuda", dtype=torch.bfloat16)
                    k = torch.randn(T, H, D, device="cuda", dtype=torch.bfloat16)
                    v = torch.randn(T, H, D, device="cuda", dtype=torch.bfloat16)

                    def run():
                        flashinfer.single_prefill_with_kv_cache(
                            q, k, v, causal=op.causal
                        )

                    backend = "flashinfer.single_prefill_with_kv_cache"

                us = bench(run, warmup=3, iters=10)
                flops = 4.0 * B * H * T * T * D * (0.5 if op.causal else 1.0)
                r = {
                    "op_id": 27,
                    "name": op.name,
                    "phase": op.phase,
                    "backend": backend,
                    "shape": f"Q/K/V:[{B},{H},{T},{D}] causal={op.causal}",
                    "us": round(us, 2),
                    "TFLOPS": round(flops / (us * 1e-6) / 1e12, 2),
                }
                results.append(r)
                print(
                    f"  27 | {op.name:28s} | {op.phase:7s} | {r['us']:10.2f} us | "
                    f"{r['TFLOPS']:8.2f} TFLOPS | {backend}"
                )
        except Exception as e:
            print(f"  {op.op_id:2d} | {op.name:28s} | FAIL: {type(e).__name__}: {e}")
            results.append({"op_id": op.op_id, "name": op.name, "error": f"{type(e).__name__}: {e}"})

    out = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "logs" / "kimi_real_kernel.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    print(f"\n# wrote {out}")


if __name__ == "__main__":
    main()
