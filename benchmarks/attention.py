"""Attention benchmarks (ops 26, 27).

op 27 — Prefill FlashAttn causal MHA (Q:[1,8,16384,256] K/V:[1,8,16384,256]):
    Real sglang path: sgl_kernel.flash_attn.flash_attn_with_kvcache (FA3).
    Proxy: flash_attn 2.x's flash_attn_func (requires head_dim <= 256).

op 26 — Flash Decoding MLA sparse MQA (q_absorbed [1,64,16,512] KV[1,1,T_kv_full,512]):
    Real sglang path: sgl_kernel.flash_mla.flash_mla_sparse_fwd (dedicated DSA sparse
    kernel; MQA k_head=1, page_size=1, topk-selected KV rows).
    Proxy: torch SDPA on the full KV OR SDPA + explicit gather to simulate topk.
    (flash_attn 2.7.4 caps head_dim at 256, so it can't run the D=512 MLA-absorbed shape.)
"""

from __future__ import annotations

import gc
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from benchmarks._util import bench, print_header, print_row
from shapes import ATTENTION_OPS

DEVICE = "cuda"


def _run_prefill_mha(op) -> dict | None:
    try:
        from flash_attn import flash_attn_func  # type: ignore
    except ImportError:
        return None
    q = torch.randn(op.B, op.T_q, op.H_q, op.D_qk, device=DEVICE, dtype=torch.float16)
    k = torch.randn(op.B, op.T_kv, op.H_kv, op.D_qk, device=DEVICE, dtype=torch.float16)
    v = torch.randn(op.B, op.T_kv, op.H_kv, op.D_v, device=DEVICE, dtype=torch.float16)
    try:
        us = bench(lambda: flash_attn_func(q, k, v, causal=op.causal),
                   warmup=3, iters=10)
    except Exception as e:
        return {"error": f"{type(e).__name__}: {str(e)[:80]}"}
    causal_factor = 0.5 if op.causal else 1.0
    flops = 4 * op.B * op.H_q * op.T_q * op.T_kv * op.D_qk * causal_factor
    return {"us": round(us, 2), "TFLOPS_fp16": round(flops / (us * 1e-6) / 1e12, 2),
            "backend": "flash_attn 2.x (proxy for FA3)"}


def _run_decode_mla_sparse(op) -> dict:
    # Dense-over-topk proxy: gather topk indices from a T_kv_full pool, then SDPA
    # over the topk subset. This models the flashmla_sparse workload with gather+SDPA.
    B, H_q, T_q, D_qk, D_v = op.B, op.H_q, op.T_q, op.D_qk, op.D_v
    topk, T_kv_full = op.T_kv, op.T_kv_full

    q = torch.randn(B, H_q, T_q, D_qk, device=DEVICE, dtype=torch.float16)
    k_full = torch.randn(B, 1, T_kv_full, D_qk, device=DEVICE, dtype=torch.float16)
    v_full = torch.randn(B, 1, T_kv_full, D_v, device=DEVICE, dtype=torch.float16)
    idx = torch.randint(0, T_kv_full, (topk,), device=DEVICE)

    def _sparse():
        k_sel = k_full[:, :, idx, :].expand(B, H_q, topk, D_qk).contiguous()
        v_sel = v_full[:, :, idx, :].expand(B, H_q, topk, D_v).contiguous()
        return F.scaled_dot_product_attention(q, k_sel, v_sel, is_causal=False)

    us_sparse = bench(_sparse, warmup=3, iters=10)
    flops = 4 * B * H_q * T_q * topk * D_qk

    # Also measure dense SDPA over the full T_kv_full for reference
    k_dense = torch.randn(B, 1, T_kv_full, D_qk, device=DEVICE, dtype=torch.float16).expand(B, H_q, T_kv_full, D_qk).contiguous()
    v_dense = torch.randn(B, 1, T_kv_full, D_v, device=DEVICE, dtype=torch.float16).expand(B, H_q, T_kv_full, D_v).contiguous()
    us_dense = bench(lambda: F.scaled_dot_product_attention(q, k_dense, v_dense, is_causal=False),
                     warmup=3, iters=10)
    flops_dense = 4 * B * H_q * T_q * T_kv_full * D_qk

    del q, k_full, v_full, k_dense, v_dense, idx
    gc.collect()
    torch.cuda.empty_cache()

    return {
        "sparse_us": round(us_sparse, 2),
        "sparse_TFLOPS": round(flops / (us_sparse * 1e-6) / 1e12, 2),
        "dense_us": round(us_dense, 2),
        "dense_TFLOPS": round(flops_dense / (us_dense * 1e-6) / 1e12, 2),
        "backend": "torch SDPA fp16 (proxy for flashmla_sparse_fwd)",
    }


results: list[dict] = []


def run() -> list[dict]:
    print_header()
    for op in ATTENTION_OPS:
        if op.phase == "prefill":
            r = _run_prefill_mha(op)
            if r is None:
                print_row(op.op_id, op.name, op.phase, "flash_attn N/A",
                          f"H={op.H_q} T={op.T_q} D={op.D_qk}", -1,
                          {"note": "flash_attn not installed"})
                continue
            if "error" in r:
                print_row(op.op_id, op.name, op.phase, "flash_attn",
                          f"H={op.H_q} T={op.T_q} D={op.D_qk}", -1, {"error": r["error"]})
                results.append({"op_id": op.op_id, "phase": op.phase, **r})
                continue
            row = {"op_id": op.op_id, "name": op.name, "phase": op.phase,
                   "shape": f"Q:[{op.B},{op.H_q},{op.T_q},{op.D_qk}] causal={op.causal}",
                   **r, "note": op.note}
            results.append(row)
            print_row(op.op_id, op.name, op.phase, r["backend"], row["shape"], r["us"],
                      {"TFLOPS_fp16": r["TFLOPS_fp16"]})
        else:
            r = _run_decode_mla_sparse(op)
            row = {"op_id": op.op_id, "name": op.name, "phase": op.phase,
                   "shape": f"Q:[{op.B},{op.H_q},{op.T_q},{op.D_qk}] KV[{op.B},1,{op.T_kv_full},{op.D_v}] topk={op.T_kv}",
                   **r, "note": op.note}
            results.append(row)
            print_row(op.op_id, op.name + " (sparse topk)", op.phase, r["backend"],
                      row["shape"], r["sparse_us"],
                      {"TFLOPS_fp16": r["sparse_TFLOPS"],
                       "dense_us": r["dense_us"]})
    return results


if __name__ == "__main__":
    run()
