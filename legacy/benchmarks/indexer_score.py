"""DSA indexer score benchmarks (ops 8, 22).

The actual formula is:
    score = sum_h( relu(Q_h @ K^T) * weights_h )     shape [M_q, M_k]
    topk_indices = argtopk(score, k=topk, dim=-1)    shape [M_q, topk]

Real path (fp8) is:
    prefill: deep_gemm.fp8_mqa_logits(q_padded_to_32heads, kv_fp8, weights, ks, ke)
    decode:  deep_gemm.fp8_paged_mqa_logits(...) OR aiter/cutedsl variants
    topk:    topk_transform_512_v2 (JIT CUDA, fused topk + page-table transform)

This fp16 einsum + torch.topk baseline is a WORST-CASE reference and typically
5-50x slower than the real fused fp8 kernel.
"""

from __future__ import annotations

import gc
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from benchmarks._util import bench, print_header, print_row
from shapes import INDEX_SCORE_OPS

DEVICE = "cuda"
DTYPE = torch.float16


def _bench_index_score(M_q: int, M_k: int, H_idx: int, D: int, topk: int) -> dict:
    q = torch.randn(M_q, H_idx, D, device=DEVICE, dtype=DTYPE)
    k = torch.randn(M_k, D, device=DEVICE, dtype=DTYPE)
    weights = torch.randn(M_q, H_idx, device=DEVICE, dtype=DTYPE)

    def _score():
        s = torch.einsum("mhd,kd->hmk", q, k).relu_()  # [H, M_q, M_k]
        s = s * weights.transpose(0, 1).unsqueeze(-1)
        return s.sum(dim=0)                             # [M_q, M_k]

    us_score = bench(_score, warmup=3, iters=10)
    score = _score()
    topk_eff = min(topk, M_k)
    us_topk = bench(lambda: torch.topk(score, k=topk_eff, dim=-1), warmup=3, iters=10)
    flops_score = 2 * H_idx * M_q * M_k * D
    tflops = flops_score / (us_score * 1e-6) / 1e12
    del q, k, weights, score
    gc.collect()
    torch.cuda.empty_cache()
    return {
        "score_us": round(us_score, 2),
        "topk_us": round(us_topk, 2),
        "total_us": round(us_score + us_topk, 2),
        "score_TFLOPS": round(tflops, 2),
        "topk_eff": topk_eff,
    }


results: list[dict] = []


def run() -> list[dict]:
    print_header()
    for op in INDEX_SCORE_OPS:
        r = _bench_index_score(op.M_q, op.M_k, op.H_idx, op.D, op.topk)
        row = {"op_id": op.op_id, "name": op.name, "phase": op.phase,
               "shape": f"Q:[{op.M_q},{op.H_idx},{op.D}] K:[{op.M_k},{op.D}] topk={r['topk_eff']}",
               **r, "note": op.note}
        results.append(row)
        print_row(op.op_id, op.name, op.phase,
                  "torch fp16 (naive; real kernel much faster)", row["shape"],
                  r["total_us"], {"score_us": r["score_us"], "topk_us": r["topk_us"],
                                  "TFLOPS_fp16": r["score_TFLOPS"]})
    return results


if __name__ == "__main__":
    run()
