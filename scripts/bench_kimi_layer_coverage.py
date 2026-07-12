#!/usr/bin/env python3
"""Benchmark Kimi layer-type variants not covered by the single-layer DSA harness.

Adds:
- Dense FFN (layers 0 .. first_k_dense_replace-1)
- MoE/shared with official Kimi K2 dimensions (7168 hidden, 384 experts)
- Embedding, RMSNorm, LM Head
- DSA index_topk skip path (metadata only — no kernel when skip_topk=True)
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

from kimi_model_config import DSA_V32, KIMI_K2, KimiProfile, dense_ffn_shapes, shared_expert_shapes  # noqa: E402
from bench_kimi_real_kernels import bench_fp8_gemm, bench_grouped_fp8, sync  # noqa: E402


def bench(fn, warmup=5, iters=20) -> float:
    for _ in range(warmup):
        fn()
    sync()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    sync()
    return (time.perf_counter() - t0) / iters * 1e6


def bench_rmsnorm(m: int, hidden: int) -> dict:
    try:
        from sgl_kernel import rmsnorm
    except Exception:
        x = torch.randn(m, hidden, device="cuda", dtype=torch.bfloat16)
        w = torch.ones(hidden, device="cuda", dtype=torch.bfloat16)

        def run():
            torch.nn.functional.rms_norm(x, (hidden,), w, eps=1e-6)

        backend = "torch.nn.functional.rms_norm"
    else:
        x = torch.randn(m, hidden, device="cuda", dtype=torch.bfloat16)
        w = torch.ones(hidden, device="cuda", dtype=torch.bfloat16)

        def run():
            rmsnorm(x, w, eps=1e-6)

        backend = "sgl_kernel.rmsnorm"

    us = bench(run, warmup=10, iters=50)
    return {
        "backend": backend,
        "shape": f"[{m},{hidden}]",
        "us": round(us, 2),
    }


def bench_embedding(m: int, vocab: int, hidden: int) -> dict:
    emb = torch.nn.Embedding(vocab, hidden, device="cuda", dtype=torch.bfloat16)
    ids = torch.randint(0, vocab, (m,), device="cuda", dtype=torch.long)

    def run():
        emb(ids)

    us = bench(run, warmup=10, iters=50)
    return {
        "backend": "torch.nn.Embedding",
        "shape": f"ids[{m}] -> [{m},{hidden}] vocab={vocab}",
        "us": round(us, 2),
    }


def bench_lm_head(m: int, hidden: int, vocab: int) -> dict:
    w = torch.randn(hidden, vocab, device="cuda", dtype=torch.bfloat16)
    x = torch.randn(m, hidden, device="cuda", dtype=torch.bfloat16)

    def run():
        torch.mm(x, w)

    us = bench(run, warmup=5, iters=20)
    flops = 2.0 * m * hidden * vocab
    return {
        "backend": "torch.mm bf16 (lm_head)",
        "shape": f"[{m},{hidden}]x[{hidden},{vocab}]",
        "us": round(us, 2),
        "TFLOPS": round(flops / (us * 1e-6) / 1e12, 2),
    }


def run_profile(p: KimiProfile) -> list[dict]:
    out: list[dict] = []
    tag = p.name

    # --- Dense FFN (layer 0 .. first_k_dense-1) ---
    for phase in ("prefill", "decode"):
        m = p.m_prefill if phase == "prefill" else p.m_decode
        gu, dn = dense_ffn_shapes(p, phase)
        for name, shape, attr in [
            ("Dense FFN GateUp", gu, "mlp.gate_up_proj"),
            ("Dense FFN Down", dn, "mlp.down_proj"),
        ]:
            m_, k, n = shape
            try:
                r = bench_fp8_gemm(m_, k, n, name, op_id=-1, phase=phase)
                r.update(
                    {
                        "profile": tag,
                        "layer_kind": "dense_ffn",
                        "layer_id": f"0-{p.first_k_dense - 1}",
                        "attr": attr,
                        "category": "ffn",
                    }
                )
                out.append(r)
                print(f"  [{tag}] dense {name:18s} {phase:7s} {r['us']:8.2f} us")
            except Exception as e:
                out.append(
                    {
                        "profile": tag,
                        "layer_kind": "dense_ffn",
                        "name": name,
                        "phase": phase,
                        "error": str(e),
                    }
                )

    # --- Shared expert on MoE layers (compare to dense FFN) ---
    for phase in ("prefill", "decode"):
        gu, dn = shared_expert_shapes(p, phase)
        for name, shape, attr in [
            ("MoE Shared GateUp", gu, "mlp.shared_experts.gate_up_proj"),
            ("MoE Shared Down", dn, "mlp.shared_experts.down_proj"),
        ]:
            m_, k, n = shape
            try:
                r = bench_fp8_gemm(m_, k, n, name, op_id=-1, phase=phase)
                r.update(
                    {
                        "profile": tag,
                        "layer_kind": "moe_shared",
                        "layer_id": f">={p.first_k_dense}",
                        "attr": attr,
                        "category": "ffn",
                    }
                )
                out.append(r)
                print(f"  [{tag}] shared {name:18s} {phase:7s} {r['us']:8.2f} us")
            except Exception as e:
                out.append({"profile": tag, "name": name, "phase": phase, "error": str(e)})

    # --- MoE grouped with correct expert count (Kimi: 384) ---
    for phase in ("prefill",):
        e, m_e, k, n = (p.n_routed, 512, p.hidden, p.moe_inter // p.tp)
        try:
            r = bench_grouped_fp8(e, m_e, k, n, "MoE GateUp GroupGEMM (profile)", -1, phase)
            r.update(
                {
                    "profile": tag,
                    "layer_kind": "moe_routed",
                    "layer_id": f">={p.first_k_dense}",
                    "attr": "mlp.experts.w13_weight",
                    "category": "moe",
                    "note": f"E={p.n_routed}",
                }
            )
            out.append(r)
            print(f"  [{tag}] moe GateUp prefill {r['us']:8.2f} us E={e}")
        except Exception as e:
            out.append({"profile": tag, "name": "MoE GateUp", "error": str(e)})

    # --- Per-layer stack: norm + embed + lm_head ---
    for phase in ("prefill", "decode"):
        m = p.m_prefill if phase == "prefill" else p.m_decode
        r = bench_rmsnorm(m, p.hidden)
        r.update(
            {
                "profile": tag,
                "name": "Input RMSNorm",
                "phase": phase,
                "layer_kind": "all_layers",
                "attr": "input_layernorm",
                "category": "norm",
            }
        )
        out.append(r)
        print(f"  [{tag}] RMSNorm {phase:7s} {r['us']:8.2f} us")

    if tag == "kimi_k2":
        for phase in ("prefill", "decode"):
            m = p.m_prefill if phase == "prefill" else p.m_decode
            r = bench_embedding(m, p.vocab, p.hidden)
            r.update(
                {
                    "profile": tag,
                    "name": "Token Embedding",
                    "phase": phase,
                    "layer_kind": "input",
                    "attr": "embed_tokens",
                    "category": "embed",
                }
            )
            out.append(r)
            print(f"  [{tag}] Embed {phase:7s} {r['us']:8.2f} us")

            r = bench_lm_head(m, p.hidden, p.vocab)
            r.update(
                {
                    "profile": tag,
                    "name": "LM Head",
                    "phase": phase,
                    "layer_kind": "output",
                    "attr": "lm_head",
                    "category": "lm_head",
                }
            )
            out.append(r)
            print(f"  [{tag}] LMHead {phase:7s} {r['us']:8.2f} us TFLOPS={r.get('TFLOPS')}")

    # --- DSA skip_topk metadata ---
    if p.has_dsa and p.index_topk_freq > 1:
        for lid in range(1, 6):
            skip = max(lid - 1, 0) % p.index_topk_freq != 0
            out.append(
                {
                    "profile": tag,
                    "name": "Index_Score",
                    "phase": "both",
                    "layer_kind": "dsa_indexer",
                    "layer_id": lid,
                    "skip_topk": skip,
                    "note": "no fp8_mqa_logits when skip_topk=True; reuses prev topk",
                    "category": "indexer_path",
                }
            )

    return out


def main():
    print(f"# GPU: {torch.cuda.get_device_name(0)}")
    results = []
    for p in (KIMI_K2, DSA_V32):
        print(f"\n### profile: {p.name} (hidden={p.hidden}, dense_layers={p.first_k_dense}) ###")
        results.extend(run_profile(p))

    out = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "logs" / "kimi_layer_coverage.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    print(f"\n# wrote {out} ({len(results)} entries)")


if __name__ == "__main__":
    main()
