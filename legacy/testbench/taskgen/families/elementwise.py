"""Elementwise / gather families: swiglu, rope, embedding, bmm, moe-combine."""
from ..config import KIMI as K, MM, GLM52 as G, recipe
from ..spec import (TaskSpec, DECODE_SWEEP, GLM52_PREFILL_SWEEP, GLM52_DECODE_SWEEP,
                    sweep_for, var, const, expr, tensor)

# Kimi swiglu: (name, op, phase, I2 = 2*intermediate/TP)
SWIGLU = [
    ("dense_ffn_swiglu_prefill",  "Dense FFN SwiGLU",  "prefill", 2 * K.dense_inter_tp),
    ("dense_ffn_swiglu_decode",   "Dense FFN SwiGLU",  "decode",  2 * K.dense_inter_tp),
    ("moe_shared_swiglu_prefill", "MoE Shared SwiGLU", "prefill", 2 * K.moe_inter_tp),
    ("moe_shared_swiglu_decode",  "MoE Shared SwiGLU", "decode",  2 * K.moe_inter_tp),
]

# rope: (name, op, phase, num_heads, kv_heads, rope_dim, max_pos, theta) + model cfg
KIMI_ROPE = [("mla_qk_rope_prefill", "MLA q/k RoPE", "prefill"), ("mla_qk_rope_decode", "MLA q/k RoPE", "decode")]
MM_ROPE = [("mm_qk_rope_prefill", "GQA q/k RoPE", "prefill"), ("mm_qk_rope_decode", "GQA q/k RoPE", "decode")]

# Kimi bmm (MLA weight-absorb, decode): (name, op, Bh, K, N)
BMM = [
    ("q_nope_absorb_bmm_decode", "q_nope-absorb BMM", K.num_heads, K.qk_nope, K.kv_lora),
    ("v_absorb_bmm_decode",      "v-absorb BMM",      K.num_heads, K.kv_lora, K.v_head),
]


def _swiglu():
    r = recipe("swiglu.py")
    for name, op, phase, i2 in SWIGLU:
        i = i2 // 2
        yield TaskSpec(
            model=K.model, name=name, op=op, family="swiglu", phase=phase, hf_id=K.hf_id,
            recipe=r, sweep=sweep_for(phase), backend="sgl_kernel silu_and_mul (blackwell)",
            meta={"I2": i2},
            description=(
                f"Kimi-K2.7 {op} ({phase}), SiLU-and-mul (sgl_kernel.silu_and_mul). "
                f"out[M,{i}] = silu(x[:, :{i}]) * x[:, {i}:], x is the [M,{i2}] gate|up projection. "
                f"Baseline = sglang production kernel; beat its latency while matching output."),
            goal=(f"Optimize solution.py against SGLang's sgl_kernel.silu_and_mul "
                  f"baseline for {op} {phase}; beat every token workload and match "
                  "SGLang output within the declared tolerance."),
            axes={"M": var(f"{phase} token count (sweep)"),
                  "I2": const(i2, "gate|up width = 2 * intermediate/TP"),
                  "I": expr("I2//2", "output width = intermediate/TP")},
            inputs={"x": tensor(["M", "I2"], "bfloat16")},
            outputs={"output": tensor(["M", "I"], "bfloat16")})


def _rope_for(cfg, table, nH, kvH, D, max_pos, theta):
    r = recipe("rope.py")
    for name, op, phase in table:
        label = "Kimi-K2.7" if cfg is K else "MiniMax-M3"
        yield TaskSpec(
            model=cfg.model, name=name, op=op, family="rope", phase=phase, hf_id=cfg.hf_id,
            recipe=r, sweep=sweep_for(phase), backend="apply_rope_with_cos_sin_cache_inplace (blackwell)",
            meta={"num_heads": nH, "kv_heads": kvH, "rope_dim": D, "interface_exact": True},
            description=(
                f"{label} {op} ({phase}), in-place RoPE on the rope slice "
                f"(apply_rope_with_cos_sin_cache_inplace). Rotates q[M,{nH},{D}] and k[M,{kvH},{D}] "
                f"by cos/sin at `positions`. INTERFACE-EXACT drop-in. Baseline = sglang production "
                f"kernel; beat its latency while matching BOTH outputs."),
            goal=(f"Optimize solution.py against SGLang's "
                  f"apply_rope_with_cos_sin_cache_inplace baseline for {label} {op} "
                  f"{phase}; beat every token workload while preserving the exact "
                  "SGLang interface and matching both outputs."),
            axes={"M": var(f"{phase} token count (sweep)"), "num_heads": const(nH),
                  "kv_heads": const(kvH), "rope_dim": const(D, "rope_dim"),
                  "max_pos": const(max_pos), "rope_theta": const(theta)},
            inputs={"q": tensor(["M", "num_heads", "rope_dim"], "bfloat16"),
                    "k": tensor(["M", "kv_heads", "rope_dim"], "bfloat16"),
                    "cos_sin_cache": tensor(["max_pos", "rope_dim"], "float32"),
                    "positions": tensor(["M"], "int64")},
            outputs={"q_out": tensor(["M", "num_heads", "rope_dim"], "bfloat16"),
                     "k_out": tensor(["M", "kv_heads", "rope_dim"], "bfloat16")})


def _embedding_for(cfg, names):
    r = recipe("embedding.py")
    label = "Kimi-K2.7" if cfg is K else "MiniMax-M3"
    for name, phase in names:
        yield TaskSpec(
            model=cfg.model, name=name, op="Input Embedding", family="embedding", phase=phase,
            hf_id=cfg.hf_id, recipe=r, sweep=sweep_for(phase), backend="torch F.embedding (blackwell)",
            meta={"V": cfg.vocab, "H": cfg.hidden},
            description=(
                f"{label} Input Embedding ({phase}), vocab gather (F.embedding, unquantized "
                f"VocabParallelEmbedding path). out[M,{cfg.hidden}] = weight[input_ids], vocab={cfg.vocab}. "
                f"Baseline = sglang production op; beat its latency while matching output."),
            goal=(f"Optimize solution.py against SGLang's F.embedding baseline for "
                  f"{label} Input Embedding {phase}; beat every token workload and "
                  "match SGLang output within the declared tolerance."),
            axes={"M": var(f"{phase} token count (sweep)"), "V": const(cfg.vocab, "vocab size"),
                  "H": const(cfg.hidden, "hidden size")},
            inputs={"input_ids": tensor(["M"], "int64"), "weight": tensor(["V", "H"], "bfloat16")},
            outputs={"output": tensor(["M", "H"], "bfloat16")})


def _bmm():
    r = recipe("bmm.py")
    for name, op, bh, kk, nn in BMM:
        yield TaskSpec(
            model=K.model, name=name, op=op, family="bmm", phase="decode", hf_id=K.hf_id,
            recipe=r, sweep=DECODE_SWEEP, backend="torch.bmm bf16 (blackwell)",
            meta={"Bh": bh, "K": kk, "N": nn},
            description=(
                f"Kimi-K2.7 {op} (decode), MLA weight-absorb batched matmul (torch.bmm, bf16 default). "
                f"out[{bh},M,{nn}] = bmm(a[{bh},M,{kk}], b[{bh},{kk},{nn}]), Bh=num_heads. "
                f"Baseline = sglang production op; beat its latency while matching output."),
            goal=(f"Optimize solution.py against SGLang's torch.bmm baseline for "
                  f"{op} decode; beat every token workload and match SGLang output "
                  "within the declared tolerance."),
            axes={"M": var("decode token count (sweep)"), "Bh": const(bh, "num_heads (batch)"),
                  "K": const(kk), "N": const(nn)},
            inputs={"a": tensor(["Bh", "M", "K"], "bfloat16"), "b": tensor(["Bh", "K", "N"], "bfloat16")},
            outputs={"output": tensor(["Bh", "M", "N"], "bfloat16")})


def _moe_combine():
    r = recipe("minimax_moe_combine.py")
    for name, phase in [("mm_moe_combine_prefill", "prefill"), ("mm_moe_combine_decode", "decode")]:
        op = "MoE Combine"
        yield TaskSpec(
            model=MM.model, name=name, op=op, family="moe-combine", phase=phase, hf_id=MM.hf_id,
            recipe=r, sweep=sweep_for(phase), backend="sgl_kernel moe_sum (blackwell)",
            meta={"top_k": MM.topk, "H": MM.hidden, "interface_exact": True},
            description=(
                f"MiniMax-M3 {op} ({phase}), sum the top_k={MM.topk} expert outputs per token "
                f"(sgl_kernel.moe_sum, in-place DPS). output[M,{MM.hidden}] = "
                f"sum(input[M,{MM.topk},{MM.hidden}], dim=1). INTERFACE-EXACT drop-in. Baseline = "
                f"sglang production kernel; beat its latency while matching output."),
            goal=(f"Optimize solution.py against SGLang's sgl_kernel.moe_sum baseline "
                  f"for MiniMax-M3 {op} {phase}; beat every token workload while "
                  "preserving the exact SGLang interface and matching output."),
            axes={"M": var(f"{phase} token count (sweep)"), "top_k": const(MM.topk),
                  "H": const(MM.hidden, "MiniMax-M3 hidden size")},
            inputs={"input_tensor": tensor(["M", "top_k", "H"], "bfloat16"),
                    "output_tensor": tensor(["M", "H"], "bfloat16")},
            outputs={"combined": tensor(["M", "H"], "bfloat16")})


def _glm52_routed_swiglu():
    """Routed expert SwiGLU fused with FP8 post-quant (production MoE runner path)."""
    r = recipe("glm52_swiglu_fp8_quant.py")
    i2 = G.gateup_n  # 4096
    for name, phase, layout, sweep in [
        ("routed_swiglu_prefill", "prefill", 0, GLM52_PREFILL_SWEEP),
        ("routed_swiglu_decode",  "decode",  1, GLM52_DECODE_SWEEP),
    ]:
        yield TaskSpec(
            model=G.model, name=name, op="Routed Expert SwiGLU+FP8 Quant",
            family="swiglu-fp8-quant", phase=phase, hf_id=G.hf_id, recipe=r,
            sweep=sweep,
            backend=("silu_and_mul_contig_post_quant" if layout == 0
                     else "silu_and_mul_masked_post_quant"),
            performance_model={"kind": "swiglu-fp8", "family": "swiglu-fp8-quant",
                               "stage": "swiglu"},
            workload_metrics=["achieved_gbps", "local_assignments", "active_experts",
                              "tokens_per_expert_cv", "us_per_token"],
            description=(
                f"GLM-5.2 Routed Expert SwiGLU + FP8 post-quant ({phase}, EP-local E={G.ep_local}). "
                f"gate|up width I2={i2}; fused act+quant matches the DeepGEMM MoE runner launch "
                f"that feeds Down. Routing = fixed-seed top-{G.moe_topk}/{G.n_routed_experts}. "
                f"Baseline = SGLang production fused kernel."),
            goal=(f"Optimize solution.py against SGLang's fused silu_and_mul_*_post_quant "
                  f"baseline for GLM-5.2 routed SwiGLU {phase}; beat the sweep and match "
                  "both FP8 payload and scale outputs."),
            axes={"M": var(f"{phase} token/batch population (sweep)"),
                  "E": const(G.ep_local),
                  "I2": const(i2, "gate|up = 2 * moe_intermediate"),
                  "I": expr("I2//2"),
                  "layout": const(layout, "0=contiguous, 1=masked"),
                  "n_global": const(G.n_routed_experts),
                  "topk": const(G.moe_topk)},
            inputs={"gate_up": tensor(None, "bfloat16"),
                    "out_fp8": tensor(None, "float8_e4m3fn"),
                    "out_scale": tensor(None, "int32"),
                    "masked_m": tensor(["E"], "int32"),
                    "m_indices": tensor(None, "int32"),
                    "layout": tensor(None, "int32"),
                    "group_size": tensor(None, "int32")},
            outputs={"q_fp8": tensor(None, "float8_e4m3fn"),
                     "scale": tensor(None, "int32")},
            meta={"E": G.ep_local, "I2": i2, "ep": G.ep, "tp": G.tp,
                  "dp": G.dp, "deployment": G.deployment})


def specs():
    out = []
    out += list(_swiglu())
    out += list(_rope_for(K, KIMI_ROPE, K.num_heads, 1, K.qk_rope, 4096, 10000))
    out += list(_rope_for(MM, MM_ROPE, MM.nq, MM.nkv, MM.rope_dim, 1048576, 5000000))
    out += list(_embedding_for(K, [("input_embedding_prefill", "prefill"), ("input_embedding_decode", "decode")]))
    out += list(_embedding_for(MM, [("mm_input_embedding_prefill", "prefill"), ("mm_input_embedding_decode", "decode")]))
    out += list(_bmm())
    out += list(_moe_combine())
    out += list(_glm52_routed_swiglu())
    return out
