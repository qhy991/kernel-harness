"""Elementwise / gather families: swiglu, rope, embedding, bmm, moe-combine."""
from ..config import KIMI as K, MM, recipe
from ..spec import TaskSpec, DECODE_SWEEP, sweep_for, var, const, expr, tensor

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
            goal=(f"优化 solution.py 相对 sglang sgl_kernel.silu_and_mul 基线（{op} {phase}）"
                  " 的 SwiGLU 延迟，跨全部 token sweep 领先；正确性对齐 sglang 输出（见 tolerance）"),
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
            goal=(f"优化 solution.py 相对 sglang apply_rope_with_cos_sin_cache_inplace 基线（{label} {op} {phase}）"
                  " 的 RoPE 延迟，跨全部 token sweep 领先；接口与 sglang 完全一致，正确性对齐两路输出"),
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
            goal=(f"优化 solution.py 相对 sglang F.embedding 基线（{label} Input Embedding {phase}）"
                  " 的 embedding gather 延迟，跨全部 token sweep 领先；正确性对齐 sglang 输出（见 tolerance）"),
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
            goal=(f"优化 solution.py 相对 sglang torch.bmm 基线（{op} decode）"
                  " 的 MLA absorb BMM 延迟，跨全部 token sweep 领先；正确性对齐 sglang 输出（见 tolerance）"),
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
            goal=(f"优化 solution.py 相对 sglang sgl_kernel.moe_sum 基线（MiniMax-M3 {op} {phase}）"
                  " 的 MoE combine 延迟，跨全部 token sweep 领先；接口与 sglang 完全一致，正确性对齐输出"),
            axes={"M": var(f"{phase} token count (sweep)"), "top_k": const(MM.topk),
                  "H": const(MM.hidden, "MiniMax-M3 hidden size")},
            inputs={"input_tensor": tensor(["M", "top_k", "H"], "bfloat16"),
                    "output_tensor": tensor(["M", "H"], "bfloat16")},
            outputs={"combined": tensor(["M", "H"], "bfloat16")})


def specs():
    out = []
    out += list(_swiglu())
    out += list(_rope_for(K, KIMI_ROPE, K.num_heads, 1, K.qk_rope, 4096, 10000))
    out += list(_rope_for(MM, MM_ROPE, MM.nq, MM.nkv, MM.rope_dim, 1048576, 5000000))
    out += list(_embedding_for(K, [("input_embedding_prefill", "prefill"), ("input_embedding_decode", "decode")]))
    out += list(_embedding_for(MM, [("mm_input_embedding_prefill", "prefill"), ("mm_input_embedding_decode", "decode")]))
    out += list(_bmm())
    out += list(_moe_combine())
    return out
