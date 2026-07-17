"""MiniMax-M3 DSA sparse-attention stack (op29-34). Uses the single shared SGLANG_DIR."""
from ..config import MM, recipe
from ..spec import (TaskSpec, DECODE_SWEEP, EXACT_TOL, sweep_for, var, const, expr, tensor)


def _qknorm_rope():
    r = recipe("minimax_dsa_qknorm_rope.py")
    qkv_dim = (MM.nq + 2 * MM.nkv) * MM.hd
    for name, phase in [("mm_dsa_qknorm_rope_prefill", "prefill"), ("mm_dsa_qknorm_rope_decode", "decode")]:
        op = "DSA fused Gemma-QK-norm + partial RoPE"
        yield TaskSpec(
            model=MM.model, name=name, op=op, family="dsa-qknorm-rope", phase=phase, hf_id=MM.hf_id,
            recipe=r, sweep=sweep_for(phase),
            backend="minimax_qknorm_rope fused C++ (blackwell)",
            description=(
                f"MiniMax-M3 {op} ({phase}), jit_kernel.minimax_qknorm_rope (fused C++, op29/35): "
                f"per-head Gemma-RMSNorm of q/k in a packed QKV[T,{qkv_dim}] (nq={MM.nq},nk=nv={MM.nkv}, "
                f"head_dim={MM.hd}) + partial RoPE (rope_dim={MM.rope_dim}, theta=5e6). Single-launch, "
                f"~2.4x vs two launches. Baseline = SGLang production kernel."),
            goal=(f"Optimize solution.py against SGLang's minimax_qknorm_rope "
                  f"baseline for {op} {phase}; beat the full sweep and match SGLang "
                  "output within the declared tolerance."),
            axes={"M": var(f"{phase} token count (sweep)"), "qkv_dim": const(qkv_dim, "(nq+nk+nv)*head_dim"),
                  "head_dim": const(MM.hd), "rope_dim": const(MM.rope_dim),
                  "max_pos": const(1048576, "M3 max_position_embeddings")},
            inputs={"qkv": tensor(["M", "qkv_dim"], "bfloat16"), "q_weight": tensor(["head_dim"], "bfloat16"),
                    "k_weight": tensor(["head_dim"], "bfloat16"),
                    "cos_sin_cache": tensor(["max_pos", "rope_dim"], "float32"),
                    "positions": tensor(["M"], "int64")},
            outputs={"qkv_out": tensor(["M", "qkv_dim"], "bfloat16")})


def _prefill_topk():
    r = recipe("minimax_dsa_prefill_topk.py")
    for name, ll in [("mm_dsa_prefill_topk_L2048", 2048), ("mm_dsa_prefill_topk_L8192", 8192)]:
        op = "DSA prefill Indexer score + top-k"
        yield TaskSpec(
            model=MM.model, name=name, op=op, family="dsa-prefill-topk", phase="prefill", hf_id=MM.hf_id,
            recipe=r, sweep=DECODE_SWEEP, tolerance=dict(EXACT_TOL),
            backend="flash_prefill_with_topk_index Triton (blackwell)",
            meta={"L": ll, "oracle": "exact-index"},
            description=(
                f"MiniMax-M3 {op} (prefill, L={ll}), flash_prefill_with_topk_index (single-launch Triton, op31): "
                f"per-block index score + top-{MM.topk_blocks} in one kernel. num_index_heads={MM.idx_heads}, "
                f"idx_dim={MM.idx_dim}, block_size={MM.block}. EXACT-index oracle. B=sweep. "
                f"Baseline = SGLang production kernel."),
            goal=(f"Optimize solution.py against SGLang's "
                  f"flash_prefill_with_topk_index baseline for {op} at L={ll}; beat "
                  "every batch workload and match block indices exactly (atol=0)."),
            axes={"M": var("prefill batch (sweep)"), "L": const(ll, "per-seq context length"),
                  "index_heads": const(MM.idx_heads), "idx_dim": const(MM.idx_dim),
                  "topk_blocks": const(MM.topk_blocks), "slots": expr("M*L", "paged index cache slots"),
                  "one": const(1), "Mp1": expr("M+1")},
            inputs={"idx_q": tensor(["M", "index_heads", "idx_dim"], "bfloat16"),
                    "idx_k": tensor(["slots", "one", "idx_dim"], "bfloat16"),
                    "req_to_token": tensor(["M", "L"], "int32"), "slot_ids": tensor(["M"], "int64"),
                    "cu_seqlens": tensor(["Mp1"], "int32"), "seq_lens": tensor(["M"], "int32"),
                    "prefix_lens": tensor(["M"], "int32"), "seqlen": tensor(None, "int32")},
            outputs={"topk_idx": tensor(["index_heads", "M", "topk_blocks"], "int32")})


def _prefill_attn():
    r = recipe("minimax_dsa_prefill_attn.py")
    for name, ll in [("mm_dsa_prefill_attn_L2048", 2048), ("mm_dsa_prefill_attn_L8192", 8192)]:
        op = "DSA prefill main sparse attention"
        yield TaskSpec(
            model=MM.model, name=name, op=op, family="dsa-prefill-attn", phase="prefill", hf_id=MM.hf_id,
            recipe=r, sweep=DECODE_SWEEP,
            backend="flash_prefill_with_gqa_share_sparse Triton (blackwell)",
            meta={"L": ll, "num_q_heads": MM.nq, "num_kv_heads": MM.nkv, "head_dim": MM.hd},
            description=(
                f"MiniMax-M3 {op} (prefill, L={ll}), flash_prefill_with_gqa_share_sparse (Triton, op33): "
                f"GQA-share sparse attention over top-{MM.topk_blocks} KV blocks. num_q_heads={MM.nq}, "
                f"num_kv_heads={MM.nkv} (GQA 16:1), head_dim={MM.hd}, block_size_k={MM.block}. B=sweep. "
                f"Baseline = SGLang production kernel."),
            goal=(f"Optimize solution.py against SGLang's "
                  f"flash_prefill_with_gqa_share_sparse baseline for {op} at L={ll}; "
                  "beat every batch workload and match SGLang output within tolerance."),
            axes={"M": var("prefill batch (sweep)"), "L": const(ll, "per-seq context length"),
                  "num_q_heads": const(MM.nq), "num_kv_heads": const(MM.nkv), "head_dim": const(MM.hd),
                  "topk_blocks": const(MM.topk_blocks), "slots": expr("M*L", "paged KV slots"),
                  "Mp1": expr("M+1")},
            inputs={"q": tensor(["M", "num_q_heads", "head_dim"], "bfloat16"),
                    "k_cache": tensor(["slots", "num_kv_heads", "head_dim"], "bfloat16"),
                    "v_cache": tensor(["slots", "num_kv_heads", "head_dim"], "bfloat16"),
                    "req_to_token": tensor(["M", "L"], "int32"), "slot_ids": tensor(["M"], "int64"),
                    "topk_idx": tensor(["num_kv_heads", "M", "topk_blocks"], "int32"),
                    "cu_seqlens": tensor(["Mp1"], "int32"), "seq_lens": tensor(["M"], "int32"),
                    "prefix_lens": tensor(["M"], "int32")},
            outputs={"o": tensor(["M", "num_q_heads", "head_dim"], "bfloat16")})


def _store_kv_index():
    yield TaskSpec(
        model=MM.model, name="mm_dsa_store_kv_index_decode", op="DSA Indexer K/V cache fused store",
        family="dsa-store-kv-index", phase="decode", hf_id=MM.hf_id,
        recipe=recipe("minimax_dsa_store_kv_index.py"), sweep=DECODE_SWEEP,
        backend="store_kv_index JIT CUDA (blackwell)",
        meta={"num_kv_heads": MM.nkv, "head_dim": MM.hd},
        description=(
            f"MiniMax-M3 DSA Indexer K/V cache fused store (decode), store_kv_index (JIT CUDA, op30): "
            f"single-launch scatter of new K/V + index-K into paged caches at per-token slots. "
            f"num_kv_heads={MM.nkv}, head_dim={MM.hd}, index_dim={MM.idx_dim}. ~12x vs separate stores. "
            f"Baseline = SGLang production kernel."),
        goal=("Optimize solution.py against SGLang's store_kv_index baseline for the "
              "DSA indexer K/V-cache fused decode store; beat the full sweep and match "
              "all three SGLang cache outputs."),
        axes={"M": var("tokens written (sweep)"), "slots": const(65536, "paged cache capacity"),
              "num_kv_heads": const(MM.nkv), "head_dim": const(MM.hd), "index_dim": const(MM.idx_dim),
              "kvd": expr("num_kv_heads*head_dim")},
        inputs={"k": tensor(["M", "kvd"], "bfloat16"), "v": tensor(["M", "kvd"], "bfloat16"),
                "k_cache": tensor(["slots", "kvd"], "bfloat16"), "v_cache": tensor(["slots", "kvd"], "bfloat16"),
                "idx_k": tensor(["M", "index_dim"], "bfloat16"),
                "idx_k_cache": tensor(["slots", "index_dim"], "bfloat16"), "indices": tensor(["M"], "int64")},
        outputs={"k_cache_out": tensor(["slots", "kvd"], "bfloat16"),
                 "v_cache_out": tensor(["slots", "kvd"], "bfloat16"),
                 "idx_k_cache_out": tensor(["slots", "index_dim"], "bfloat16")})


def _decode_attn():
    r = recipe("minimax_dsa_decode_attn.py")
    for name, ctx in [("mm_dsa_decode_attn_ctx4096", 4096), ("mm_dsa_decode_attn_ctx32768", 32768)]:
        op = "DSA decode main sparse attention"
        yield TaskSpec(
            model=MM.model, name=name, op=op, family="dsa-decode-attn", phase="decode", hf_id=MM.hf_id,
            recipe=r, sweep=DECODE_SWEEP,
            backend="flash_decode_with_gqa_share_sparse Triton (blackwell)",
            meta={"ctx": ctx, "num_q_heads": MM.nq, "num_kv_heads": MM.nkv, "head_dim": MM.hd},
            description=(
                f"MiniMax-M3 {op} (decode, ctx={ctx}), flash_decode_with_gqa_share_sparse (Triton, op34): "
                f"GQA-share sparse attention over the top-{MM.topk_blocks} KV blocks. num_q_heads={MM.nq}, "
                f"num_kv_heads={MM.nkv} (GQA 16:1), head_dim={MM.hd}, block_size={MM.block}. Batch=sweep, "
                f"paged KV. Baseline = SGLang production kernel."),
            goal=(f"Optimize solution.py against SGLang's "
                  f"flash_decode_with_gqa_share_sparse baseline for {op} at context "
                  f"length {ctx}; beat every batch workload and match SGLang output."),
            axes={"M": var("decode batch (sweep)"), "ctx": const(ctx, "context length"),
                  "num_q_heads": const(MM.nq), "num_kv_heads": const(MM.nkv), "head_dim": const(MM.hd),
                  "topk_blocks": const(MM.topk_blocks), "slots": expr("M*ctx", "paged KV slots")},
            inputs={"q": tensor(["M", "num_q_heads", "head_dim"], "bfloat16"),
                    "k_cache": tensor(["slots", "num_kv_heads", "head_dim"], "bfloat16"),
                    "v_cache": tensor(["slots", "num_kv_heads", "head_dim"], "bfloat16"),
                    "req_to_token": tensor(["M", "ctx"], "int32"), "seq_lens": tensor(["M"], "int32"),
                    "slot_ids": tensor(["M"], "int64"),
                    "topk_idx": tensor(["num_kv_heads", "M", "topk_blocks"], "int32")},
            outputs={"o": tensor(["M", "num_q_heads", "head_dim"], "bfloat16")})


def _decode_topk():
    r = recipe("minimax_dsa_decode_topk.py")
    for name, ctx in [("mm_dsa_decode_topk_ctx4096", 4096), ("mm_dsa_decode_topk_ctx32768", 32768)]:
        op = "DSA decode Indexer top-k blocks"
        yield TaskSpec(
            model=MM.model, name=name, op=op, family="dsa-decode-topk", phase="decode", hf_id=MM.hf_id,
            recipe=r, sweep=DECODE_SWEEP, tolerance=dict(EXACT_TOL),
            backend="minimax_decode_topk JIT radix (blackwell)",
            meta={"index_heads": MM.idx_heads, "block_size": MM.block, "ctx": ctx, "oracle": "exact-index"},
            description=(
                f"MiniMax-M3 {op} (decode, ctx={ctx}), minimax_decode_topk (JIT radix, op32): select "
                f"top-{MM.topk_blocks} KV blocks per (index_head,seq) from block scores. "
                f"index_heads={MM.idx_heads}, block_size={MM.block}. EXACT-index oracle. "
                f"Baseline = SGLang production kernel."),
            goal=(f"Optimize solution.py against SGLang's minimax_decode_topk baseline "
                  f"for {op} at context length {ctx}; beat every batch workload and "
                  "match block indices exactly (atol=0)."),
            axes={"M": var("decode batch (sweep)"), "index_heads": const(MM.idx_heads),
                  "block_size": const(MM.block), "ctx": const(ctx, "context length"),
                  "topk_blocks": const(MM.topk_blocks)},
            inputs={"score": tensor(["index_heads", "M", "ctx"], "float32"),
                    "seq_lens": tensor(["M"], "int32")},
            outputs={"topk_idx": tensor(["index_heads", "M", "topk_blocks"], "int32")})


def specs():
    out = []
    for gen in (_qknorm_rope, _prefill_topk, _prefill_attn, _store_kv_index, _decode_attn, _decode_topk):
        out.extend(gen())
    return out
