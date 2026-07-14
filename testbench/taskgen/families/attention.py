"""Kimi MLA attention + GLM-5.2 Sparse MLA decode."""
from ..config import KIMI as K, GLM52 as G, recipe
from ..spec import (TaskSpec, DECODE_SWEEP, PREFILL_SWEEP,
                    GLM52_DECODE_SWEEP, GLM52_SPARSE_MLA_CTX,
                    var, const, expr, tensor)


def _mla_decode():
    r = recipe("kimi_mla_decode.py")
    for name, seq_len in [("mla_decode_seq2048", 2048), ("mla_decode_seq32768", 32768)]:
        block_num_v = (((seq_len + 127) // 128) + 0) // 1  # pack=1 (block_size=128)
        yield TaskSpec(
            model=K.model, name=name, op="MLA decode attention", family="mla-attention",
            phase="decode", hf_id=K.hf_id, recipe=r, sweep=DECODE_SWEEP,
            backend="sgl_kernel cutlass_mla_decode (blackwell)",
            meta={"num_heads": K.num_heads, "kv_lora": K.kv_lora, "seq_len": seq_len},
            description=(
                f"Kimi-K2.7 MLA decode attention (decode, seq_len={seq_len}), sgl_kernel.cutlass_mla_decode: "
                f"Multi-head Latent Attention over the compressed latent KV cache. num_heads={K.num_heads}, "
                f"kv_lora_rank={K.kv_lora}, qk_rope={K.qk_rope} -> head_dim=576, block_size=128. batch=sweep, "
                f"paged latent KV. Baseline = sglang production kernel; beat its latency while matching output."),
            goal=(f"Optimize solution.py against SGLang's "
                  f"sgl_kernel.cutlass_mla_decode baseline for MLA decode at "
                  f"sequence length {seq_len}; beat every batch workload and match "
                  "SGLang output within the declared tolerance."),
            axes={"M": var("decode batch (sweep)"), "seq_len": const(seq_len, "KV context length"),
                  "num_heads": const(K.num_heads), "kv_lora": const(K.kv_lora),
                  "qk_rope": const(K.qk_rope), "block_size": const(128),
                  "head_dim": const(K.kv_lora + K.qk_rope),
                  "block_num": const(block_num_v, "blocks per seq"),
                  "blocks": expr("M*block_num", "total paged blocks"),
                  "ws": const(1, "workspace bytes (custom, sized in get_inputs)")},
            inputs={"q_nope": tensor(["M", "num_heads", "kv_lora"], "bfloat16"),
                    "q_pe": tensor(["M", "num_heads", "qk_rope"], "bfloat16"),
                    "kv_cache": tensor(["blocks", "block_size", "head_dim"], "bfloat16"),
                    "seq_lens": tensor(["M"], "int32"),
                    "block_table": tensor(["M", "block_num"], "int32"),
                    "workspace": tensor(["ws"], "int8"), "scale": tensor(None, "float32")},
            outputs={"o": tensor(["M", "num_heads", "kv_lora"], "bfloat16")})


def _mla_prefill():
    yield TaskSpec(
        model=K.model, name="mla_prefill", op="MLA prefill attention", family="mla-attention",
        phase="prefill", hf_id=K.hf_id, recipe=recipe("kimi_mla_prefill.py"), sweep=PREFILL_SWEEP,
        backend="flashinfer BatchPrefillWithRaggedKVCacheWrapper (blackwell)",
        meta={"num_heads": K.num_heads, "qk_head_dim": 192, "v_head_dim": 128},
        description=(
            f"Kimi-K2.7 MLA prefill attention (prefill), flashinfer BatchPrefillWithRaggedKVCacheWrapper: "
            f"MHA over full q/k/v (FA3/flash_attn_varlen is not built for B200 sm100, so the ragged "
            f"flashinfer path is production here). num_heads={K.num_heads}, qk_head_dim=192, v_head_dim=128, "
            f"causal. seqlen=sweep, single sequence. Wrapper .plan() is untimed setup; only .run() is timed. "
            f"Baseline = sglang production kernel; beat its latency while matching output."),
        goal=("Optimize solution.py against SGLang's FlashInfer ragged-prefill "
              "baseline for MLA prefill; beat every sequence-length workload and "
              "match SGLang output within the declared tolerance."),
        axes={"M": var("prefill sequence length (sweep)"), "num_heads": const(K.num_heads),
              "qk_head_dim": const(192), "v_head_dim": const(128)},
        inputs={"q": tensor(["M", "num_heads", "qk_head_dim"], "bfloat16"),
                "k": tensor(["M", "num_heads", "qk_head_dim"], "bfloat16"),
                "v": tensor(["M", "num_heads", "v_head_dim"], "bfloat16"),
                "wrapper": tensor(None, "int32")},
        outputs={"o": tensor(["M", "num_heads", "v_head_dim"], "bfloat16")})


def _glm52_sparse_mla_decode():
    """B200 TRT-LLM sparse MLA decode — one task; ctx×batch are workloads.

    Matches SGLang: a single production kernel; context length is runtime metadata,
    not a separate operator. WIN requires beating every (ctx, bs) point.
    """
    r = recipe("glm52_sparse_mla_decode.py")
    workloads = [{"M": m, "ctx": ctx}
                 for ctx in GLM52_SPARSE_MLA_CTX
                 for m in GLM52_DECODE_SWEEP]
    yield TaskSpec(
        model=G.model, name="sparse_mla_decode", op="Sparse MLA Decode",
        family="sparse-mla-decode", phase="decode", hf_id=G.hf_id, recipe=r,
        sweep=GLM52_DECODE_SWEEP,
        workloads=workloads,
        backend="flashinfer trtllm_batch_decode_with_kv_cache_mla (trtllm-gen, blackwell)",
        performance_model={"kind": "sparse-mla", "family": "sparse-mla-decode"},
        workload_metrics=["valid_selected_kv", "effective_topk", "effective_topk_ratio",
                          "selected_token_head_per_s", "effective_kv_gbps",
                          "cache_footprint_bytes", "us_per_token"],
        meta={"num_heads": G.local_heads, "kv_lora": G.kv_lora,
              "index_topk": G.index_topk, "page_size": G.page_size,
              "ctx_sweep": list(GLM52_SPARSE_MLA_CTX),
              "tp": G.tp, "deployment": "B200-TP8-EP8",
              "backend_note": "B200 FP8 KV auto-routes to TRT-LLM, not flashmla_sparse"},
        description=(
            f"GLM-5.2 Sparse MLA Decode (TP8 local heads={G.local_heads}), "
            f"flashinfer TRT-LLM sparse MLA (B200 production path). "
            f"Same SGLang kernel for all contexts; sweep ctx={GLM52_SPARSE_MLA_CTX} × "
            f"batch={GLM52_DECODE_SWEEP}. top-k={G.index_topk} (effective min(ctx,topk)); "
            f"page_size={G.page_size}; latent dim={G.head_dim}; out kv_lora={G.kv_lora}. "
            f"Baseline = SGLang production kernel on Blackwell."),
        goal=("Optimize solution.py against SGLang's TRT-LLM sparse MLA decode "
              "baseline across every (ctx, batch) workload and match SGLang output "
              "within the FP8 DSA tolerance."),
        axes={"M": var("decode batch (sweep)"),
              "ctx": var("KV context length (sweep)"),
              "num_heads": const(G.local_heads, "local heads under TP8"),
              "head_dim": const(G.head_dim, "kv_lora + qk_rope"),
              "kv_lora": const(G.kv_lora),
              "topk": const(G.index_topk),
              "page_size": const(G.page_size),
              "qk_nope": const(G.qk_nope),
              "qk_rope": const(G.qk_rope)},
        inputs={"q": tensor(["M", 1, "num_heads", "head_dim"], "float8_e4m3fn"),
                "kv_cache": tensor(None, "float8_e4m3fn"),
                "block_tables": tensor(["M", 1, "topk"], "int32"),
                "seq_lens": tensor(["M"], "int32"),
                "workspace": tensor(None, "uint8"),
                "bmm1_scale": tensor(None, "float32"),
                "max_seq_len": tensor(None, "int32")},
        outputs={"o": tensor(["M", "num_heads", "kv_lora"], "bfloat16")})


def specs():
    return list(_mla_decode()) + list(_mla_prefill()) + list(_glm52_sparse_mla_decode())
