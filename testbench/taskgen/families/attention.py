"""Kimi MLA attention: decode (cutlass_mla_decode) + prefill (flashinfer ragged)."""
from ..config import KIMI as K, recipe
from ..spec import TaskSpec, DECODE_SWEEP, PREFILL_SWEEP, var, const, expr, tensor


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


def specs():
    return list(_mla_decode()) + list(_mla_prefill())
