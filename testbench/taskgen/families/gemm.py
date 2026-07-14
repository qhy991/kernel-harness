"""GEMM / quant families: fp8-linear, bf16-linear, router, lm-head, grouped-MoE, act-quant."""
from ..config import KIMI as K, MM, GLM52 as G, recipe
from ..spec import (TaskSpec, DECODE_SWEEP, PREFILL_SWEEP, ROUTER_SWEEP, MASKED_SWEEP,
                    GLM52_PREFILL_SWEEP, GLM52_DECODE_SWEEP,
                    EXACT_TOL, sweep_for, var, const, expr, tensor)

# fp8-linear-gemm: (name, op, phase, K, N, csv_wallclock_us)
FP8_LINEAR = [
    ("dense_ffn_gateup_prefill",  "Dense FFN GateUp",  "prefill", 7168, 4608, 1044.69),
    ("dense_ffn_gateup_decode",   "Dense FFN GateUp",  "decode",  7168, 4608,   56.50),
    ("dense_ffn_down_prefill",    "Dense FFN Down",    "prefill", 2304, 7168,  451.28),
    ("dense_ffn_down_decode",     "Dense FFN Down",    "decode",  2304, 7168,   52.92),
    ("qa_kva_fused_prefill",      "Q_a+KV_a fused",    "prefill", 7168, 2176,  436.81),
    ("qa_kva_fused_decode",       "Q_a+KV_a fused",    "decode",  7168, 2176,   56.60),
    ("q_b_prefill",               "Q_b",               "prefill", 1536, 1536,   77.83),
    ("q_b_decode",                "Q_b",               "decode",  1536, 12288,  52.52),
    ("kv_b_prefill",              "KV_b",              "prefill",  512, 2048,   73.39),
    ("o_proj_prefill",            "O_proj",            "prefill", 1024, 7168,  264.18),
    ("o_proj_decode",             "O_proj",            "decode",  8192, 7168,   57.57),
    ("moe_shared_gateup_prefill", "MoE Shared GateUp", "prefill", 7168,  512,  148.43),
    ("moe_shared_gateup_decode",  "MoE Shared GateUp", "decode",  7168,  512,   55.77),
    ("moe_shared_down_prefill",   "MoE Shared Down",   "prefill",  256, 7168,  240.41),
    ("moe_shared_down_decode",    "MoE Shared Down",   "decode",   256, 7168,   51.12),
]

# minimax bf16-linear (inventory op28): (name, op, phase, K, N)
MM_BF16_LINEAR = [
    ("mm_dense_ffn_gateup_prefill", "Dense FFN GateUp (layers 0-2)", "prefill", MM.hidden, 2 * MM.dense_inter),
    ("mm_dense_ffn_gateup_decode",  "Dense FFN GateUp (layers 0-2)", "decode",  MM.hidden, 2 * MM.dense_inter),
    ("mm_dense_ffn_down_prefill",   "Dense FFN Down (layers 0-2)",   "prefill", MM.dense_inter, MM.hidden),
    ("mm_dense_ffn_down_decode",    "Dense FFN Down (layers 0-2)",   "decode",  MM.dense_inter, MM.hidden),
    ("mm_qkv_proj_prefill",         "Main Q/K/V proj (GQA)",         "prefill", MM.hidden, MM.qkv_out),
    ("mm_qkv_proj_decode",          "Main Q/K/V proj (GQA)",         "decode",  MM.hidden, MM.qkv_out),
    ("mm_o_proj_prefill",           "Attention O_proj",              "prefill", MM.o_in, MM.hidden),
    ("mm_o_proj_decode",            "Attention O_proj",              "decode",  MM.o_in, MM.hidden),
    ("mm_shared_gateup_decode",     "Shared expert GateUp",          "decode",  MM.hidden, 2 * MM.shared_inter),
    ("mm_shared_down_decode",       "Shared expert Down",            "decode",  MM.shared_inter, MM.hidden),
    ("mm_shared_gateup_prefill",    "Shared expert GateUp",          "prefill", MM.hidden, 2 * MM.shared_inter),
    ("mm_shared_down_prefill",      "Shared expert Down",            "prefill", MM.shared_inter, MM.hidden),
]


def _fp8_linear():
    r = recipe("fp8_linear_gemm.py")
    for name, op, phase, kk, nn, base_us in FP8_LINEAR:
        yield TaskSpec(
            model=K.model, name=name, op=op, family="fp8-linear-gemm", phase=phase,
            hf_id=K.hf_id, recipe=r, sweep=sweep_for(phase), baseline_us=base_us,
            backend="deep_gemm w8a8_block_fp8 (blackwell)",
            description=(
                f"Kimi-K2.7 {op} ({phase}), FP8 blockwise GEMM "
                f"(deep_gemm w8a8_block_fp8, Blackwell). out[M,{nn}] = x_fp8[M,{kk}] @ "
                f"w_fp8[{nn},{kk}].T with 1x128 act / 128x128 weight scales. Baseline = "
                f"sglang production kernel (~{base_us:.1f}us at canonical shape); beat its "
                f"latency across the sweep while matching its output."),
            goal=(f"Optimize solution.py against SGLang's DeepGEMM w8a8_block_fp8 "
                  f"baseline for {op} {phase}; beat every shape in the sweep and match "
                  "SGLang output within the declared tolerance."),
            axes={"M": var(f"{phase} token count (sweep)"), "K": const(kk), "N": const(nn),
                  "K_scale_blocks": expr("K//512", "ue8m0-packed K scale blocks (K/128/4)")},
            inputs={"x_fp8": tensor(["M", "K"], "float8_e4m3fn"),
                    "x_scale": tensor(["M", "K_scale_blocks"], "int32"),
                    "w_fp8": tensor(["N", "K"], "float8_e4m3fn"),
                    "w_scale": tensor(["N", "K_scale_blocks"], "int32")},
            outputs={"output": tensor(["M", "N"], "bfloat16")},
            meta={"K": kk, "N": nn})


def _bf16_linear():
    r = recipe("bf16_linear.py")
    for name, op, phase, kk, nn in MM_BF16_LINEAR:
        yield TaskSpec(
            model=MM.model, name=name, op=op, family="bf16-linear", phase=phase,
            hf_id=MM.hf_id, recipe=r, sweep=sweep_for(phase),
            backend="cuBLAS bf16 GEMM (blackwell)",
            description=(
                f"MiniMax-M3 {op} ({phase}), bf16 cuBLAS GEMM (base checkpoint; "
                f"inventory op28 standard Linear). out[M,{nn}] = x[M,{kk}] @ weight[{nn},{kk}].T. "
                f"Baseline = sglang production path; beat its latency across the sweep while "
                f"matching output. (-MXFP8 checkpoint would use deep_gemm/flashinfer — separate.)"),
            goal=(f"Optimize solution.py against SGLang's cuBLAS BF16 GEMM baseline "
                  f"for MiniMax-M3 {op} {phase}; beat the full sweep and match SGLang "
                  "output within the declared tolerance."),
            axes={"M": var(f"{phase} token count (sweep)"), "K": const(kk), "N": const(nn)},
            inputs={"x": tensor(["M", "K"], "bfloat16"), "weight": tensor(["N", "K"], "bfloat16")},
            outputs={"output": tensor(["M", "N"], "bfloat16")},
            meta={"K": kk, "N": nn})


def _router():
    # Kimi: dsv3_router_gemm (M<=16, N=384, fp32 out)
    yield TaskSpec(
        model=K.model, name="moe_router_gemm_decode", op="MoE Router GEMM",
        family="router-gemm", phase="decode", hf_id=K.hf_id, recipe=recipe("kimi_router_gemm.py"),
        sweep=ROUTER_SWEEP, backend="sgl_kernel dsv3_router_gemm (blackwell)",
        description=(
            f"Kimi-K2.7 MoE Router GEMM (decode), sgl_kernel.dsv3_router_gemm: "
            f"out[M,{K.n_routed_experts}] = hidden[M,{K.hidden}] @ router_w[{K.n_routed_experts},{K.hidden}].T in fp32. "
            f"Kernel requires num_tokens<=16 (DP32xEP32 M_local=16), so the sweep is "
            f"architecture-true M in {ROUTER_SWEEP}. Baseline = sglang production kernel; "
            f"beat its latency across the sweep while matching its output."),
        goal=("Optimize solution.py against SGLang's sgl_kernel.dsv3_router_gemm "
              "baseline for MoE Router GEMM decode (M<=16); beat every shape and match "
              "SGLang output within the declared tolerance."),
        axes={"M": var("decode token count (<=16, sweep)"), "H": const(K.hidden, "hidden size"),
              "N": const(K.n_routed_experts, "n_routed_experts")},
        inputs={"hidden_states": tensor(["M", "H"], "bfloat16"),
                "router_weights": tensor(["N", "H"], "bfloat16")},
        outputs={"logits": tensor(["M", "N"], "float32")},
        meta={"H": K.hidden, "N": K.n_routed_experts})
    # MiniMax: fp32 cuBLAS matmul (prefill + decode)
    r = recipe("minimax_router.py")
    for name, phase in [("mm_moe_router_prefill", "prefill"), ("mm_moe_router_decode", "decode")]:
        yield TaskSpec(
            model=MM.model, name=name, op="MoE Router GEMM", family="router-gemm", phase=phase,
            hf_id=MM.hf_id, recipe=r, sweep=sweep_for(phase), backend="cuBLAS fp32 GEMM (blackwell)",
            description=(
                f"MiniMax-M3 MoE Router GEMM ({phase}), fp32 cuBLAS matmul (inventory op37): "
                f"logits[M,{MM.experts}] = hidden[M,{MM.hidden}] @ gate_w[{MM.experts},{MM.hidden}].T. "
                f"Baseline = sglang production path; beat its latency while matching output."),
            goal=(f"Optimize solution.py against SGLang's FP32 cuBLAS baseline for "
                  f"MiniMax-M3 MoE Router GEMM {phase}; beat the full sweep and match "
                  "SGLang output within the declared tolerance."),
            axes={"M": var(f"{phase} token count (sweep)"), "H": const(MM.hidden, "hidden size"),
                  "E": const(MM.experts, "n_routed_experts")},
            inputs={"hidden_states": tensor(["M", "H"], "float32"),
                    "gate_weight": tensor(["E", "H"], "float32")},
            outputs={"logits": tensor(["M", "E"], "float32")},
            meta={"H": MM.hidden, "E": MM.experts})


def _lm_head():
    for cfg, name in [(K, "lm_head_decode"), (MM, "mm_lm_head_decode")]:
        label = "Kimi-K2.7" if cfg is K else "MiniMax-M3"
        yield TaskSpec(
            model=cfg.model, name=name, op="LM-head logits GEMM", family="lm-head", phase="decode",
            hf_id=cfg.hf_id, recipe=recipe("kimi_lm_head.py"), sweep=DECODE_SWEEP,
            backend="torch.matmul bf16 (blackwell)",
            description=(
                f"{label} LM-head logits GEMM (decode), torch.matmul bf16: "
                f"out[M,{cfg.vocab}] = hidden[M,{cfg.hidden}] @ lm_head_w[{cfg.vocab},{cfg.hidden}].T. "
                f"Bandwidth-bound. Baseline = sglang production path; beat its latency while matching output."),
            goal=(f"Optimize solution.py against SGLang's torch.matmul LM-head "
                  f"baseline for {label} decode; beat the full sweep and match SGLang "
                  "output within the declared tolerance."),
            axes={"M": var("sampled positions (decode, sweep)"), "H": const(cfg.hidden, "hidden size"),
                  "V": const(cfg.vocab, "vocab size")},
            inputs={"hidden_states": tensor(["M", "H"], "bfloat16"),
                    "lm_head_weight": tensor(["V", "H"], "bfloat16")},
            outputs={"logits": tensor(["M", "V"], "bfloat16")},
            meta={"H": cfg.hidden, "V": cfg.vocab})


def _grouped_moe_masked():
    r = recipe("grouped_moe_masked.py")
    kimi = [("moe_gateup_grouped_decode", "MoE GateUp GroupGEMM (masked)", K.hidden, 2 * K.moe_inter_full),
            ("moe_down_grouped_decode",   "MoE Down GroupGEMM (masked)",   K.moe_inter_full, K.hidden)]
    for name, op, kk, nn in kimi:
        yield TaskSpec(
            model=K.model, name=name, op=op, family="grouped-moe", phase="decode",
            hf_id=K.hf_id, recipe=r, sweep=MASKED_SWEEP,
            backend="deep_gemm fp8_m_grouped_gemm_nt_masked (blackwell)",
            description=(
                f"Kimi-K2.7 {op} (decode), deep_gemm.fp8_m_grouped_gemm_nt_masked over "
                f"E={K.ep_local_experts} local experts (EP32 of 384 routed), K={kk}, N={nn}. Sweep "
                f"= tokens/expert (masked). FP8 blockwise (per-token act, per-block weight, "
                f"UE8M0), quant done offline in get_inputs; only the grouped GEMM is timed. "
                f"Baseline = sglang production kernel; beat its latency while matching output."),
            goal=(f"Optimize solution.py against SGLang's "
                  f"deep_gemm.fp8_m_grouped_gemm_nt_masked baseline for {op} decode; "
                  "beat every tokens-per-expert workload and match SGLang output."),
            axes={"M": var("tokens per expert (masked, sweep)"),
                  "E": const(K.ep_local_experts, "local experts (EP32)"), "K": const(kk), "N": const(nn)},
            inputs={"a_fp8": tensor(["E", "M", "K"], "float8_e4m3fn"), "a_s": tensor(["E", "M", "K"], "int32"),
                    "b_fp8": tensor(["E", "N", "K"], "float8_e4m3fn"), "b_s": tensor(["E", "N", "K"], "int32"),
                    "out": tensor(["E", "M", "N"], "bfloat16"), "masked_m": tensor(["E"], "int32"),
                    "expected_m": tensor(None, "int32")},
            outputs={"grouped_out": tensor(["E", "M", "N"], "bfloat16")},
            meta={"E": K.ep_local_experts, "K": kk, "N": nn})
    mm = [("mm_moe_gateup_grouped_decode", "MoE GateUp GroupGEMM (masked)", MM.hidden, 2 * MM.inter),
          ("mm_moe_down_grouped_decode",   "MoE Down GroupGEMM (masked)",   MM.inter, MM.hidden)]
    for name, op, kk, nn in mm:
        yield TaskSpec(
            model=MM.model, name=name, op=op, family="grouped-moe", phase="decode",
            hf_id=MM.hf_id, recipe=r, sweep=MASKED_SWEEP,
            backend="deep_gemm fp8_m_grouped_gemm_nt_masked (blackwell)",
            description=(
                f"MiniMax-M3 {op} (decode), deep_gemm.fp8_m_grouped_gemm_nt_masked over "
                f"E={MM.ep_local} local experts (128 routed / EP8), K={kk}, N={nn}. Sweep = "
                f"tokens/expert (masked). FP8 blockwise (per-token act, per-block weight, UE8M0), "
                f"quant offline in get_inputs; only the grouped GEMM is timed. Baseline = sglang "
                f"production kernel; beat its latency while matching output."),
            goal=(f"Optimize solution.py against SGLang's "
                  f"deep_gemm.fp8_m_grouped_gemm_nt_masked baseline for MiniMax-M3 "
                  f"{op} decode; beat every tokens-per-expert workload and match output."),
            axes={"M": var("tokens per expert (masked, sweep)"),
                  "E": const(MM.ep_local, "local experts (EP8 of 128)"), "K": const(kk), "N": const(nn)},
            inputs={"a_fp8": tensor(["E", "M", "K"], "float8_e4m3fn"), "a_s": tensor(["E", "M", "K"], "int32"),
                    "b_fp8": tensor(["E", "N", "K"], "float8_e4m3fn"), "b_s": tensor(["E", "N", "K"], "int32"),
                    "out": tensor(["E", "M", "N"], "bfloat16"), "masked_m": tensor(["E"], "int32"),
                    "expected_m": tensor(None, "int32")},
            outputs={"grouped_out": tensor(["E", "M", "N"], "bfloat16")},
            meta={"E": MM.ep_local, "K": kk, "N": nn})


def _grouped_moe_contiguous():
    r = recipe("grouped_moe_contiguous.py")
    for name, op, kk, nn in [("mm_moe_gateup_grouped_prefill", "MoE GateUp GroupGEMM (contiguous)", MM.hidden, 2 * MM.inter),
                             ("mm_moe_down_grouped_prefill",   "MoE Down GroupGEMM (contiguous)",   MM.inter, MM.hidden)]:
        yield TaskSpec(
            model=MM.model, name=name, op=op, family="grouped-moe-contiguous", phase="prefill",
            hf_id=MM.hf_id, recipe=r, sweep=MASKED_SWEEP,
            backend="deep_gemm m_grouped_fp8_gemm_nt_contiguous (blackwell)",
            description=(
                f"MiniMax-M3 {op} (prefill), deep_gemm.m_grouped_fp8_gemm_nt_contiguous over "
                f"E={MM.ep_local} local experts, K={kk}, N={nn}. Sweep = tokens/expert (contiguous "
                f"layout, m_indices routing). FP8 blockwise, quant offline in get_inputs; only the "
                f"grouped GEMM is timed. Baseline = sglang production kernel; beat its latency."),
            goal=(f"Optimize solution.py against SGLang's "
                  f"deep_gemm.m_grouped_fp8_gemm_nt_contiguous baseline for MiniMax-M3 "
                  f"{op} prefill; beat every tokens-per-expert workload and match output."),
            axes={"M": var("tokens per expert (contiguous, sweep)"),
                  "E": const(MM.ep_local, "local experts"), "EM": expr("E*M", "E*M flattened rows"),
                  "K": const(kk), "N": const(nn)},
            inputs={"a_fp8": tensor(["EM", "K"], "float8_e4m3fn"), "a_s": tensor(["EM", "K"], "int32"),
                    "b_fp8": tensor(["E", "N", "K"], "float8_e4m3fn"), "b_s": tensor(["E", "N", "K"], "int32"),
                    "out": tensor(["EM", "N"], "bfloat16"), "m_indices": tensor(["EM"], "int32")},
            outputs={"grouped_out": tensor(["EM", "N"], "bfloat16")},
            meta={"E": MM.ep_local, "K": kk, "N": nn})


def _act_quant():
    r = recipe("minimax_act_quant.py")
    for name, phase in [("mm_act_fp8_quant_prefill", "prefill"), ("mm_act_fp8_quant_decode", "decode")]:
        yield TaskSpec(
            model=MM.model, name=name, op="Act FP8 quant (per-token)", family="act-fp8-quant", phase=phase,
            hf_id=MM.hf_id, recipe=r, sweep=sweep_for(phase),
            backend="sglang_per_token_group_quant_fp8 (blackwell)",
            description=(
                f"MiniMax-M3 Act FP8 quant (per-token) ({phase}), sglang_per_token_group_quant_fp8: bf16 "
                f"activation[M,{MM.hidden}] -> fp8_e4m3 + 1x128 ue8m0/tma-aligned scales. The "
                f"quant that precedes every block-FP8 GEMM. Baseline = sglang production kernel; "
                f"beat its latency while matching output."),
            goal=(f"Optimize solution.py against SGLang's "
                  f"sglang_per_token_group_quant_fp8 baseline for MiniMax-M3 per-token "
                  f"FP8 activation quantization {phase}; beat the full sweep and match "
                  "both SGLang outputs."),
            axes={"M": var(f"{phase} token count (sweep)"), "K": const(MM.hidden, "hidden size"),
                  "K_scale": expr("K//512", "ue8m0 scale blocks")},
            inputs={"x": tensor(["M", "K"], "bfloat16")},
            outputs={"q_fp8": tensor(["M", "K"], "float8_e4m3fn"), "scale": tensor(["M", "K_scale"], "int32")},
            meta={"K": MM.hidden})


def _glm52_o_proj():
    """Attention O Projection under TP8: K=local_heads*v_head=2048, N=hidden=6144."""
    r = recipe("fp8_linear_gemm.py")
    kk, nn = G.o_in, G.hidden
    for name, phase, sweep in [
        ("o_proj_prefill", "prefill", GLM52_PREFILL_SWEEP),
        ("o_proj_decode", "decode", GLM52_DECODE_SWEEP),
    ]:
        yield TaskSpec(
            model=G.model, name=name, op="Attention O Projection",
            family="fp8-linear-gemm", phase=phase, hf_id=G.hf_id, recipe=r,
            sweep=sweep, backend="deep_gemm w8a8_block_fp8 (blackwell)",
            flops_expr="2*M*K*N",
            performance_model={"kind": "gemm", "family": "fp8-linear-gemm"},
            workload_metrics=["achieved_tflops", "achieved_gbps", "arithmetic_intensity",
                              "bound", "us_per_token"],
            description=(
                f"GLM-5.2 Attention O Projection ({phase}, TP8 local), FP8 blockwise GEMM "
                f"(deep_gemm w8a8_block_fp8). out[M,{nn}] = x_fp8[M,{kk}] @ w_fp8[{nn},{kk}].T. "
                f"K={kk}=local_heads*v_head ({G.local_heads}*{G.v_head}), N={nn}=hidden. "
                f"Baseline = SGLang production DeepGEMM; beat every shape while matching output."),
            goal=(f"Optimize solution.py against SGLang's DeepGEMM w8a8_block_fp8 "
                  f"baseline for GLM-5.2 O Projection {phase}; beat the sweep and match "
                  "SGLang output within the declared FP8 tolerance."),
            axes={"M": var(f"{phase} token/batch count (sweep)"),
                  "K": const(kk, "local O_proj input = heads/TP * v_head"),
                  "N": const(nn, "hidden size"),
                  "K_scale_blocks": expr("K//512", "ue8m0-packed K scale blocks")},
            inputs={"x_fp8": tensor(["M", "K"], "float8_e4m3fn"),
                    "x_scale": tensor(["M", "K_scale_blocks"], "int32"),
                    "w_fp8": tensor(["N", "K"], "float8_e4m3fn"),
                    "w_scale": tensor(["N", "K_scale_blocks"], "int32")},
            outputs={"output": tensor(["M", "N"], "bfloat16")},
            meta={"K": kk, "N": nn, "tp": G.tp, "deployment": "B200-TP8-EP8"})


def _glm52_routed_experts():
    """Routed Gate+Up / Down with EP-local realistic top-8 routing."""
    r = recipe("glm52_routed_moe_gemm.py")
    # (name, op, phase, layout 0=contig/1=masked, K, N, sweep)
    table = [
        ("routed_gateup_prefill", "Routed Expert Gate+Up", "prefill", 0,
         G.hidden, G.gateup_n, GLM52_PREFILL_SWEEP),
        ("routed_gateup_decode",  "Routed Expert Gate+Up", "decode",  1,
         G.hidden, G.gateup_n, GLM52_DECODE_SWEEP),
        ("routed_down_prefill",   "Routed Expert Down",    "prefill", 0,
         G.moe_inter, G.hidden, GLM52_PREFILL_SWEEP),
        ("routed_down_decode",    "Routed Expert Down",    "decode",  1,
         G.moe_inter, G.hidden, GLM52_DECODE_SWEEP),
    ]
    for name, op, phase, layout, kk, nn, sweep in table:
        family = "grouped-moe-contiguous" if layout == 0 else "grouped-moe"
        backend = ("deep_gemm m_grouped_fp8_gemm_nt_contiguous (blackwell)" if layout == 0
                   else "deep_gemm fp8_m_grouped_gemm_nt_masked (blackwell)")
        routing_note = (
            "prefill normalizes local assignment count to the expected M for stable shapes. "
            if layout == 0 else "decode uses sampled EP-local counts and masked per-expert padding. "
        )
        yield TaskSpec(
            model=G.model, name=name, op=op, family=family, phase=phase,
            hf_id=G.hf_id, recipe=r, sweep=sweep, backend=backend,
            flops_expr="2*M*K*N" if layout == 0 else None,
            performance_model={"kind": "routed-expert", "family": family,
                               "stage": "gateup" if "gateup" in name else "down"},
            workload_metrics=["local_assignments", "active_experts", "empty_experts",
                              "tokens_per_expert_cv", "useful_vs_padded_util",
                              "useful_tflops", "us_per_assignment"],
            description=(
                f"GLM-5.2 {op} ({phase}, EP{G.ep} local E={G.ep_local}), FP8 grouped GEMM. "
                f"Routing = fixed-seed top-{G.moe_topk}/{G.n_routed_experts} filtered to this "
                f"rank; {routing_note}K={kk}, N={nn}. Quant offline; only the grouped GEMM is timed. "
                f"Baseline = SGLang DeepGEMM production kernel."),
            goal=(f"Optimize solution.py against SGLang's DeepGEMM grouped-GEMM baseline "
                  f"for GLM-5.2 {op} {phase}; beat every workload and match SGLang output."),
            axes={"M": var(f"{phase} token/batch population (sweep)"),
                  "E": const(G.ep_local, f"local experts (EP{G.ep} of {G.n_routed_experts})"),
                  "K": const(kk), "N": const(nn),
                  "layout": const(layout, "0=contiguous prefill, 1=masked decode"),
                  "n_global": const(G.n_routed_experts),
                  "topk": const(G.moe_topk)},
            inputs={"a_fp8": tensor(None, "float8_e4m3fn"),
                    "a_s": tensor(None, "int32"),
                    "b_fp8": tensor(None, "float8_e4m3fn"),
                    "b_s": tensor(None, "int32"),
                    "out": tensor(None, "bfloat16"),
                    "masked_m": tensor(["E"], "int32"),
                    "expected_m": tensor(None, "int32"),
                    "m_indices": tensor(None, "int32"),
                    "layout": tensor(None, "int32")},
            outputs={"grouped_out": tensor(None, "bfloat16")},
            meta={"E": G.ep_local, "K": kk, "N": nn, "ep": G.ep, "tp": G.tp,
                  "deployment": "B200-TP8-EP8"})


def specs():
    out = []
    for gen in (_fp8_linear, _bf16_linear, _router, _lm_head,
                _grouped_moe_masked, _grouped_moe_contiguous, _act_quant,
                _glm52_o_proj, _glm52_routed_experts):
        out.extend(gen())
    return out
