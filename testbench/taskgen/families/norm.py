"""Normalization families: rmsnorm, fused-add-rmsnorm (Kimi) + Gemma variants (MiniMax)."""
from ..config import KIMI as K, MM, recipe
from ..spec import TaskSpec, sweep_for, var, const, tensor

# Kimi standalone RMSNorm (q_a/kv_a layernorm): (name, op, phase, H)
KIMI_RMSNORM = [
    ("q_a_layernorm_prefill",  "Q_a LayerNorm (RMSNorm)",  "prefill", K.q_lora),
    ("q_a_layernorm_decode",   "Q_a LayerNorm (RMSNorm)",  "decode",  K.q_lora),
    ("kv_a_layernorm_prefill", "KV_a LayerNorm (RMSNorm)", "prefill", K.kv_lora),
    ("kv_a_layernorm_decode",  "KV_a LayerNorm (RMSNorm)", "decode",  K.kv_lora),
]


def _kimi_rmsnorm():
    r = recipe("rmsnorm.py")
    for name, op, phase, h in KIMI_RMSNORM:
        yield TaskSpec(
            model=K.model, name=name, op=op, family="rmsnorm", phase=phase, hf_id=K.hf_id,
            recipe=r, sweep=sweep_for(phase), backend="sgl_kernel rmsnorm (blackwell)",
            description=(
                f"Kimi-K2.7 {op} ({phase}), standalone RMSNorm over hidden={h} "
                f"(sgl_kernel.rmsnorm). out[M,{h}] = x * rsqrt(mean(x^2,-1)+eps) * weight. "
                f"Baseline = sglang production kernel; beat its latency while matching output."),
            goal=(f"Optimize solution.py against SGLang's sgl_kernel.rmsnorm baseline "
                  f"for {op} {phase}; beat every token workload and match SGLang output "
                  "within the declared tolerance."),
            axes={"M": var(f"{phase} token count (sweep)"), "H": const(h, "hidden size")},
            inputs={"x": tensor(["M", "H"], "bfloat16"), "weight": tensor(["H"], "bfloat16")},
            outputs={"output": tensor(["M", "H"], "bfloat16")},
            meta={"H": h})


def _kimi_fused_add_rmsnorm():
    r = recipe("fused_add_rmsnorm.py")
    for name, phase in [("input_layernorm_fused_prefill", "prefill"), ("input_layernorm_fused_decode", "decode")]:
        op = "Input LayerNorm (fused add-RMSNorm)"
        yield TaskSpec(
            model=K.model, name=name, op=op, family="fused-add-rmsnorm", phase=phase, hf_id=K.hf_id,
            recipe=r, sweep=sweep_for(phase), backend="sgl_kernel fused_add_rmsnorm (blackwell)",
            meta={"H": K.hidden, "interface_exact": True},
            description=(
                f"Kimi-K2.7 {op} ({phase}), fused residual-add + RMSNorm over hidden={K.hidden} "
                f"(sgl_kernel.fused_add_rmsnorm, in-place). residual_out = x + residual; "
                f"normed = rmsnorm(residual_out) * weight. INTERFACE-EXACT drop-in. Baseline = "
                f"sglang production kernel; beat its latency while matching BOTH outputs."),
            goal=(f"Optimize solution.py against SGLang's "
                  f"sgl_kernel.fused_add_rmsnorm baseline for {op} {phase}; beat every "
                  "token workload while preserving the exact interface and both outputs."),
            axes={"M": var(f"{phase} token count (sweep)"), "H": const(K.hidden, "hidden size")},
            inputs={"x": tensor(["M", "H"], "bfloat16"), "residual": tensor(["M", "H"], "bfloat16"),
                    "weight": tensor(["H"], "bfloat16"), "eps": tensor(None, "float32")},
            outputs={"normed": tensor(["M", "H"], "bfloat16"), "residual_out": tensor(["M", "H"], "bfloat16")})


def _mm_gemma_rmsnorm():
    r = recipe("minimax_gemma_rmsnorm.py")
    for name, phase in [("mm_gemma_rmsnorm_prefill", "prefill"), ("mm_gemma_rmsnorm_decode", "decode")]:
        op, h = "Gemma-RMSNorm", MM.hidden
        yield TaskSpec(
            model=MM.model, name=name, op=op, family="gemma-rmsnorm", phase=phase, hf_id=MM.hf_id,
            recipe=r, sweep=sweep_for(phase), backend="sgl_kernel gemma_rmsnorm (blackwell)",
            meta={"H": h, "interface_exact": True},
            description=(
                f"MiniMax-M3 {op} ({phase}), Gemma-RMSNorm over hidden={h} (sgl_kernel.gemma_rmsnorm; "
                f"scales by 1+weight). out[M,{h}] = x * rsqrt(mean(x^2,-1)+eps) * (1 + weight). "
                f"INTERFACE-EXACT drop-in. Baseline = sglang production kernel; beat its latency while matching output."),
            goal=(f"Optimize solution.py against SGLang's sgl_kernel.gemma_rmsnorm "
                  f"baseline for MiniMax-M3 {op} {phase}; beat every token workload "
                  "while preserving the exact interface and matching output."),
            axes={"M": var(f"{phase} token count (sweep)"), "H": const(h, "MiniMax-M3 hidden size")},
            inputs={"x": tensor(["M", "H"], "bfloat16"), "weight": tensor(["H"], "bfloat16"),
                    "eps": tensor(None, "float32")},
            outputs={"output": tensor(["M", "H"], "bfloat16")})


def _mm_gemma_fused_add():
    r = recipe("minimax_gemma_fused_add_rmsnorm.py")
    for name, phase in [("mm_gemma_fused_add_rmsnorm_prefill", "prefill"), ("mm_gemma_fused_add_rmsnorm_decode", "decode")]:
        op, h = "Gemma fused add-RMSNorm", MM.hidden
        yield TaskSpec(
            model=MM.model, name=name, op=op, family="gemma-fused-add-rmsnorm", phase=phase, hf_id=MM.hf_id,
            recipe=r, sweep=sweep_for(phase), backend="sgl_kernel gemma_fused_add_rmsnorm (blackwell)",
            meta={"H": h, "interface_exact": True},
            description=(
                f"MiniMax-M3 {op} ({phase}), fused residual-add + Gemma-RMSNorm over hidden={h} "
                f"(sgl_kernel.gemma_fused_add_rmsnorm, in-place). residual_out = x + residual; "
                f"normed = rmsnorm(residual_out) * (1 + weight). INTERFACE-EXACT drop-in. Baseline = "
                f"sglang production kernel; beat its latency while matching BOTH outputs."),
            goal=(f"Optimize solution.py against SGLang's "
                  f"sgl_kernel.gemma_fused_add_rmsnorm baseline for MiniMax-M3 {op} "
                  f"{phase}; beat every token workload while preserving the exact "
                  "interface and both outputs."),
            axes={"M": var(f"{phase} token count (sweep)"), "H": const(h, "MiniMax-M3 hidden size")},
            inputs={"x": tensor(["M", "H"], "bfloat16"), "residual": tensor(["M", "H"], "bfloat16"),
                    "weight": tensor(["H"], "bfloat16"), "eps": tensor(None, "float32")},
            outputs={"normed": tensor(["M", "H"], "bfloat16"), "residual_out": tensor(["M", "H"], "bfloat16")})


def specs():
    out = []
    for gen in (_kimi_rmsnorm, _kimi_fused_add_rmsnorm, _mm_gemma_rmsnorm, _mm_gemma_fused_add):
        out.extend(gen())
    return out
