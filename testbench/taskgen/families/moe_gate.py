"""MoE gate / routing families (exact-index oracle): Kimi biased-topk + MiniMax sigmoid-topk."""
from ..config import KIMI as K, MM, recipe
from ..spec import TaskSpec, DECODE_SWEEP, EXACT_TOL, var, const, tensor


def _kimi_gate():
    yield TaskSpec(
        model=K.model, name="moe_gate_topk_decode", op="MoE Gate + biased TopK (routing)",
        family="moe-gate", phase="decode", hf_id=K.hf_id, recipe=recipe("kimi_moe_gate.py"),
        sweep=DECODE_SWEEP, tolerance=dict(EXACT_TOL),
        backend="sgl_kernel kimi_k2_moe_fused_gate (blackwell)",
        meta={"num_experts": K.n_routed_experts, "topk": K.moe_topk, "oracle": "exact-index"},
        description=(
            f"Kimi-K2.7 MoE Gate + biased TopK (decode), sgl_kernel.kimi_k2_moe_fused_gate: bias + "
            f"renormalized top-{K.moe_topk} selection over {K.n_routed_experts} experts. Outputs the "
            f"selected expert INDICES[M,{K.moe_topk}] (int32); EXACT-match oracle (atol=0,rtol=0,ratio=1.0). "
            f"Baseline = sglang production kernel; beat its latency while matching routing exactly."),
        goal=("Optimize solution.py against SGLang's "
              "sgl_kernel.kimi_k2_moe_fused_gate baseline for MoE Gate decode; beat "
              "the full sweep and match SGLang routing indices exactly (atol=0)."),
        axes={"M": var("decode token count (sweep)"), "num_experts": const(K.n_routed_experts),
              "topk": const(K.moe_topk)},
        inputs={"input_tensor": tensor(["M", "num_experts"], "float32"),
                "bias": tensor(["num_experts"], "float32")},
        outputs={"indices": tensor(["M", "topk"], "int32")})


def _mm_gate():
    yield TaskSpec(
        model=MM.model, name="mm_moe_gate_topk_decode", op="MoE Gate + sigmoid Top-4 (routing)",
        family="moe-gate", phase="decode", hf_id=MM.hf_id, recipe=recipe("minimax_moe_gate.py"),
        sweep=DECODE_SWEEP, tolerance=dict(EXACT_TOL),
        backend="sgl_kernel topk_sigmoid (blackwell)",
        meta={"num_experts": MM.experts, "topk": MM.topk, "oracle": "exact-index"},
        description=(
            f"MiniMax-M3 MoE Gate + sigmoid Top-{MM.topk} (decode), sgl_kernel.topk_sigmoid: sigmoid + "
            f"correction bias + renormalized top-{MM.topk} over {MM.experts} experts "
            f"(routed_scaling={MM.routed_scaling}). Outputs selected expert INDICES[M,{MM.topk}] (int32); "
            f"EXACT-match oracle. Baseline = sglang production kernel; beat its latency while matching routing."),
        goal=("Optimize solution.py against SGLang's sgl_kernel.topk_sigmoid baseline "
              "for MiniMax-M3 MoE Gate decode; beat the full sweep and match SGLang "
              "routing indices exactly (atol=0)."),
        axes={"M": var("decode token count (sweep)"), "num_experts": const(MM.experts),
              "topk": const(MM.topk)},
        inputs={"gating_output": tensor(["M", "num_experts"], "float32"),
                "correction_bias": tensor(["num_experts"], "float32")},
        outputs={"indices": tensor(["M", "topk"], "int32")})


def specs():
    return list(_kimi_gate()) + list(_mm_gate())
