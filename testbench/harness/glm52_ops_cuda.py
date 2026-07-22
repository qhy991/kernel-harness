"""GLM-5.2 operator definitions — **CUDA/NVIDIA ONLY** (B200, sm100).

The AMD counterpart is `glm52_ops_amd.py`. `glm52_ops.py` is a shim that
routes to whichever of these two the current backend needs, so existing
`from testbench.harness import glm52_ops` imports keep working transparently.

This file assumes `torch.float8_e4m3fn`, TMA/UE8M0 scale format, deep_gemm +
sgl_kernel as the operator provider, and B200 peaks. Every `IS_ROCM` branch
from the merged module is pre-collapsed to its False path; there is no ROCm
code path left inside. Keep it that way when editing — any conditional on
platform belongs in the shim, not here.

One file owns everything a task needs, so a task directory carries no definition
of its own and cannot drift from what actually runs:

    spec(op, phase)                     -> the full contract (shapes, thresholds, cost)
    build_inputs(op, phase, M, S, ...)  -> the frozen input dict (seeded, quantized once)
    reference(op, phase, inputs)        -> the baseline kernel == the correctness oracle
    poison(inputs)                      -> destroy the reference's answer before the candidate
    compare(ref, cand, op, phase, ins)  -> every metric plus a decided pass/reason
    calc_diff(x, y)                     -> deep_gemm.testing.numeric.calc_diff, verbatim
    cost(op, phase, M, S)               -> (flops, bytes_hbm, compute_dtype)
    reward(latency_ms, *cost)           -> bound-aware roofline utilisation
    problem(op, phase, device)          -> the whole problem definition as a dict
    describe(op, phase, device)         -> problem() rendered as text

Provenance
----------
Merged from opbench (PR1) and rewardbench (PR2), which were compared op-by-op.
Three of the five families are identical in both: gemm (same quant helpers, same
tensors, same backend), bmm (byte-identical per-tensor quant math), and mla
(identical q/kv/indices construction). The remaining two differ, and neither PR
is right about both:

  * moe   -> PR2 is right. PR1 sizes the per-expert slab as
             round128(ceil(total_m/E)) with no guard on the actual histogram, so
             every prefill shape overflows (max bin 1055 vs 1024 at M=1024, 2096
             vs 2048, 4198 vs 4096). deep_gemm's masked kernel then indexes past
             each expert's slab into its neighbour's rows: no crash, no NaN,
             finite garbage — and compare's `out[e, :masked_m[e]]` slice gets
             silently clamped back, so nothing anywhere reports an error. PR2's
             `max(..., max(counts))` guard is adopted here.
  * score -> PR1 is right. fp8_mqa_logits takes no separate q scale; real sglang
             (dsa_indexer.py) folds the per-token q_scale and the index
             softmax_scale into `weights` before calling. PR2 omits that fold, so
             its logits are off by a per-token factor — it has no correctness
             gate, so it never noticed. PR1's fold is adopted here.

Cost model and peaks follow PR2 (verified bit-exact against rewardbench across
all original ops x 5 shapes): its byte model additionally counts the fp8 scale
side-bands and the MLA index buffer that PR1 omits (+0.00%..+5.19%). Peaks are
HBM 8.0e12 / FP8 4.5e15 / BF16 2.25e15 — PR2 and testbench/harness/profile.py
agree on 8.0e12; opbench/mfu.py's 7.7e12 is the lone outlier.

Baseline caveat
---------------
The backend provider owns the production baseline. On AMD/ROCm the MI300X
provider routes dense FP8 GEMM through SGLang's
`aiter_w8a8_block_fp8_linear` wrapper, so the benchmark measures the same
gfx942 AITER Triton path SGLang dispatches at runtime instead of the faster
AITER CK proxy. If the SGLang source tree is unavailable, the provider may fall
back to AITER CK or torch-native kernels, and calibration will mark that as not
a full SGLang baseline reproduction.
"""
from __future__ import annotations

import math
import random
import sys
from pathlib import Path

import torch

# evaluate_task loads this file via importlib path; ensure package imports work.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from testbench.harness.backends import get_backend

BACKEND_BUNDLE = get_backend()
DEVICE_PROFILE = BACKEND_BUNDLE.profile
if DEVICE_PROFILE.platform != "cuda":
    raise RuntimeError(
        f"glm52_ops_cuda requires a CUDA backend; got platform={DEVICE_PROFILE.platform!r} "
        f"(profile={DEVICE_PROFILE.id!r}). Set KERNEL_HARNESS_PLATFORM=cuda or import "
        "the shim `testbench.harness.glm52_ops` instead."
    )
OPERATOR_PROVIDER = BACKEND_BUNDLE.provider
FP8_DTYPE = getattr(torch, DEVICE_PROFILE.fp8_dtype_name)
IS_ROCM = False
USE_UE8M0 = True

# ── model constants (GLM-5.2) — identical in PR1 and PR2 (9/9 verified) ──
HIDDEN_SIZE = 6144
Q_LORA_RANK = 2048
KV_LORA_RANK = 512
QK_NOPE_HEAD_DIM = 192
QK_HEAD_DIM = 256
V_HEAD_DIM = 256
NUM_HEADS = 64
D_QK = 576
D_V = 512
TOPK = 2048
INDEX_N_HEADS = 32
INDEX_HEAD_DIM = 128
MOE_INTERMEDIATE_SIZE = 2048
N_EXPERT = 8              # single-GPU shard (EP32)
EXPERTS_PER_TOK = 8
FUSED_QKV_A_OUT = 2624
BLOCK_SIZE_KV = 64
HEAD_DIM_WITH_SF = 132    # 128 fp8 bytes + 4-byte inline f32 scale

FP8_MAX = torch.finfo(FP8_DTYPE).max
DEFAULT_S = 65536
DEFAULT_SEED = 0
DEFAULT_SWEEP = {"prefill": (1024, 2048, 4096), "decode": (16, 32)}

# ── roofline peaks — selected device profile, not part of the model contract ──
FP8_B, BF16_B, F32_B = 1, 2, 4
HBM_BYTES_PER_S = DEVICE_PROFILE.peaks["hbm_bytes_per_s"]
PEAK_FLOPS = {
    "fp8": DEVICE_PROFILE.peaks["fp8"],
    "bf16": DEVICE_PROFILE.peaks["bf16"],
}
PEAKS = DEVICE_PROFILE.as_dict()

# ── operator tables ──
GEMM_OPS = {
    "fused_qkv_a":    dict(K=HIDDEN_SIZE, N=FUSED_QKV_A_OUT,        rows="M"),
    "q_b":            dict(K=Q_LORA_RANK, N=NUM_HEADS * QK_HEAD_DIM, rows="M"),
    "o_proj":         dict(K=NUM_HEADS * V_HEAD_DIM, N=HIDDEN_SIZE,  rows="M"),
    "index_q_upproj": dict(K=Q_LORA_RANK, N=INDEX_N_HEADS * INDEX_HEAD_DIM, rows="M"),
    # index_k projects every KV token, so prefill drives it with S, not M: all three
    # prefill shapes collapse to one [65536, 6144] x [6144, 128] GEMM. Intended, and
    # identical in PR1 (rows="S_or_M") and PR2 (C.gemm_fp8_cost(a["S"], H, IHD)).
    "index_k":        dict(K=HIDDEN_SIZE, N=INDEX_HEAD_DIM,          rows="S_or_M"),
}
BMM_OPS = {
    "absorbed_W_UK": dict(K=QK_NOPE_HEAD_DIM, N=KV_LORA_RANK),
    "absorbed_W_UV": dict(K=KV_LORA_RANK,     N=V_HEAD_DIM),
}
MOE_OPS = {
    "moe_gate": dict(K=HIDDEN_SIZE,           N=MOE_INTERMEDIATE_SIZE),
    "moe_up":   dict(K=HIDDEN_SIZE,           N=MOE_INTERMEDIATE_SIZE),
    "moe_down": dict(K=MOE_INTERMEDIATE_SIZE, N=HIDDEN_SIZE),
}
MOE_FUSED_OPS = {
    "moe_total": dict(K=HIDDEN_SIZE, I=MOE_INTERMEDIATE_SIZE, E=N_EXPERT),
}
MLA_OPS = ("dsa_attn",)
SCORE_OPS = ("index_score",)

ALL_OPS = (
    list(GEMM_OPS)
    + list(BMM_OPS)
    + list(MOE_OPS)
    + list(MOE_FUSED_OPS)
    + list(MLA_OPS)
    + list(SCORE_OPS)
)

_LABEL = {
    "fused_qkv_a": "Fused QKV-A Projection", "q_b": "Q-B Projection",
    "o_proj": "Attention O Projection", "index_q_upproj": "Indexer Q Up-Projection",
    "index_k": "Indexer K Projection", "absorbed_W_UK": "Absorbed W_UK BMM",
    "absorbed_W_UV": "Absorbed W_UV BMM", "moe_gate": "MoE Gate Projection",
    "moe_up": "MoE Up Projection", "moe_down": "MoE Down Projection",
    "moe_total": "Routed Expert Gate+Up/Down Total",
    "dsa_attn": "DSA Sparse Attention", "index_score": "Indexer Score (MQA logits)",
}


def family(op: str) -> str:
    if op in GEMM_OPS:  return "gemm"
    if op in BMM_OPS:   return "bmm"
    if op in MOE_OPS:   return "moe"
    if op in MOE_FUSED_OPS: return "moe_fused"
    if op in MLA_OPS:   return "mla"
    if op in SCORE_OPS: return "score"
    raise KeyError(f"unknown op {op!r}; known: {', '.join(ALL_OPS)}")


def infer_phase(M: int) -> str:
    return "prefill" if M >= 1024 else "decode"


def _round128(x: int) -> int:
    return ((x + 127) // 128) * 128


# ── correctness tolerances ───────────────────────────────────────────────────
# Structure and values come from upstream, not from us.
#
# DIFF_TOL — the aggregate gate. calc_diff is *literally the same function* in
#   deep_gemm.testing.numeric.calc_diff and FlashMLA's kernelkit get_cos_diff.
#   5e-6 is FlashMLA's own cos_diff_tol for sparse-MLA decode output, i.e. direct
#   provenance for this exact kernel family. It leaves ~3 orders of headroom over
#   what a legitimate independent implementation actually produces here (measured
#   4.5e-9 for o_proj: dequantize fp8 -> f32 -> torch.matmul -> bf16).
#   DeepGEMM's own 1e-3 is NOT used: that budget exists to absorb fp8-vs-bf16
#   quantization error, and we have none — candidate and reference consume the
#   same fp8 bytes, so the only divergence left is accumulation order.
#
# REL_TOL — FlashMLA's 2.01/128. bf16 carries 8 mantissa bits, so its relative
#   ulp is 2^-8; the legitimate implementation above needs >= 7.8125e-3 (= 2^-7,
#   exactly two ulps), leaving 2x. Being dtype-derived, it ports across ops.
#
# ABS_TOL_FACTOR — abs_tol CANNOT be a constant. Output magnitude spans seven
#   orders across these GLM-5.2 ops (|ref|max: dsa_attn 0.285, o_proj 564,
#   index_score 1.5e7 in f32), because build_inputs draws weights from a plain
#   randn with no 1/sqrt(K) scaling. FlashMLA's fixed 1e-3 is calibrated for O(1)
#   attention output and would forgive nothing at o_proj's scale. So abs_tol is
#   derived per shape from the reference itself: abs_tol = factor * |ref|.max().
#   Its only job is to forgive near-zero elements whose relative error explodes
#   (measured: exactly 1 element of 98304 needs it, at abs_err 4.6e-5), and
#   1e-4 * 564 = 0.056 covers that with three orders to spare.
DIFF_TOL = 5e-6
REL_TOL = 2.01 / 128
ABS_TOL_FACTOR = 1e-4
NEAR_ZERO_FLOOR = 1e-3  # abs_tol floor: forgive near-zero fp-reorder noise (sparse-MLA); calc_diff (5e-6) still gates aggregate


def calc_diff(x: torch.Tensor, y: torch.Tensor) -> float:
    """Aggregate difference, verbatim from deep_gemm.testing.numeric.calc_diff
    (FlashMLA's get_cos_diff is the same function).

    Algebraically ||x-y||^2 / (||x||^2 + ||y||^2), so unlike plain cosine it is
    scale-SENSITIVE: y = k*x gives 1 - 2k/(1+k^2), i.e. 0.2 at k=0.5 or k=2.
    That is why no separate magnitude check is needed alongside it.
    """
    x, y = x.double(), y.double()
    denominator = (x * x + y * y).sum()
    if denominator == 0:
        return 0.0
    return (1 - 2 * (x * y).sum() / denominator).item()


def spec(op: str, phase: str) -> dict:
    """The complete contract for one (op, phase). Nothing else may declare these."""
    fam = family(op)
    kind = {"gemm": "dense", "bmm": "dense", "moe": "masked_grouped",
            "moe_fused": "dense",
            "mla": "mla_sparse"}.get(fam) or (
        "logits_ksrange" if phase == "prefill" else "logits_paged")
    d = dict(op=op, phase=phase, label=_LABEL[op], family=fam,
             backend=OPERATOR_PROVIDER.baseline_name(fam, phase), output_kind=kind,
             peak_dtype="bf16" if fam == "mla" else "fp8",
             diff_tol=DIFF_TOL, rel_tol=REL_TOL, abs_tol_factor=ABS_TOL_FACTOR,
             elementwise_gate=True,
             sweep=list(DEFAULT_SWEEP[phase]), S=DEFAULT_S, seed=DEFAULT_SEED,
             has_output_buffer=fam in ("gemm", "moe"))
    if fam == "gemm":
        d.update(K=GEMM_OPS[op]["K"], N=GEMM_OPS[op]["N"], rows=GEMM_OPS[op]["rows"])
    elif fam == "bmm":
        d.update(K=BMM_OPS[op]["K"], N=BMM_OPS[op]["N"], batch=NUM_HEADS)
    elif fam == "moe":
        d.update(K=MOE_OPS[op]["K"], N=MOE_OPS[op]["N"], E=N_EXPERT,
                 experts_per_tok=EXPERTS_PER_TOK)
    elif fam == "moe_fused":
        d.update(diff_tol=1e-5, elementwise_gate=False)
        d.update(K=HIDDEN_SIZE, N=HIDDEN_SIZE, intermediate=MOE_INTERMEDIATE_SIZE,
                 E=N_EXPERT, experts_per_tok=EXPERTS_PER_TOK)
    return d


# ══════════════════════════════════════════════════════════════════════════
# Quantization helpers
# ══════════════════════════════════════════════════════════════════════════
def cast_to_fp8_per_tensor(x: torch.Tensor):
    """Per-tensor quant for bmm_fp8's cuBLAS path. PR1's cast_to_fp8_per_tensor
    and PR2's quant_per_tensor are the same math; this is it."""
    amax = x.abs().float().amax()
    scale = (amax / FP8_MAX).float().clamp(min=1e-12)
    return (x.float() / scale).clamp(-FP8_MAX, FP8_MAX).to(FP8_DTYPE), scale.view(1).to(x.device)


# ══════════════════════════════════════════════════════════════════════════
# Frozen inputs — ONE dict feeds both the reference and the candidate
# ══════════════════════════════════════════════════════════════════════════
def build_inputs(op: str, phase: str, M: int, S: int = DEFAULT_S,
                 device=None, seed: int = DEFAULT_SEED) -> dict:
    device = torch.device(device or "cuda:0")
    torch.manual_seed(seed)
    random.seed(seed)
    fam = family(op)
    if fam == "gemm":  return _build_gemm(op, phase, M, S, device)
    if fam == "bmm":   return _build_bmm(op, M, device)
    if fam == "moe":   return _build_moe(op, M, device)
    if fam == "moe_fused": return _build_moe_total(M, device)
    if fam == "mla":   return _build_mla(M, S, device)
    return _build_score(phase, M, S, device)


def _build_gemm(op, phase, M, S, device):
    cfg = GEMM_OPS[op]
    K, N = cfg["K"], cfg["N"]
    rows = S if (cfg["rows"] == "S_or_M" and phase == "prefill") else M
    x_bf16 = torch.randn(rows, K, dtype=torch.bfloat16, device=device)
    x_fp8, x_scale = OPERATOR_PROVIDER.per_token_cast(x_bf16, use_ue8m0=USE_UE8M0)
    x_scale = OPERATOR_PROVIDER.align_scale(x_scale)
    w_bf16 = torch.randn(N, K, dtype=torch.bfloat16, device=device)
    w_fp8, w_scale = OPERATOR_PROVIDER.per_block_cast(w_bf16, use_ue8m0=USE_UE8M0)
    # Pre-allocated so the timed call does not also measure an allocation. It is
    # poisoned between the reference and the candidate — see poison().
    out = torch.empty(rows, N, dtype=torch.bfloat16, device=device)
    return dict(x_fp8=x_fp8, x_scale=x_scale, w_fp8=w_fp8, w_scale=w_scale,
                rows=rows, N=N, device=device, out=out)


def _build_bmm(op, M, device):
    cfg = BMM_OPS[op]
    K, N = cfg["K"], cfg["N"]
    B = NUM_HEADS
    A_fp8, A_scale = cast_to_fp8_per_tensor(
        torch.randn(B, M, K, dtype=torch.bfloat16, device=device))
    B_fp8, B_scale = cast_to_fp8_per_tensor(
        torch.randn(B, K, N, dtype=torch.bfloat16, device=device))
    return dict(A_fp8=A_fp8.view(B, M, K), B_fp8=B_fp8.view(B, K, N),
                A_scale=A_scale, B_scale=B_scale)


def _build_moe(op, M, device):
    cfg = MOE_OPS[op]
    K, N = cfg["K"], cfg["N"]
    E = N_EXPERT
    total_m = M * EXPERTS_PER_TOK
    counts = [0] * E
    rng = random
    for _ in range(total_m):
        counts[rng.randint(0, E - 1)] += 1
    # PR2's fix, adopted. deep_gemm's masked kernel walks ceil(masked_m[e]/BLOCK_M)
    # M-blocks for expert e and indexes into [E, expected_m, .]; sizing the slab to
    # ceil(total_m/E) alone overflows whenever the multinomial is skewed, which it
    # is at every prefill shape (max bin 1055 vs 1024 at M=1024, 2096 vs 2048,
    # 4198 vs 4096). The overflow reads into the NEXT expert's rows — finite
    # garbage, no crash, and compare's slice clamps silently — so it is invisible
    # without this guard. PR1 lacks it.
    expected_m = _round128(max((total_m + E - 1) // E, max(counts)))
    x_bf16 = torch.randn(E, expected_m, K, dtype=torch.bfloat16, device=device)
    x_fp8 = torch.empty_like(x_bf16, dtype=FP8_DTYPE)
    x_scale = torch.empty(E, expected_m, K // 128, dtype=torch.float32, device=device)
    for e in range(E):
        x_fp8[e], x_scale[e] = OPERATOR_PROVIDER.per_token_cast(
            x_bf16[e], use_ue8m0=USE_UE8M0
        )
    w_bf16 = torch.randn(E, N, K, dtype=torch.bfloat16, device=device)
    w_fp8 = torch.empty(E, N, K, dtype=FP8_DTYPE, device=device)
    w_scale = torch.empty(E, _round128(N) // 128, K // 128, dtype=torch.float32, device=device)
    for e in range(E):
        w_fp8[e], w_scale[e] = OPERATOR_PROVIDER.per_block_cast(
            w_bf16[e], use_ue8m0=USE_UE8M0
        )
    out = torch.empty(E, expected_m, N, dtype=torch.bfloat16, device=device)
    return dict(x_fp8=x_fp8, x_scale=x_scale, w_fp8=w_fp8, w_scale=w_scale,
                masked_m=torch.tensor(counts, dtype=torch.int32, device=device),
                expected_m=expected_m, E=E, N=N, device=device, out=out)


def _cast_to_fp8_rows(x: torch.Tensor):
    scale = x.abs().float().amax(dim=-1, keepdim=True).clamp(min=1e-12) / FP8_MAX
    q = (x.float() / scale).clamp(-FP8_MAX, FP8_MAX).to(FP8_DTYPE)
    return q.contiguous(), scale.squeeze(-1).contiguous()


def _build_moe_total(M, device):
    E, H, I, topk = N_EXPERT, HIDDEN_SIZE, MOE_INTERMEDIATE_SIZE, EXPERTS_PER_TOK
    hidden_states = torch.randn(M, H, dtype=torch.bfloat16, device=device)
    w1 = torch.randn(E, 2 * I, H, dtype=torch.bfloat16, device=device).to(FP8_DTYPE).contiguous()
    w2 = torch.randn(E, H, I, dtype=torch.bfloat16, device=device).to(FP8_DTYPE).contiguous()
    topk_ids = torch.arange(E, dtype=torch.int32, device=device).repeat(M, 1)
    topk_weights = torch.softmax(
        torch.randn(M, topk, dtype=torch.float32, device=device), dim=-1
    ).contiguous()
    router_logits = torch.zeros(M, E, dtype=torch.float32, device=device)
    moe_config_kwargs = dict(
        num_experts=E,
        num_local_experts=E,
        hidden_size=H,
        intermediate_size_per_partition=I,
        top_k=topk,
        activation="silu",
        is_gated=True,
        inplace=False,
    )
    helpers = {}
    try:
        from sglang.srt.layers.moe.moe_runner import MoeRunnerConfig
        from sglang.srt.layers.moe.topk import StandardTopKOutput

        helpers["topk_output"] = StandardTopKOutput(
            topk_weights,
            topk_ids,
            router_logits,
        )
        helpers["moe_runner_config"] = MoeRunnerConfig(
            **moe_config_kwargs,
            params_dtype=hidden_states.dtype,
        )
    except Exception:
        pass
    return dict(
        hidden_states=hidden_states,
        w1=w1,
        w2=w2,
        topk_weights=topk_weights,
        topk_ids=topk_ids,
        router_logits=router_logits,
        w1_scale=torch.ones(E, dtype=torch.float32, device=device),
        w2_scale=torch.ones(E, dtype=torch.float32, device=device),
        a1_scale=torch.ones(1, dtype=torch.float32, device=device),
        a2_scale=torch.ones(1, dtype=torch.float32, device=device),
        moe_config_kwargs=moe_config_kwargs,
        **helpers,
    )


def _build_mla(M, S, device):
    q = torch.randn(M, NUM_HEADS, D_QK, dtype=torch.bfloat16, device=device)
    kv_shape = (S, 1, D_QK)
    kv = torch.randn(*kv_shape, dtype=torch.bfloat16, device=device)
    tk = min(TOPK, S)
    indices = torch.stack([torch.randperm(S, device=device)[:tk]
                           for _ in range(M)])
    indices = indices.view(M, 1, tk).to(torch.int32)
    return dict(q=q, kv=kv, indices=indices, sm_scale=D_QK ** -0.5, d_v=D_V)


def _build_score(phase, M, S, device):
    softmax_scale = INDEX_HEAD_DIM ** -0.5
    if phase == "prefill":
        q_bf16 = torch.randn(M, INDEX_N_HEADS, INDEX_HEAD_DIM,
                             dtype=torch.bfloat16, device=device)
        q_view = q_bf16.view(M, INDEX_N_HEADS, INDEX_HEAD_DIM // 128, 128)
        q_scale = (q_view.abs().float().amax(dim=-1) / FP8_MAX).clamp(min=1e-12)
        q_fp8 = (q_view.float() / q_scale.unsqueeze(-1)).to(FP8_DTYPE).view(
            M, INDEX_N_HEADS, INDEX_HEAD_DIM)
        k_bf16 = torch.randn(S, INDEX_HEAD_DIM, dtype=torch.bfloat16, device=device)
        k_view = k_bf16.view(S, INDEX_HEAD_DIM // 128, 128)
        k_scale = (k_view.abs().float().amax(dim=-1) / FP8_MAX).clamp(min=1e-12)
        k_fp8 = (k_view.float() / k_scale.unsqueeze(-1)).to(FP8_DTYPE).view(
            S, INDEX_HEAD_DIM)
        # PR1's fold, adopted. fp8_mqa_logits takes NO separate q scale: real
        # sglang (dsa_indexer.py) folds the per-token q_scale and the index
        # softmax_scale into `weights` before the call. PR2 skips this, so its
        # logits are off by a per-token factor — it has no correctness gate, so
        # nothing caught it.
        weights = (torch.randn(M, INDEX_N_HEADS, dtype=torch.float32, device=device)
                   * q_scale.squeeze(-1) * softmax_scale)
        return dict(q_fp8=q_fp8, k_fp8=k_fp8, k_scale=k_scale.squeeze(-1),
                    weights=weights,
                    ks=torch.zeros(M, dtype=torch.int32, device=device),
                    ke=torch.full((M,), S, dtype=torch.int32, device=device))

    # decode: paged KV, one page table per request
    nbps = (S + BLOCK_SIZE_KV - 1) // BLOCK_SIZE_KV
    total_blocks = nbps * M
    q_bf16 = torch.randn(M, 1, INDEX_N_HEADS, INDEX_HEAD_DIM,
                         dtype=torch.bfloat16, device=device)
    q_view = q_bf16.view(M * INDEX_N_HEADS, INDEX_HEAD_DIM // 128, 128)
    q_scale = (q_view.abs().float().amax(dim=-1) / FP8_MAX).clamp(min=1e-12)
    q_fp8 = (q_view.float() / q_scale.unsqueeze(-1)).to(FP8_DTYPE).view(
        M, 1, INDEX_N_HEADS, INDEX_HEAD_DIM)
    # Build the paged cache from REAL fp8-quantized data with scale=1.0: random
    # uint8 would contain fp8-NaN bit patterns and every logit would come back NaN.
    data = torch.randn(total_blocks, BLOCK_SIZE_KV, 1, 128, dtype=torch.bfloat16,
                       device=device).to(FP8_DTYPE).view(torch.uint8)
    sf = torch.ones(total_blocks, BLOCK_SIZE_KV, 1, dtype=torch.float32,
                    device=device).view(torch.uint8).reshape(total_blocks, BLOCK_SIZE_KV, 1, 4)
    kv_cache_fp8 = torch.cat([data, sf], dim=-1).contiguous()
    weights = (torch.randn(M, INDEX_N_HEADS, dtype=torch.float32, device=device)
               * q_scale.view(M, INDEX_N_HEADS) * softmax_scale)
    seqlens = torch.full((M, 1), S, dtype=torch.int32, device=device)
    block_tables = torch.arange(total_blocks, dtype=torch.int32,
                                device=device).view(M, nbps)
    sched = OPERATOR_PROVIDER.paged_mqa_metadata(seqlens, BLOCK_SIZE_KV)
    return dict(q_fp8=q_fp8, kv_cache_fp8=kv_cache_fp8, weights=weights,
                seqlens=seqlens, block_tables=block_tables, schedule_metadata=sched,
                max_seq_len=nbps * BLOCK_SIZE_KV)


# ══════════════════════════════════════════════════════════════════════════
# Reference == the baseline kernel == the correctness oracle
# ══════════════════════════════════════════════════════════════════════════
def reference(op: str, phase: str, inputs: dict):
    fam = family(op)
    if not OPERATOR_PROVIDER.supports(op, phase):
        raise NotImplementedError(
            f"provider {OPERATOR_PROVIDER.id!r} does not support {op}/{phase}"
        )
    return OPERATOR_PROVIDER.reference(op, phase, fam, inputs)


def poison(inputs: dict) -> bool:
    """Destroy the reference's answer in the shared output buffer.

    gemm/moe pre-allocate `out` and reference() writes into it, so cloning the
    reference output is not enough: the correct answer is still sitting in
    inputs["out"], and a candidate whose whole body is `return inputs["out"]`
    scores cosine 1.000000 having computed nothing (verified on B200). Filling
    with NaN makes that candidate fail instead. Returns whether anything was
    poisoned (False for the families that allocate their own output).
    """
    out = inputs.get("out")
    if torch.is_tensor(out):
        out.fill_(float("nan"))
        return True
    return False


# ══════════════════════════════════════════════════════════════════════════
# Comparison — mask by output_kind, then the three upstream layers
# ══════════════════════════════════════════════════════════════════════════
def _main(x):
    return x[0] if isinstance(x, (tuple, list)) else x


def prepare(out, kind: str, inputs: dict) -> torch.Tensor:
    """Reduce a raw output to the flat vector that is actually compared."""
    out = _main(out)
    if kind in ("dense", "mla_sparse"):
        return out.reshape(-1)
    if kind == "masked_grouped":
        masked_m = inputs["masked_m"]
        parts = [out[e, : int(masked_m[e].item())].reshape(-1)
                 for e in range(out.shape[0]) if int(masked_m[e].item()) > 0]
        return torch.cat(parts) if parts else out.reshape(-1)
    if kind == "logits_ksrange":
        S = out.shape[-1]
        col = torch.arange(S, device=out.device).view(1, -1)
        mask = (col >= inputs["ks"].view(-1, 1)) & (col < inputs["ke"].view(-1, 1))
        return (out * mask).reshape(-1)
    if kind == "logits_paged":
        S = out.shape[-1]
        col = torch.arange(S, device=out.device).view(1, -1)
        return (out[..., :S] * (col < inputs["seqlens"].view(-1, 1))).reshape(-1)
    raise ValueError(f"unknown output_kind {kind!r}")


def compare(ref_out, cand_out, op: str, phase: str, inputs: dict) -> dict:
    """Three-layer correctness check, structured after FlashMLA's
    kernelkit.check_is_allclose (which is a superset of DeepGEMM's calc_diff gate):

      1. inf / -inf / nan must occupy the SAME positions in both.
      2. every element must satisfy (abs_err < abs_tol) OR (rel_err < rel_tol).
         The OR is load-bearing: large elements pass on relative error, near-zero
         elements pass on absolute error. Neither alone works — a small absolute
         perturbation near zero drives relative error to ~92%, while at |ref|~500
         a single bf16 ulp is an absolute error of 2.
      3. the aggregate calc_diff must be <= diff_tol.

    Returns every metric plus a decided `pass` and, on failure, a `reason` that
    names which layer rejected it.
    """
    s = spec(op, phase)
    kind = s["output_kind"]
    r = prepare(ref_out, kind, inputs).reshape(-1).float()
    c = prepare(cand_out, kind, inputs).reshape(-1).float()
    if r.shape != c.shape:
        raise ValueError(
            f"shape mismatch: reference {tuple(r.shape)} vs candidate {tuple(c.shape)}")

    out = {"diff_tol": s["diff_tol"], "rel_tol": s["rel_tol"],
           "elements": int(r.numel())}

    # ── 1. anomalies, positionally ──
    # The harness NaN-poisons the shared output buffer, so a candidate that never
    # wrote its output arrives here as all-NaN and is named as such here, rather
    # than silently poisoning every downstream metric.
    for label, rm, cm in (("nan", torch.isnan(r), torch.isnan(c)),
                          ("+inf", r == float("inf"), c == float("inf")),
                          ("-inf", r == float("-inf"), c == float("-inf"))):
        if not torch.equal(rm, cm):
            n_c, n_r = int(cm.sum()), int(rm.sum())
            out.update(pass_=False, anomaly_ok=False, calc_diff=None,
                       max_abs_err=None, max_rel_err=None, abs_tol=None,
                       elementwise_failed=None)
            out["reason"] = (
                f"{label} in {n_c} candidate elements vs {n_r} reference "
                f"({100.0 * n_c / max(int(r.numel()), 1):.1f}% of the output)"
                + (" — the candidate never wrote its output"
                   if label == "nan" and n_r == 0 and n_c == int(r.numel()) else ""))
            return _finish(out)
        # Zero matched anomalies in both so they cannot poison the metrics below.
        r = r.masked_fill(rm, 0.0)
        c = c.masked_fill(cm, 0.0)
    out["anomaly_ok"] = True

    # ── diagnostic only: cosine + best-fit scale ──
    # Cosine is NOT a gate here, and cannot be: it is scale-invariant, so
    # reference*k scores 1.000000 for every positive k (verified at k = 0.5, 2.0,
    # 1000.0), and a wrong FP8 scale convention is exactly that shape of bug. But
    # paired with calc_diff it *names* the failure, which neither does alone:
    # direction right + magnitude wrong is a scale error, direction wrong is an
    # algorithm/indexing/layout error, and those want completely different fixes.
    # best_fit_scale is the least-squares k for cand ~= k*ref, so a scale error reports
    # the factor to look for rather than leaving it to be guessed.
    rd, cd_ = r.double(), c.double()
    rn = torch.linalg.vector_norm(rd)
    cn = torch.linalg.vector_norm(cd_)
    dot = (rd * cd_).sum()
    out["cosine"] = (dot / (rn * cn)).item() if rn > 0 and cn > 0 else float("nan")
    out["best_fit_scale"] = (dot / (rn * rn)).item() if rn > 0 else float("nan")

    # ── 2. elementwise: abs OR rel ──
    abs_tol = max(s["abs_tol_factor"] * r.abs().max().item(), NEAR_ZERO_FLOOR)
    raw_abs = (c - r).abs()
    raw_rel = raw_abs / (r.abs() + 1e-6)
    pass_mask = (raw_abs < abs_tol) | (raw_rel < s["rel_tol"])
    n_fail = int((~pass_mask).sum())
    out.update(abs_tol=abs_tol,
               max_abs_err=raw_abs.max().item(), max_rel_err=raw_rel.max().item(),
               elementwise_failed=n_fail)

    # ── 3. aggregate ──
    out["calc_diff"] = calc_diff(r, c)
    elementwise_gate = bool(s.get("elementwise_gate", True))
    out["elementwise_gate"] = elementwise_gate

    if n_fail and elementwise_gate:
        bad = ~pass_mask
        out.update(pass_=False, reason=(
            f"{n_fail}/{int(r.numel())} elements fail both tolerances "
            f"(abs<{abs_tol:.3e} or rel<{s['rel_tol']:.3e}); worst offender "
            f"abs={raw_abs[bad].max().item():.3e} rel={raw_rel[bad].max().item():.3e}"
            + _diagnose(out)))
        return _finish(out)
    if abs(out["calc_diff"]) > s["diff_tol"]:
        out.update(pass_=False, reason=(
            f"calc_diff {out['calc_diff']:.3e} > {s['diff_tol']:.1e} — every element is "
            f"within tolerance but the output is systematically off" + _diagnose(out)))
        return _finish(out)
    out.update(pass_=True, reason=None)
    return _finish(out)


def _diagnose(out: dict) -> str:
    """Turn cosine + best_fit_scale into the sentence that says which bug this is."""
    cos, k = out.get("cosine"), out.get("best_fit_scale")
    if cos is None or math.isnan(cos):
        return ""
    if cos > 0.999:
        return (f". Direction is right (cosine {cos:.6f}) and the output is ~{k:.4g}x the "
                f"reference — a magnitude error, not an algorithm one. Look at the "
                f"dequant scale, the UE8M0 exponent, or the FP8 max "
                f"({FP8_MAX:g} for this backend)")
    return (f". Direction is wrong (cosine {cos:.6f}), so this is not a scale error — "
            f"look at the algorithm, the indexing, or the layout")


def _finish(out: dict) -> dict:
    out["pass"] = out.pop("pass_")
    return out


# ══════════════════════════════════════════════════════════════════════════
# Cost model (PR2) — original ops verified bit-exact against rewardbench.
# ══════════════════════════════════════════════════════════════════════════
def cost(op: str, phase: str, M: int, S: int = DEFAULT_S):
    """(flops, bytes_hbm, compute_dtype) for one shape."""
    fam = family(op)
    if fam == "gemm":
        cfg = GEMM_OPS[op]
        K, N = cfg["K"], cfg["N"]
        rows = S if (cfg["rows"] == "S_or_M" and phase == "prefill") else M
        byts = (rows * K * FP8_B + N * K * FP8_B + rows * N * BF16_B
                + rows * (K // 128) * F32_B                    # per-token x scale
                + math.ceil(N / 128) * (K // 128) * F32_B)     # per-block w scale
        return 2.0 * rows * K * N, float(byts), "fp8"
    if fam == "bmm":
        cfg = BMM_OPS[op]
        K, N, B = cfg["K"], cfg["N"], NUM_HEADS
        byts = B * M * K * FP8_B + B * K * N * FP8_B + B * M * N * BF16_B
        return 2.0 * B * M * K * N, float(byts), "fp8"
    if fam == "moe":
        cfg = MOE_OPS[op]
        K, N, E = cfg["K"], cfg["N"], N_EXPERT
        total_m = M * EXPERTS_PER_TOK
        byts = (total_m * K * FP8_B + E * N * K * FP8_B + total_m * N * BF16_B
                + total_m * (K // 128) * F32_B
                + E * math.ceil(N / 128) * (K // 128) * F32_B)
        return 2.0 * total_m * K * N, float(byts), "fp8"
    if fam == "moe_fused":
        E, H, I, topk = N_EXPERT, HIDDEN_SIZE, MOE_INTERMEDIATE_SIZE, EXPERTS_PER_TOK
        total_m = M * topk
        flops = 2.0 * total_m * H * (2 * I) + 2.0 * total_m * I * H
        byts = (
            M * H * BF16_B
            + E * (2 * I) * H * FP8_B
            + E * H * I * FP8_B
            + total_m * (2 * I) * BF16_B
            + total_m * I * BF16_B
            + M * H * BF16_B
            + E * F32_B * 2
            + 2 * F32_B
        )
        return flops, float(byts), "fp8"
    if fam == "mla":
        tk = min(TOPK, S)
        # KV dedup: gathered rows saturate at the latent cache, so KV traffic stops
        # growing with batch once tk*M >= S. PR1 and PR2 agree on the dedup here.
        kv_rows = min(tk * M, S)
        byts = (M * NUM_HEADS * D_QK * BF16_B + kv_rows * D_QK * BF16_B
                + M * NUM_HEADS * D_V * BF16_B + M * tk * F32_B)   # int32 indices
        return 2.0 * NUM_HEADS * M * tk * (D_QK + D_V), float(byts), "bf16"
    # score
    h, hd = INDEX_N_HEADS, INDEX_HEAD_DIM
    q_b, w_b, logits_b = M * h * hd * FP8_B, M * h * F32_B, M * S * F32_B
    if phase == "prefill":
        byts = q_b + S * hd * FP8_B + S * F32_B + w_b + logits_b
    else:
        nbps = (S + BLOCK_SIZE_KV - 1) // BLOCK_SIZE_KV
        byts = q_b + (nbps * M) * BLOCK_SIZE_KV * HEAD_DIM_WITH_SF + w_b + logits_b
    return 2.0 * M * S * h * hd, float(byts), "fp8"


def reward(latency_ms: float, flops: float, bytes_hbm: float, compute_dtype: str,
           attainable_bw: float | None = None) -> dict:
    """Bound-aware roofline utilisation (PR2). Collapses exactly to compute_util
    when compute-bound and bw_util when memory-bound. NOT clamped: a value > 1
    means the byte model undercounts real traffic, and callers should surface it
    rather than hide it.

    `attainable_bw` (bytes/s), when given, is a measured pure-copy ceiling for this
    byte footprint (see bin/bw_ceiling.py). It adds achievability-honest fields
    ALONGSIDE the spec-peak reward — small transfers cannot reach spec HBM, so a
    memory-bound op near its attainable ceiling is at its real roof — and never
    changes the existing reward number."""
    peak = PEAK_FLOPS[compute_dtype]
    lat_s = latency_ms * 1e-3
    ai = flops / bytes_hbm
    ridge = peak / HBM_BYTES_PER_S
    achieved_flops, achieved_bw = flops / lat_s, bytes_hbm / lat_s
    ceiling = min(peak, ai * HBM_BYTES_PER_S)
    out = {
        "latency_ms": latency_ms, "tflops": achieved_flops / 1e12,
        "gbps": achieved_bw / 1e9, "arithmetic_intensity": ai, "ridge": ridge,
        "bound": "compute" if ai >= ridge else "memory",
        "compute_util": achieved_flops / peak, "bw_util": achieved_bw / HBM_BYTES_PER_S,
        "reward": achieved_flops / ceiling if ceiling > 0 else 0.0,
        "compute_dtype": compute_dtype,
    }
    if attainable_bw and attainable_bw > 0:
        out["attainable_bw_gbps"] = attainable_bw / 1e9
        out["attainable_frac_of_peak"] = attainable_bw / HBM_BYTES_PER_S
        # honest memory-bound utilisation: achieved vs what a pure copy of this
        # footprint actually sustains, not vs the physically-unreachable spec peak.
        out["bw_util_attainable"] = achieved_bw / attainable_bw
    return out


# ══════════════════════════════════════════════════════════════════════════
# The agent-facing problem statement — generated, so it cannot drift
# ══════════════════════════════════════════════════════════════════════════
PROBLEM_SCHEMA_VERSION = "1.0"

BASELINE_CAVEAT = OPERATOR_PROVIDER.baseline_caveat
ACCEPTED_CANDIDATE_FORMS = list(OPERATOR_PROVIDER.accepted_candidate_forms)


def problem(op: str, phase: str, device=None) -> dict:
    """The complete problem definition as data. `describe()` renders exactly this, so
    the prose and the JSON cannot disagree.

    Tensor shapes and dtypes are read off a REAL build_inputs() call rather than
    transcribed, so neither can disagree with what the harness actually runs. Pass
    `device` to include them; without a GPU that section is omitted, not guessed.
    """
    s = spec(op, phase)
    fam = s["family"]

    math_notes = []
    if fam == "gemm":
        rows = "S" if s["rows"] == "S_or_M" and phase == "prefill" else "M"
        expr = f"out[{rows},{s['N']}] = x_fp8[{rows},{s['K']}] @ w_fp8[{s['N']},{s['K']}].T"
        dims = {"K": s["K"], "N": s["N"], "rows": rows}
        if rows == "S":
            math_notes.append(
                f"rows = S = {s['S']} (every KV token), not M — all prefill shapes are "
                f"the same single GEMM. Intended; PR1 and PR2 agree.")
    elif fam == "bmm":
        expr = (f"out[{s['batch']},M,{s['N']}] = A[{s['batch']},M,{s['K']}] @ "
                f"B[{s['batch']},{s['K']},{s['N']}]")
        dims = {"K": s["K"], "N": s["N"], "batch": s["batch"]}
        math_notes.append("per-tensor fp8 scales (cuBLAS path), not blockwise")
    elif fam == "moe":
        expr = (f"masked grouped GEMM over E={s['E']} experts, K={s['K']} N={s['N']}, "
                f"top_k={s['experts_per_tok']}")
        dims = {"K": s["K"], "N": s["N"], "E": s["E"], "top_k": s["experts_per_tok"]}
        math_notes.append(
            f"rows are replicated M*{s['experts_per_tok']} and bucketed into experts by a "
            f"seeded multinomial (masked_m); expected_m is the per-expert slab capacity "
            f"and is sized to hold the largest bin.")
    elif fam == "moe_fused":
        expr = (
            f"SGLang fused MoE total: hidden[M,{HIDDEN_SIZE}] -> "
            f"w1[E,{2 * MOE_INTERMEDIATE_SIZE},{HIDDEN_SIZE}] -> SiLU*Up -> "
            f"w2[E,{HIDDEN_SIZE},{MOE_INTERMEDIATE_SIZE}] -> out[M,{HIDDEN_SIZE}], "
            f"top_k={s['experts_per_tok']}"
        )
        dims = {
            "hidden": HIDDEN_SIZE,
            "intermediate": MOE_INTERMEDIATE_SIZE,
            "E": s["E"],
            "top_k": s["experts_per_tok"],
        }
        math_notes.append(
            "This is the production-equivalent total Routed Expert Gate+Up/Down metric. "
            "The split gate/up/down tasks remain diagnostics only and should not be "
            "reported as the fused MoE total.")
    elif fam == "mla":
        kv_expr = f"kv[{s['S']},1,{D_QK}]"
        expr = (f"sparse MLA: q[M,{NUM_HEADS},{D_QK}] attends the top-{TOPK} of "
                f"{kv_expr} -> out[M,{NUM_HEADS},{D_V}]")
        dims = {"heads": NUM_HEADS, "d_qk": D_QK, "d_v": D_V, "topk": TOPK}
    else:
        expr = (f"indexer logits[M,{s['S']}] = sum_h weights[M,h] * "
                f"(q[M,h,{INDEX_HEAD_DIM}] . k[{s['S']},{INDEX_HEAD_DIM}]),  h={INDEX_N_HEADS}")
        dims = {"heads": INDEX_N_HEADS, "head_dim": INDEX_HEAD_DIM}
        math_notes.append(
            f"`weights` already carries the per-token q_scale and the index softmax_scale "
            f"({INDEX_HEAD_DIM}**-0.5): fp8_mqa_logits takes no separate q scale, and real "
            f"sglang folds them in before the call. Do not apply them again.")

    tensors, tensors_error = None, None
    if device is not None:
        try:
            M0 = s["sweep"][0]
            ins = build_inputs(op, phase, M0, s["S"], device, s["seed"])
            tensors = {"at_M": M0, "read_from": "a real build_inputs() call", "items": []}
            import json as _json
            for k, v in ins.items():
                if torch.is_tensor(v):
                    tensors["items"].append({"name": k, "shape": list(v.shape),
                                             "dtype": str(v.dtype).replace("torch.", "")})
                else:
                    # Only include when JSON-serialisable. Skips MoeRunnerConfig,
                    # StandardTopKOutput (a NamedTuple carrying Tensors),
                    # torch.device, and anything else problem.json can't render.
                    try:
                        _json.dumps(v)
                    except (TypeError, ValueError):
                        continue
                    tensors["items"].append({"name": k, "value": v})
        except Exception as e:
            tensors_error = f"{type(e).__name__}: {e}"

    return {
        "schema_version": PROBLEM_SCHEMA_VERSION,
        "generated_by": "testbench/harness/glm52_ops.py:problem() — do not hand-edit",
        "operator": op,
        "phase": phase,
        "label": s["label"],
        "family": fam,
        "model": "GLM-5.2",
        "deployment": DEVICE_PROFILE.deployment,
        "S": s["S"],
        "seed": s["seed"],

        "math": {"expression": expr, "dims": dims, "notes": math_notes},

        "workload": {
            "axis": "M",
            "sweep": s["sweep"],
            "rule": "every shape must pass correctness AND be beaten on latency",
        },

        "baseline": {
            "backend": s["backend"],
            "platform": DEVICE_PROFILE.platform,
            "profile": DEVICE_PROFILE.id,
            "provider": OPERATOR_PROVIDER.id,
            "call": f"glm52_ops.reference({op!r}, {phase!r}, inputs)",
            "role": "the correctness oracle AND the latency denominator — the same call, "
                    "on the same frozen inputs, timed under the same protocol",
            "caveat": BASELINE_CAVEAT,
        },

        "contract": {
            "entrypoint": "run(inputs: dict) -> output",
            "where": "candidate.py in this directory, or any file/directory passed to "
                     "--candidate (it may live anywhere; testing a kernel does not "
                     "require editing the task)",
            "inputs_call": f"glm52_ops.build_inputs({op!r}, {phase!r}, M, S, device, seed)",
            "frozen": "the very same dict feeds the reference — do NOT re-quantize, "
                      "re-seed, or rebuild any tensor inside run(), or you measure a "
                      "different problem than the one the gate checked",
            "tensors": tensors,
            "tensors_error": tensors_error,
            "output_buffer": (
                {"key": "out",
                 "may_write_in_place": True,
                 "poisoned": "NaN-filled before run() is called",
                 "why": "reference() writes into this shared buffer, so without poisoning "
                        "a candidate whose whole body is `return inputs['out']` inherits "
                        "the reference's answer and scores a perfect match having computed "
                        "nothing. Returning it unwritten now FAILS."}
                if s["has_output_buffer"] else None),
            "accepted_forms": ACCEPTED_CANDIDATE_FORMS,
        },

        "correctness": {
            "output_kind": s["output_kind"],
            "structure": "FlashMLA kernelkit.check_is_allclose; the aggregate is "
                         "deep_gemm.testing.numeric.calc_diff verbatim",
            "layers": [
                {"order": 1, "check": "inf / -inf / nan occupy the same positions in both"},
                {"order": 2,
                 "check": "every element: abs_err < abs_tol OR rel_err < rel_tol",
                 "gate": bool(s.get("elementwise_gate", True)),
                 "rel_tol": s["rel_tol"],
                 "abs_tol": f"{s['abs_tol_factor']:.0e} * |ref|.max(), computed per shape",
                 "why_or": "large elements pass on relative error, near-zero elements on "
                           "absolute — neither alone works",
                 "why_derived_abs_tol": "output magnitude spans seven orders across these "
                                        "GLM-5.2 ops (dsa_attn 0.285, o_proj 564, index_score "
                                        "1.5e7), so a fixed abs_tol cannot port"},
                {"order": 3,
                 "check": "calc_diff <= diff_tol",
                 "diff_tol": s["diff_tol"],
                 "formula": "||x-y||^2 / (||x||^2 + ||y||^2)",
                 "why": "scale-SENSITIVE, unlike cosine — a uniform k*reference is caught "
                        "here (k=0.5 or 2 both give 0.2)"},
            ],
            "post_timing_recheck": "correctness is re-checked on freshly built inputs "
                                   "after timing, to catch a kernel that mutates its "
                                   "inputs or drifts across the timed iterations",
            "diagnostics": {
                "cosine": "reported, never gated — it is scale-invariant, so "
                          "reference*k scores 1.000000 for every k and it cannot "
                          "catch the likeliest FP8 bug on its own",
                "best_fit_scale": "least-squares k for candidate ~= k*reference",
                "why": "paired with calc_diff they name the failure: cosine ~1 with a "
                       "large calc_diff is a magnitude error (check the dequant scale, "
                       "the UE8M0 exponent, or the backend FP8 max) and best_fit_scale "
                       "is the factor to look for; a low cosine is an algorithm, "
                       "indexing or layout error instead. calc_diff alone cannot tell "
                       "them apart — flipping the output and applying a uniform FP8 "
                       "scale error both score similarly badly",
            },
        },

        "performance": {
            "timing": getattr(
                BACKEND_BUNDLE.timer, "contract_description", BACKEND_BUNDLE.timer.description
            ),
            "timer_id": getattr(
                BACKEND_BUNDLE.timer, "contract_id", BACKEND_BUNDLE.timer.id
            ),
            "why_device_time": "the reward is a hardware-utilisation ratio, so it must be "
                               "paired with device time; a per-call wall-clock timer "
                               "reports ~99us for this op's ~47us kernel, and the "
                               "difference is host dispatch stall",
            "gate": "at least one shape WINS and no shape REGRESSES",
            "shape_verdict": {
                "win": "reference p10 / candidate p90 > 1.0 — the candidate is ahead "
                       "even on the reading least favourable to it",
                "regress": "reference p90 / candidate p10 < 1.0 — the candidate is "
                           "behind even on the reading most favourable to it",
                "neutral": "neither — inside the noise band; does not veto the run",
                "why_quantiles": "not max/min: dividing two extremes lets one artifact "
                                 "sample decide the verdict, which at repeat=10 is "
                                 "likely rather than rare",
                "why_not_all_shapes": "requiring every shape to win is unreachable the "
                                      "moment one shape merely matches — an "
                                      "identical-to-reference candidate measures "
                                      "sp_cons 0.855-0.989, never above 1.0 — so it "
                                      "made per-shape fallback impossible to express",
            },
            "fallback_is_allowed": (
                "run() may branch on the shape and hand the losing shapes to "
                "glm52_ops.reference(op, phase, inputs). That is what SGLang itself "
                "does (deepgemm_w8a8_block_fp8_linear_with_fallback), and it is the "
                "expected answer when a kernel wins in one regime only: the fallback "
                "shapes land as `neutral` and no longer veto the win. Falling back on "
                "EVERY shape scores zero wins and still fails, so this buys nothing "
                "unless something real is gained somewhere. Do the dispatch inside "
                "run() — the harness will not do it for you, because then it would be "
                "measuring a kernel the candidate does not contain."),
            "defaults": {"warmup": 3, "repeat": 10, "iterations": 30},
            "repeat_note": "--repeat 1 is a probe, not a verdict: at 1 the conservative "
                           "margin collapses to the median one and a candidate identical "
                           "to the reference passes a >1.0 gate a good fraction of the time",
            "reward": "bound-aware roofline utilisation: (flops/latency) / "
                      "min(peak_flops, ai*peak_bw); unclamped",
            "peaks": PEAKS,
        },

        "verdict": {
            "exit_0": "correct on every shape AND performance gate met",
            "exit_1": "correct on every shape, performance gate not met",
            "exit_2": "incorrect, incomplete sweep, or correctness did not survive timing",
            "exit_3": "infrastructure error, or task.json disagrees with glm52_ops",
        },

        "run": {
            "gate": "./run.sh",
            "describe": "./run.sh --describe        (this text)",
            "describe_json": "./run.sh --describe --json",
            "external_candidate": "./run.sh --candidate PATH",
            "one_shape": "./run.sh --M <M>",
        },
    }


def describe(op: str, phase: str, device=None) -> str:
    """Render problem() as the text an agent or a human reads."""
    p = problem(op, phase, device)
    c, k, f = p["contract"], p["correctness"], p["performance"]
    L = [f"TASK  {p['operator']}/{p['phase']} — {p['label']}",
         f"  {p['model']}, {p['deployment']}.  family={p['family']}  S={p['S']}  seed={p['seed']}",
         "",
         f"  MATH   {p['math']['expression']}"]
    for n in p["math"]["notes"]:
        L += [f"         NOTE {line}" for line in _wrap(n, 66)]
    L += ["",
          f"  WORKLOAD   M in {p['workload']['sweep']}   ({p['workload']['rule']})",
          "",
          f"  BASELINE   {p['baseline']['backend']}",
          f"             {p['baseline']['call']}"]
    L += [f"             {line}" for line in _wrap(p["baseline"]["role"], 64)]
    L += [f"             CAVEAT {line}" if i == 0 else f"                    {line}"
          for i, line in enumerate(_wrap(p["baseline"]["caveat"], 60))]
    L += ["",
          f"  CONTRACT   {c['entrypoint']} — that function is the entire ABI."]
    L += [f"             {line}" for line in _wrap(c["where"], 64)]
    L += [f"             inputs = {c['inputs_call']}"]
    L += [f"             {line}" for line in _wrap(c["frozen"], 64)]
    if c["tensors"]:
        L.append(f"             tensors at M={c['tensors']['at_M']} "
                 f"(read from {c['tensors']['read_from']}):")
        for t in c["tensors"]["items"]:
            if "shape" in t:
                L.append(f"               {t['name']:<18} {str(tuple(t['shape'])):<26} {t['dtype']}")
            else:
                L.append(f"               {t['name']:<18} {t['value']!r}")
    elif c["tensors_error"]:
        L.append(f"             (could not build inputs: {c['tensors_error']})")
    if c["output_buffer"]:
        L.append(f"             inputs['out'] is pre-allocated and MAY be written in place,")
        L.append(f"             but is {c['output_buffer']['poisoned']}: returning it "
                 f"unwritten FAILS.")
    L.append("             accepted candidate forms:")
    for form in c["accepted_forms"]:
        w = _wrap(form, 60)
        L.append(f"               - {w[0]}")
        L += [f"                 {line}" for line in w[1:]]
    L += ["",
          f"  CORRECT    masked by output_kind={k['output_kind']}, then all three, in order"]
    L += [f"             ({line})" for line in _wrap(k["structure"], 62)]
    for lyr in k["layers"]:
        L.append(f"               {lyr['order']}. {lyr['check']}")
        for key in ("rel_tol", "abs_tol", "diff_tol", "formula"):
            if key in lyr:
                L.append(f"                  {key} = {lyr[key]}")
        for key in ("why", "why_or", "why_derived_abs_tol"):
            if key in lyr:
                L += [f"                  {line}" for line in _wrap(lyr[key], 58)]
    L += [f"             {line}" for line in _wrap(k["post_timing_recheck"], 64)]
    L.append("             diagnostics (reported, never gated):")
    L += [f"               {line}" for line in _wrap(k["diagnostics"]["why"], 60)]
    L += ["", f"  FAST       {f['timing'].split(':')[0]}"]
    L += [f"             gate: {f['gate']}"]
    sv = f["shape_verdict"]
    for key in ("win", "regress", "neutral"):
        L += [f"               {key:<8} {line}" if i == 0 else f"                        {line}"
              for i, line in enumerate(_wrap(sv[key], 54))]
    L += [f"             {line}" for line in _wrap(f["fallback_is_allowed"], 64)]
    L += [f"             defaults: " + ", ".join(f"{a}={b}" for a, b in f["defaults"].items())]
    L += ["",
          "  RUN        ./run.sh                        the gate",
          "             ./run.sh --describe [--json]    this text, or it as JSON",
          "             ./run.sh --candidate PATH       any .py/dir defining run(inputs),",
          "                                             from anywhere — no task edit needed",
          "             ./run.sh --M <M>                one shape",
          "             exit 0=correct+fast  1=correct  2=wrong  3=infra/contract",
          ]
    return "\n".join(L)


def _wrap(text: str, width: int) -> list[str]:
    import textwrap
    return textwrap.wrap(text, width) or [""]
