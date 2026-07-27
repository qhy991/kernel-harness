"""GLM-5.2 DSA Sparse Attention (PREFILL) — native-64-head Triton flash kernel.

Task: sparse MLA. q[M,64,576] attends its per-query top-2048 rows of kv[32768,576]
-> out[M,64,512], bf16, sm_scale = 576**-0.5 = 1/24, S=32768, M in {1024,2048,4096}.

WHY THIS EXISTS (see .humanize/rlcr/2026-07-27_11-57-03/task3-direction.md):
The production baseline `glm52_ops.reference('dsa_attn','prefill')` is aiter's ASM
`mla_decode_fwd`, which has NO bf16 gqa=64 kernel on gfx942 and so pads q 64->128 heads
-> ~2x wasted QK^T+P.V FLOPs (measured ~15% useful MFU). A native-64 kernel does HALF the
compute. An index-locality probe on that ASM path showed collapsing the gather to a single
cached row saves only ~15% (allzero/random 0.853@M2048, 0.850@M4096) => the kernel is
COMPUTE/padding-bound, not gather-bound => a native-64 kernel has real headroom (Codex-
vetted GO, 2026-07-27).

PRECISION (the 5e-6 calc_diff gate, vs the fully-f32 math oracle `_sparse_mla_math_oracle`):
  * QK: bf16xbf16 MFMA with fp32 accumulation (tl.dot default). The inputs are already bf16,
    so `tl.dot(q_bf16, k_bf16)` computes each product in fp32 and accumulates in fp32 =
    exactly the f32 einsum of the bf16 inputs, at matrix-core speed. Do NOT upcast q/k to
    fp32 first (materializes q_f32[64,576], uses the ~8x-slower fp32 MFMA) and do NOT round
    logits to bf16 (drifts calc_diff to ~6.5e-6 > gate — BL-20260723-aiter-fp32yq-mfma-qk).
    Span all 576 dims (512 latent + 64 rope); dropping to 512 is wrong.
  * softmax: fp32 online (running max/sum), matches the oracle's single-pass full softmax.
    Never cast the max/sum/acc normalization state to bf16.
  * PV: default bf16 MFMA with fp32 accumulation (PV_F32=0), ~2x faster than fp32 PV. This
    was the sibling-kernel's opt-in-only path (it left a few near-zero output elems over the
    abs_tol floor at the decode kernel's tiny M); here it is CONFIRMED gate-safe at every
    prefill shape by the authoritative evaluate_task.py run — calc_diff 1.87e-6 (2.7x under
    the 5e-6 gate) and elementwise_failed=0 at M=1024/2048/4096 (the few near-zero elems pass
    on rel_tol, not abs). The fp32-accumulator PV path (DSA_PV_F32=1) is still available as a
    fallback but is not needed. Never cast the accumulator itself to bf16.

ABI: run(inputs)->out on the FROZEN input dict (no re-quantize/re-seed/rebuild; no sort/
dedup/prune of indices). On any unexpected shape/dtype or kernel error, fall back to the
reference (AC-3). MLA layout: the first d_v=512 dims of the 576 are the value/latent dims,
the last 64 are rope — QK spans all 576, PV/out use only the first 512.
"""
from __future__ import annotations
import os
import torch
import triton
import triton.language as tl

from testbench.harness import glm52_ops


@triton.jit
def _dsa_prefill_fused(
    Q, KV, IDX, O, M, H, TK,
    sqm, sqh, sqd, skv_s, skv_d, sim, sik, som, soh, sod,
    sm_scale,
    BH: tl.constexpr, BK: tl.constexpr, DV: tl.constexpr, DR: tl.constexpr,
    PV_F32: tl.constexpr,
):
    """One program = one query row x BH heads; single-pass online softmax over all TK
    top-k keys, fp32 (or bf16) PV accumulation, normalize, write bf16. No intermediate HBM.
    This is the sibling dsa_attn_decode `_dsa_split` with NS=1 and the combine step inlined
    (single split => weight exp(m-m)=1, so out = acc / l_i) — mathematically identical."""
    pid_m = tl.program_id(0)
    pid_h = tl.program_id(1)
    offs_h = pid_h * BH + tl.arange(0, BH)
    hmask = offs_h < H
    d0 = tl.arange(0, DV)        # value/latent dims [0:512]
    d1 = tl.arange(0, DR)        # rope dims [512:576]
    # q halves for these heads (loaded once, reused across all key tiles); kept bf16.
    q0 = tl.load(Q + pid_m * sqm + offs_h[:, None] * sqh + d0[None, :] * sqd,
                 mask=hmask[:, None], other=0.)
    q1 = tl.load(Q + pid_m * sqm + offs_h[:, None] * sqh + (DV + d1)[None, :] * sqd,
                 mask=hmask[:, None], other=0.)
    m_i = tl.full((BH,), -float('inf'), tl.float32)
    l_i = tl.zeros((BH,), tl.float32)
    acc = tl.zeros((BH, DV), tl.float32)
    for k0 in range(0, TK, BK):
        offs_k = k0 + tl.arange(0, BK)
        kmask = offs_k < TK
        krow = tl.load(IDX + pid_m * sim + offs_k * sik, mask=kmask, other=0)
        kt0 = tl.load(KV + krow[:, None] * skv_s + d0[None, :] * skv_d,
                      mask=kmask[:, None], other=0.)          # [BK, 512] bf16
        kt1 = tl.load(KV + krow[:, None] * skv_s + (DV + d1)[None, :] * skv_d,
                      mask=kmask[:, None], other=0.)          # [BK, 64] bf16
        # QK: bf16 MFMA, fp32 accumulate (fp32-precision logits over all 576 dims)
        qk = (tl.dot(q0, tl.trans(kt0)) + tl.dot(q1, tl.trans(kt1))) * sm_scale
        qk = tl.where(kmask[None, :], qk, -float('inf'))      # [BH, BK] fp32
        m_new = tl.maximum(m_i, tl.max(qk, 1))
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(qk - m_new[:, None])                       # [BH, BK] fp32
        l_i = l_i * alpha + tl.sum(p, 1)
        # PV. fp32 (PV_F32=1) is the proven-safe recipe; bf16 PV is ~2x faster MFMA and only
        # enabled after the FULL 3-layer gate passes with margin on every shape.
        if PV_F32:
            acc = acc * alpha[:, None] + tl.dot(p, kt0.to(tl.float32))
        else:
            acc = acc * alpha[:, None] + tl.dot(p.to(tl.bfloat16), kt0)
        m_i = m_new
    out = acc / l_i[:, None]
    tl.store(O + pid_m * som + offs_h[:, None] * soh + d0[None, :] * sod,
             out.to(tl.bfloat16), mask=hmask[:, None])


# Config knobs. Defaults = the task4 op-level GATE-1 WINNER (event-sweep + authoritative
# evaluate_task.py hipgraph run, 2026-07-27): BH=64, BK=64, nw=4, ns=1, wpe=1, bf16 PV
# (DSA_PV_F32=0), and matrix_instr_nonkdim=16 (DSA_MFMA=16). The MFMA knob is the decisive
# lever: Triton's default MFMA selection (32x32x8) for this skinny-M (64-head) sparse-MLA
# dot is ~40% slower than forcing the 16x16x16 instruction. With mfma16 the native-64 kernel
# WINS all three shapes vs the ASM padded-128 baseline by the conservative p90<p10 gate:
#   M=1024 1235us (asm 1677, 1.36x)  M=2048 2217us (asm 2876, 1.30x)  M=4096 4485us (asm 5835,
#   1.30x); calc_diff 1.87e-6 (gate 5e-6), elementwise_failed=0 on every shape.
# Env vars override each knob for re-tuning; the gate runs with these defaults (no env set).
# NOTE BK must be a power of 2 (Triton block-dim constraint); BK=48 fails to compile.
def _cfg():
    def _i(name, default):
        v = os.environ.get(name)
        return int(v) if v is not None else default
    return dict(
        BH=_i("DSA_BH", 64),
        BK=_i("DSA_BK", 64),
        num_warps=_i("DSA_NW", 4),
        num_stages=_i("DSA_NS", 1),
        waves_per_eu=_i("DSA_WPE", 1),
        pv_f32=_i("DSA_PV_F32", 0),   # bf16 PV — gate-confirmed safe (see docstring)
        mfma=_i("DSA_MFMA", 16),      # matrix_instr_nonkdim=16 — the winning MFMA knob
        kpack=_i("DSA_KPACK", 0),
    )


_DV, _DR = 512, 64


def _run_triton(q, kv2, idx, sm, dv):
    M, H, _d_qk = q.shape
    TK = idx.shape[1]
    dev = q.device
    out = torch.empty(M, H, dv, dtype=torch.bfloat16, device=dev)
    c = _cfg()
    BH = c["BH"]
    amd = {}
    if c["waves_per_eu"]:
        amd["waves_per_eu"] = c["waves_per_eu"]
    if c["mfma"]:
        amd["matrix_instr_nonkdim"] = c["mfma"]
    if c["kpack"]:
        amd["kpack"] = c["kpack"]
    grid = (M, triton.cdiv(H, BH))
    _dsa_prefill_fused[grid](
        q, kv2, idx, out, M, H, TK,
        q.stride(0), q.stride(1), q.stride(2),
        kv2.stride(0), kv2.stride(1),
        idx.stride(0), idx.stride(1),
        out.stride(0), out.stride(1), out.stride(2),
        sm,
        BH=BH, BK=c["BK"], DV=_DV, DR=_DR, PV_F32=c["pv_f32"],
        num_warps=c["num_warps"], num_stages=c["num_stages"], **amd,
    )
    return out


def run(inputs: dict):
    q = inputs["q"]
    kv = inputs["kv"]
    idx_in = inputs["indices"]
    try:
        # Accept both AMD (2-D kv, 2-D int64 idx) and CUDA (singleton kv-head axis) schemas.
        kv2 = (kv[:, 0, :] if kv.ndim == 3 else kv).contiguous()
        idx = (idx_in[:, 0, :] if idx_in.ndim == 3 else idx_in).to(torch.int32).contiguous()
        sm = float(inputs["sm_scale"])
        dv = int(inputs["d_v"])
        assert q.ndim == 3 and q.dtype == torch.bfloat16 and q.is_cuda
        assert kv2.shape[-1] == q.shape[-1] and dv == _DV and q.shape[-1] == _DV + _DR
        return _run_triton(q.contiguous(), kv2, idx, sm, dv)
    except Exception:
        # AC-3 safety net for unexpected shape/dtype or a kernel error. NOTE: at M=1024 the
        # ASM reference itself fails the oracle, so this net must not be relied on there —
        # the Triton path must succeed at M=1024 (it does on the frozen inputs).
        # DSA_DEBUG_RAISE=1 re-raises instead (so a kernel bug can't hide as a silent
        # fallback during bring-up); it is OFF by default and never set during the gate.
        if os.environ.get("DSA_DEBUG_RAISE"):
            raise
        return glm52_ops.reference("dsa_attn", "prefill", inputs)
