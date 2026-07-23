# DEFAULT CANDIDATE = ported MI300X tuned winner (0723 campaign), re-validated
# 2026-07-23 against the corrected bench standard (math-oracle gate + aiter-triton
# baseline). Provenance: rewardbench/amd/tuned/dsa_attn_decode.py. Edit freely or pass
# ./run.sh --candidate PATH to test another kernel.
"""Kernel variants for the MI300X 12h optimization campaign.

Each factory returns a run(inputs)->out closure for a given config. The campaign driver
searches (variant x config) per op, gating correctness (calc_diff < 5e-6) and maximizing
the bound-aware roofline reward toward 1.0 (the hardware roofline).

GEMM variants (o_proj / index_k), consuming glm52_ops blockwise-fp8 frozen inputs:
  - bf16_dot   : load fp8, upcast to bf16, tl.dot (bf16 matrix core, 1.3 PF)  [baseline]
  - fp8_dot    : native fp8 tl.dot (e4m3fnuz matrix core, 2.6 PF)             [compute lever]
  - fp8_splitk : native fp8 + split-K reduction (parallelism for skinny/small-M)
MLA variant (dsa_attn): tk-split flash-DECODING + fused combine (config-parametrized).
"""
from __future__ import annotations
import torch, triton, triton.language as tl

# ───────────────────────── blockwise-fp8 GEMM (bf16 or native-fp8 dot) ──────────
@triton.jit
def _gemm_k(A, As, B, Bs, C, M, N, K,
           sam, sak, sbn, sbk, scm, scn, sasm, sask, sbsn, sbsk,
           BM: tl.constexpr, BN: tl.constexpr, GROUP_M: tl.constexpr,
           NATIVE_FP8: tl.constexpr):
    pid = tl.program_id(0)
    grid_m = tl.cdiv(M, BM); grid_n = tl.cdiv(N, BN)
    width = GROUP_M * grid_n
    gid = pid // width
    pid_m = gid * GROUP_M + (pid % width) % GROUP_M
    pid_n = (pid % width) // GROUP_M
    offs_m = pid_m * BM + tl.arange(0, BM)
    offs_n = pid_n * BN + tl.arange(0, BN)
    offs_k = tl.arange(0, 128)
    acc = tl.zeros((BM, BN), tl.float32)
    nkb = K // 128
    for kb in range(nkb):
        k = kb * 128 + offs_k
        a = tl.load(A + offs_m[:, None]*sam + k[None, :]*sak, mask=offs_m[:, None] < M, other=0.)
        b = tl.load(B + offs_n[:, None]*sbn + k[None, :]*sbk, mask=offs_n[:, None] < N, other=0.)
        a_s = tl.load(As + offs_m*sasm + kb*sask, mask=offs_m < M, other=0.)
        b_s = tl.load(Bs + (offs_n // 128)*sbsn + kb*sbsk, mask=offs_n < N, other=0.)
        if NATIVE_FP8:
            acc += tl.dot(a, tl.trans(b)) * a_s[:, None] * b_s[None, :]
        else:
            acc += tl.dot(a.to(tl.bfloat16), tl.trans(b.to(tl.bfloat16))) * a_s[:, None] * b_s[None, :]
    tl.store(C + offs_m[:, None]*scm + offs_n[None, :]*scn, acc.to(tl.bfloat16),
             mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


@triton.jit
def _gemm_splitk_k(A, As, B, Bs, C, M, N, K, SPLIT,
                  sam, sak, sbn, sbk, scm, scn, sasm, sask, sbsn, sbsk,
                  BM: tl.constexpr, BN: tl.constexpr, NATIVE_FP8: tl.constexpr):
    pid_m = tl.program_id(0); pid_n = tl.program_id(1); pid_k = tl.program_id(2)
    offs_m = pid_m * BM + tl.arange(0, BM)
    offs_n = pid_n * BN + tl.arange(0, BN)
    offs_k = tl.arange(0, 128)
    nkb = K // 128
    per = tl.cdiv(nkb, SPLIT)
    kb0 = pid_k * per; kb1 = min(kb0 + per, nkb)
    acc = tl.zeros((BM, BN), tl.float32)
    for kb in range(kb0, kb1):
        k = kb * 128 + offs_k
        a = tl.load(A + offs_m[:, None]*sam + k[None, :]*sak, mask=offs_m[:, None] < M, other=0.)
        b = tl.load(B + offs_n[:, None]*sbn + k[None, :]*sbk, mask=offs_n[:, None] < N, other=0.)
        a_s = tl.load(As + offs_m*sasm + kb*sask, mask=offs_m < M, other=0.)
        b_s = tl.load(Bs + (offs_n // 128)*sbsn + kb*sbsk, mask=offs_n < N, other=0.)
        if NATIVE_FP8:
            acc += tl.dot(a, tl.trans(b)) * a_s[:, None] * b_s[None, :]
        else:
            acc += tl.dot(a.to(tl.bfloat16), tl.trans(b.to(tl.bfloat16))) * a_s[:, None] * b_s[None, :]
    tl.atomic_add(C + offs_m[:, None]*scm + offs_n[None, :]*scn, acc,
                  mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


def gemm_factory(native_fp8: bool, split_k: int = 1):
    def make(cfg):
        BM = cfg["BM"]; BN = cfg["BN"]; GROUP_M = cfg.get("GROUP_M", 8)
        nw = cfg.get("num_warps", 4); ns = cfg.get("num_stages", 2)
        # AMD MI300X tuning knobs (waves_per_eu / MFMA instr size / kpack)
        amd = {k: cfg[k] for k in ("waves_per_eu", "matrix_instr_nonkdim", "kpack") if k in cfg}

        def run(inputs):
            x, xs, w, ws, out = inputs["x_fp8"], inputs["x_scale"], inputs["w_fp8"], inputs["w_scale"], inputs["out"]
            M, K = x.shape; N = w.shape[0]
            xs = xs.contiguous(); ws = ws.contiguous()
            if split_k > 1:
                scratch = torch.zeros(M, N, dtype=torch.float32, device=x.device)
                grid = (triton.cdiv(M, BM), triton.cdiv(N, BN), split_k)
                _gemm_splitk_k[grid](x, xs, w, ws, scratch, M, N, K, split_k,
                    x.stride(0), x.stride(1), w.stride(0), w.stride(1), scratch.stride(0), scratch.stride(1),
                    xs.stride(0), xs.stride(1), ws.stride(0), ws.stride(1),
                    BM=BM, BN=BN, NATIVE_FP8=native_fp8, num_warps=nw, num_stages=ns, **amd)
                out.copy_(scratch)
            else:
                grid = (triton.cdiv(M, BM) * triton.cdiv(N, BN),)
                _gemm_k[grid](x, xs, w, ws, out, M, N, K,
                    x.stride(0), x.stride(1), w.stride(0), w.stride(1), out.stride(0), out.stride(1),
                    xs.stride(0), xs.stride(1), ws.stride(0), ws.stride(1),
                    BM=BM, BN=BN, GROUP_M=GROUP_M, NATIVE_FP8=native_fp8, num_warps=nw, num_stages=ns, **amd)
            return out
        return run
    return make


# ───────────────────────── dsa_attn tk-split flash-decode + fused combine ───────
@triton.jit
def _dsa_split(Q, KV, IDX, ACC, LSE, MAX, M, H, TK, NS,
              sqm, sqh, sqd, skv_s, skv_d, sim, sik,
              sa_m, sa_s, sa_h, sa_d, sl_m, sl_s, sl_h, sm_scale,
              BH: tl.constexpr, BK: tl.constexpr):
    pid_m = tl.program_id(0); pid_s = tl.program_id(1); pid_h = tl.program_id(2)
    offs_h = pid_h * BH + tl.arange(0, BH); hmask = offs_h < H
    d0 = tl.arange(0, 512); d1 = tl.arange(0, 64)
    split = TK // NS; k_start = pid_s * split; k_end = k_start + split
    q0 = tl.load(Q + pid_m*sqm + offs_h[:, None]*sqh + d0[None, :]*sqd, mask=hmask[:, None], other=0.)
    q1 = tl.load(Q + pid_m*sqm + offs_h[:, None]*sqh + (512+d1)[None, :]*sqd, mask=hmask[:, None], other=0.)
    m_i = tl.full((BH,), -float('inf'), tl.float32); l_i = tl.zeros((BH,), tl.float32)
    acc = tl.zeros((BH, 512), tl.float32)
    for k0 in range(k_start, k_end, BK):
        offs_k = k0 + tl.arange(0, BK); kmask = offs_k < k_end
        krow = tl.load(IDX + pid_m*sim + offs_k*sik, mask=kmask, other=0)
        kt0 = tl.load(KV + krow[:, None]*skv_s + d0[None, :]*skv_d, mask=kmask[:, None], other=0.)
        kt1 = tl.load(KV + krow[:, None]*skv_s + (512+d1)[None, :]*skv_d, mask=kmask[:, None], other=0.)
        qk = (tl.dot(q0, tl.trans(kt0)) + tl.dot(q1, tl.trans(kt1))) * sm_scale
        qk = tl.where(kmask[None, :], qk, -float('inf'))
        m_new = tl.maximum(m_i, tl.max(qk, 1)); alpha = tl.exp(m_i - m_new)
        p = tl.exp(qk - m_new[:, None]); l_i = l_i*alpha + tl.sum(p, 1)
        # f32 PV accumulate: rounding p to bf16 here leaves 2-4 near-zero output
        # elements just over the gate (abs 1.22e-3 vs 1e-3 floor; calc_diff 6.4e-6 vs
        # 5e-6). The QK dot is already f32; matching it on PV clears the gate.
        acc = acc*alpha[:, None] + tl.dot(p, kt0.to(tl.float32)); m_i = m_new
    tl.store(ACC + pid_m*sa_m + pid_s*sa_s + offs_h[:, None]*sa_h + d0[None, :]*sa_d, acc, mask=hmask[:, None])
    tl.store(LSE + pid_m*sl_m + pid_s*sl_s + offs_h*sl_h, l_i, mask=hmask)
    tl.store(MAX + pid_m*sl_m + pid_s*sl_s + offs_h*sl_h, m_i, mask=hmask)


@triton.jit
def _dsa_combine(ACC, LSE, MAX, O, M, H, NS,
                sa_m, sa_s, sa_h, sa_d, sl_m, sl_s, sl_h, som, soh, sod, BH: tl.constexpr):
    pid_m = tl.program_id(0); pid_h = tl.program_id(1)
    offs_h = pid_h * BH + tl.arange(0, BH); hmask = offs_h < H
    d0 = tl.arange(0, 512)
    m_g = tl.full((BH,), -float('inf'), tl.float32)
    for s in range(NS):
        m_g = tl.maximum(m_g, tl.load(MAX + pid_m*sl_m + s*sl_s + offs_h*sl_h, mask=hmask, other=-float('inf')))
    num = tl.zeros((BH, 512), tl.float32); den = tl.zeros((BH,), tl.float32)
    for s in range(NS):
        ms = tl.load(MAX + pid_m*sl_m + s*sl_s + offs_h*sl_h, mask=hmask, other=-float('inf'))
        ls = tl.load(LSE + pid_m*sl_m + s*sl_s + offs_h*sl_h, mask=hmask, other=0.)
        w = tl.exp(ms - m_g)
        a = tl.load(ACC + pid_m*sa_m + s*sa_s + offs_h[:, None]*sa_h + d0[None, :]*sa_d, mask=hmask[:, None], other=0.)
        num += a * w[:, None]; den += ls * w
    tl.store(O + pid_m*som + offs_h[:, None]*soh + d0[None, :]*sod, (num/den[:, None]).to(tl.bfloat16), mask=hmask[:, None])


def dsa_factory():
    def make(cfg):
        BH = cfg["BH"]; BK = cfg["BK"]; NS = cfg["NS"]; nw = cfg.get("num_warps", 4)
        ns = cfg.get("num_stages", 1); BH2 = cfg.get("BH2", 32)
        amd = {k: cfg[k] for k in ("waves_per_eu",) if k in cfg}

        def run(inputs):
            q = inputs["q"]
            # AMD glm52_ops build_inputs yields 2-D kv (S, D_QK) and 2-D int64 indices
            # (M, tk). Older/CUDA builds carry a singleton kv-head axis (S,1,D_QK) /
            # (M,1,tk); collapse it when present so the wrapper accepts both schemas.
            kv = inputs["kv"]
            kv2 = (kv[:, 0, :] if kv.ndim == 3 else kv).contiguous()
            idx_in = inputs["indices"]
            idx = (idx_in[:, 0, :] if idx_in.ndim == 3 else idx_in).to(torch.int32).contiguous()
            sm = inputs["sm_scale"]; dv = inputs["d_v"]
            M, H, _ = q.shape; TK = idx.shape[1]; dev = q.device
            ns_ = NS
            while TK % ns_ != 0:
                ns_ //= 2
            acc = torch.empty(M, ns_, H, dv, dtype=torch.float32, device=dev)
            lse = torch.empty(M, ns_, H, dtype=torch.float32, device=dev)
            mx = torch.empty(M, ns_, H, dtype=torch.float32, device=dev)
            _dsa_split[(M, ns_, triton.cdiv(H, BH))](q, kv2, idx, acc, lse, mx, M, H, TK, ns_,
                q.stride(0), q.stride(1), q.stride(2), kv2.stride(0), kv2.stride(1),
                idx.stride(0), idx.stride(1), acc.stride(0), acc.stride(1), acc.stride(2), acc.stride(3),
                lse.stride(0), lse.stride(1), lse.stride(2), sm, BH=BH, BK=BK, num_warps=nw, num_stages=ns, **amd)
            out = torch.empty(M, H, dv, dtype=torch.bfloat16, device=dev)
            _dsa_combine[(M, triton.cdiv(H, BH2))](acc, lse, mx, out, M, H, ns_,
                acc.stride(0), acc.stride(1), acc.stride(2), acc.stride(3),
                lse.stride(0), lse.stride(1), lse.stride(2), out.stride(0), out.stride(1), out.stride(2),
                BH=BH2, num_warps=4)
            return out
        return run
    return make


# ==================== MI300X 12h-campaign winner: dsa_attn_decode ====================
# variant=flash_split  geomean_reward=0.08476
# per-shape: {"16": {"lat_us": 123.22, "reward": 0.06141, "pct_roofline": 6.14, "bound": "memory", "speedup_vs_ref": 2.089, "reward_vs_ref": 2.089}, "32": {"lat_us": 129.39, "reward": 0.11697, "pct_roofline": 11.7, "bound": "memory", "speedup_vs_ref": 2.974, "reward_vs_ref": 2.974}}
_CFG = {"NS": 8, "BH": 64, "BK": 16, "BH2": 16, "num_warps": 4, "num_stages": 1, "waves_per_eu": 1}
_run = dsa_factory()(_CFG)
def run(inputs):
    return _run(inputs)
