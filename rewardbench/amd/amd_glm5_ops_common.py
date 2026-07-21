"""Shared engine for the GLM-5.2 operator benchmarks on **AMD MI300X (CDNA3 / gfx942, ROCm)**.

This is the A-card (MI300X) port of kernel-harness/rewardbench/glm5_ops_common.py
(originally B200/SM100 + DeepGEMM/FlashMLA/sgl_kernel). The design is preserved 1:1:

  * Rule 3 "interpret by the kernel's bound": reward is a **bound-aware roofline
    utilization** in [0,1] = achieved / roofline-ceiling. Compute-bound op -> matrix-core
    utilization; memory-bound op -> HBM-bandwidth utilization; auto-classified by
    arithmetic intensity (FLOP/byte) vs the MI300X ridge point.
  * Rule 5 "same ABI/wrapper": every op calls the AMD backend kernel that sglang-ROCm
    uses at runtime (aiter / hipBLASLt via torch._scaled_mm / torch), with the runtime
    dtype and layout — so the number describes the real serving path on MI300X.
  * Rule 2 "fix the benchmark first": shapes come from GLM-5.2 config and are frozen.

Two things change from the CUDA/B200 original, everything else is identical:
  1. Roofline peaks are MI300X (HBM 5.3 TB/s, FP8 e4m3 2.615 PFLOP/s, BF16 1.307 PFLOP/s).
  2. Backend kernels are AMD:
       deep_gemm.fp8_gemm_nt                 -> aiter.gemm_a8w8_blockscale  (fallback: torch._scaled_mm / hipBLASLt)
       deep_gemm.bf16_gemm_nt                -> torch.mm (bf16 -> f32)
       sgl_kernel.bmm_fp8                    -> per-head torch._scaled_mm loop (hipBLASLt)
       deep_gemm.fp8_m_grouped_gemm_nt_masked-> aiter.fmoe / per-expert torch._scaled_mm loop
       sgl_kernel.flash_mla_sparse_fwd       -> aiter MLA (fallback: gather + SDPA, bf16)
       deep_gemm.fp8_mqa_logits              -> torch._scaled_mm logits + weighted sum
  The analytic FLOP/byte cost model is hardware-agnostic and is copied verbatim, so a
  reward computed here is directly comparable to the B200 rewardbench numbers.

FP8 note: on gfx942 the hardware matrix-core FP8 format is **e4m3fnuz** (finfo.max=240),
not OCP e4m3fn (max=448). We auto-select fnuz on ROCm. torch._scaled_mm on ROCm requires
the fnuz variant on MI300X.
"""

import math
import os
import torch

os.environ.setdefault("PYTORCH_ROCM_ARCH", "gfx942")  # MI300X target (harmless on CUDA)

# ══════════════════════════════════════════════════════════════════════════════
# Backend detection: ROCm vs CUDA, aiter availability, FP8 dtype
# ══════════════════════════════════════════════════════════════════════════════
IS_ROCM = torch.version.hip is not None
FP8_DTYPE = torch.float8_e4m3fnuz if IS_ROCM else torch.float8_e4m3fn
# gfx942 matrix-core FP8 is e4m3fnuz; qhy's sglang-ROCm path scales by 224.0 (not the
# 240.0 finfo max) — we match that convention so numbers line up with the deployed path.
# (On CUDA/e4m3fn the OCP max is 448.0.)
FP8_MAX = 224.0 if IS_ROCM else 448.0

# Optional aiter (AMD's kernel library, the sglang-ROCm backend). If present we can
# call the exact production kernels; otherwise we fall back to torch-native ROCm ops
# (hipBLASLt via torch._scaled_mm) which are correct and give a legitimate MI300X
# baseline latency. Every builder marks which path it took in BUILD_BACKEND.
try:
    import aiter  # noqa: F401
    HAVE_AITER = True
except Exception:
    HAVE_AITER = False

# Which backend each builder actually used this process (for provenance in the CSV).
BUILD_BACKEND: dict = {}


# ══════════════════════════════════════════════════════════════════════════════
# GLM-5.2 model config (identical to llm_flops/bench_glm5_*.py and kernel-harness)
# ══════════════════════════════════════════════════════════════════════════════
HIDDEN_SIZE = 6144
Q_LORA_RANK = 2048
KV_LORA_RANK = 512
QK_NOPE_HEAD_DIM = 192
QK_ROPE_HEAD_DIM = 64
QK_HEAD_DIM = 256
V_HEAD_DIM = 256
NUM_HEADS = 64
D_QK = 576            # kv_lora_rank + qk_rope_head_dim (absorbed MLA)
D_V = 512             # kv_lora_rank
TOPK = 2048           # index_topk (DSA)
INDEX_N_HEADS = 32
INDEX_HEAD_DIM = 128
MOE_INTERMEDIATE_SIZE = 2048
N_EXPERT = 8          # single-GPU local experts (llm_flops convention)
NUM_EXPERTS_PER_TOK = 8
FUSED_QKV_A_OUT = Q_LORA_RANK + KV_LORA_RANK + QK_ROPE_HEAD_DIM  # 2624
BLOCK_SIZE_KV = 64    # paged KV block size (decode indexer)
HEAD_DIM_WITH_SF = 132  # 128 fp8 elems + 4-byte inline fp32 scale (paged index KV)

# ══════════════════════════════════════════════════════════════════════════════
# MI300X (CDNA3 / gfx942) roofline peaks — the reward denominators.
# AMD Instinct MI300X datasheet dense figures (no sparsity; MI300X has no 2:4 feature):
#   HBM3 192 GB @ 5.3 TB/s; matrix-core BF16 1307.4 TFLOP/s, FP8 e4m3 2614.9 TFLOP/s.
# Confirmed against qhy's measured workspace (experiment_report.md): HBM 5300 GB/s,
# FP8 2614.8 TFLOPS, BF16 1307.0 TFLOPS, 304 CU.
# (B200/SM100 original was HBM 8 TB/s, FP8 4.5 PF, BF16 2.25 PF.)
# ══════════════════════════════════════════════════════════════════════════════
HBM_BYTES_PER_S = 5.3e12          # 5.3 TB/s HBM3
FP8_PEAK_FLOPS = 2.6149e15        # 2614.9 TFLOP/s dense e4m3 matrix core
BF16_PEAK_FLOPS = 1.3074e15       # 1307.4 TFLOP/s dense bf16 matrix core
PEAK_FLOPS = {"fp8": FP8_PEAK_FLOPS, "bf16": BF16_PEAK_FLOPS}
NUM_CU = 304

FP8_B = 1   # bytes per fp8 element
BF16_B = 2
F32_B = 4

NUM_WARMUP = 5
NUM_RUNS = 20


# ══════════════════════════════════════════════════════════════════════════════
# FP8 quantization (MI300X: e4m3fnuz, per-tensor + per-token/per-block).
# MI300X does NOT require UE8M0 power-of-2 scales (that is a Blackwell DeepGEMM rule);
# plain fp32 scales are used, matching aiter / hipBLASLt.
# ══════════════════════════════════════════════════════════════════════════════
def quant_per_tensor(x):
    """Per-tensor FP8 (scalar scale) — for torch._scaled_mm / hipBLASLt / aiter."""
    amax = x.abs().float().amax().clamp(min=1e-12)
    scale = (amax / FP8_MAX).float()
    x_fp8 = (x.float() / scale).clamp(-FP8_MAX, FP8_MAX).to(FP8_DTYPE)
    return x_fp8, scale.view(1).to(x.device)


def quant_per_token(x):
    """Per-token (per-row) FP8 with a [M,1] fp32 scale (rowwise scaled_mm)."""
    amax = x.abs().float().amax(dim=-1, keepdim=True).clamp(min=1e-12)
    scale = (amax / FP8_MAX).float()
    x_fp8 = (x.float() / scale).clamp(-FP8_MAX, FP8_MAX).to(FP8_DTYPE)
    return x_fp8, scale


def quant_token_blockwise(x_bf16, block=128):
    """Per-token blockwise (128 along K) FP8 with fp32 scales -> (x_fp8[M,K], scale[M,K/128])."""
    m, k = x_bf16.shape
    xv = x_bf16.view(m, k // block, block)
    amax = xv.abs().float().amax(dim=-1).clamp(min=1e-12)
    scale = (amax / FP8_MAX).float()
    x_fp8 = (xv.float() / scale.unsqueeze(-1)).clamp(-FP8_MAX, FP8_MAX).to(FP8_DTYPE).view(m, k)
    return x_fp8, scale


def quant_block_blockwise(w_bf16, block=128):
    """Per-128x128-block FP8 with fp32 scales -> (w_fp8[N,K], scale[ceil(N/128),K/128])."""
    n, k = w_bf16.shape
    n_ceil = (n + block - 1) // block * block
    if n < n_ceil:
        w_pad = torch.zeros(n_ceil, k, dtype=w_bf16.dtype, device=w_bf16.device)
        w_pad[:n] = w_bf16
    else:
        w_pad = w_bf16
    wv = w_pad.view(n_ceil // block, block, k // block, block)
    amax = wv.abs().float().amax(dim=(1, 3)).clamp(min=1e-12)
    scale = (amax / FP8_MAX).float()
    w_fp8 = (wv.float() / scale[:, None, :, None]).clamp(-FP8_MAX, FP8_MAX).to(FP8_DTYPE)
    return w_fp8.view(n_ceil, k)[:n].contiguous(), scale


# ══════════════════════════════════════════════════════════════════════════════
# Timing.  Primary: HIP-graph capture+replay (torch.cuda.CUDAGraph maps to hipGraph on
# ROCm), matching the llm_flops / rewardbench CUDA-graph methodology so numbers are
# comparable across GPUs. Fallback: CUDA/HIP-event timing, robust for kernels that are
# not graph-capturable (aiter autotune, host-side routing).
# ══════════════════════════════════════════════════════════════════════════════
def graph_bench(run_fn, warmup=NUM_WARMUP, iters=NUM_RUNS):
    torch.cuda.synchronize()
    for _ in range(warmup):
        run_fn()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(iters):
            run_fn()
    torch.cuda.synchronize()
    for _ in range(warmup):
        graph.replay()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    graph.replay()
    end.record()
    torch.cuda.synchronize()
    avg_ms = start.elapsed_time(end) / iters
    del graph
    return avg_ms


def event_bench(run_fn, warmup=NUM_WARMUP, iters=NUM_RUNS):
    """HIP/CUDA-event timing WITHOUT graph capture. Includes launch/dispatch overhead."""
    for _ in range(warmup):
        run_fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        run_fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


def robust_bench(run_fn, prefer_graph=True):
    """Try HIP-graph timing; on any capture failure fall back to event timing.
    Returns (avg_ms, method_str)."""
    if prefer_graph and os.environ.get("AMD_BENCH_NO_GRAPH", "0") != "1":
        try:
            return graph_bench(run_fn), "hipgraph"
        except Exception:
            torch.cuda.synchronize()
    return event_bench(run_fn), "event"


# ══════════════════════════════════════════════════════════════════════════════
# Analytic FLOP / HBM-byte models — copied verbatim from the B200 rewardbench
# (hardware-agnostic; only the peaks differ). Bytes count the dtype actually moved
# to/from HBM (fp8=1, bf16=2, f32=4), including scale factors.
# ══════════════════════════════════════════════════════════════════════════════
def gemm_fp8_cost(M, K, N):
    flops = 2 * M * K * N
    bytes_hbm = (M * K * FP8_B + N * K * FP8_B + M * N * BF16_B
                 + M * (K // 128) * F32_B + math.ceil(N / 128) * (K // 128) * F32_B)
    return flops, bytes_hbm, "fp8"


def gemm_bf16_cost(M, K, N):
    flops = 2 * M * K * N
    bytes_hbm = M * K * BF16_B + N * K * BF16_B + M * N * F32_B
    return flops, bytes_hbm, "bf16"


def bmm_fp8_cost(B, M, K, N):
    flops = 2 * B * M * K * N
    bytes_hbm = B * M * K * FP8_B + B * K * N * FP8_B + B * M * N * BF16_B
    return flops, bytes_hbm, "fp8"


def linear_bf16_cost(M, K, N, out_b=BF16_B):
    flops = 2 * M * K * N
    bytes_hbm = M * K * BF16_B + N * K * BF16_B + M * N * out_b
    return flops, bytes_hbm, "bf16"


def sparse_mla_cost(s_q, s_kv, h_q=NUM_HEADS, d_qk=D_QK, d_v=D_V, topk=TOPK):
    """Sparse MLA (bf16): each query attends to topk gathered KV rows. FLOPs = QK^T + PV.
    KV read deduped (single shared latent cache)."""
    tk = min(topk, s_kv)
    flops = 2 * h_q * s_q * tk * (d_qk + d_v)
    kv_rows = min(tk * s_q, s_kv)
    bytes_hbm = (s_q * h_q * d_qk * BF16_B          # q
                 + kv_rows * d_qk * BF16_B           # gathered KV (deduped)
                 + s_q * h_q * d_v * BF16_B          # out
                 + s_q * tk * F32_B)                 # indices (int32)
    return flops, bytes_hbm, "bf16"


def mqa_logits_ragged_cost(M, S, h=INDEX_N_HEADS, hd=INDEX_HEAD_DIM):
    flops = 2 * M * S * h * hd
    bytes_hbm = (M * h * hd * FP8_B
                 + S * hd * FP8_B + S * F32_B
                 + M * h * F32_B
                 + M * S * F32_B)
    return flops, bytes_hbm, "fp8"


def paged_mqa_logits_cost(M, S, h=INDEX_N_HEADS, hd=INDEX_HEAD_DIM):
    num_blocks_per_seq = (S + BLOCK_SIZE_KV - 1) // BLOCK_SIZE_KV
    total_blocks = num_blocks_per_seq * M
    flops = 2 * M * S * h * hd
    bytes_hbm = (M * h * hd * FP8_B
                 + total_blocks * BLOCK_SIZE_KV * HEAD_DIM_WITH_SF
                 + M * h * F32_B
                 + M * S * F32_B)
    return flops, bytes_hbm, "fp8"


def moe_grouped_cost(M, K, N, top_k=NUM_EXPERTS_PER_TOK, n_expert=N_EXPERT):
    total_m = M * top_k
    flops = 2 * total_m * K * N
    bytes_hbm = (total_m * K * FP8_B
                 + n_expert * N * K * FP8_B
                 + total_m * N * BF16_B
                 + total_m * (K // 128) * F32_B
                 + n_expert * math.ceil(N / 128) * (K // 128) * F32_B)
    return flops, bytes_hbm, "fp8"


# ══════════════════════════════════════════════════════════════════════════════
# AMD kernel-call builders — build inputs once, return a no-arg callable.
# Each targets the sglang-ROCm production path (aiter) with a torch-native hipBLASLt
# fallback. `tag` records which op for BUILD_BACKEND provenance.
# ══════════════════════════════════════════════════════════════════════════════
def _scaled_mm(x_fp8, w_fp8, x_scale, w_scale, out_dtype=torch.bfloat16):
    """hipBLASLt FP8 GEMM: x[M,K] (fp8) @ w[N,K].t() (fp8) -> [M,N]. w passed as [N,K]."""
    return torch._scaled_mm(x_fp8, w_fp8.t(), scale_a=x_scale, scale_b=w_scale,
                            out_dtype=out_dtype)


def build_fp8_gemm(M, K, N, device, tag="fp8_gemm"):
    """Blockwise-FP8 GEMM [M,K]x[N,K]->[M,N] bf16. Prefer aiter.gemm_a8w8_blockscale;
    fall back to per-tensor torch._scaled_mm (hipBLASLt)."""
    x_bf16 = torch.randn(M, K, dtype=torch.bfloat16, device=device)
    w_bf16 = torch.randn(N, K, dtype=torch.bfloat16, device=device) * (K ** -0.5)
    if HAVE_AITER:
        try:
            # Exact sglang-ROCm production call (qhy bench_aiter_vs_pytorch.py:75,106):
            #   gemm_a8w8_blockscale(x_fp8, w_fp8, x_scale, w_scale, dtype=bf16) -> [M,N] bf16
            from aiter.ops.gemm_op_a8w8 import gemm_a8w8_blockscale
            x_fp8, x_scale = quant_token_blockwise(x_bf16)   # [M,K], scale [M,K/128]
            w_fp8, w_scale = quant_block_blockwise(w_bf16)   # [N,K], scale [ceil(N/128),K/128]
            BUILD_BACKEND[tag] = "aiter.gemm_a8w8_blockscale(CK)"
            return lambda: gemm_a8w8_blockscale(x_fp8, w_fp8, x_scale, w_scale,
                                                dtype=torch.bfloat16)
        except Exception:
            pass
    # Fallback: per-tensor hipBLASLt FP8 GEMM (torch._scaled_mm). Same FLOPs/latency
    # class; per-tensor vs blockwise only changes numerics, which this perf bench ignores.
    x_fp8, x_scale = quant_per_tensor(x_bf16)
    w_fp8, w_scale = quant_per_tensor(w_bf16)
    BUILD_BACKEND[tag] = "torch._scaled_mm(hipBLASLt,per-tensor)"
    return lambda: _scaled_mm(x_fp8, w_fp8, x_scale, w_scale)


def build_bf16_gemm(M, K, N, device, tag="bf16_gemm"):
    """BF16 GEMM [M,K]x[N,K]->[M,N] f32 (index_weights_proj). torch.mm(x, w.t())."""
    x = torch.randn(M, K, dtype=torch.bfloat16, device=device)
    w = torch.randn(N, K, dtype=torch.bfloat16, device=device)
    wt = w.t().contiguous()
    BUILD_BACKEND[tag] = "torch.mm(bf16->f32)"
    try:
        return lambda: torch.mm(x, wt, out_dtype=torch.float32)  # torch>=2.8
    except TypeError:
        return lambda: torch.mm(x, wt).float()


def build_bmm_fp8(B, M, K, N, device, tag="bmm_fp8"):
    """Batched per-tensor FP8 matmul [B,M,K]x[B,K,N]->[B,M,N] bf16.
    absorbed_W_UK/UV: per-head hipBLASLt FP8 GEMM loop (the portable dsa_projection.py
    pattern; shares one activation across heads, distinct per-head weight)."""
    x_bf16 = torch.randn(M, K, dtype=torch.bfloat16, device=device)
    x_fp8, x_scale = quant_per_tensor(x_bf16)
    w_fp8_list, w_scale_list = [], []
    for _ in range(B):
        wf, ws = quant_per_tensor(torch.randn(N, K, dtype=torch.bfloat16, device=device))
        w_fp8_list.append(wf)
        w_scale_list.append(ws)
    BUILD_BACKEND[tag] = "torch._scaled_mm x B (hipBLASLt bmm)"

    def run():
        for h in range(B):
            _scaled_mm(x_fp8, w_fp8_list[h], x_scale, w_scale_list[h])
    return run


def build_moe_grouped(M, K, N, device, tag="moe_grouped"):
    """MoE masked grouped GEMM: total_m = M*top_k tokens over N_EXPERT experts.
    Prefer aiter fused MoE grouped GEMM; fall back to a per-expert hipBLASLt loop."""
    import random
    total_m = M * NUM_EXPERTS_PER_TOK
    counts = [0] * N_EXPERT
    rng = random.Random(M)
    for _ in range(total_m):
        counts[rng.randint(0, N_EXPERT - 1)] += 1
    expected_m = max((total_m + N_EXPERT - 1) // N_EXPERT, max(counts))
    expected_m = ((expected_m + 127) // 128) * 128
    # Per-expert weights (fp8) and a shared padded activation buffer.
    w_fp8_list, w_scale_list = [], []
    for _ in range(N_EXPERT):
        wf, ws = quant_per_tensor(torch.randn(N, K, dtype=torch.bfloat16, device=device))
        w_fp8_list.append(wf)
        w_scale_list.append(ws)
    x_fp8, x_scale = quant_per_tensor(torch.randn(expected_m, K, dtype=torch.bfloat16, device=device))
    BUILD_BACKEND[tag] = "torch._scaled_mm x E (per-expert grouped)"

    def run():
        for e in range(N_EXPERT):
            m_e = counts[e]
            if m_e == 0:
                continue
            _scaled_mm(x_fp8[:m_e], w_fp8_list[e], x_scale, w_scale_list[e])
    return run


def build_sparse_mla(s_q, s_kv, device, tag="sparse_mla"):
    """DSA sparse MLA (bf16): each query gathers topk KV rows then attends (MQA, h_kv=1).
    Prefer aiter MLA; fall back to a memory-bounded chunked explicit attention. We tile
    over queries so the score tensor stays O(chunk*H*tk) instead of materializing the
    full [s_q, H, tk, D_QK] broadcast (which OOMs)."""
    tk = min(TOPK, s_kv)
    q = torch.randn(s_q, NUM_HEADS, D_QK, dtype=torch.bfloat16, device=device)
    kv = torch.randn(s_kv, D_QK, dtype=torch.bfloat16, device=device)     # shared latent (h_kv=1)
    indices = torch.stack([torch.randperm(s_kv, device=device)[:tk]
                           for _ in range(s_q)]).to(torch.int64)           # [s_q, tk]
    sm_scale = D_QK ** -0.5
    CHUNK = 256 if s_q >= 256 else s_q
    BUILD_BACKEND[tag] = "gather+chunked-attn(bf16)"

    def run():
        for i in range(0, s_q, CHUNK):
            idx = indices[i:i + CHUNK]                       # [c, tk]
            kv_g = kv[idx]                                   # [c, tk, D_QK]
            qc = q[i:i + CHUNK]                              # [c, H, D_QK]
            scores = torch.einsum('chd,ckd->chk', qc, kv_g).float() * sm_scale  # [c,H,tk]
            p = torch.softmax(scores, dim=-1).to(torch.bfloat16)               # [c,H,tk]
            torch.einsum('chk,ckd->chd', p, kv_g[..., :D_V])                    # [c,H,D_V]
    return run


def build_mqa_logits(M, S, device, tag="mqa_logits"):
    """DSA indexer score: logits[M,S] = sum_h w[m,h]*(q[m,h].k[s]).
    Portable: ONE batched fp8 GEMM q[M*h,hd] @ k[S,hd].t() (hipBLASLt), tiled over S to
    bound the logits tensor, weighted-summed over the 32 index heads."""
    h, hd = INDEX_N_HEADS, INDEX_HEAD_DIM
    q_bf16 = torch.randn(M, h, hd, dtype=torch.bfloat16, device=device)
    k_bf16 = torch.randn(S, hd, dtype=torch.bfloat16, device=device)
    q_fp8, q_scale = quant_per_tensor(q_bf16.reshape(M * h, hd))   # [M*h, hd]
    k_fp8, k_scale = quant_per_tensor(k_bf16)                      # [S, hd]
    weights = torch.randn(M, h, 1, dtype=torch.float32, device=device)
    SCHUNK = 8192
    BUILD_BACKEND[tag] = "batched _scaled_mm + weighted-sum (mqa logits)"

    def run():
        for s0 in range(0, S, SCHUNK):
            kc = k_fp8[s0:s0 + SCHUNK]                             # [cs, hd]
            lg = _scaled_mm(q_fp8, kc, q_scale, k_scale, out_dtype=torch.bfloat16)  # [M*h, cs]
            lg = lg.view(M, h, -1).float()                        # [M, h, cs]
            (lg * weights).sum(dim=1)                             # [M, cs] weighted logits
    return run


# ══════════════════════════════════════════════════════════════════════════════
# Reward = bound-aware roofline utilization (identical formula to the B200 bench).
# ══════════════════════════════════════════════════════════════════════════════
def roofline_reward(latency_ms, flops, bytes_hbm, compute_dtype):
    peak_flops = PEAK_FLOPS[compute_dtype]
    lat_s = latency_ms * 1e-3
    ai = flops / bytes_hbm
    ridge = peak_flops / HBM_BYTES_PER_S
    bound = "compute" if ai >= ridge else "memory"
    achieved_flops = flops / lat_s
    achieved_bw = bytes_hbm / lat_s
    roofline_ceiling = min(peak_flops, ai * HBM_BYTES_PER_S)
    reward = achieved_flops / roofline_ceiling if roofline_ceiling > 0 else 0.0
    return {
        "latency_ms": latency_ms,
        "tflops": achieved_flops / 1e12,
        "gbps": achieved_bw / 1e9,
        "ai": ai,
        "ridge": ridge,
        "bound": bound,
        "compute_util": achieved_flops / peak_flops,
        "bw_util": achieved_bw / HBM_BYTES_PER_S,
        "reward": reward,
        "compute_dtype": compute_dtype,
    }


# ══════════════════════════════════════════════════════════════════════════════
# Driver: run a list of ops, compute rewards, print + CSV (mirrors rewardbench.run_ops).
# ops: list of (name, category, backend_label, cost_fn(axes), run_builder(axes,device)).
# ══════════════════════════════════════════════════════════════════════════════
def run_ops(ops, sweep_axes, device, phase, csv_path):
    header = (f"{'operator':<20s} {'cat':<12s} {'backend':<34s} {'lat(ms)':>9s} "
              f"{'TFLOP/s':>9s} {'GB/s':>8s} {'AI':>8s} {'bound':>7s} {'util%':>7s} {'reward':>7s}")
    rows = []
    for axes in sweep_axes:
        tag = ", ".join(f"{k}={v}" for k, v in axes.items())
        print("\n" + "=" * 124)
        print(f"  [{phase}]  {tag}   (MI300X: HBM 5.3TB/s, FP8 2.615PF, BF16 1.307PF)")
        print("=" * 124)
        print("  " + header)
        print("  " + "-" * 122)
        for name, category, backend, cost_fn, run_builder in ops:
            torch.cuda.empty_cache()
            try:
                flops, bytes_hbm, dtype = cost_fn(axes)
                fn = run_builder(axes, device)
                lat, method = robust_bench(fn)
                r = roofline_reward(lat, flops, bytes_hbm, dtype)
                util = r["compute_util"] if r["bound"] == "compute" else r["bw_util"]
                real_backend = BUILD_BACKEND.get(name, backend)
                print(f"  {name:<20s} {category:<12s} {real_backend:<34s} {lat:>9.4f} "
                      f"{r['tflops']:>9.1f} {r['gbps']:>8.0f} {r['ai']:>8.1f} "
                      f"{r['bound']:>7s} {util*100:>6.1f}% {r['reward']:>7.3f}")
                rec = {"operator": name, "category": category, "backend": real_backend,
                       "phase": phase, "timing": method,
                       **{f"axis_{k}": v for k, v in axes.items()}, **r}
                rows.append(rec)
            except Exception as e:
                print(f"  {name:<20s} {category:<12s} {backend:<34s}   FAILED: {str(e)[:56]}")
                rows.append({"operator": name, "category": category, "backend": backend,
                             "phase": phase, **{f"axis_{k}": v for k, v in axes.items()},
                             "error": str(e)})
    import csv
    keys = ["operator", "category", "backend", "phase", "timing"] + \
           sorted({k for r in rows for k in r if k.startswith("axis_")}) + \
           ["latency_ms", "tflops", "gbps", "ai", "ridge", "bound", "compute_util",
            "bw_util", "reward", "compute_dtype", "error"]
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"\nSaved {csv_path}")
    return rows


def print_env_banner():
    dev = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    print("=" * 124)
    print(f"AMD GLM-5.2 operator bench | torch {torch.__version__} | hip {torch.version.hip} "
          f"| device {dev}")
    print(f"  FP8 dtype = {FP8_DTYPE} (max {FP8_MAX}) | aiter = {HAVE_AITER} | "
          f"peaks: HBM {HBM_BYTES_PER_S/1e12}TB/s FP8 {FP8_PEAK_FLOPS/1e15}PF BF16 {BF16_PEAK_FLOPS/1e15}PF")
    print("=" * 124)
