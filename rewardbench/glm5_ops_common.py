"""Shared utilities for the GLM-5 operator reward benchmarks (prefill + decode).

Design (per SGLang official practice rules, https://www.lmsys.org/blog/2026-07-02-agent-assisted-sglang-development#6-practice-rules):
  * Rule 3 "interpret NCU by the kernel's bound": the reward is a **bound-aware
    roofline utilization** — achieved / roofline-ceiling in [0,1]. For a
    compute-bound op this equals tensor-core utilization; for a memory-bound op it
    equals HBM-bandwidth utilization. The op is auto-classified by arithmetic
    intensity (FLOP/byte) vs the B200 ridge point. Maximizing the reward drives a
    kernel to the roofline regardless of which resource binds it.
  * Rule 5 "same ABI/wrapper": every op calls the exact sglang backend kernel
    (deep_gemm / sgl_kernel) with the same dtype, scaling, and layout the model
    uses at runtime — so the number describes the real serving path.
  * Rule 2 "fix the benchmark first": shapes come from GLM-5 config + the assignment
    sheet and are fixed; we never drop shapes after seeing results.

B200 (SM100) alignment note: on Blackwell the DeepGEMM blockwise-FP8 path REQUIRES
UE8M0 (power-of-2) scale factors (DEEPGEMM_SCALE_UE8M0=True in sglang). The
llm_flops reference casts use raw fp32 scales, which assert on this deep_gemm build;
here we use DeepGEMM's canonical `per_token/per_block_cast_to_fp8(use_ue8m0=True)`,
which is exactly what sglang emits at runtime on B200.
"""

import math
import torch
import deep_gemm
from deep_gemm.utils.math import per_token_cast_to_fp8, per_block_cast_to_fp8
from deep_gemm.utils.layout import get_mn_major_tma_aligned_tensor
# ══════════════════════════════════════════════════════════════════════════════
# GLM-5 model config (identical to llm_flops/bench_glm5_*.py)
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
# B200 (SM100) roofline peaks — the reward denominators.
# HBM3e ~8 TB/s; dense tensor-core BF16 ~2.25 PFLOP/s, FP8 ~4.5 PFLOP/s.
# These are the standard NVIDIA B200 dense figures; sglang's harness uses the same
# HBM=8 TB/s / BF16=2250 TFLOP/s constants (kernel-harness/testbench/harness/profile.py).
# ══════════════════════════════════════════════════════════════════════════════
HBM_BYTES_PER_S = 8.0e12          # 8 TB/s
FP8_PEAK_FLOPS = 4.5e15           # 4.5 PFLOP/s dense e4m3 tensor-core
BF16_PEAK_FLOPS = 2.25e15         # 2.25 PFLOP/s dense bf16 tensor-core
PEAK_FLOPS = {"fp8": FP8_PEAK_FLOPS, "bf16": BF16_PEAK_FLOPS}

FP8_B = 1   # bytes per fp8 element
BF16_B = 2
F32_B = 4

NUM_WARMUP = 5
NUM_RUNS = 20


# ══════════════════════════════════════════════════════════════════════════════
# FP8 quantization (B200 UE8M0 blockwise + per-tensor)
# ══════════════════════════════════════════════════════════════════════════════
def quant_token_blockwise(x_bf16):
    """Per-token blockwise (128) FP8 with UE8M0 scales, mn-major TMA layout for x."""
    x_fp8, x_scale = per_token_cast_to_fp8(x_bf16, use_ue8m0=True)
    return x_fp8, get_mn_major_tma_aligned_tensor(x_scale)


def quant_block_blockwise(w_bf16):
    """Per-128x128-block FP8 with UE8M0 scales (weights)."""
    return per_block_cast_to_fp8(w_bf16, use_ue8m0=True)


def quant_per_tensor(x):
    """Per-tensor FP8 (for sgl_kernel.bmm_fp8, which uses cuBLAS per-tensor scale)."""
    amax = x.abs().float().amax()
    scale = (amax / torch.finfo(torch.float8_e4m3fn).max).float().clamp(min=1e-12)
    x_fp8 = (x.float() / scale).to(torch.float8_e4m3fn)
    return x_fp8, scale.view(1).to(x.device)


def _ceil_to_ue8m0(x):
    bits = x.abs().float().view(torch.int)
    exp = ((bits >> 23) & 0xFF) + (bits & 0x7FFFFF).bool().int()
    return (exp.clamp(1, 254) << 23).view(torch.float)


# ══════════════════════════════════════════════════════════════════════════════
# CUDA-graph timing (matches llm_flops: warmup, capture NUM_RUNS replays, 1 timed replay)
# This captures the real dispatch path (launch + kernel), the same methodology the
# llm_flops reference uses, so numbers are comparable to it.
# ══════════════════════════════════════════════════════════════════════════════
def cuda_graph_bench(run_fn):
    torch.cuda.synchronize()
    for _ in range(NUM_WARMUP):
        run_fn()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(NUM_RUNS):
            run_fn()
    torch.cuda.synchronize()
    for _ in range(NUM_WARMUP):
        graph.replay()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    graph.replay()
    end.record()
    torch.cuda.synchronize()
    avg_ms = start.elapsed_time(end) / NUM_RUNS
    del graph
    return avg_ms


def event_bench(run_fn, warmup=NUM_WARMUP, iters=NUM_RUNS):
    """CUDA-event timing WITHOUT graph capture — robust for arbitrary candidate kernels
    (Triton autotune, host-side routing, .item(), dynamic launches) that are not
    graph-capturable. Times the mean over `iters` back-to-back launches, so it includes
    real launch/dispatch overhead (closer to the served call path per KDA-Pilot). run_fn
    must operate on fixed pre-built inputs (no per-iter alloc)."""
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


# ══════════════════════════════════════════════════════════════════════════════
# Analytic FLOP / HBM-byte models (audited per op). Bytes count the dtype actually
# moved to/from HBM (fp8=1, bf16=2, f32=4) including scale factors.
# ══════════════════════════════════════════════════════════════════════════════
def gemm_fp8_cost(M, K, N):
    """Blockwise-FP8 GEMM [M,K]x[N,K]->[M,N] bf16 (fp8 in, bf16 out, fp32 scales)."""
    flops = 2 * M * K * N
    bytes_hbm = (M * K * FP8_B + N * K * FP8_B + M * N * BF16_B
                 + M * (K // 128) * F32_B + math.ceil(N / 128) * (K // 128) * F32_B)
    return flops, bytes_hbm, "fp8"


def gemm_bf16_cost(M, K, N):
    """BF16 GEMM [M,K]x[N,K]->[M,N] f32 (index_weights_proj)."""
    flops = 2 * M * K * N
    bytes_hbm = M * K * BF16_B + N * K * BF16_B + M * N * F32_B
    return flops, bytes_hbm, "bf16"


def bmm_fp8_cost(B, M, K, N):
    """Batched per-tensor FP8 matmul [B,M,K]x[B,K,N]->[B,M,N] bf16 (cuBLAS bmm_fp8)."""
    flops = 2 * B * M * K * N
    bytes_hbm = B * M * K * FP8_B + B * K * N * FP8_B + B * M * N * BF16_B
    return flops, bytes_hbm, "fp8"


def bmm_bf16_cost(B, M, K, N):
    """Batched BF16 matmul [B,M,K]x[B,K,N]->[B,M,N] bf16 (torch.bmm).
    This is the real B200 path for absorbed_W_UK/UV: on Blackwell DEEPGEMM_BLACKWELL=True
    dequantizes kv_b_proj to bf16 (deepseek_weight_loader.py), so w_kc/w_vc are bf16 and
    forward_mla.py runs torch.bmm, not bmm_fp8."""
    flops = 2 * B * M * K * N
    bytes_hbm = B * M * K * BF16_B + B * K * N * BF16_B + B * M * N * BF16_B
    return flops, bytes_hbm, "bf16"


def linear_bf16_cost(M, K, N, out_b=BF16_B):
    """BF16 GEMM [M,K]x[N,K]->[M,N] via torch F.linear / torch.mm (cuBLAS).
    out_b = output element bytes (bf16=2, or f32=4 for weights_proj's f32 output)."""
    flops = 2 * M * K * N
    bytes_hbm = M * K * BF16_B + N * K * BF16_B + M * N * out_b
    return flops, bytes_hbm, "bf16"


def sparse_mla_cost(s_q, s_kv, h_q=NUM_HEADS, d_qk=D_QK, d_v=D_V, topk=TOPK):
    """flash_mla_sparse_fwd (bf16, --kv-cache-dtype bfloat16 path): each query attends to
    topk gathered KV rows. FLOPs = QK^T + PV. KV read deduped (single shared latent cache)."""
    tk = min(topk, s_kv)
    flops = 2 * h_q * s_q * tk * (d_qk + d_v)
    kv_rows = min(tk * s_q, s_kv)
    bytes_hbm = (s_q * h_q * d_qk * BF16_B          # q
                 + kv_rows * d_qk * BF16_B           # gathered KV (deduped)
                 + s_q * h_q * d_v * BF16_B          # out
                 + s_q * tk * F32_B)                 # indices (int32)
    return flops, bytes_hbm, "bf16"


def sparse_mla_trtllm_cost(s_q, s_kv, h_q=NUM_HEADS, d_qk=D_QK, d_v=D_V, topk=TOPK):
    """trtllm-gen sparse MLA (B200 DEFAULT, FP8 KV). No cross-query dedup — StaticTokenSparse
    re-reads each query's topk KV rows (physically realistic: real queries pick different
    sparse tokens). This makes it KV-read (memory) bound at these shapes."""
    num_slots = ((s_kv + BLOCK_SIZE_KV - 1) // BLOCK_SIZE_KV) * BLOCK_SIZE_KV
    tk = min(topk, num_slots)
    flops = 2 * h_q * s_q * tk * (d_qk + d_v)
    bytes_hbm = (s_q * tk * d_qk * FP8_B       # KV read (dominant, no dedup)
                 + s_q * h_q * d_qk * FP8_B     # query fp8
                 + s_q * h_q * d_v * BF16_B     # output bf16
                 + s_q * tk * F32_B)            # block_tables int32
    return flops, bytes_hbm, "fp8"


def mqa_logits_ragged_cost(M, S, h=INDEX_N_HEADS, hd=INDEX_HEAD_DIM):
    """deep_gemm.fp8_mqa_logits (prefill): logits[M,S] = sum_h w[m,h]*(q[m,h].k[s]).
    FLOPs ~ 2*M*S*h*hd. Reads q(fp8) + k(fp8)+scale, writes logits[M,S] f32."""
    flops = 2 * M * S * h * hd
    bytes_hbm = (M * h * hd * FP8_B          # q fp8
                 + S * hd * FP8_B + S * F32_B  # k fp8 + per-row scale
                 + M * h * F32_B               # weights
                 + M * S * F32_B)              # logits out (f32)
    return flops, bytes_hbm, "fp8"


def paged_mqa_logits_cost(M, S, h=INDEX_N_HEADS, hd=INDEX_HEAD_DIM):
    """deep_gemm.fp8_paged_mqa_logits (decode): per-batch paged fp8 KV read dominates.
    total_blocks = ceil(S/64)*M pages of 64x132 bytes."""
    num_blocks_per_seq = (S + BLOCK_SIZE_KV - 1) // BLOCK_SIZE_KV
    total_blocks = num_blocks_per_seq * M
    flops = 2 * M * S * h * hd
    bytes_hbm = (M * h * hd * FP8_B                                   # q fp8
                 + total_blocks * BLOCK_SIZE_KV * HEAD_DIM_WITH_SF     # paged fp8 KV (uint8, incl sf)
                 + M * h * F32_B                                      # weights
                 + M * S * F32_B)                                     # logits out
    return flops, bytes_hbm, "fp8"


def moe_grouped_cost(M, K, N, top_k=NUM_EXPERTS_PER_TOK, n_expert=N_EXPERT):
    """fp8_m_grouped_gemm_nt_masked: total_m = M*top_k tokens over n_expert experts.
    Weights (E*N*K fp8) are streamed once; activations are total_m rows."""
    total_m = M * top_k
    flops = 2 * total_m * K * N
    bytes_hbm = (total_m * K * FP8_B                              # x fp8 (active rows)
                 + n_expert * N * K * FP8_B                       # all expert weights
                 + total_m * N * BF16_B                           # out
                 + total_m * (K // 128) * F32_B                   # x scale
                 + n_expert * math.ceil(N / 128) * (K // 128) * F32_B)  # w scale
    return flops, bytes_hbm, "fp8"


# ══════════════════════════════════════════════════════════════════════════════
# Kernel-call builders — build inputs once, return a no-arg callable (for CUDA graph).
# Each calls the EXACT sglang backend kernel with the runtime dtype/layout (Rule 5).
# ══════════════════════════════════════════════════════════════════════════════
def build_fp8_gemm(M, K, N, device):
    x_fp8, x_scale = quant_token_blockwise(torch.randn(M, K, dtype=torch.bfloat16, device=device))
    w_fp8, w_scale = quant_block_blockwise(torch.randn(N, K, dtype=torch.bfloat16, device=device))
    out = torch.empty(M, N, dtype=torch.bfloat16, device=device)
    return lambda: deep_gemm.fp8_gemm_nt((x_fp8, x_scale), (w_fp8, w_scale), out)


def build_bf16_gemm(M, K, N, device):
    x = torch.randn(M, K, dtype=torch.bfloat16, device=device)
    w = torch.randn(N, K, dtype=torch.bfloat16, device=device)
    out = torch.empty(M, N, dtype=torch.float32, device=device)
    return lambda: deep_gemm.bf16_gemm_nt(x, w, out)


def build_bmm_fp8(B, M, K, N, device):
    from sgl_kernel import bmm_fp8
    A_fp8, A_scale = quant_per_tensor(torch.randn(B, M, K, dtype=torch.bfloat16, device=device))
    B_fp8, B_scale = quant_per_tensor(torch.randn(B, K, N, dtype=torch.bfloat16, device=device))
    A_fp8 = A_fp8.view(B, M, K)
    B_fp8 = B_fp8.view(B, K, N)
    return lambda: bmm_fp8(A_fp8, B_fp8, A_scale, B_scale, torch.bfloat16)


def build_bmm_bf16(B, M, K, N, device):
    """torch.bmm bf16 — the real B200 absorbed_W_UK/UV path (bf16 dequantized weights)."""
    A = torch.randn(B, M, K, dtype=torch.bfloat16, device=device)
    W = torch.randn(B, K, N, dtype=torch.bfloat16, device=device)
    return lambda: torch.bmm(A, W)


def build_linear_bf16(M, K, N, device, out_f32=False):
    """torch F.linear / torch.mm in BF16 (cuBLAS) — the GLM-5.2 indexer wk / weights path
    (fused wk_weights_proj is a single bf16 ReplicatedLinear; here modeled at per-op
    granularity). out_f32=True matches weights_proj's bf16-in/f32-out (torch.mm out_dtype)."""
    x = torch.randn(M, K, dtype=torch.bfloat16, device=device)
    w = torch.randn(N, K, dtype=torch.bfloat16, device=device)
    if out_f32:
        try:
            return lambda: torch.mm(x, w.t(), out_dtype=torch.float32)  # torch>=2.8
        except TypeError:
            wt = w.t().contiguous()
            return lambda: torch.mm(x, wt).float()
    return lambda: torch.nn.functional.linear(x, w)


def build_sparse_mla(s_q, s_kv, device):
    from sgl_kernel.flash_mla import flash_mla_sparse_fwd
    q = torch.randn(s_q, NUM_HEADS, D_QK, dtype=torch.bfloat16, device=device)
    kv = torch.randn(s_kv, 1, D_QK, dtype=torch.bfloat16, device=device)
    tk = min(TOPK, s_kv)
    indices = torch.stack([torch.randperm(s_kv, device=device)[:tk]
                           for _ in range(s_q)]).view(s_q, 1, tk).to(torch.int32)
    sm_scale = D_QK ** -0.5
    return lambda: flash_mla_sparse_fwd(q, kv, indices, sm_scale, D_V)


def build_sparse_mla_trtllm(s_q, s_kv, device):
    """trtllm-gen sparse MLA (B200 default fp8 path). Sparse top-k is passed via
    block_tables of shape [num_seqs, 1, topk] (global physical KV slots); KV pages shared
    across queries to avoid OOM. Mirrors sglang dsa_backend.py::_forward_trtllm."""
    import flashinfer.decode
    device = torch.device(device)
    PAGE = BLOCK_SIZE_KV
    num_pages = (s_kv + PAGE - 1) // PAGE
    num_slots = num_pages * PAGE
    topk = min(TOPK, num_slots)
    FP8 = torch.float8_e4m3fn
    query = (torch.randn(s_q, 1, NUM_HEADS, D_QK, dtype=torch.bfloat16, device=device) * 0.3).to(FP8)
    kv_cache = (torch.randn(num_pages, 1, PAGE, D_QK, dtype=torch.bfloat16, device=device) * 0.3).to(FP8)
    stride = max(1, num_slots // topk)
    base = (torch.arange(topk, dtype=torch.int32, device=device) * stride).clamp_(max=num_slots - 1)
    block_tables = base.view(1, 1, topk).expand(s_q, 1, topk).contiguous()
    seq_lens = torch.full((s_q,), s_kv, dtype=torch.int32, device=device)
    ws = torch.zeros(128 * 1024 * 1024, dtype=torch.int8, device=device)
    return lambda: flashinfer.decode.trtllm_batch_decode_with_kv_cache_mla(
        query=query, kv_cache=kv_cache, workspace_buffer=ws,
        qk_nope_head_dim=128, kv_lora_rank=KV_LORA_RANK, qk_rope_head_dim=QK_ROPE_HEAD_DIM,
        block_tables=block_tables, seq_lens=seq_lens, max_seq_len=s_kv,
        sparse_mla_top_k=topk, bmm1_scale=D_QK ** -0.5, bmm2_scale=1.0, backend="trtllm-gen")


def build_mqa_ragged(M, S, device):
    """prefill index_score: deep_gemm.fp8_mqa_logits (UE8M0 fp8 q/k)."""
    q = torch.randn(M, INDEX_N_HEADS, INDEX_HEAD_DIM, dtype=torch.bfloat16, device=device)
    qv = q.view(M * INDEX_N_HEADS, 1, INDEX_HEAD_DIM)
    qsf = _ceil_to_ue8m0(qv.abs().float().amax(-1).clamp(1e-4) / 448.0)
    q_fp8 = (qv.float() / qsf.unsqueeze(-1)).to(torch.float8_e4m3fn).view(M, INDEX_N_HEADS, INDEX_HEAD_DIM)
    k = torch.randn(S, INDEX_HEAD_DIM, dtype=torch.bfloat16, device=device)
    kv = k.view(S, 1, INDEX_HEAD_DIM)
    ksf = _ceil_to_ue8m0(kv.abs().float().amax(-1).clamp(1e-4) / 448.0)
    k_fp8 = (kv.float() / ksf.unsqueeze(-1)).to(torch.float8_e4m3fn).view(S, INDEX_HEAD_DIM)
    ksf = ksf.view(S)
    weights = torch.randn(M, INDEX_N_HEADS, dtype=torch.float32, device=device)
    ks = torch.zeros(M, dtype=torch.int32, device=device)
    ke = torch.full((M,), S, dtype=torch.int32, device=device)
    return lambda: deep_gemm.fp8_mqa_logits(q_fp8, (k_fp8, ksf), weights, ks, ke, clean_logits=False)


def build_paged_mqa(M, S, device):
    """decode index_score: deep_gemm.fp8_paged_mqa_logits (2D context_lens, finite fp8 KV)."""
    num_blocks_per_seq = (S + BLOCK_SIZE_KV - 1) // BLOCK_SIZE_KV
    total_blocks = num_blocks_per_seq * M
    q = torch.randn(M, 1, INDEX_N_HEADS, INDEX_HEAD_DIM, dtype=torch.bfloat16, device=device)
    qv = q.view(M * INDEX_N_HEADS, 1, INDEX_HEAD_DIM)
    qsf = _ceil_to_ue8m0(qv.abs().float().amax(-1).clamp(1e-4) / 448.0)
    q_fp8 = (qv.float() / qsf.unsqueeze(-1)).to(torch.float8_e4m3fn).view(M, 1, INDEX_N_HEADS, INDEX_HEAD_DIM)
    kvb = torch.randn(total_blocks, BLOCK_SIZE_KV, 1, INDEX_HEAD_DIM, dtype=torch.bfloat16, device=device)
    flat = kvb.view(-1, INDEX_HEAD_DIM)
    ksf = _ceil_to_ue8m0(flat.abs().float().amax(-1).clamp(1e-4) / 448.0)
    kfp8 = (flat.float() / ksf.unsqueeze(-1)).to(torch.float8_e4m3fn).view(total_blocks, BLOCK_SIZE_KV, 1, INDEX_HEAD_DIM)
    kv_cache = torch.empty(total_blocks, BLOCK_SIZE_KV, 1, HEAD_DIM_WITH_SF, dtype=torch.uint8, device=device)
    kv_cache[..., :INDEX_HEAD_DIM] = kfp8.view(torch.uint8)
    kv_cache[..., INDEX_HEAD_DIM:] = ksf.view(total_blocks, BLOCK_SIZE_KV, 1, 1).view(torch.uint8)
    weights = torch.randn(M, INDEX_N_HEADS, dtype=torch.float32, device=device)
    seqlens = torch.full((M, 1), S, dtype=torch.int32, device=device)
    block_tables = torch.arange(total_blocks, dtype=torch.int32, device=device).view(M, num_blocks_per_seq)
    max_seq_len = num_blocks_per_seq * BLOCK_SIZE_KV
    sched = deep_gemm.get_paged_mqa_logits_metadata(seqlens, BLOCK_SIZE_KV, deep_gemm.get_num_sms())
    return lambda: deep_gemm.fp8_paged_mqa_logits(
        q_fp8, kv_cache, weights, seqlens, block_tables, sched, max_seq_len, clean_logits=False)


def build_moe_grouped(M, K, N, device):
    import random
    total_m = M * NUM_EXPERTS_PER_TOK
    counts = [0] * N_EXPERT
    rng = random.Random(M)
    for _ in range(total_m):
        counts[rng.randint(0, N_EXPERT - 1)] += 1
    # Capacity must hold the largest per-expert bin (deep_gemm's masked kernel processes
    # ceil(masked_m/BLOCK_M) M-blocks per expert and indexes into [E, expected_m, .]);
    # sizing to ceil(total_m/E) alone can overflow when the random multinomial is skewed.
    expected_m = max((total_m + N_EXPERT - 1) // N_EXPERT, max(counts))
    expected_m = ((expected_m + 127) // 128) * 128
    x_bf16 = torch.randn(N_EXPERT, expected_m, K, dtype=torch.bfloat16, device=device)
    x_fp8 = torch.empty_like(x_bf16, dtype=torch.float8_e4m3fn)
    x_scale = torch.empty(N_EXPERT, expected_m, K // 128, dtype=torch.float32, device=device)
    for i in range(N_EXPERT):
        x_fp8[i], x_scale[i] = per_token_cast_to_fp8(x_bf16[i], use_ue8m0=True)
    w_bf16 = torch.randn(N_EXPERT, N, K, dtype=torch.bfloat16, device=device)
    n_ceil = (N + 127) // 128 * 128
    w_fp8 = torch.empty(N_EXPERT, N, K, dtype=torch.float8_e4m3fn, device=device)
    w_scale = torch.empty(N_EXPERT, n_ceil // 128, K // 128, dtype=torch.float32, device=device)
    for i in range(N_EXPERT):
        w_fp8[i], w_scale[i] = per_block_cast_to_fp8(w_bf16[i], use_ue8m0=True)
    out = torch.empty(N_EXPERT, expected_m, N, dtype=torch.bfloat16, device=device)
    masked_m = torch.tensor(counts, dtype=torch.int32, device=device)
    return lambda: deep_gemm.fp8_m_grouped_gemm_nt_masked(
        (x_fp8, x_scale), (w_fp8, w_scale), out, masked_m, expected_m)


# ══════════════════════════════════════════════════════════════════════════════
# Reward = bound-aware roofline utilization
# ══════════════════════════════════════════════════════════════════════════════
def roofline_reward(latency_ms, flops, bytes_hbm, compute_dtype):
    """Return the reward record. reward in [0,1] = achieved_flops / roofline_ceiling.
    - ridge = peak_flops/peak_bw; AI = flops/bytes.
    - bound = 'compute' if AI>=ridge else 'memory'.
    - roofline_ceiling = min(peak_flops, AI*peak_bw)   (Williams et al. roofline).
    - reward = achieved_flops / roofline_ceiling  == compute-util (compute-bound)
               or bandwidth-util (memory-bound).
    """
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
# Driver: run a list of ops, compute rewards, print + CSV
# ══════════════════════════════════════════════════════════════════════════════
def run_ops(ops, sweep_axes, device, phase, csv_path):
    """ops: list of (name, category, backend, cost_fn, run_builder).
       cost_fn(axes) -> (flops, bytes, dtype). run_builder(axes, device) -> callable.
       sweep_axes: list of dict axis assignments (e.g. [{'M':1024,'S':65536}, ...]).
    """
    header = (f"{'operator':<20s} {'cat':<12s} {'backend':<30s} {'lat(ms)':>9s} "
              f"{'TFLOP/s':>9s} {'GB/s':>8s} {'AI':>8s} {'bound':>7s} {'util%':>7s} {'reward':>7s}")
    rows = []
    for axes in sweep_axes:
        tag = ", ".join(f"{k}={v}" for k, v in axes.items())
        print("\n" + "=" * 120)
        print(f"  [{phase}]  {tag}")
        print("=" * 120)
        print("  " + header)
        print("  " + "-" * 118)
        for name, category, backend, cost_fn, run_builder in ops:
            torch.cuda.empty_cache()
            try:
                flops, bytes_hbm, dtype = cost_fn(axes)
                fn = run_builder(axes, device)
                lat = cuda_graph_bench(fn)
                r = roofline_reward(lat, flops, bytes_hbm, dtype)
                util = r["compute_util"] if r["bound"] == "compute" else r["bw_util"]
                print(f"  {name:<20s} {category:<12s} {backend:<30s} {lat:>9.4f} "
                      f"{r['tflops']:>9.1f} {r['gbps']:>8.0f} {r['ai']:>8.1f} "
                      f"{r['bound']:>7s} {util*100:>6.1f}% {r['reward']:>7.3f}")
                rec = {"operator": name, "category": category, "backend": backend,
                       "phase": phase, **{f"axis_{k}": v for k, v in axes.items()}, **r}
                rows.append(rec)
            except Exception as e:
                print(f"  {name:<20s} {category:<12s} {backend:<30s}   FAILED: {str(e)[:60]}")
                rows.append({"operator": name, "category": category, "backend": backend,
                             "phase": phase, **{f"axis_{k}": v for k, v in axes.items()},
                             "error": str(e)})
    # CSV
    import csv
    keys = ["operator", "category", "backend", "phase"] + \
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


# ══════════════════════════════════════════════════════════════════════════════
# Candidate reward-bench engine — test a FOLDER of optimized operators -> big CSV.
# Shared by bench_GLM5_ops_prefill.py / _decode.py (phase-filtered). The 13-op class
# is the target; a candidate folder may be INCOMPLETE (only some ops provided).
# ══════════════════════════════════════════════════════════════════════════════
import csv as _csv
import datetime as _dt
import inspect as _inspect
import json as _json
import re as _re
from pathlib import Path as _Path

# per reward operator -> default GEMM dims (used only when task.json omits them)
OP_DEFAULTS = {
    "o_proj": dict(K=16384, N=6144), "q_b_proj": dict(K=2048, N=16384),
    "fused_qkv_a_proj": dict(K=6144, N=2624), "index_k_proj": dict(K=6144, N=128),
    "index_q_upproj": dict(K=2048, N=4096), "index_weights_proj": dict(K=6144, N=32),
    "moe_gate_proj": dict(K=6144, N=2048, E=8), "moe_up_proj": dict(K=6144, N=2048, E=8),
    "moe_down_proj": dict(K=2048, N=6144, E=8),
    "dsa_prefill_attn": dict(S=65536), "dsa_decode_attn": dict(S=65536),
    "index_score": dict(S=65536),
}
_ALL_OPS = ("fused_qkv_a_proj", "q_b_proj", "o_proj", "absorbed_W_UK", "absorbed_W_UV",
            "dsa_prefill_attn", "dsa_decode_attn", "index_k_proj", "index_q_upproj",
            "index_weights_proj", "index_score", "moe_gate_proj", "moe_up_proj", "moe_down_proj")
_PREFILL_SWEEP = [1024, 2048, 4096]
_DECODE_SWEEP = [16, 32]


def load_solution(sol_path):
    """exec a candidate solution.py in a fresh namespace (like the harness driver)."""
    sol_path = _Path(sol_path)
    ns = {"__name__": f"sol_{sol_path.parent.name}", "__file__": str(sol_path)}
    exec(compile(sol_path.read_text(), str(sol_path), "exec"), ns)
    return ns


def read_task(folder):
    tj = _Path(folder) / "task.json"
    if tj.exists() and tj.stat().st_size > 0:
        try:
            return _json.loads(tj.read_text())
        except Exception:
            pass
    return {}


def reward_operator(folder, task):
    """Map a candidate folder -> its reward-bench operator (META.md > task.json > name)."""
    folder = _Path(folder)
    meta = folder / "META.md"
    if meta.exists():
        m = _re.search(r"reward operator:\s*`?([A-Za-z0-9_]+)`?", meta.read_text())
        if m:
            return m.group(1)
    if task.get("llm_flops_op"):
        return task["llm_flops_op"]
    for op in _ALL_OPS:
        if op.lower() in folder.name.lower():
            return op
    return folder.name


def candidate_phase(folder, task):
    return task.get("phase") or ("decode" if "decode" in _Path(folder).name.lower() else "prefill")


def detect_family(task, op, run_params):
    fam = (task.get("family") or "").lower()
    rp = set(run_params)
    if "grouped" in fam or "moe" in fam or rp >= {"a_fp8", "b_fp8", "masked_m"}:
        return "grouped-moe"
    if "sparse-mla-decode" in fam or "seq_lens" in rp or "block_tables" in rp:
        return "sparse-mla-decode"
    if "prefill-attn" in fam or "dsa-prefill" in fam or rp >= {"q", "kv_cache", "indices"}:
        return "dsa-prefill-attn"
    if "bf16" in fam and "linear" in fam:
        return "bf16-linear"
    if rp >= {"x_fp8", "w_fp8"}:
        return "fp8-linear"
    if op in ("moe_gate_proj", "moe_up_proj", "moe_down_proj"):
        return "grouped-moe"
    if op == "dsa_decode_attn":
        return "sparse-mla-decode"
    if op == "dsa_prefill_attn":
        return "dsa-prefill-attn"
    if op == "index_weights_proj":
        return "bf16-linear"
    return "fp8-linear"


def _clone_inputs(d):
    return {k: (v.clone() if torch.is_tensor(v) else v) for k, v in d.items()}


def call_by_name(fn, inputs):
    """Bind fn's parameters positionally from the inputs dict by name."""
    params = list(_inspect.signature(fn).parameters)
    return fn(*[inputs[p] for p in params])


def canonical_get_inputs(family):
    """Bench reference get_inputs when a candidate bundles none & no sibling provides one."""
    if family == "fp8-linear":
        def gi(axes, device):
            M, K, N = axes["M"], axes["K"], axes["N"]
            xf, xs = quant_token_blockwise(torch.randn(M, K, dtype=torch.bfloat16, device=device))
            wf, ws = quant_block_blockwise(torch.randn(N, K, dtype=torch.bfloat16, device=device) * (K ** -0.5))
            return {"x_fp8": xf, "x_scale": xs, "w_fp8": wf, "w_scale": ws}
        return gi
    if family == "bf16-linear":
        def gi(axes, device):
            M, K, N = axes["M"], axes["K"], axes["N"]
            return {"x": torch.randn(M, K, dtype=torch.bfloat16, device=device),
                    "w": torch.randn(N, K, dtype=torch.bfloat16, device=device),
                    "out": torch.empty(M, N, dtype=torch.float32, device=device)}
        return gi
    if family == "dsa-prefill-attn":
        def gi(axes, device):
            s_q, s_kv = axes["M"], axes.get("S", 65536)
            tk = min(TOPK, s_kv)
            return {"q": torch.randn(s_q, NUM_HEADS, D_QK, dtype=torch.bfloat16, device=device),
                    "kv_cache": torch.randn(s_kv, 1, D_QK, dtype=torch.bfloat16, device=device),
                    "indices": torch.stack([torch.randperm(s_kv, device=device)[:tk]
                                            for _ in range(s_q)]).view(s_q, 1, tk).to(torch.int32)}
        return gi
    if family == "grouped-moe":
        def gi(axes, device):
            import random
            E, K, N, M = axes.get("E", N_EXPERT), axes["K"], axes["N"], axes["M"]
            total_m = M * NUM_EXPERTS_PER_TOK
            counts = [0] * E
            rng = random.Random(M)
            for _ in range(total_m):
                counts[rng.randint(0, E - 1)] += 1
            Mp = ((max(max(counts), 1) + 127) // 128) * 128
            a = torch.randn(E, Mp, K, dtype=torch.bfloat16, device=device)
            b = torch.randn(E, N, K, dtype=torch.bfloat16, device=device) * (K ** -0.5)
            af, as_ = zip(*[per_token_cast_to_fp8(a[e], use_ue8m0=True) for e in range(E)])
            bf, bs = zip(*[per_block_cast_to_fp8(b[e], use_ue8m0=True) for e in range(E)])
            return {"a_fp8": torch.stack(list(af)), "a_s": torch.stack(list(as_)),
                    "b_fp8": torch.stack(list(bf)), "b_s": torch.stack(list(bs)),
                    "out": torch.empty(E, Mp, N, dtype=torch.bfloat16, device=device),
                    "masked_m": torch.tensor(counts, dtype=torch.int32, device=device),
                    "expected_m": torch.tensor(max(max(counts), 1), dtype=torch.int32, device=device),
                    "m_indices": torch.empty(0, dtype=torch.int32, device=device),
                    "layout": torch.tensor(1, dtype=torch.int32, device=device)}
        return gi
    if family == "sparse-mla-decode":
        raise RuntimeError("sparse-mla-decode needs the candidate's own get_inputs")
    raise ValueError(f"no canonical get_inputs for {family}")


def baseline_run(family, inputs):
    """Live sglang/deep_gemm/flashinfer reference kernel for the speedup denominator."""
    if family == "fp8-linear":
        x, xs, w, ws = inputs["x_fp8"], inputs["x_scale"], inputs["w_fp8"], inputs["w_scale"]
        out = torch.empty(x.shape[0], w.shape[0], dtype=torch.bfloat16, device=x.device)
        deep_gemm.fp8_gemm_nt((x, xs), (w, ws), out); return out
    if family == "bf16-linear":
        deep_gemm.bf16_gemm_nt(inputs["x"], inputs["w"], inputs["out"]); return inputs["out"]
    if family == "grouped-moe":
        layout = int(inputs["layout"]) if torch.is_tensor(inputs["layout"]) else int(inputs["layout"])
        a, as_, b, bs, out = inputs["a_fp8"], inputs["a_s"], inputs["b_fp8"], inputs["b_s"], inputs["out"]
        if layout == 1:
            em = int(inputs["expected_m"]) if torch.is_tensor(inputs["expected_m"]) else int(inputs["expected_m"])
            deep_gemm.fp8_m_grouped_gemm_nt_masked((a, as_), (b, bs), out, inputs["masked_m"], em)
        else:
            deep_gemm.m_grouped_fp8_gemm_nt_contiguous((a, as_), (b, bs), out, inputs["m_indices"])
        return out
    if family == "dsa-prefill-attn":
        from sgl_kernel.flash_mla import flash_mla_sparse_fwd
        q, kv, idx = inputs["q"], inputs["kv_cache"], inputs["indices"]
        if kv.dim() == 2:
            kv = kv.view(kv.shape[0], 1, kv.shape[1])
        if idx.dim() == 2:
            idx = idx.view(idx.shape[0], 1, idx.shape[1])
        o, _, _ = flash_mla_sparse_fwd(q, kv, idx, D_QK ** -0.5, D_V); return o
    if family == "sparse-mla-decode":
        import flashinfer.decode
        q, kv, bt, sl = inputs["q"], inputs["kv_cache"], inputs["block_tables"], inputs["seq_lens"]
        return flashinfer.decode.trtllm_batch_decode_with_kv_cache_mla(
            query=q, kv_cache=kv, workspace_buffer=inputs["workspace"],
            qk_nope_head_dim=128, kv_lora_rank=KV_LORA_RANK, qk_rope_head_dim=QK_ROPE_HEAD_DIM,
            block_tables=bt, seq_lens=sl, max_seq_len=int(inputs["max_seq_len"]),
            sparse_mla_top_k=bt.shape[-1], bmm1_scale=float(inputs["bmm1_scale"]), bmm2_scale=1.0,
            backend="trtllm-gen")
    raise ValueError(family)


def cost_from_inputs(family, inputs):
    """Analytic (flops, hbm_bytes, compute_dtype) derived from the ACTUAL built tensors."""
    if family in ("fp8-linear", "bf16-linear"):
        x = inputs.get("x_fp8", inputs.get("x")); w = inputs.get("w_fp8", inputs.get("w"))
        M, K, N = x.shape[-2], x.shape[-1], w.shape[-2]
        return gemm_fp8_cost(M, K, N) if family == "fp8-linear" else gemm_bf16_cost(M, K, N)
    if family == "grouped-moe":
        a, b = inputs["a_fp8"], inputs["b_fp8"]
        E, K, N = a.shape[0], a.shape[-1], b.shape[-2]
        mm = inputs.get("masked_m")
        total = int(mm.sum()) if (torch.is_tensor(mm) and mm.numel()) else a.shape[0] * a.shape[1]
        flops = 2 * total * K * N
        bytes_hbm = (total * K * FP8_B + E * N * K * FP8_B + total * N * BF16_B
                     + total * (K // 128) * F32_B + E * math.ceil(N / 128) * (K // 128) * F32_B)
        return flops, bytes_hbm, "fp8"
    if family == "dsa-prefill-attn":
        q, idx = inputs["q"], inputs["indices"]
        return sparse_mla_cost(q.shape[0], inputs["kv_cache"].shape[0], h_q=q.shape[1],
                               d_qk=q.shape[2], d_v=D_V, topk=idx.shape[-1])
    if family == "sparse-mla-decode":
        q, bt, sl = inputs["q"], inputs["block_tables"], inputs["seq_lens"]
        return sparse_mla_trtllm_cost(q.shape[0], int(sl.max()), h_q=q.shape[-2],
                                      d_qk=q.shape[-1], d_v=D_V, topk=bt.shape[-1])
    raise ValueError(family)


def _axes_list(task, op, phase):
    d = OP_DEFAULTS.get(op, {})
    sweep = task.get("sweep") or (_PREFILL_SWEEP if phase == "prefill" else _DECODE_SWEEP)
    base = {k: task.get(k, d.get(k)) for k in ("K", "N", "E", "S") if (k in task or k in d)}
    ctxs = task.get("ctx_sweep")
    out = []
    for m in sweep:
        a = dict(base); a["M"] = m; a.setdefault("S", base.get("S", 65536))
        if ctxs:
            for c in ctxs:
                aa = dict(a); aa["ctx"] = c; aa["S"] = c; out.append(aa)
        else:
            out.append(a)
    return out


_CANDIDATE_COLS = ["ts", "round", "candidate", "task", "operator", "phase", "family",
                   "M", "K", "N", "S", "sol_us", "base_us", "speedup", "achieved_tflops",
                   "achieved_gbps", "arithmetic_intensity", "bound", "pct_fp_peak",
                   "util_pct", "reward"]


def _write_candidate_csv(path, rows, append=False):
    """Write rows (list of dicts) to `path`. append=True keeps a timestamped history
    (header written only when the file is new/empty)."""
    path = _Path(path)
    need_header = (not append) or (not path.exists()) or path.stat().st_size == 0
    with open(path, "a" if append else "w", newline="") as fh:
        w = _csv.DictWriter(fh, fieldnames=_CANDIDATE_COLS, extrasaction="ignore")
        if need_header:
            w.writeheader()
        for r in rows:
            w.writerow(r)


def run_candidate_folder(kernels_dir, phase, device, out_csv, repeat=1, no_baseline=False, rnd=0):
    """Benchmark optimized operator candidates and report the bound-aware roofline reward.

    `kernels_dir` may be EITHER:
      * a parent folder holding many <candidate>/ subdirs (aggregate mode), OR
      * a single candidate folder that directly contains solution.py (single-op mode) —
        the common case in a flow where one operator is tested at a time.

    Each candidate's rows are printed to the terminal AND appended (with a UTC timestamp)
    to a per-operator CSV inside that candidate's own directory (`reward_bench.csv`), in
    addition to the aggregate `out_csv`. Returns all rows.
    The 13-op class is the target; ops without a provided candidate are simply absent."""
    root = _Path(kernels_dir)
    if not (root / "README.md").exists() and (root / "best-kernels-reward-bench").exists():
        root = root / "best-kernels-reward-bench"
    if (root / "solution.py").exists():
        folders = [root]                                   # single candidate folder
    else:
        folders = sorted([p for p in root.iterdir() if p.is_dir() and (p / "solution.py").exists()])
    single = len(folders) == 1

    loaded, fam_gi = {}, {}
    for f in folders:
        try:
            ns = load_solution(f / "solution.py"); loaded[f.name] = ns
            task = read_task(f); op = reward_operator(f, task)
            fam = detect_family(task, op, list(_inspect.signature(ns["run"]).parameters))
            if "get_inputs" in ns and fam not in fam_gi:
                fam_gi[fam] = ns["get_inputs"]
        except Exception as e:
            print(f"  ! load {f.name}: {str(e)[:80]}")

    TS = _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    print(f"  timestamp (UTC): {TS}\n")
    rows = []
    n_cand = 0
    for f in folders:
        ns = loaded.get(f.name)
        if ns is None:
            continue
        task = read_task(f); op = reward_operator(f, task)
        cand_phase = candidate_phase(f, task)
        if cand_phase != phase:
            if single:
                print(f"  ! {f.name}: candidate phase is '{cand_phase}', not '{phase}'. "
                      f"Run bench_GLM5_ops_{cand_phase}.py instead.")
            continue
        n_cand += 1
        run_fn = ns["run"]
        fam = detect_family(task, op, list(_inspect.signature(run_fn).parameters))
        gi = ns.get("get_inputs") or fam_gi.get(fam)
        if gi is None:
            try:
                gi = canonical_get_inputs(fam)
            except Exception as e:
                print(f"  ! {f.name}: no get_inputs for {fam}: {str(e)[:60]}"); continue
        cand_rows = []
        for axes in _axes_list(task, op, phase):
            tag = f"{op}/{phase} M={axes.get('M')} S={axes.get('S')}"
            try:
                inputs0 = gi(dict(axes), device)
            except Exception as e:
                print(f"  ! {f.name} {tag}: get_inputs: {str(e)[:70]}"); continue
            try:
                flops, bytes_hbm, dtype = cost_from_inputs(fam, inputs0)
            except Exception:
                flops = bytes_hbm = None; dtype = "fp8"
            # PERF ONLY: this module benchmarks a given optimized kernel. Correctness is a
            # separate upstream gate we do not implement here — no allclose / no `correct`.
            sol_us = base_us = None
            try:
                if not no_baseline:
                    try:
                        bi = _clone_inputs(inputs0); baseline_run(fam, bi)
                        torch.cuda.synchronize()
                        base_us = event_bench(lambda: baseline_run(fam, bi)) * 1e3
                    except Exception:
                        base_us = None
                sol_us = min(event_bench(lambda: call_by_name(run_fn, inputs0))
                             for _ in range(max(1, repeat))) * 1e3
            except Exception as e:
                print(f"  ! {f.name} {tag}: run: {str(e)[:90]}")
            rec = {"ts": TS, "round": rnd, "candidate": f.name, "task": f"glm52/{op}_{phase}",
                   "operator": op, "phase": phase, "family": fam, "M": axes.get("M"),
                   "K": axes.get("K"), "N": axes.get("N"), "S": axes.get("S"),
                   "sol_us": round(sol_us, 3) if sol_us else None,
                   "base_us": round(base_us, 3) if base_us else None,
                   "speedup": round(base_us / sol_us, 4) if (base_us and sol_us) else None}
            if flops and sol_us:
                r = roofline_reward(sol_us / 1e3, flops, bytes_hbm, dtype)
                util = r["compute_util"] if r["bound"] == "compute" else r["bw_util"]
                rec.update({"achieved_tflops": round(r["tflops"], 2), "achieved_gbps": round(r["gbps"], 1),
                            "arithmetic_intensity": round(r["ai"], 1), "bound": r["bound"],
                            "pct_fp_peak": round(r["compute_util"] * 100, 2),
                            "util_pct": round(util * 100, 2), "reward": round(r["reward"], 4)})
            cand_rows.append(rec)
            print(f"  {TS}  {f.name:<28s} {tag:<30s} sol={rec['sol_us']} base={rec['base_us']} "
                  f"sp={rec['speedup']} reward(roofline util ratio)={rec.get('reward')} ({rec.get('bound')})")
        # per-operator CSV inside the candidate's own directory (timestamped history)
        if cand_rows:
            op_csv = f / "reward_bench.csv"
            _write_candidate_csv(op_csv, cand_rows, append=True)
            print(f"    -> {op_csv}")
        rows.extend(cand_rows)

    _write_candidate_csv(out_csv, rows, append=False)
    print(f"\n[{phase}] {n_cand} matching candidate(s), {len(rows)} rows -> {out_csv}")
    return rows
