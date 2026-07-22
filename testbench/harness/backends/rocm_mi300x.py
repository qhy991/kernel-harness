"""AMD Instinct MI300X (CDNA3 / gfx942, ROCm) backend — the A-card analog of cuda_b200.

Self-contained: depends only on torch (ROCm build) + triton. This node has **no**
aiter / deep_gemm / sgl_kernel, and sglang's own fp8 kernels can't be imported here
(an unrelated transformers registration conflict breaks the import chain), so the
provider brings its own references:

  * gemm / moe  -> a triton **blockwise-fp8** matmul with the SAME blockwise scale
                   semantics deep_gemm uses on B200 (per-token 1x128 A scales,
                   per-128x128 B scales), so the frozen inputs glm52_ops builds feed
                   it unchanged. A torch dequant-matmul is kept as a proven-correct
                   fallback and as the independent oracle the mark-validator checks
                   this kernel against.
  * mla         -> chunked gather + explicit sparse attention (bf16), memory-bounded.
  * bmm / score -> torch (per-tensor dequant bmm; folded-weight MQA logits).

FP8 on gfx942 is ``float8_e4m3fnuz`` (scale by 224.0, aiter's safe max — finfo reports
240 but 224 keeps a margin below the fnuz NaN-adjacent code). Timing is HIP-event
cold-L2 median: ROCm has no CUPTI device-span path, so the harness's CUDA-event timer
(which is HIP events on a ROCm torch) is the primitive, wrapped as a rocm-platform Timer.

Roofline peaks (reward denominator): HBM 5.3 TB/s, FP8 2.6149 PFLOP/s, BF16 1.3074
PFLOP/s — the MI300X datasheet figures, identical to the AMD4GLM52 rewardbench so the
two marks share one cost model.
"""
from __future__ import annotations

import importlib.metadata
from functools import lru_cache

import torch

from .base import DeviceProfile
from ._load import load_harness_module

# ── device profile ────────────────────────────────────────────────────────────
PROFILE = DeviceProfile(
    id="rocm-mi300x",
    platform="rocm",
    accelerator="AMD Instinct MI300X",
    deployment="MI300X DP1/TP1/EP32",
    fp8_dtype_name="float8_e4m3fnuz",
    peaks={
        "hbm_bytes_per_s": 5.3e12,
        "fp8": 2.6149e15,
        "bf16": 1.3074e15,
    },
    peaks_source=(
        "AMD Instinct MI300X datasheet (HBM3 5.3 TB/s; matrix-core FP8 e4m3 2614.9 "
        "TFLOP/s, BF16 1307.4 TFLOP/s); identical to AMD4GLM52 rewardbench peaks"
    ),
)

FP8_DTYPE = torch.float8_e4m3fnuz
FP8_MAX = 224.0          # aiter/sglang-ROCm safe max for e4m3fnuz
_BLK = 128               # blockwise scale granularity (matches deep_gemm)


BASELINE_CAVEAT = (
    "MI300X reference is a self-contained torch+triton blockwise-fp8 path, NOT aiter's "
    "production gemm_a8w8_blockscale / fused_moe (aiter is not installed on this node). "
    "The triton blockwise-fp8 matmul carries the same per-token/per-block scale semantics "
    "deep_gemm uses on B200 and is validated bit-close to a bf16 dequant oracle by "
    "AMD4GLM52/validate_marks.py, so it is a legitimate correctness oracle AND a real "
    "MFMA-backed latency denominator — but it is not the last word in MI300X GEMM "
    "throughput. The aiter ASM path (gemm_a8w8_blockscale_bpreshuffle_asm) is ~2.6x "
    "faster at M>=4096; a candidate that merely routes to it is a real win here."
)

ACCEPTED_CANDIDATE_FORMS = (
    "Python / PyTorch — a .py defining run(inputs)",
    "Triton-ROCm — @triton.jit / @triton.autotune in that same .py (gfx942 backend); "
    "nothing special needed",
    "HIP C++ / CK — pass a directory holding candidate.py + the .hip/.cpp, and let "
    "candidate.py torch.utils.cpp_extension.load() it (hipcc --offload-arch=gfx942) at "
    "import time so compilation stays outside the timed window. A bare .hip cannot be "
    "passed: nothing in it says which kernel to launch or how the inputs dict maps to "
    "its arguments. run(inputs) is that missing statement, and it is the whole ABI.",
)


# ══════════════════════════════════════════════════════════════════════════════
# Blockwise-fp8 quantization (module-level so the provider and the mark-validator
# share one definition). Shapes match what glm52_ops.build_inputs/cost expect:
#   per_token_cast(x[R,K]) -> x_fp8[R,K], scale[R, K//128]   (1x128 blocks)
#   per_block_cast(w[N,K]) -> w_fp8[N,K], scale[ceil(N/128), K//128]  (128x128 blocks)
# ``use_ue8m0`` is ignored: CDNA3 has no TMA/UE8M0 path, scales stay plain f32.
# ══════════════════════════════════════════════════════════════════════════════
def per_token_cast(x: torch.Tensor, *, use_ue8m0: bool = False):
    del use_ue8m0
    R, K = x.shape
    assert K % _BLK == 0, f"K={K} not a multiple of {_BLK}"
    xv = x.float().view(R, K // _BLK, _BLK)
    scale = (xv.abs().amax(dim=-1) / FP8_MAX).clamp(min=1e-12)      # [R, K//128]
    x_fp8 = (xv / scale.unsqueeze(-1)).clamp(-FP8_MAX, FP8_MAX).to(FP8_DTYPE).view(R, K)
    return x_fp8, scale.contiguous()


def per_block_cast(w: torch.Tensor, *, use_ue8m0: bool = False):
    del use_ue8m0
    N, K = w.shape
    assert K % _BLK == 0, f"K={K} not a multiple of {_BLK}"
    NB = (N + _BLK - 1) // _BLK
    Npad = NB * _BLK
    wpad = w
    if Npad != N:
        wpad = torch.zeros(Npad, K, dtype=w.dtype, device=w.device)
        wpad[:N] = w
    wv = wpad.float().view(NB, _BLK, K // _BLK, _BLK)
    scale = (wv.abs().amax(dim=(1, 3)) / FP8_MAX).clamp(min=1e-12)  # [NB, K//128]
    w_fp8 = (wv / scale[:, None, :, None]).clamp(-FP8_MAX, FP8_MAX).to(FP8_DTYPE)
    w_fp8 = w_fp8.view(Npad, K)[:N].contiguous()
    return w_fp8, scale.contiguous()


# ══════════════════════════════════════════════════════════════════════════════
# Triton blockwise-fp8 GEMM: C[M,N] = A[M,K] @ B[N,K].T, blockwise dequant.
#   As: [M, K//128] per-token ; Bs: [ceil(N/128), K//128] per-128x128-block.
# BLOCK_K is pinned to 128 = the scale block, so each block's scale factors out of
# the inner sum exactly (sum_k (a*as)(b*bs) = as*bs*sum_k a*b).
# ══════════════════════════════════════════════════════════════════════════════
import triton  # noqa: E402
import triton.language as tl  # noqa: E402


@triton.jit
def _bw_fp8_gemm_kernel(
    A, As, B, Bs, C,
    M, N, K,
    stride_am, stride_ak,
    stride_bn, stride_bk,
    stride_cm, stride_cn,
    stride_asm, stride_ask,
    stride_bsn, stride_bsk,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, 128)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    nkb = K // 128
    for kb in range(nkb):
        k = kb * 128 + offs_k
        a = tl.load(A + offs_m[:, None] * stride_am + k[None, :] * stride_ak,
                    mask=offs_m[:, None] < M, other=0.0).to(tl.bfloat16)
        b = tl.load(B + offs_n[:, None] * stride_bn + k[None, :] * stride_bk,
                    mask=offs_n[:, None] < N, other=0.0).to(tl.bfloat16)
        a_s = tl.load(As + offs_m * stride_asm + kb * stride_ask,
                      mask=offs_m < M, other=0.0)
        b_s = tl.load(Bs + (offs_n // 128) * stride_bsn + kb * stride_bsk,
                      mask=offs_n < N, other=0.0)
        acc += tl.dot(a, tl.trans(b)) * a_s[:, None] * b_s[None, :]
    c = acc.to(tl.bfloat16)
    tl.store(C + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn, c,
             mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


def _blockwise_fp8_gemm(x_fp8, x_scale, w_fp8, w_scale, out=None):
    """C[M,N] = x[M,K] @ w[N,K].T with blockwise fp8 dequant. triton, MFMA-backed."""
    M, K = x_fp8.shape
    N = w_fp8.shape[0]
    if out is None:
        out = torch.empty(M, N, dtype=torch.bfloat16, device=x_fp8.device)
    x_scale = x_scale.contiguous()
    w_scale = w_scale.contiguous()
    BLOCK_M, BLOCK_N = 64, 64
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    _bw_fp8_gemm_kernel[grid](
        x_fp8, x_scale, w_fp8, w_scale, out,
        M, N, K,
        x_fp8.stride(0), x_fp8.stride(1),
        w_fp8.stride(0), w_fp8.stride(1),
        out.stride(0), out.stride(1),
        x_scale.stride(0), x_scale.stride(1),
        w_scale.stride(0), w_scale.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
    )
    return out


def _blockwise_fp8_gemm_torch(x_fp8, x_scale, w_fp8, w_scale, out=None):
    """Proven-correct dequant-matmul reference. Slow; the oracle for validate_marks."""
    M, K = x_fp8.shape
    N = w_fp8.shape[0]
    xd = (x_fp8.float().view(M, K // _BLK, _BLK) * x_scale.unsqueeze(-1)).view(M, K)
    NB = w_scale.shape[0]
    wp = torch.zeros(NB * _BLK, K, dtype=torch.float32, device=w_fp8.device)
    wp[:N] = w_fp8.float()
    wd = (wp.view(NB, _BLK, K // _BLK, _BLK) * w_scale[:, None, :, None]).view(NB * _BLK, K)[:N]
    res = (xd @ wd.t()).to(torch.bfloat16)
    if out is not None:
        out.copy_(res)
        return out
    return res


# ══════════════════════════════════════════════════════════════════════════════
# Per-family references — consume the frozen glm52_ops.build_inputs dict.
# ══════════════════════════════════════════════════════════════════════════════
def _ref_gemm(inputs: dict):
    out = inputs["out"]
    try:
        return _blockwise_fp8_gemm(inputs["x_fp8"], inputs["x_scale"],
                                   inputs["w_fp8"], inputs["w_scale"], out=out)
    except Exception:
        return _blockwise_fp8_gemm_torch(inputs["x_fp8"], inputs["x_scale"],
                                         inputs["w_fp8"], inputs["w_scale"], out=out)


def _ref_moe(inputs: dict):
    out = inputs["out"]                       # [E, expected_m, N]
    E = out.shape[0]
    for e in range(E):
        _blockwise_fp8_gemm(inputs["x_fp8"][e], inputs["x_scale"][e],
                            inputs["w_fp8"][e], inputs["w_scale"][e], out=out[e])
    return out


def _ref_bmm(inputs: dict):
    A = inputs["A_fp8"].float() * inputs["A_scale"]       # [B,M,K]
    B = inputs["B_fp8"].float() * inputs["B_scale"]       # [B,K,N]
    return torch.bmm(A, B).to(torch.bfloat16)


def _ref_mla(inputs: dict):
    """Chunked sparse MLA: each query gathers topk KV rows, then attends (MQA, h_kv=1)."""
    q = inputs["q"]                 # [M, H, D_QK]
    kv = inputs["kv"]               # [S, 1, D_QK]
    indices = inputs["indices"]     # [M, 1, tk] int32
    sm_scale = inputs["sm_scale"]
    d_v = inputs["d_v"]
    M, H, D_QK = q.shape
    kv2 = kv[:, 0, :]                                        # [S, D_QK]
    idx = indices[:, 0, :].long()                           # [M, tk]
    chunk = 256 if M >= 256 else M
    out = torch.empty(M, H, d_v, dtype=torch.bfloat16, device=q.device)
    for i in range(0, M, chunk):
        gi = idx[i:i + chunk]                               # [c, tk]
        kv_g = kv2[gi]                                      # [c, tk, D_QK]
        qc = q[i:i + chunk].float()                        # [c, H, D_QK]
        scores = torch.einsum('chd,ckd->chk', qc, kv_g.float()) * sm_scale   # [c,H,tk]
        p = torch.softmax(scores, dim=-1)                                    # [c,H,tk]
        oc = torch.einsum('chk,ckd->chd', p, kv_g[..., :d_v].float())        # [c,H,d_v]
        out[i:i + chunk] = oc.to(torch.bfloat16)
    return out


def _ref_score(op, phase, inputs: dict):
    weights = inputs["weights"]                             # [M, h]
    if phase == "prefill":
        q = inputs["q_fp8"].float()                        # [M, h, hd]
        k = inputs["k_fp8"].float()                        # [S, hd]
        k_scale = inputs["k_scale"].float()                # [S]
        Q = torch.einsum('mh,mhd->md', weights.float(), q)  # fold heads: [M, hd]
        logits = (Q @ k.t()) * k_scale[None, :]            # [M, S]
        return logits
    # decode: paged KV cache, reconstruct k[s] per request from its blocks.
    q = inputs["q_fp8"].float()                            # [M,1,h,hd]
    M, _, h, hd = q.shape
    cache = inputs["kv_cache_fp8"]                          # [tot,BLK,1,132] uint8
    seqlens = inputs["seqlens"].view(-1)                   # [M]
    block_tables = inputs["block_tables"]                  # [M, nbps]
    blk = cache.shape[1]
    max_s = int(seqlens.max().item())
    logits = torch.zeros(M, block_tables.shape[1] * blk, dtype=torch.float32, device=q.device)
    Qm = torch.einsum('mahd,mah->md', q, weights.view(M, 1, h).float())  # [M, hd]
    for m in range(M):
        s = int(seqlens[m].item())
        rows = []
        for bt in block_tables[m].tolist():
            data = cache[bt, :, 0, :hd].view(torch.float8_e4m3fnuz).float()   # [blk, hd]
            sf = cache[bt, :, 0, hd:hd + 4].contiguous().view(torch.float32)  # [blk]
            rows.append(data * sf[:, None])
        kfull = torch.cat(rows, dim=0)[:s]                 # [s, hd]
        logits[m, :s] = Qm[m] @ kfull.t()
    return logits


# ══════════════════════════════════════════════════════════════════════════════
# Provider
# ══════════════════════════════════════════════════════════════════════════════
def _pkg_version(dist: str):
    try:
        return importlib.metadata.version(dist)
    except importlib.metadata.PackageNotFoundError:
        return None


class TorchTritonRocmProvider:
    id = "torch-triton-rocm"
    platform = "rocm"
    capabilities = frozenset({"gemm", "bmm", "moe", "mla", "score"})
    required_modules = ("torch", "triton")
    baseline_caveat = BASELINE_CAVEAT
    accepted_candidate_forms = ACCEPTED_CANDIDATE_FORMS

    def supports(self, op: str, phase: str) -> bool:
        del op, phase
        return True

    def baseline_name(self, family: str, phase: str) -> str:
        if family == "gemm":
            return "triton blockwise-fp8 matmul (rocm)"
        if family == "bmm":
            return "torch per-tensor dequant bmm (rocm)"
        if family == "moe":
            return "triton blockwise-fp8 grouped matmul, per-expert (rocm)"
        if family == "mla":
            return "torch chunked gather + sparse attention (rocm)"
        return ("torch folded-weight MQA logits (rocm, prefill)" if phase == "prefill"
                else "torch paged MQA logits (rocm, decode)")

    def per_token_cast(self, tensor, *, use_ue8m0: bool):
        return per_token_cast(tensor, use_ue8m0=use_ue8m0)

    def per_block_cast(self, tensor, *, use_ue8m0: bool):
        return per_block_cast(tensor, use_ue8m0=use_ue8m0)

    def align_scale(self, scale):
        return scale                       # no TMA alignment on CDNA3

    def paged_mqa_metadata(self, seqlens, block_size: int):
        del block_size
        return seqlens                     # torch reference reads seqlens/block_tables directly

    def reference(self, op: str, phase: str, family: str, inputs: dict):
        if family == "gemm":
            return _ref_gemm(inputs)
        if family == "moe":
            return _ref_moe(inputs)
        if family == "bmm":
            return _ref_bmm(inputs)
        if family == "mla":
            return _ref_mla(inputs)
        return _ref_score(op, phase, inputs)

    def version_info(self):
        return {
            "torch": _pkg_version("torch"),
            "triton": _pkg_version("triton") or _pkg_version("pytorch-triton-rocm"),
            "hip": getattr(torch.version, "hip", None),
        }


# ══════════════════════════════════════════════════════════════════════════════
# Timer — HIP-event cold-L2 median (ROCm has no CUPTI device-span path; the harness's
# CUDA-event timer is HIP events on a ROCm torch build).
# ══════════════════════════════════════════════════════════════════════════════
class RocmKernelTimer:
    platform = "rocm"
    contract_id = "hip-event-cold-l2-median"
    contract_description = (
        "HIP-event cold-L2 median: inputs cloned per iteration and L2 flushed before "
        "each, both outside the measured window (ROCm has no CUPTI device-span path)"
    )

    @property
    def id(self) -> str:
        return self.contract_id

    @property
    def description(self) -> str:
        return self.contract_description

    def available(self) -> bool:
        return bool(torch.cuda.is_available() and getattr(torch.version, "hip", None))

    def measure(self, fn, *, setup, warmup, rep, device):
        timing = load_harness_module("timing")
        return timing.bench_time_with_cuda_events(
            fn, warmup=warmup, rep=rep, setup=setup, device=device
        )


PROVIDER = TorchTritonRocmProvider()
TIMER = RocmKernelTimer()
