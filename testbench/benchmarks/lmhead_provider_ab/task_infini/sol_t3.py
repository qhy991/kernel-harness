import torch, triton, triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor

# TMA-based weight-once GEMM for large M. Host-built descriptors; TMA streams the
# 2.35GB weight at near-peak HBM bandwidth with deep pipelining. M covered in one
# BLOCK_M pass -> weight read from HBM exactly once (cold-L2-safe).
@triton.jit
def _k(a_desc, w_desc, c_ptr, M, N, K, scm, scn,
       BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    pid_n = tl.program_id(0)
    n0 = pid_n * BLOCK_N
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, K, BLOCK_K):
        a = a_desc.load([0, k])          # [BLOCK_M, BLOCK_K]
        b = w_desc.load([n0, k])         # [BLOCK_N, BLOCK_K]
        acc += tl.dot(a, tl.trans(b), out_dtype=tl.float32)
    offs_m = tl.arange(0, BLOCK_M)
    offs_n = n0 + tl.arange(0, BLOCK_N)
    c_ptrs = c_ptr + offs_m[:, None]*scm + offs_n[None, :]*scn
    tl.store(c_ptrs, acc.to(c_ptr.dtype.element_ty), mask=offs_m[:, None] < M)

def run(hidden_states, lm_head_weight):
    a = hidden_states.bfloat16(); w = lm_head_weight.bfloat16()
    M, K = a.shape; N, _ = w.shape
    out = torch.empty((M, N), device=a.device, dtype=torch.bfloat16)
    BM, BN, BK, wr, st = 256,128,64,8,3
    a_desc = TensorDescriptor.from_tensor(a, block_shape=[BM, BK])
    w_desc = TensorDescriptor.from_tensor(w, block_shape=[BN, BK])
    grid = (triton.cdiv(N, BN),)
    _k[grid](a_desc, w_desc, out, M, N, K, out.stride(0), out.stride(1),
             BLOCK_M=BM, BLOCK_N=BN, BLOCK_K=BK, num_warps=wr, num_stages=st)
    return out
