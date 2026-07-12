import torch, triton, triton.language as tl

# Compute out.T[N,M] = W[N,K] @ hidden.T[K,M], then it's stored into out[M,N].
# W (big, contiguous [N,K]) is the stationary "A" operand with a fat row-tile BN,
# giving tensor cores a large M dimension. hidden (tiny) is the "B" operand.
@triton.jit
def _k(w_ptr, h_ptr, c_ptr, N, M, K, swn,swk,shm,shk,scm,scn,
       BLOCK_N: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_K: tl.constexpr):
    pid_n = tl.program_id(0)
    offs_n = pid_n*BLOCK_N + tl.arange(0, BLOCK_N)
    offs_m = tl.arange(0, BLOCK_M)
    offs_k = tl.arange(0, BLOCK_K)
    w_ptrs = w_ptr + offs_n[:,None]*swn + offs_k[None,:]*swk      # [BN, BK] coalesced
    h_ptrs = h_ptr + offs_k[:,None]*shk + offs_m[None,:]*shm      # [BK, BM]  (hidden.T)
    m_mask = offs_m[None,:] < M
    acc = tl.zeros((BLOCK_N, BLOCK_M), dtype=tl.float32)
    for k in range(0, K, BLOCK_K):
        w = tl.load(w_ptrs)
        h = tl.load(h_ptrs, mask=m_mask, other=0.0)
        acc += tl.dot(w, h, out_dtype=tl.float32)                # [BN, BM]
        w_ptrs += BLOCK_K*swk; h_ptrs += BLOCK_K*shk
    # store transposed into out[M, N]: c[offs_m, offs_n] = acc[offs_n, offs_m]
    c_ptrs = c_ptr + offs_m[None,:]*scm + offs_n[:,None]*scn      # [BN, BM]
    tl.store(c_ptrs, acc.to(c_ptr.dtype.element_ty), mask=(offs_m[None,:] < M))

def run(hidden_states, lm_head_weight):
    a = hidden_states.bfloat16(); w = lm_head_weight.bfloat16()
    M,K = a.shape; N,_ = w.shape
    out = torch.empty((M,N), device=a.device, dtype=torch.bfloat16)
    BN,BM,BK,wr,st = 128,256,128,16,2
    grid = (triton.cdiv(N,BN),)
    _k[grid](w,a,out,N,M,K,w.stride(0),w.stride(1),a.stride(0),a.stride(1),out.stride(0),out.stride(1),
             BLOCK_N=BN,BLOCK_M=BM,BLOCK_K=BK,num_warps=wr,num_stages=st)
    return out
