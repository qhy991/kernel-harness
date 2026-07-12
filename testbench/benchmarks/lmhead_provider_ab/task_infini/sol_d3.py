import torch, triton, triton.language as tl

# Weight-read-once for large M: each program owns one BN column tile and loads it
# ONCE per k-step, then multiplies it against TWO M-halves (each MMA stays at the
# well-behaved BLOCK_M=128 shape). Cold-L2-safe: the 2.35GB weight is streamed
# from HBM exactly once, unlike a 2-pass grid that re-reads it.
@triton.jit
def _k2(a_ptr, w_ptr, c_ptr, M, N, K, sam,sak,swn,swk,scm,scn,
        HALF: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    pid_n = tl.program_id(0)
    offs_n = pid_n*BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    offs_m0 = tl.arange(0, HALF)
    offs_m1 = HALF + tl.arange(0, HALF)
    a0 = a_ptr + offs_m0[:,None]*sam + offs_k[None,:]*sak
    a1 = a_ptr + offs_m1[:,None]*sam + offs_k[None,:]*sak
    w_ptrs = w_ptr + offs_n[:,None]*swn + offs_k[None,:]*swk
    m0_mask = offs_m0[:,None] < M
    m1_mask = offs_m1[:,None] < M
    acc0 = tl.zeros((HALF, BLOCK_N), dtype=tl.float32)
    acc1 = tl.zeros((HALF, BLOCK_N), dtype=tl.float32)
    for k in range(0, K, BLOCK_K):
        b = tl.load(w_ptrs)                       # [BN,BK] loaded ONCE per k
        bt = tl.trans(b)
        acc0 += tl.dot(tl.load(a0, mask=m0_mask, other=0.0), bt, out_dtype=tl.float32)
        acc1 += tl.dot(tl.load(a1, mask=m1_mask, other=0.0), bt, out_dtype=tl.float32)
        a0 += BLOCK_K*sak; a1 += BLOCK_K*sak; w_ptrs += BLOCK_K*swk
    c0 = c_ptr + offs_m0[:,None]*scm + offs_n[None,:]*scn
    c1 = c_ptr + offs_m1[:,None]*scm + offs_n[None,:]*scn
    tl.store(c0, acc0.to(c_ptr.dtype.element_ty), mask=m0_mask)
    tl.store(c1, acc1.to(c_ptr.dtype.element_ty), mask=m1_mask)

def run(hidden_states, lm_head_weight):
    a = hidden_states.bfloat16(); w = lm_head_weight.bfloat16()
    M,K = a.shape; N,_ = w.shape
    out = torch.empty((M,N), device=a.device, dtype=torch.bfloat16)
    HALF,BN,BK,wr,st = 128,256,64,8,3
    grid = (triton.cdiv(N,BN),)
    _k2[grid](a,w,out,M,N,K,a.stride(0),a.stride(1),w.stride(0),w.stride(1),out.stride(0),out.stride(1),
              HALF=HALF,BLOCK_N=BN,BLOCK_K=BK,num_warps=wr,num_stages=st)
    return out
