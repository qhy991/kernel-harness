import torch, triton, triton.language as tl

@triton.jit
def _k(a_ptr, w_ptr, c_ptr, M, N, K, sam,sak,swn,swk,scm,scn,
       NSUB: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    pid_n = tl.program_id(0)
    offs_n = pid_n*BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    w_base = w_ptr + offs_n[:,None]*swn + offs_k[None,:]*swk
    # process M in NSUB sequential sub-tiles of BLOCK_M rows; weight column tile
    # stays L2-resident so only the first sub-pass touches HBM for the weight.
    for sub in range(NSUB):
        offs_m = sub*BLOCK_M + tl.arange(0, BLOCK_M)
        a_ptrs = a_ptr + offs_m[:,None]*sam + offs_k[None,:]*sak
        w_ptrs = w_base
        m_mask = offs_m[:,None] < M
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for k in range(0, K, BLOCK_K):
            a = tl.load(a_ptrs, mask=m_mask, other=0.0)
            b = tl.load(w_ptrs)
            acc += tl.dot(a, tl.trans(b), out_dtype=tl.float32)
            a_ptrs += BLOCK_K*sak; w_ptrs += BLOCK_K*swk
        c_ptrs = c_ptr + offs_m[:,None]*scm + offs_n[None,:]*scn
        tl.store(c_ptrs, acc.to(c_ptr.dtype.element_ty), mask=m_mask)

def run(hidden_states, lm_head_weight):
    a = hidden_states.bfloat16(); w = lm_head_weight.bfloat16()
    M,K = a.shape; N,_ = w.shape
    out = torch.empty((M,N), device=a.device, dtype=torch.bfloat16)
    BM,BN,BK,wr,st = 128,512,64,8,3
    NSUB = triton.cdiv(M, BM)
    grid = (triton.cdiv(N,BN),)
    _k[grid](a,w,out,M,N,K,a.stride(0),a.stride(1),w.stride(0),w.stride(1),out.stride(0),out.stride(1),
             NSUB=NSUB,BLOCK_M=BM,BLOCK_N=BN,BLOCK_K=BK,num_warps=wr,num_stages=st)
    return out
