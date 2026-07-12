import torch, triton, triton.language as tl

# Split-K: partition K into SPLIT chunks; each program computes a partial [BM,BN]
# and atomic-adds into fp32 out. More concurrent programs -> higher occupancy ->
# better tensor-core utilization for the balanced M=256 GEMM. Weight still read once.
@triton.jit
def _k(a_ptr, w_ptr, c_ptr, M, N, K, sam,sak,swn,swk,scm,scn,
       SPLIT: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    pid_n = tl.program_id(0)
    pid_m = tl.program_id(1)
    pid_k = tl.program_id(2)
    offs_m = pid_m*BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n*BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = pid_k*BLOCK_K + tl.arange(0, BLOCK_K)
    KPER = K // SPLIT
    kstart = pid_k * KPER
    a_ptrs = a_ptr + offs_m[:,None]*sam + (kstart+tl.arange(0,BLOCK_K))[None,:]*sak
    w_ptrs = w_ptr + offs_n[:,None]*swn + (kstart+tl.arange(0,BLOCK_K))[None,:]*swk
    m_mask = offs_m[:,None] < M
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, KPER, BLOCK_K):
        a = tl.load(a_ptrs, mask=m_mask, other=0.0)
        b = tl.load(w_ptrs)
        acc += tl.dot(a, tl.trans(b), out_dtype=tl.float32)
        a_ptrs += BLOCK_K*sak; w_ptrs += BLOCK_K*swk
    c_ptrs = c_ptr + offs_m[:,None]*scm + offs_n[None,:]*scn
    tl.atomic_add(c_ptrs, acc, mask=m_mask)

def run(hidden_states, lm_head_weight):
    a = hidden_states.bfloat16(); w = lm_head_weight.bfloat16()
    M,K = a.shape; N,_ = w.shape
    BM,BN,BK,SP,wr,st = __CFG__
    out = torch.zeros((M,N), device=a.device, dtype=torch.float32)
    grid = (triton.cdiv(N,BN), triton.cdiv(M,BM), SP)
    _k[grid](a,w,out,M,N,K,a.stride(0),a.stride(1),w.stride(0),w.stride(1),out.stride(0),out.stride(1),
             SPLIT=SP,BLOCK_M=BM,BLOCK_N=BN,BLOCK_K=BK,num_warps=wr,num_stages=st)
    return out.bfloat16()
