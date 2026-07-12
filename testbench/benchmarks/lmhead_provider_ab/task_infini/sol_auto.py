import torch, triton, triton.language as tl

def _cfgs():
    out=[]
    for BM in (16,32,64,128,256):
        for BN in (64,128,256):
            for BK in (32,64,128):
                for w in (4,8):
                    for s in (2,3,4):
                        out.append(triton.Config(
                            {'BLOCK_M':BM,'BLOCK_N':BN,'BLOCK_K':BK,'GROUP_M':8},
                            num_warps=w, num_stages=s))
    return out

@triton.autotune(configs=_cfgs(), key=['M','N','K'])
@triton.jit
def _mm(a_ptr, w_ptr, c_ptr, M, N, K, sam,sak,swn,swk,scm,scn,
        GROUP_M: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    pid = tl.program_id(0)
    npm = tl.cdiv(M, BLOCK_M); npn = tl.cdiv(N, BLOCK_N)
    ng = GROUP_M * npn
    gid = pid // ng
    fm = gid * GROUP_M
    gsm = min(npm - fm, GROUP_M)
    pid_m = fm + ((pid % ng) % gsm)
    pid_n = (pid % ng) // gsm
    offs_m = pid_m*BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n*BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    a_ptrs = a_ptr + offs_m[:,None]*sam + offs_k[None,:]*sak
    w_ptrs = w_ptr + offs_n[:,None]*swn + offs_k[None,:]*swk
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
    grid = lambda meta: (triton.cdiv(M, meta['BLOCK_M']) * triton.cdiv(N, meta['BLOCK_N']),)
    _mm[grid](a,w,out,M,N,K,a.stride(0),a.stride(1),w.stride(0),w.stride(1),out.stride(0),out.stride(1))
    return out
