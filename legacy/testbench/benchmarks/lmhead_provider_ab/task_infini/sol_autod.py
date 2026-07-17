import torch, triton, triton.language as tl

def _cfgs():
    out=[]
    for BN in (128,256):
        for BK in (32,64):
            for w in (4,8):
                for s in (2,3,4):
                    out.append(triton.Config({'HALF':128,'BLOCK_N':BN,'BLOCK_K':BK}, num_warps=w, num_stages=s))
    return out

@triton.autotune(configs=_cfgs(), key=['M','N','K'])
@triton.jit
def _gd(a_ptr, w_ptr, c_ptr, M, N, K, sam,sak,swn,swk,scm,scn,
        HALF: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    pid_n = tl.program_id(0)
    offs_n = pid_n*BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    m0 = tl.arange(0, HALF); m1 = HALF + tl.arange(0, HALF)
    a0 = a_ptr + m0[:,None]*sam + offs_k[None,:]*sak
    a1 = a_ptr + m1[:,None]*sam + offs_k[None,:]*sak
    w_ptrs = w_ptr + offs_n[:,None]*swn + offs_k[None,:]*swk
    mm0 = m0[:,None] < M; mm1 = m1[:,None] < M
    ac0 = tl.zeros((HALF,BLOCK_N),dtype=tl.float32); ac1 = tl.zeros((HALF,BLOCK_N),dtype=tl.float32)
    for k in range(0,K,BLOCK_K):
        bt = tl.trans(tl.load(w_ptrs))
        ac0 += tl.dot(tl.load(a0,mask=mm0,other=0.0), bt, out_dtype=tl.float32)
        ac1 += tl.dot(tl.load(a1,mask=mm1,other=0.0), bt, out_dtype=tl.float32)
        a0+=BLOCK_K*sak; a1+=BLOCK_K*sak; w_ptrs+=BLOCK_K*swk
    et=c_ptr.dtype.element_ty
    tl.store(c_ptr+m0[:,None]*scm+offs_n[None,:]*scn, ac0.to(et), mask=mm0)
    tl.store(c_ptr+m1[:,None]*scm+offs_n[None,:]*scn, ac1.to(et), mask=mm1)

def run(hidden_states, lm_head_weight):
    a=hidden_states.bfloat16(); w=lm_head_weight.bfloat16()
    M,K=a.shape; N,_=w.shape
    out=torch.empty((M,N),device=a.device,dtype=torch.bfloat16)
    grid=lambda meta:(triton.cdiv(N,meta['BLOCK_N']),)
    _gd[grid](a,w,out,M,N,K,a.stride(0),a.stride(1),w.stride(0),w.stride(1),out.stride(0),out.stride(1))
    return out
