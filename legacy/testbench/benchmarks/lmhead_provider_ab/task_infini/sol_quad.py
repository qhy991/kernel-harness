import torch, triton, triton.language as tl

# Quad accumulator: weight column tile loaded ONCE per k-step, multiplied against
# FOUR BLOCK_M=QM row-quarters. HBM weight read once; each acc is [QM,BN] (small
# registers -> high occupancy). Covers M up to 4*QM in a single launch.
@triton.jit
def _k(a_ptr, w_ptr, c_ptr, M, N, K, sam,sak,swn,swk,scm,scn,
       QM: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    pid_n = tl.program_id(0)
    offs_n = pid_n*BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    r = tl.arange(0, QM)
    m0=r; m1=QM+r; m2=2*QM+r; m3=3*QM+r
    a0=a_ptr+m0[:,None]*sam+offs_k[None,:]*sak
    a1=a_ptr+m1[:,None]*sam+offs_k[None,:]*sak
    a2=a_ptr+m2[:,None]*sam+offs_k[None,:]*sak
    a3=a_ptr+m3[:,None]*sam+offs_k[None,:]*sak
    w_ptrs=w_ptr+offs_n[:,None]*swn+offs_k[None,:]*swk
    ac0=tl.zeros((QM,BLOCK_N),dtype=tl.float32); ac1=tl.zeros((QM,BLOCK_N),dtype=tl.float32)
    ac2=tl.zeros((QM,BLOCK_N),dtype=tl.float32); ac3=tl.zeros((QM,BLOCK_N),dtype=tl.float32)
    for k in range(0,K,BLOCK_K):
        bt=tl.trans(tl.load(w_ptrs))
        ac0+=tl.dot(tl.load(a0),bt,out_dtype=tl.float32)
        ac1+=tl.dot(tl.load(a1),bt,out_dtype=tl.float32)
        ac2+=tl.dot(tl.load(a2),bt,out_dtype=tl.float32)
        ac3+=tl.dot(tl.load(a3),bt,out_dtype=tl.float32)
        a0+=BLOCK_K*sak; a1+=BLOCK_K*sak; a2+=BLOCK_K*sak; a3+=BLOCK_K*sak; w_ptrs+=BLOCK_K*swk
    et=c_ptr.dtype.element_ty
    tl.store(c_ptr+m0[:,None]*scm+offs_n[None,:]*scn, ac0.to(et), mask=m0[:,None]<M)
    tl.store(c_ptr+m1[:,None]*scm+offs_n[None,:]*scn, ac1.to(et), mask=m1[:,None]<M)
    tl.store(c_ptr+m2[:,None]*scm+offs_n[None,:]*scn, ac2.to(et), mask=m2[:,None]<M)
    tl.store(c_ptr+m3[:,None]*scm+offs_n[None,:]*scn, ac3.to(et), mask=m3[:,None]<M)

def run(hidden_states, lm_head_weight):
    a=hidden_states.bfloat16(); w=lm_head_weight.bfloat16()
    M,K=a.shape; N,_=w.shape
    out=torch.empty((M,N),device=a.device,dtype=torch.bfloat16)
    QM,BN,BK,wr,st=__CFG__
    grid=(triton.cdiv(N,BN),)
    _k[grid](a,w,out,M,N,K,a.stride(0),a.stride(1),w.stride(0),w.stride(1),out.stride(0),out.stride(1),
             QM=QM,BLOCK_N=BN,BLOCK_K=BK,num_warps=wr,num_stages=st)
    return out
