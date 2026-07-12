import torch, triton, triton.language as tl

@triton.jit
def _kq(a_ptr, w_ptr, c_ptr, M, N, K, sam,sak,swn,swk,scm,scn,
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
    k0=m0[:,None]<M; k1=m1[:,None]<M; k2=m2[:,None]<M; k3=m3[:,None]<M
    ac0=tl.zeros((QM,BLOCK_N),dtype=tl.float32); ac1=tl.zeros((QM,BLOCK_N),dtype=tl.float32)
    ac2=tl.zeros((QM,BLOCK_N),dtype=tl.float32); ac3=tl.zeros((QM,BLOCK_N),dtype=tl.float32)
    for k in range(0,K,BLOCK_K):
        bt=tl.trans(tl.load(w_ptrs))
        ac0+=tl.dot(tl.load(a0,mask=k0,other=0.0),bt,out_dtype=tl.float32)
        ac1+=tl.dot(tl.load(a1,mask=k1,other=0.0),bt,out_dtype=tl.float32)
        ac2+=tl.dot(tl.load(a2,mask=k2,other=0.0),bt,out_dtype=tl.float32)
        ac3+=tl.dot(tl.load(a3,mask=k3,other=0.0),bt,out_dtype=tl.float32)
        a0+=BLOCK_K*sak; a1+=BLOCK_K*sak; a2+=BLOCK_K*sak; a3+=BLOCK_K*sak; w_ptrs+=BLOCK_K*swk
    et=c_ptr.dtype.element_ty
    tl.store(c_ptr+m0[:,None]*scm+offs_n[None,:]*scn, ac0.to(et), mask=k0)
    tl.store(c_ptr+m1[:,None]*scm+offs_n[None,:]*scn, ac1.to(et), mask=k1)
    tl.store(c_ptr+m2[:,None]*scm+offs_n[None,:]*scn, ac2.to(et), mask=k2)
    tl.store(c_ptr+m3[:,None]*scm+offs_n[None,:]*scn, ac3.to(et), mask=k3)

# minimal _gemm fallback for M<256 (reuse proven configs)
@triton.jit
def _gemm(a_ptr, w_ptr, c_ptr, M, N, K, sam,sak,swn,swk,scm,scn,
          BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    pid_n = tl.program_id(0)
    offs_m = tl.arange(0, BLOCK_M)
    offs_n = pid_n*BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    a_ptrs = a_ptr + offs_m[:,None]*sam + offs_k[None,:]*sak
    w_ptrs = w_ptr + offs_n[:,None]*swn + offs_k[None,:]*swk
    m_mask = offs_m[:,None] < M
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, K, BLOCK_K):
        acc += tl.dot(tl.load(a_ptrs,mask=m_mask,other=0.0), tl.trans(tl.load(w_ptrs)), out_dtype=tl.float32)
        a_ptrs += BLOCK_K*sak; w_ptrs += BLOCK_K*swk
    tl.store(c_ptr + offs_m[:,None]*scm + offs_n[None,:]*scn, acc.to(c_ptr.dtype.element_ty), mask=m_mask)

def run(hidden_states, lm_head_weight):
    a=hidden_states.bfloat16(); w=lm_head_weight.bfloat16()
    M,K=a.shape; N,_=w.shape
    out=torch.empty((M,N),device=a.device,dtype=torch.bfloat16)
    sa=(a.stride(0),a.stride(1)); sw=(w.stride(0),w.stride(1)); so=(out.stride(0),out.stride(1))
    if M >= 256:
        QM,BN,BK,wr,st = 64,256,32,8,4
        grid=(triton.cdiv(N,BN),)
        _kq[grid](a,w,out,M,N,K,*sa,*sw,*so,QM=QM,BLOCK_N=BN,BLOCK_K=BK,num_warps=wr,num_stages=st)
        return out
    BM=max(16,triton.next_power_of_2(M))
    if M<=16: BN,BK,wr,st=64,128,4,4
    elif M<=32: BN,BK,wr,st=128,128,4,3
    elif M<=64: BN,BK,wr,st=128,128,8,4
    else: BN,BK,wr,st=256,64,8,4
    grid=(triton.cdiv(N,BN),)
    _gemm[grid](a,w,out,M,N,K,*sa,*sw,*so,BLOCK_M=BM,BLOCK_N=BN,BLOCK_K=BK,num_warps=wr,num_stages=st)
    return out
