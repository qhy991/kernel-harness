import torch, triton, triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor

# TMA weight-once + dual accumulator: weight column tile loaded ONCE per k-step via
# TMA, multiplied against two BLOCK_M row-halves. HBM weight read exactly once, and
# each MMA keeps the well-behaved BLOCK_M shape (no register-spilling 256-row acc).
@triton.jit
def _k(a_desc, w_desc, c_ptr, M, N, K, scm, scn,
       HALF: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    pid_n = tl.program_id(0)
    n0 = pid_n * BLOCK_N
    acc0 = tl.zeros((HALF, BLOCK_N), dtype=tl.float32)
    acc1 = tl.zeros((HALF, BLOCK_N), dtype=tl.float32)
    for k in range(0, K, BLOCK_K):
        b = w_desc.load([n0, k])                 # [BLOCK_N, BLOCK_K] once per k
        bt = tl.trans(b)
        a0 = a_desc.load([0, k])                 # [HALF, BLOCK_K]
        a1 = a_desc.load([HALF, k])
        acc0 += tl.dot(a0, bt, out_dtype=tl.float32)
        acc1 += tl.dot(a1, bt, out_dtype=tl.float32)
    offs_n = n0 + tl.arange(0, BLOCK_N)
    om0 = tl.arange(0, HALF); om1 = HALF + tl.arange(0, HALF)
    tl.store(c_ptr + om0[:,None]*scm + offs_n[None,:]*scn, acc0.to(c_ptr.dtype.element_ty), mask=om0[:,None]<M)
    tl.store(c_ptr + om1[:,None]*scm + offs_n[None,:]*scn, acc1.to(c_ptr.dtype.element_ty), mask=om1[:,None]<M)

def run(hidden_states, lm_head_weight):
    a = hidden_states.bfloat16(); w = lm_head_weight.bfloat16()
    M, K = a.shape; N, _ = w.shape
    out = torch.empty((M, N), device=a.device, dtype=torch.bfloat16)
    HALF, BN, BK, wr, st = 128,256,64,4,4
    a_desc = TensorDescriptor.from_tensor(a, block_shape=[HALF, BK])
    w_desc = TensorDescriptor.from_tensor(w, block_shape=[BN, BK])
    grid = (triton.cdiv(N, BN),)
    _k[grid](a_desc, w_desc, out, M, N, K, out.stride(0), out.stride(1),
             HALF=HALF, BLOCK_N=BN, BLOCK_K=BK, num_warps=wr, num_stages=st)
    return out
