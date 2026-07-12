import torch, triton, triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor

# Persistent warp-specialized TMA matmul (Blackwell native): a fixed pool of
# programs (one per SM) each sweep multiple output tiles. TMA + software pipeline
# overlaps weight streaming with MMA the way cuBLAS's warp-specialized kernel does.
@triton.jit
def _k(a_desc, w_desc, c_ptr, M, N, K, scm, scn, NUM_SM,
       BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    start = tl.program_id(0)
    num_n = tl.cdiv(N, BLOCK_N)
    num_m = tl.cdiv(M, BLOCK_M)
    total = num_n * num_m
    k_tiles = tl.cdiv(K, BLOCK_K)
    for tile in range(start, total, NUM_SM):
        pid_m = tile % num_m
        pid_n = tile // num_m
        m0 = pid_m * BLOCK_M
        n0 = pid_n * BLOCK_N
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for ki in range(k_tiles):
            a = a_desc.load([m0, ki*BLOCK_K])
            b = w_desc.load([n0, ki*BLOCK_K])
            acc += tl.dot(a, tl.trans(b), out_dtype=tl.float32)
        offs_m = m0 + tl.arange(0, BLOCK_M)
        offs_n = n0 + tl.arange(0, BLOCK_N)
        c_ptrs = c_ptr + offs_m[:,None]*scm + offs_n[None,:]*scn
        tl.store(c_ptrs, acc.to(c_ptr.dtype.element_ty), mask=offs_m[:,None] < M)

def run(hidden_states, lm_head_weight):
    a = hidden_states.bfloat16(); w = lm_head_weight.bfloat16()
    M, K = a.shape; N, _ = w.shape
    out = torch.empty((M, N), device=a.device, dtype=torch.bfloat16)
    BM, BN, BK, wr, st = 256,64,64,8,4
    BMp = max(16, triton.next_power_of_2(M)) if M < BM else BM
    a_desc = TensorDescriptor.from_tensor(a, block_shape=[BMp, BK])
    w_desc = TensorDescriptor.from_tensor(w, block_shape=[BN, BK])
    NUM_SM = torch.cuda.get_device_properties(0).multi_processor_count
    grid = (NUM_SM,)
    _k[grid](a_desc, w_desc, out, M, N, K, out.stride(0), out.stride(1), NUM_SM,
             BLOCK_M=BMp, BLOCK_N=BN, BLOCK_K=BK, num_warps=wr, num_stages=st)
    return out
