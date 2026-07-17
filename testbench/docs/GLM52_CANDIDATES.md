# GLM-5.2 candidates: what you can pass, and how

The entire ABI is one function:

```python
def run(inputs: dict): ...   # -> the output tensor
```

Anything that can be reached from Python can be a candidate — PyTorch, Triton, a
hand-written `.cu`, CUTLASS, a CUDA graph, a different deep_gemm config. The task
does not care how `run` is implemented, only that it consumes the frozen `inputs`
and returns the output.

You do **not** have to edit the task to be measured:

```bash
T=testbench/tasks/glm52/o_proj_decode

$T/run.sh --describe               # what is this problem? (tensor table included)
$T/run.sh --describe --json        # ...the same, machine-readable (== problem.json)

$T/run.sh                                     # tests $T/candidate.py
$T/run.sh --candidate ~/kernels/o_proj.py     # tests any .py, anywhere
$T/run.sh --candidate ~/kernels/o_proj_dir/   # or a dir holding candidate.py
```

A directory is searched for `candidate.py`, then `solution.py`, then `impl.py`.
`result.json` records the candidate's absolute path and SHA-256, and the run
directory keeps a byte-exact copy of the file that ran, so an external candidate is
just as reproducible as an in-tree one.

Read `problem.json` (or `--describe`) for the tensor names, shapes and dtypes. They
are read off a real `build_inputs()` call, so they cannot be stale.

## Rules that bite

- `inputs` is frozen and **shared byte-for-byte with the reference**. Re-quantizing,
  re-seeding, or rebuilding a tensor inside `run()` means you measured a different
  problem than the one the gate checked. Changing *layout* (`.contiguous()`,
  `.view()`) is fine — that is your kernel's business, and it is timed.
- `inputs["out"]`, where present, is **NaN-poisoned** before `run()` is called.
  Write it or return a fresh tensor; returning it unwritten fails.
- Setup that is not the kernel — JIT compilation, autotune warmup, building a CUDA
  graph — should happen at **import time**, not inside `run()`. The harness imports
  your file once and then calls `run()` under CUPTI, so import-time work is outside
  the measured window. Work you do inside `run()` is your latency.

## Triton

Nothing special: Triton kernels are Python, so a Triton candidate is just a `.py`.
`@triton.autotune` works — the harness's warmup absorbs the tuning sweep.

```python
# ~/kernels/triton_o_proj.py
import triton
import triton.language as tl


@triton.autotune(
    configs=[triton.Config({"BLOCK_M": bm, "BLOCK_N": bn}, num_warps=w, num_stages=s)
             for bm in (16, 32) for bn in (64, 128) for w in (4, 8) for s in (3, 4)],
    key=["M", "N", "K"],
)
@triton.jit
def _fp8_blk_gemm(x_ptr, xs_ptr, w_ptr, ws_ptr, o_ptr, M, N, K,
                  BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    pm, pn = tl.program_id(0), tl.program_id(1)
    offs_m = pm * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pn * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, 128)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    nkb = K // 128
    for kb in range(nkb):                      # one 128-wide K block = one scale
        k = kb * 128 + offs_k
        x = tl.load(x_ptr + offs_m[:, None] * K + k[None, :],
                    mask=offs_m[:, None] < M, other=0.0).to(tl.float32)
        w = tl.load(w_ptr + offs_n[:, None] * K + k[None, :],
                    mask=offs_n[:, None] < N, other=0.0).to(tl.float32)
        xs = tl.load(xs_ptr + offs_m * nkb + kb, mask=offs_m < M, other=0.0)
        ws = tl.load(ws_ptr + (offs_n // 128) * nkb + kb, mask=offs_n < N, other=0.0)
        acc += tl.dot(x, tl.trans(w)) * xs[:, None] * ws[None, :]
    tl.store(o_ptr + offs_m[:, None] * N + offs_n[None, :], acc.to(tl.bfloat16),
             mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


def run(inputs: dict):
    x, w, out = inputs["x_fp8"], inputs["w_fp8"], inputs["out"]
    M, K = x.shape
    N = w.shape[0]
    xs = inputs["x_scale"].contiguous()     # layout only — NOT re-quantizing
    ws = inputs["w_scale"].contiguous()
    grid = lambda meta: (triton.cdiv(M, meta["BLOCK_M"]),
                         triton.cdiv(N, meta["BLOCK_N"]))
    _fp8_blk_gemm[grid](x, xs, w, ws, out, M, N, K)
    return out
```

Measured on B200, `o_proj_decode` M=16:

```
PASS  calc_diff=3.17e-09  max_rel=2.92e-02  cand=69.98us  ref=53.31us  0.762x  exit=1
```

Correct, and honestly slower than deep_gemm — which is the point of the gate.

## CUDA `.cu`

Pass the **directory**. `candidate.py` compiles the `.cu` at import time and `run()`
only launches.

```
~/kernels/o_proj_cu/
  candidate.py
  o_proj.cu
```

```python
# candidate.py
from pathlib import Path
from torch.utils.cpp_extension import load

_HERE = Path(__file__).resolve().parent
# Compilation happens at import — outside the timed window.
_ext = load(name="o_proj_cu", sources=[str(_HERE / "o_proj.cu")],
            extra_cuda_cflags=["-O3", "--use_fast_math",
                               "-gencode=arch=compute_100,code=sm_100"],
            verbose=False)


def run(inputs: dict):
    out = inputs["out"]
    _ext.launch(inputs["x_fp8"], inputs["x_scale"].contiguous(),
                inputs["w_fp8"], inputs["w_scale"].contiguous(), out)
    return out
```

```cuda
// o_proj.cu  — deliberately naive; it is a mechanism example, not a fast kernel
#include <torch/extension.h>
#include <cuda_fp8.h>
#include <cuda_bf16.h>

__global__ void naive_fp8_blk_gemm(const __nv_fp8_storage_t* __restrict__ x,
                                   const float* __restrict__ xs,
                                   const __nv_fp8_storage_t* __restrict__ w,
                                   const float* __restrict__ ws,
                                   __nv_bfloat16* __restrict__ out,
                                   int M, int N, int K) {
  int n = blockIdx.x * blockDim.x + threadIdx.x;
  int m = blockIdx.y;
  if (n >= N || m >= M) return;
  const int nkb = K / 128;
  float acc = 0.f;
  for (int kb = 0; kb < nkb; ++kb) {
    float s = xs[m * nkb + kb] * ws[(n / 128) * nkb + kb];
    float part = 0.f;
    for (int j = 0; j < 128; ++j) {
      int k = kb * 128 + j;
      float xv = __half2float(__nv_cvt_fp8_to_halfraw(x[(size_t)m * K + k], __NV_E4M3));
      float wv = __half2float(__nv_cvt_fp8_to_halfraw(w[(size_t)n * K + k], __NV_E4M3));
      part = fmaf(xv, wv, part);
    }
    acc = fmaf(part, s, acc);
  }
  out[(size_t)m * N + n] = __float2bfloat16(acc);
}

void launch(torch::Tensor x, torch::Tensor xs, torch::Tensor w, torch::Tensor ws,
            torch::Tensor out) {
  int M = x.size(0), K = x.size(1), N = w.size(0);
  dim3 blk(256), grd((N + 255) / 256, M);
  naive_fp8_blk_gemm<<<grd, blk>>>(
      reinterpret_cast<const __nv_fp8_storage_t*>(x.data_ptr()), xs.data_ptr<float>(),
      reinterpret_cast<const __nv_fp8_storage_t*>(w.data_ptr()), ws.data_ptr<float>(),
      reinterpret_cast<__nv_bfloat16*>(out.data_ptr()), M, N, K);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("launch", &launch); }
```

Measured on B200, `o_proj_decode` M=16:

```
PASS  calc_diff=3.25e-09  cand=6765.36us  ref=64.38us  0.010x  reward=0.0019  exit=1
```

Correct and 100x slower, as a naive kernel should be. `reward=0.0019` says the same
thing in roofline terms.

## Why a bare `.cu` cannot be passed

A `.cu` file is not self-describing. Nothing in it states which `__global__` is the
entry point, what grid and block to launch it with, or how the `inputs` dict maps to
its arguments — including how the fp8 scales are laid out. Only glue code can say
that, and `run(inputs)` **is** that glue. So `run(inputs)` is not a Python
restriction; it is the minimum a candidate must declare to be launchable at all.
(rewardbench's `call_by_name` binds by parameter name and is likewise Python.)

## What the two examples above happen to prove

Both are independent third-party implementations, and both land at
`calc_diff ≈ 3.2e-09` — the same order as the f32 dequantize+matmul reference used
to calibrate the gate (4.5e-09). Three unrelated implementations agreeing three
orders of magnitude below the `5e-6` threshold is why that threshold is defensible.

Both also produce `max_rel = 2.92e-02`, which exceeds `rel_tol = 0.0157` and passes
only because `abs_tol` forgives the near-zero elements. Had `abs_tol` been the fixed
`1e-3` that FlashMLA uses for its O(1) attention outputs, **both correct
implementations would have been failed** at this op's O(100) output scale. That is
why `abs_tol` is derived per shape from `|ref|.max()`.
