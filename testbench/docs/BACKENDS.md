# Backends

Kernel-Harness separates **contract** (shapes, correctness, FLOPs/bytes) from
**backend** pieces that vary by hardware:

| Piece | Owns |
|-------|------|
| `DeviceProfile` | peaks, FP8 dtype name, deployment label |
| `OperatorProvider` | reference kernels, quant/layout helpers |
| `Timer` | measurement protocol |

Only one bundle is registered today:

```text
cuda / cuda-b200 / deep-gemm-sgl-kernel / auto
```

Select it explicitly (or leave defaults):

```bash
export KERNEL_HARNESS_PLATFORM=cuda
export KERNEL_HARNESS_PROFILE=cuda-b200
export KERNEL_HARNESS_PROVIDER=deep-gemm-sgl-kernel
export KERNEL_HARNESS_TIMER=auto
```

Or put the same keys in `testbench/harness.env` (see `harness.env.example`).

Unsupported combinations (including any ROCm/AITER request) raise at
`get_backend()` time — there is no silent fallback.

Code layout:

```text
testbench/harness/backends/
  base.py          protocols + BackendBundle
  registry.py      selection
  cuda_b200.py     current B200 / DeepGEMM / CUPTI-or-Event implementation
```

Adding AMD later means registering a new `(platform, profile, provider, timer)`
bundle without copying the 24 task directories. The runner, candidate ABI, experience
bank, and GPU timing lock stay shared.
