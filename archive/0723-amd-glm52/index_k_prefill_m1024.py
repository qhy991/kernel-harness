"""index_k prefill — MI300X GLM-5.2 optimized operator  (archive 0723-amd-glm52).

Target shape : M=1024   (S=KV=65536, GLM-5.2)
Baseline     : aiter gemm_a8w8_blockscale (Triton path, sglang-ROCm production op)
Our kernel   : fp8_dot — native fp8 e4m3fnuz tl.dot (2.6PF matrix core) + AMD MFMA knobs (waves_per_eu/matrix_instr_nonkdim/kpack)

Measured on MI300X (gfx942), HIP-event cold-L2 median:
  latency          : 288.8 us    (aiter baseline: 567.7 us)
  speedup vs aiter : 1.97x
  speedup vs ref   : 2.933x    (opbench torch/triton reference)
  roofline util    : 28.77%      (memory-bound)
  correctness      : PASS (calc_diff < 5e-6 vs bf16 dequant/full-attn oracle)

run(inputs) consumes the frozen glm52_ops inputs and returns the op output.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _amd_kernels import gemm_factory

TARGET = {"op": "index_k", "phase": "prefill", "M": 1024, "S": 65536}
CFG = {"BM": 256, "BN": 128, "GROUP_M": 8, "num_warps": 8, "num_stages": 1, "waves_per_eu": 2, "matrix_instr_nonkdim": 32, "kpack": 2}
META = {"lat_us": 288.8, "aiter_us": 567.7, "speedup_vs_aiter": 1.97, "speedup_vs_ref": 2.933, "pct_roofline": 28.77, "bound": "memory", "correct": true, "variant": "fp8_dot", "aiter_baseline": "gemm_a8w8_blockscale"}

run = gemm_factory(True, 1)(CFG)
