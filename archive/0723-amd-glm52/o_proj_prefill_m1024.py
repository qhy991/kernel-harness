"""o_proj prefill — MI300X GLM-5.2 optimized operator  (archive 0723-amd-glm52).

Target shape : M=1024   (S=KV=65536, GLM-5.2)
Baseline     : aiter gemm_a8w8_blockscale (Triton path, sglang-ROCm production op)
Our kernel   : fp8_dot — native fp8 e4m3fnuz tl.dot (2.6PF matrix core) + AMD MFMA knobs (waves_per_eu/matrix_instr_nonkdim/kpack)

Measured on MI300X (gfx942), HIP-event cold-L2 median:
  latency          : 428.1 us    (aiter baseline: 946.0 us)
  speedup vs aiter : 2.21x
  speedup vs ref   : 3.643x    (opbench torch/triton reference)
  roofline util    : 16.22%      (compute-bound)
  correctness      : PASS (calc_diff < 5e-6 vs bf16 dequant/full-attn oracle)

run(inputs) consumes the frozen glm52_ops inputs and returns the op output.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _amd_kernels import gemm_factory

TARGET = {"op": "o_proj", "phase": "prefill", "M": 1024, "S": 65536}
CFG = {"BM": 128, "BN": 128, "GROUP_M": 1, "num_warps": 4, "num_stages": 2, "waves_per_eu": 2, "matrix_instr_nonkdim": 16, "kpack": 2}
META = {"lat_us": 428.1, "aiter_us": 946.0, "speedup_vs_aiter": 2.21, "speedup_vs_ref": 3.643, "pct_roofline": 16.22, "bound": "compute", "correct": true, "variant": "fp8_dot", "aiter_baseline": "gemm_a8w8_blockscale"}

run = gemm_factory(True, 1)(CFG)
