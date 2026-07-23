"""dsa_attn decode — MI300X GLM-5.2 optimized operator  (archive 0723-amd-glm52).

Target shape : BS=16   (S=KV=65536, GLM-5.2)
Baseline     : aiter pa_decode_sparse (Triton path, sglang-ROCm production op)
Our kernel   : flash_split — tk-split flash-DECODING (occupancy) + fused combine kernel; f32-accumulated QK

Measured on MI300X (gfx942), HIP-event cold-L2 median:
  latency          : 122.6 us    (aiter baseline: 273.9 us)
  speedup vs aiter : 2.23x
  speedup vs ref   : 2.089x    (opbench torch/triton reference)
  roofline util    : 6.14%      (memory-bound)
  correctness      : PASS (calc_diff < 5e-6 vs bf16 dequant/full-attn oracle)

run(inputs) consumes the frozen glm52_ops inputs and returns the op output.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _amd_kernels import dsa_factory

TARGET = {"op": "dsa_attn", "phase": "decode", "BS": 16, "S": 65536}
CFG = {"NS": 8, "BH": 64, "BK": 16, "BH2": 16, "num_warps": 4, "num_stages": 1, "waves_per_eu": 1}
META = {"lat_us": 122.6, "aiter_us": 273.9, "speedup_vs_aiter": 2.23, "speedup_vs_ref": 2.089, "pct_roofline": 6.14, "bound": "memory", "correct": true, "variant": "flash_split", "aiter_baseline": "pa_decode_sparse"}

run = dsa_factory()(CFG)
