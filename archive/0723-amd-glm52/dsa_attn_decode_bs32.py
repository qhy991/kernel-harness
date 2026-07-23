"""dsa_attn decode — MI300X GLM-5.2 optimized operator  (archive 0723-amd-glm52).

Target shape : BS=32   (S=KV=65536, GLM-5.2)
Baseline     : aiter pa_decode_sparse (Triton path, sglang-ROCm production op)
Our kernel   : flash_split — tk-split flash-DECODING (occupancy) + fused combine kernel; f32-accumulated QK

Measured on MI300X (gfx942), HIP-event cold-L2 median:
  latency          : 128.2 us    (aiter baseline: 432.8 us)
  speedup vs aiter : 3.38x
  speedup vs ref   : 2.974x    (opbench torch/triton reference)
  roofline util    : 11.7%      (memory-bound)
  correctness      : PASS (calc_diff < 5e-6 vs bf16 dequant/full-attn oracle)

run(inputs) consumes the frozen glm52_ops inputs and returns the op output.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _amd_kernels import dsa_factory

TARGET = {"op": "dsa_attn", "phase": "decode", "BS": 32, "S": 65536}
CFG = {"NS": 8, "BH": 64, "BK": 16, "BH2": 16, "num_warps": 4, "num_stages": 1, "waves_per_eu": 1}
META = {"lat_us": 128.2, "aiter_us": 432.8, "speedup_vs_aiter": 3.38, "speedup_vs_ref": 2.974, "pct_roofline": 11.7, "bound": "memory", "correct": True, "variant": "flash_split", "aiter_baseline": "pa_decode_sparse"}

run = dsa_factory()(CFG)
