#!/usr/bin/env python3
"""GLM-5.2 PREFILL operator **reward** benchmark on **AMD MI300X (ROCm)**.

A-card port of kernel-harness/rewardbench/bench_GLM5_ops_prefill.py. Scores the 13
GLM-5.2 prefill operators against the **MI300X roofline** and reports a bound-aware
roofline-utilization reward in [0,1] (compute-util for compute-bound ops, HBM-bandwidth
util for memory-bound ops, auto-classified by arithmetic intensity). This is the
reward denominator (baseline) that operator-optimization candidates are later measured
against — the same role the B200 rewardbench plays, but with MI300X peaks and AMD kernels.

Peaks (MI300X / CDNA3): HBM 5.3 TB/s, FP8 e4m3 2.615 PF, BF16 1.307 PF.

Run:  python bench_AMD_GLM5_ops_prefill.py               # M sweep 1024/2048/4096 -> reward CSV
      python bench_AMD_GLM5_ops_prefill.py --m 4096       # single M
      AMD_BENCH_NO_GRAPH=1 python bench_AMD_GLM5_ops_prefill.py   # event timing
"""
import argparse
import torch

import amd_glm5_ops_common as C
from amd_bench_glm5_prefill import build_ops


def main():
    ap = argparse.ArgumentParser(description="GLM-5.2 PREFILL reward benchmark (MI300X)")
    ap.add_argument("--m", type=int, default=None, help="single prefill M (default 1024,2048,4096)")
    ap.add_argument("--s", type=int, default=65536)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--csv", default="amd_glm5_ops_prefill_reward.csv")
    args = ap.parse_args()

    device = torch.device(args.device)
    torch.cuda.set_device(device)
    C.print_env_banner()
    print("GLM-5 PREFILL reward-bench [MI300X, sglang-ROCm ABI, bound-aware roofline util]")

    m_list = [args.m] if args.m else [1024, 2048, 4096]
    sweep = [{"M": m, "S": args.s} for m in m_list]
    C.run_ops(build_ops(), sweep, device, "prefill", args.csv)


if __name__ == "__main__":
    main()
