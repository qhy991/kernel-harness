#!/usr/bin/env python3
"""GLM-5.2 DECODE operator **reward** benchmark on **AMD MI300X (ROCm)**.

A-card port of kernel-harness/rewardbench/bench_GLM5_ops_decode.py. Scores the 13
GLM-5.2 decode operators against the MI300X roofline (batch sweep 1/4/8/16/32/64,
S=65536). At decode's tiny batch the dense GEMMs are weight-memory-bound, so most
rewards are HBM-bandwidth utilization — the reward correctly rewards a kernel that
streams weights closer to 5.3 TB/s rather than one that maximizes (already-trivial)
matrix-core work. This CSV is the decode reward denominator (baseline).

Run:  python bench_AMD_GLM5_ops_decode.py               # batch sweep -> reward CSV
      python bench_AMD_GLM5_ops_decode.py --m 32         # single batch
"""
import argparse
import torch

import amd_glm5_ops_common as C
from amd_bench_glm5_decode import build_ops


def main():
    ap = argparse.ArgumentParser(description="GLM-5.2 DECODE reward benchmark (MI300X)")
    ap.add_argument("--m", type=int, default=None, help="single decode batch (default 1,4,8,16,32,64)")
    ap.add_argument("--s", type=int, default=65536)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--csv", default="amd_glm5_ops_decode_reward.csv")
    args = ap.parse_args()

    device = torch.device(args.device)
    torch.cuda.set_device(device)
    C.print_env_banner()
    print("GLM-5 DECODE reward-bench [MI300X, sglang-ROCm ABI, bound-aware roofline util]")

    m_list = [args.m] if args.m else [1, 4, 8, 16, 32, 64]
    sweep = [{"M": m, "S": args.s} for m in m_list]
    C.run_ops(build_ops(), sweep, device, "decode", args.csv)


if __name__ == "__main__":
    main()
