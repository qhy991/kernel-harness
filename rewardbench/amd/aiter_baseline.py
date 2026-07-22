#!/usr/bin/env python3
"""Re-test the MI300X GLM-5.2 optimization degree against the AITER production baseline.

The goal's harder baseline: instead of our torch/triton reference, use AITER's own
optimized operators (the sglang-ROCm production path) as the baseline, and measure how
much our tuned kernels still improve over them.

  o_proj / index_k : aiter.ops.triton.gemm.basic.gemm_a8w8_blockscale (a8w8 blockwise fp8)
  dsa_attn         : aiter.ops.triton.attention.pa_decode_sparse (sparse-MLA paged decode)

Both consume our frozen glm52_ops inputs directly (aiter's blockscale scale layout is
identical: x_scale [M,K//128], w_scale [ceil(N/128),K//128]; pa_decode_sparse takes
q[N,H,D] + a flat kv pool + kv_indices/indptr). Correctness of each aiter op is checked
against our validated oracle before timing.

CAVEAT (reported in the output): only aiter's **Triton** path is runnable on this node —
the CK / ASM path (gemm_a8w8_blockscale_bpreshuffle_asm, ~2.64x over CK) needs a C++ JIT
build into aiter's package dir, which is not writable here, and no prebuilt code objects
ship. So "aiter baseline" here = aiter's Triton kernels, not its ASM peak.

Env:  AITER_TRITON_ONLY=1  PYTHONPATH=<aiter repo>  (aiter lives at
      /opt/devmachine/*/repos/aiter on this node; pass --aiter-path to override)
Run:  .venv/bin/python rewardbench/amd/aiter_baseline.py [--aiter-path P]
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import os
import statistics
import sys
from pathlib import Path

os.environ.setdefault("KERNEL_HARNESS_PLATFORM", "rocm")
os.environ.setdefault("KERNEL_HARNESS_PROFILE", "rocm-mi300x")
os.environ.setdefault("KERNEL_HARNESS_PROVIDER", "torch-triton-rocm")
os.environ.setdefault("KERNEL_HARNESS_TIMER", "event")
os.environ.setdefault("AITER_TRITON_ONLY", "1")
os.environ.setdefault("AITER_LOG_LEVEL", "ERROR")

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parents[1]
sys.path.insert(0, str(_REPO))

import torch  # noqa: E402


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


ops = _load("glm52_ops", _REPO / "testbench/harness/glm52_ops.py")
tb_timing = _load("tb_timing", _REPO / "testbench/harness/timing.py")
from testbench.harness.backends import rocm_mi300x as R  # noqa: E402


def med_ms(fn, ins, dev, reps=5, iters=20):
    setup = lambda: {k: (v.clone() if torch.is_tensor(v) else v) for k, v in ins.items()}  # noqa: E731
    return statistics.median(
        [statistics.median(tb_timing.bench_time_with_cuda_events(
            lambda a: fn(a), warmup=3, rep=iters, setup=setup, device=dev)) for _ in range(reps)]
    ) * 1e3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--aiter-path", default=None,
                    help="aiter repo root (default: autodetect under /opt/devmachine/*/repos/aiter)")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    ap_path = args.aiter_path
    if ap_path is None:
        import glob
        cands = glob.glob("/opt/devmachine/*/repos/aiter")
        ap_path = cands[0] if cands else None
    if ap_path and ap_path not in sys.path:
        sys.path.insert(0, ap_path)
    from aiter.ops.triton.gemm.basic.gemm_a8w8_blockscale import gemm_a8w8_blockscale
    from aiter.ops.triton.attention.pa_decode_sparse import pa_decode_sparse
    print(f"aiter from: {ap_path}  (Triton path; CK/ASM not built here)\n")

    dev = torch.device(args.device); torch.cuda.set_device(dev)
    S = ops.DEFAULT_S

    def aiter_gemm(a):
        return gemm_a8w8_blockscale(a["x_fp8"], a["w_fp8"], a["x_scale"], a["w_scale"],
                                    dtype=torch.bfloat16)

    def aiter_mla(a):
        M = a["q"].shape[0]; tk = a["indices"].shape[-1]
        idx = a["indices"][:, 0, :].to(torch.int32).contiguous().reshape(-1)
        indptr = torch.arange(0, (M + 1) * tk, tk, dtype=torch.int32, device=dev)
        sink = torch.full((a["q"].shape[1],), -1e30, dtype=torch.float32, device=dev)
        return pa_decode_sparse(a["q"].contiguous(), a["kv"][:, 0, :].contiguous(),
                                idx, indptr, sink, a["sm_scale"], has_invalid=False)[..., :512]

    def tuned(task):
        m = _load(task, _HERE / "tuned" / f"{task}.py")
        return m.run

    JOBS = [
        ("o_proj", "prefill", (1024, 2048, 4096), "o_proj_prefill", aiter_gemm, "gemm_a8w8_blockscale"),
        ("index_k", "prefill", (1024, 2048, 4096), "index_k_prefill", aiter_gemm, "gemm_a8w8_blockscale"),
        ("dsa_attn", "decode", (16, 32), "dsa_attn_decode", aiter_mla, "pa_decode_sparse"),
    ]
    rows = []
    hdr = f"{'op':>10} {'phase':>7} {'M':>5} {'aiter_us':>9} {'ourRef_us':>10} {'tuned_us':>9} {'tuned/aiter':>11} {'aiter/ourRef':>12} {'correct':>8}"
    print(hdr)
    for op, ph, Ms, task, aiterfn, aiter_name in JOBS:
        tfn = tuned(task)
        for M in Ms:
            ins = ops.build_inputs(op, ph, M, S, dev, 0)
            # correctness of the aiter baseline vs our validated oracle
            aout = aiterfn(ins)
            if op == "dsa_attn":
                oracle = R._ref_mla(ins).float()
            else:
                oracle = R._blockwise_fp8_gemm_torch(ins["x_fp8"], ins["x_scale"],
                                                     ins["w_fp8"], ins["w_scale"]).float()
            cd = ops.calc_diff(aout.float(), oracle)
            correct = cd < 5e-6
            la = med_ms(aiterfn, ins, dev)
            lr = med_ms(lambda a: ops.reference(op, ph, a), ins, dev)
            lt = med_ms(tfn, ins, dev)
            rows.append(dict(op=op, phase=ph, M=M, aiter_kernel=aiter_name,
                             aiter_us=round(la, 1), our_ref_us=round(lr, 1), tuned_us=round(lt, 1),
                             tuned_vs_aiter=round(la / lt, 3), aiter_vs_ourref=round(lr / la, 3),
                             aiter_correct=correct, aiter_calc_diff=f"{cd:.2e}"))
            print(f"{op:>10} {ph:>7} {M:>5} {la:>9.1f} {lr:>10.1f} {lt:>9.1f} "
                  f"{la/lt:>10.2f}x {lr/la:>11.2f}x {'OK' if correct else 'BAD':>8}")

    out = _HERE / "amd_glm5_aiter_baseline.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    gm = lambda xs: (statistics.geometric_mean(xs) if xs else 0)  # noqa: E731
    print(f"\nwrote {out.relative_to(_REPO)}")
    print(f"geomean tuned-vs-aiter speedup: {gm([r['tuned_vs_aiter'] for r in rows]):.2f}x "
          f"(aiter is {gm([r['aiter_vs_ourref'] for r in rows]):.2f}x faster than our old torch/triton reference)")
    print("CAVEAT: aiter here = Triton path only; CK/ASM (2.64x over CK) not buildable on this node.")


if __name__ == "__main__":
    main()
