"""Launch exactly one stock and one candidate W2 kernel for NCU capture.

Used to answer a single survivor question: the BM16 candidate clears the graph
leaf gate but sits on the 1.03 boundary for the containing region, so where do
its microseconds go relative to the weight-streaming DRAM floor?

Run under Nsight Compute with ``--kernel-name-base demangled`` and a kernel
filter, inside a ``with_hotspot_gpu.sh`` lease.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
CANDIDATE_PATH = HERE / "candidates" / "moe_w2_hotspot_bm16_auto.py"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-m", type=int, default=4, choices=(4, 5, 8, 9))
    parser.add_argument("--arm", choices=("stock", "candidate", "both"), default="both")
    parser.add_argument("--warmup", type=int, default=3)
    args = parser.parse_args()

    sys.path.insert(0, str(HERE.parent))
    spec = importlib.util.spec_from_file_location("ncu_probe_candidate", CANDIDATE_PATH)
    assert spec is not None and spec.loader is not None
    candidate_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(candidate_module)
    candidate_module.prepare_process()

    from serving_native.runner import Runtime
    from serving_native.workloads import get_workload

    decode_m = 16 if args.expected_m in (4, 5) else 32
    workload = get_workload(f"moe_w2_hotspot_decode_em{args.expected_m}")
    runtime = Runtime(workload)
    candidate_module.prepare_runtime(
        SimpleNamespace(workload=workload, local_rank=0)
    )
    torch = runtime.torch
    inputs = runtime.build_inputs()
    assert workload.params["decode_m"] == decode_m

    import deep_gemm

    lhs = (inputs["activation_fp8"], inputs["activation_scale"])
    rhs = (inputs["weight_fp8"], inputs["weight_scale"])

    def launch(bm16: bool) -> None:
        deep_gemm.fp8_m_grouped_gemm_nt_masked(
            lhs,
            rhs,
            inputs["out"],
            inputs["masked_m"],
            inputs["expected_m"],
            disable_ue8m0_cast=True,
            glm52_moe_w2_decode_bm16_auto=bm16,
        )

    arms = ("stock", "candidate") if args.arm == "both" else (args.arm,)
    for _ in range(args.warmup):
        for arm in arms:
            launch(arm == "candidate")
    torch.cuda.synchronize()

    # One profiled launch per arm, in a fixed order.
    torch.cuda.profiler.start()
    for arm in arms:
        launch(arm == "candidate")
    torch.cuda.synchronize()
    torch.cuda.profiler.stop()
    print(f"profiled arms={arms} expected_m={args.expected_m} decode_m={decode_m}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
