#!/usr/bin/env python3
"""Calibrate GLM-5.2 ROCm bench references against SGLang production call sites.

This is deliberately stricter than "the benchmark runs": every task is labelled
as production-matched, production-unavailable, ABI-mismatched, or proxy-only.
The output is meant to decide which benchmark numbers may be treated as SGLang
baseline reproduction and which ones are only optimization proxies.
"""

from __future__ import annotations

import argparse
import csv
import importlib
import importlib.metadata
import json
import os
import statistics
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Callable


REPO = Path(__file__).resolve().parents[2]
DEFAULT_TASKSET = REPO / "tasksets" / "glm52_rocm_local.json"
SGLANG_ROOT = Path(os.environ.get("SGLANG_DIR", str(REPO.parent / "sglang")))
AITER_ROOT = Path(os.environ.get("AITER_PATH", str(REPO.parent / "aiter")))

# These must be set before importing SGLang fp8_utils.
os.environ.setdefault("KERNEL_HARNESS_PLATFORM", "rocm")
os.environ.setdefault("KERNEL_HARNESS_PROFILE", "amd-mi300x")
os.environ.setdefault("KERNEL_HARNESS_PROVIDER", "aiter-torch-reference")
os.environ.setdefault("KERNEL_HARNESS_TIMER", "event")
os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
os.environ.setdefault("SGLANG_USE_AITER", "1")

if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
sglang_python = SGLANG_ROOT / "python"
if sglang_python.exists() and str(sglang_python) not in sys.path:
    sys.path.insert(0, str(sglang_python))
if AITER_ROOT.exists() and str(AITER_ROOT) not in sys.path:
    sys.path.insert(0, str(AITER_ROOT))

import torch  # noqa: E402

from testbench.harness import glm52_ops as ops  # noqa: E402


GEMM_TASKS = {"fused_qkv_a", "q_b", "o_proj", "index_q_upproj", "index_k"}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--taskset", type=Path, default=DEFAULT_TASKSET)
    ap.add_argument("--phase", choices=("all", "prefill", "decode"), default="all")
    ap.add_argument("--task", action="append", default=None)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--M", type=int, default=None)
    ap.add_argument("--S", type=int, default=None)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--rep", type=int, default=64,
                    help="paired latency samples per shape; averaged after interleaving bench/prod")
    ap.add_argument("--latency-tolerance", type=float, default=0.15,
                    help="relative tolerance for production-matched latency ratio")
    ap.add_argument("--json-out", type=Path,
                    default=Path("/tmp/glm52_rocm_selected_baseline_calibration.json"))
    ap.add_argument("--csv-out", type=Path,
                    default=Path("/tmp/glm52_rocm_selected_baseline_calibration.csv"))
    args = ap.parse_args()

    taskset = load_taskset(args.taskset)
    selected = select_tasks(taskset["tasks"], args.phase, args.task)
    defaults = taskset.get("defaults", {})
    s_value = int(args.S if args.S is not None else defaults.get("S", 65536))

    device = torch.device(args.device)
    torch.cuda.set_device(device)

    report = {
        "taskset": taskset["name"],
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "environment": environment(),
        "parameters": {
            "device": str(device),
            "warmup": args.warmup,
            "rep": args.rep,
            "latency_method": "paired_interleaved_mean",
            "latency_tolerance": args.latency_tolerance,
            "S": s_value,
            "smoke": args.smoke,
        },
        "results": [],
    }

    for task in selected:
        for m_value in m_values(task, args, defaults):
            row = calibrate_one(task, int(m_value), s_value, device, args)
            report["results"].append(row)
            status = row["classification"]
            ratio = row.get("latency_ratio")
            ratio_s = f" ratio={ratio:.3f}" if isinstance(ratio, (int, float)) else ""
            print(f"{task['id']}[M={m_value}]: {status}{ratio_s} - {row.get('reason', '')}")

    report["summary"] = summarize(report["results"])
    write_outputs(report, args.json_out, args.csv_out)
    print(json.dumps({"summary": report["summary"], "json": str(args.json_out), "csv": str(args.csv_out)},
                     indent=2, sort_keys=True))
    return 0


def load_taskset(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text())
    if not data.get("tasks"):
        raise SystemExit(f"taskset has no tasks: {path}")
    return data


def select_tasks(tasks: list[dict[str, Any]], phase: str, requested: list[str] | None):
    selected = [task for task in tasks if phase == "all" or task["phase"] == phase]
    if requested:
        wanted = set(requested)
        selected = [
            task for task in selected
            if task["id"] in wanted
            or task["harness_task"] in wanted
            or task["reward_operator"] in wanted
            or task.get("harness_operator") in wanted
        ]
    if not selected:
        raise SystemExit("no tasks selected")
    return selected


def m_values(task: dict[str, Any], args, defaults: dict[str, Any]) -> list[int]:
    if args.M is not None:
        return [args.M]
    if args.smoke:
        smoke = defaults.get("smoke") or {}
        key = "prefill_M" if task["phase"] == "prefill" else "decode_M"
        return [int(smoke.get(key, 1024 if task["phase"] == "prefill" else 16))]
    key = "prefill_M" if task["phase"] == "prefill" else "decode_M"
    fallback = [1024, 2048, 4096] if task["phase"] == "prefill" else [1, 4, 8, 16, 32, 64]
    return [int(v) for v in defaults.get(key, fallback)]


def calibrate_one(task: dict[str, Any], m_value: int, s_value: int, device: torch.device, args) -> dict[str, Any]:
    op = task["harness_operator"]
    phase = task["phase"]
    fam = ops.family(op)
    base = {
        "task_id": task["id"],
        "harness_task": task["harness_task"],
        "harness_operator": op,
        "reward_operator": task["reward_operator"],
        "phase": phase,
        "M": m_value,
        "S": s_value,
        "family": fam,
        "bench_backend": ops.spec(op, phase)["backend"],
    }

    route = production_route(task, fam)
    base.update({
        "sglang_route": route["name"],
        "sglang_backend": route.get("backend"),
    })
    if route["classification"] != "runnable":
        base.update(
            classification=route["classification"],
            correct=None,
            reason=route["reason"],
        )
        return base

    try:
        inputs = ops.build_inputs(op, phase, m_value, s_value, device=device, seed=0)
        prod_fn = route["builder"](inputs)
        prod_out = clone_output(prod_fn())
        torch.cuda.synchronize()
        ref_out = clone_output(ops.reference(op, phase, inputs))
        comparison = ops.compare(ref_out, prod_out, op, phase, inputs)

        base.update(flatten_comparison(comparison))
        base["correct"] = bool(comparison["pass"])
        if not comparison["pass"]:
            base.update(
                classification="correctness_mismatch",
                reason=comparison["reason"],
            )
            return base

        bench_latency, prod_latency = measure_pair_ms(
            lambda: ops.reference(op, phase, inputs),
            prod_fn,
            args.warmup,
            args.rep,
        )
        ratio = prod_latency / bench_latency if bench_latency > 0 else None
        base.update(
            bench_latency_ms=bench_latency,
            sglang_latency_ms=prod_latency,
            latency_ratio=ratio,
        )
        if ratio is not None and abs(ratio - 1.0) <= args.latency_tolerance:
            classification = "production_matched"
            reason = "same ABI output passes and latency is within tolerance"
        else:
            classification = "production_correct_latency_mismatch"
            reason = "same ABI output passes but latency differs from bench reference"
        base.update(classification=classification, reason=reason)
        return base
    except Exception as exc:
        base.update(
            classification="production_unavailable",
            correct=None,
            reason=f"{type(exc).__name__}: {str(exc)[:500]}",
            traceback=traceback.format_exc(limit=8),
        )
        return base


def production_route(task: dict[str, Any], fam: str) -> dict[str, Any]:
    op = task["harness_operator"]
    phase = task["phase"]
    if fam == "gemm" and op in GEMM_TASKS:
        return {
            "classification": "runnable",
            "name": "sglang.srt.layers.quantization.fp8_utils.aiter_w8a8_block_fp8_linear",
            "backend": describe_sglang_fp8_backend(),
            "builder": build_sglang_gemm,
            "reason": "",
        }
    if op == "dsa_attn":
        return {
            "classification": "runnable",
            "name": "sglang.srt.layers.attention.dsa.tilelang_kernel.tilelang_sparse_fwd",
            "backend": "SGLang AMD default DSA tilelang backend",
            "builder": build_tilelang_sparse,
            "reason": "",
        }
    if op == "index_score":
        return {
            "classification": "runnable",
            "name": "aiter.ops.triton.fp8_mqa_logits.fp8_mqa_logits",
            "backend": "SGLang HIP DSA indexer ragged MQA logits kernel",
            "builder": build_aiter_mqa_logits,
            "reason": "",
        }
    if fam == "moe_fused":
        return {
            "classification": "runnable",
            "name": "sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe",
            "backend": "SGLang fused MoE total over gate/up, activation, routing weights, and down",
            "builder": build_sglang_fused_moe,
            "reason": "",
        }
    if fam == "moe":
        return {
            "classification": "abi_mismatch",
            "name": "SGLang fused MoE runner",
            "backend": "aiter.fused_moe / fused_moe runner",
            "reason": (
                "bench task is a split per-projection grouped GEMM; SGLang production "
                "path is fused MoE over gate/up, activation, routing weights, and down "
                "projection, so outputs and ABI are not comparable one-to-one"
            ),
        }
    return {
        "classification": "proxy_only",
        "name": "unknown",
        "backend": None,
        "reason": f"no production route registered for {op}/{phase}/{fam}",
    }


def build_sglang_gemm(inputs: dict[str, Any]) -> Callable[[], torch.Tensor]:
    from sglang.srt.layers.quantization.fp8_utils import aiter_w8a8_block_fp8_linear

    return lambda: aiter_w8a8_block_fp8_linear(
        inputs["x_fp8"],
        inputs["w_fp8"],
        [128, 128],
        inputs["w_scale"],
        inputs["x_scale"],
    )


def build_tilelang_sparse(inputs: dict[str, Any]) -> Callable[[], torch.Tensor]:
    from sglang.srt.layers.attention.dsa.tilelang_kernel import tilelang_sparse_fwd

    q = inputs["q"]
    kv = inputs["kv"]
    if kv.ndim == 2:
        kv = kv.unsqueeze(1)
    indices = inputs["indices"]
    if indices.ndim == 2:
        indices = indices.unsqueeze(1)
    indices = indices.to(torch.int32)
    def run():
        out = tilelang_sparse_fwd(
            q=q,
            kv=kv,
            indices=indices,
            sm_scale=float(inputs["sm_scale"]),
            d_v=int(inputs["d_v"]),
        )
        return out.squeeze(0) if out.ndim == 4 and out.shape[0] == 1 else out

    return run


def build_aiter_mqa_logits(inputs: dict[str, Any]) -> Callable[[], torch.Tensor]:
    from aiter.ops.triton.fp8_mqa_logits import fp8_mqa_logits

    q = inputs["q_fp8"]
    weights = inputs["weights"]
    if weights.ndim == 3:
        weights = weights.squeeze(-1)
    if q.ndim == 2:
        q = q.view(weights.shape[0], weights.shape[1], -1)
    k_scale = inputs["k_scale"]
    if k_scale.numel() == 1:
        k_scale = k_scale.expand(inputs["k_fp8"].shape[0]).contiguous()

    return lambda: fp8_mqa_logits(
        q,
        inputs["k_fp8"],
        k_scale,
        weights,
        inputs["ks"],
        inputs["ke"],
    )


def build_sglang_fused_moe(inputs: dict[str, Any]) -> Callable[[], torch.Tensor]:
    from sglang.srt.server_args import (
        ServerArgs,
        get_global_server_args,
        set_global_server_args_for_scheduler,
    )
    from sglang.srt.layers.moe.moe_runner import MoeRunnerConfig
    from sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe import fused_moe
    from sglang.srt.layers.moe.topk import StandardTopKOutput

    try:
        get_global_server_args()
    except ValueError:
        set_global_server_args_for_scheduler(
            ServerArgs(model_path=os.environ.get("SGLANG_DUMMY_MODEL_PATH", "dummy"))
        )

    topk_output = inputs.get("topk_output")
    if topk_output is None:
        topk_output = StandardTopKOutput(
            inputs["topk_weights"],
            inputs["topk_ids"],
            inputs["router_logits"],
        )
    cfg = inputs.get("moe_runner_config")
    if cfg is None:
        cfg = MoeRunnerConfig(
            **inputs["moe_config_kwargs"],
            params_dtype=inputs["hidden_states"].dtype,
        )

    return lambda: fused_moe(
        hidden_states=inputs["hidden_states"],
        w1=inputs["w1"],
        w2=inputs["w2"],
        topk_output=topk_output,
        moe_runner_config=cfg,
        use_fp8_w8a8=True,
        w1_scale=inputs["w1_scale"],
        w2_scale=inputs["w2_scale"],
        a1_scale=inputs["a1_scale"],
        a2_scale=inputs["a2_scale"],
    )


def describe_sglang_fp8_backend() -> str:
    try:
        mod = importlib.import_module("sglang.srt.layers.quantization.fp8_utils")
        use_aiter = bool(getattr(mod, "_use_aiter", False))
        use_gfx95 = bool(getattr(mod, "_use_aiter_gfx95", False))
        use_bp = bool(getattr(mod, "_use_aiter_bpreshuffle_gfx95", False))
        if use_aiter and not use_gfx95:
            return "AITER enabled; fp8_utils selects aiter Triton blockscale on non-gfx95 HIP"
        if use_bp:
            return "AITER bpreshuffle gfx95"
        if use_gfx95:
            return "AITER CK/bpreshuffle gfx95"
        return "SGLang fp8_utils fallback (AITER disabled)"
    except Exception as exc:
        return f"unavailable: {type(exc).__name__}: {exc}"


def measure_pair_ms(
    bench_fn: Callable[[], Any],
    prod_fn: Callable[[], Any],
    warmup: int,
    rep: int,
) -> tuple[float, float]:
    for _ in range(warmup):
        bench_fn()
        prod_fn()
    torch.cuda.synchronize()
    bench_samples = []
    prod_samples = []
    for i in range(rep):
        if i % 2 == 0:
            bench_samples.append(measure_one_ms(bench_fn))
            prod_samples.append(measure_one_ms(prod_fn))
        else:
            prod_samples.append(measure_one_ms(prod_fn))
            bench_samples.append(measure_one_ms(bench_fn))
    return statistics.fmean(bench_samples), statistics.fmean(prod_samples)


def measure_one_ms(fn: Callable[[], Any]) -> float:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end)


def clone_output(out):
    if torch.is_tensor(out):
        return out.clone()
    return tuple(x.clone() if torch.is_tensor(x) else x for x in out)


def flatten_comparison(comparison: dict[str, Any]) -> dict[str, Any]:
    keys = [
        "calc_diff",
        "max_abs_err",
        "max_rel_err",
        "abs_tol",
        "rel_tol",
        "diff_tol",
        "cosine",
        "best_fit_scale",
        "elementwise_failed",
        "elements",
    ]
    return {k: comparison.get(k) for k in keys if k in comparison}


def environment() -> dict[str, Any]:
    return {
        "torch": torch.__version__,
        "hip": getattr(torch.version, "hip", None),
        "tilelang": dist_version("tilelang"),
        "apache_tvm_ffi": dist_version("apache-tvm-ffi"),
        "torch_c_dlpack_ext": dist_version("torch-c-dlpack-ext"),
        "cuda_available": torch.cuda.is_available(),
        "device_count": torch.cuda.device_count(),
        "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "kernel_harness_commit": git_rev(REPO),
        "sglang_root": str(SGLANG_ROOT),
        "sglang_commit": git_rev(SGLANG_ROOT),
        "SGLANG_USE_AITER": os.environ.get("SGLANG_USE_AITER"),
        "AITER_CONFIG_DIR": os.environ.get("AITER_CONFIG_DIR"),
        "ROCM_USER_HOME": os.environ.get("ROCM_USER_HOME"),
        "CPATH": os.environ.get("CPATH"),
        "PYTORCH_ROCM_ARCH": os.environ.get("PYTORCH_ROCM_ARCH"),
        "backend_identity": getattr(ops, "BACKEND_BUNDLE").identity(),
    }


def dist_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def git_rev(path: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=path,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return None


def summarize(results: list[dict[str, Any]]) -> dict[str, int]:
    keys = [
        "production_matched",
        "production_correct_latency_mismatch",
        "correctness_mismatch",
        "production_unavailable",
        "abi_mismatch",
        "proxy_only",
    ]
    return {"total": len(results), **{k: sum(r["classification"] == k for r in results) for k in keys}}


def write_outputs(report: dict[str, Any], json_out: Path, csv_out: Path) -> None:
    json_out.parent.mkdir(parents=True, exist_ok=True)
    csv_out.parent.mkdir(parents=True, exist_ok=True)
    json_out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")

    rows = report["results"]
    fieldnames = [
        "task_id", "harness_task", "harness_operator", "reward_operator", "phase",
        "M", "S", "family", "classification", "correct", "bench_backend",
        "sglang_route", "sglang_backend", "bench_latency_ms", "sglang_latency_ms",
        "latency_ratio", "calc_diff", "max_abs_err", "max_rel_err", "cosine",
        "best_fit_scale", "elementwise_failed", "elements", "reason",
    ]
    with csv_out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


if __name__ == "__main__":
    raise SystemExit(main())
