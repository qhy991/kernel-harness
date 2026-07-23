#!/usr/bin/env python
"""GLM-5.2 end-to-end benchmark aligned with llm_flops scenarios.

Wraps `sglang.bench_one_batch` and reports TTFT (prefill) + decode
throughput on the two scenario sets llm_flops uses:

  prefill (KV cache = 64k, TP = 8):
    input_len ∈ {1024, 2048, 4096}, output_len = 1  →  measure TTFT
    → matches `llm_flops/bench_glm5_prefill.py` M sweep

  decode  (TP = 8):
    batch_size ∈ {128, 256}                          →  measure decode throughput
    (input_len small so TTFT doesn't dominate; --input-len configurable)
    → matches the deployed serving scenario llm_flops targets

Operator replacement:
  --overrides <file.py>   the file's `register()` runs after the compat shim
                          and before sglang boots. See operator_overrides.py.
                          Omit to run vanilla sglang production dispatch.

Outputs:
  · per-scenario one-line summary printed live
  · full JSONL per shape at $KDA_E2E_OUT/results_<stamp>.jsonl (default
    /tmp/glm5_e2e/results_YYYYmmdd_HHMMSS.jsonl)
  · every RESULT_JSON line from bench_one_batch is captured verbatim so
    downstream tools (rewardbench, plotting) can consume it

Environment prerequisites (this script asserts them at start):
  · 8× MI300X on a single node (visible via HIP_VISIBLE_DEVICES=0,...,7 or ROCR)
  · GLM-5.2-FP8 model at $KDA_E2E_MODEL (default /mnt/public/qinhaiyan/models/GLM-5.2-FP8)
  · /root/venvs/rocm-torch  (the ROCm PyTorch venv; override with ROCM_TORCH_VENV)
  · sglang + aiter checkouts on PYTHONPATH (see benchmarks/glm5_e2e/README.md)

Typical invocations:

  # baseline (no overrides) — full prefill TTFT sweep
  ./bench_glm5_e2e.py prefill

  # baseline — decode throughput, both batch sizes
  ./bench_glm5_e2e.py decode

  # your kernel swap in
  ./bench_glm5_e2e.py prefill --overrides ./examples/example_huyan_o_proj.py

  # single shape probe
  ./bench_glm5_e2e.py prefill --input-len 4096

Exit code: 0 = all scenarios completed; 1 = one or more failed (see log).
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_SHIM = _HERE / "shim"


# ── scenario definitions (aligned with llm_flops) ────────────────────────────
DEFAULT_MODEL_PATH = "/mnt/public/qinhaiyan/models/GLM-5.2-FP8"
DEFAULT_TP = 8
DEFAULT_KV_CACHE_TOKENS = 65536
DEFAULT_MEM_FRAC = 0.95
DEFAULT_DECODE_INPUT_LEN = 128

PREFILL_INPUT_LENS = [1024, 2048, 4096]     # llm_flops M sweep
DECODE_BATCH_SIZES = [128, 256]              # user-requested


def _venv_python() -> str:
    """The ROCm PyTorch used by sglang (aiter etc.). Overridable via env."""
    v = os.environ.get("ROCM_TORCH_PYTHON")
    if v and Path(v).is_file():
        return v
    root = os.environ.get("ROCM_TORCH_VENV", "/root/venvs/rocm-torch")
    p = Path(root) / "bin" / "python"
    if p.is_file():
        return str(p)
    fallback = shutil.which("python3")
    if fallback:
        return fallback
    sys.exit("[bench_glm5_e2e] no ROCm python found — set ROCM_TORCH_VENV or ROCM_TORCH_PYTHON")


def _shim_pythonpath() -> str:
    """PYTHONPATH additions for TP worker forks: shim + benchmarks dir + repo root."""
    parts = [str(_SHIM), str(_HERE)]
    repo_root = _HERE.parents[1]
    parts.append(str(repo_root))
    # Also honor the caller's PYTHONPATH.
    existing = os.environ.get("PYTHONPATH", "")
    if existing:
        parts.append(existing)
    return ":".join(parts)


def _prepare_env(overrides_path: str | None) -> dict:
    """Assemble the environment sglang subprocesses will inherit."""
    env = os.environ.copy()
    env["PYTHONPATH"] = _shim_pythonpath()
    env.setdefault("HIP_VISIBLE_DEVICES", "0,1,2,3,4,5,6,7")
    env.setdefault("PYTORCH_ROCM_ARCH", "gfx942")
    env.setdefault("SGLANG_USE_AITER", "1")
    env.setdefault("SGLANG_DSA_FUSE_TOPK", "0")
    env.setdefault("SGLANG_OPT_USE_AITER_SILU_MUL", "1")
    env.setdefault("SGLANG_DISABLE_GFX942_BPRESHUFFLE", "1")
    env.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
    env.setdefault("TORCHDYNAMO_DISABLE", "1")
    env.setdefault("TORCH_COMPILE_DISABLE", "1")
    # Reduce coredump risk that previously filled the local disk on crashes.
    env.setdefault("HSA_ENABLE_COREDUMP_ON_EXCEPTION", "0")
    if overrides_path:
        env["KDA_E2E_OVERRIDES"] = str(Path(overrides_path).resolve())
    return env


# ── entrypoint template ──────────────────────────────────────────────────────
_ENTRY_TEMPLATE = r'''
"""Auto-generated entry — imports shim + user overrides, then hands off to sglang.bench_one_batch."""
import os
import sys

# 1. compat shim (loads on import; also exports apply() for re-entry).
sys.path.insert(0, {shim_dir!r})
import glm52_gfx942_shim  # noqa: F401 — side-effect: patches applied

# 2. user overrides (optional).
overrides_path = os.environ.get("KDA_E2E_OVERRIDES", "")
if overrides_path:
    sys.path.insert(0, {benchmarks_dir!r})
    from operator_overrides import load_overrides, apply_overrides
    apply_overrides(load_overrides(overrides_path))
else:
    print("[bench_glm5_e2e] no --overrides given; running vanilla sglang", flush=True)

# 3. hand off to sglang.bench_one_batch.main with argv already set.
import argparse
from sglang.bench_one_batch import BenchArgs, main
from sglang.srt.server_args import ServerArgs

sys.argv = {argv!r}

parser = argparse.ArgumentParser()
ServerArgs.add_cli_args(parser)
BenchArgs.add_cli_args(parser)
cli = parser.parse_args()
server_args = ServerArgs.from_cli_args(cli)
bench_args = BenchArgs.from_cli_args(cli)

import logging
logging.basicConfig(level=logging.INFO, format="%(message)s")
main(server_args, bench_args)
'''


def _bench_argv(*, model_path: str, tp: int, input_len: int, output_len: int,
                batch_size: int, kv_tokens: int, mem_frac: float,
                dsa_prefill: str, dsa_decode: str, dsa_topk: str, moe_runner: str,
                result_filename: str) -> list[str]:
    """Assemble the sglang.bench_one_batch argv for one scenario point."""
    argv = [
        "bench_one_batch",
        "--model-path", model_path,
        "--tp", str(tp),
        "--batch-size", str(batch_size),
        "--input-len", str(input_len),
        "--output-len", str(output_len),
        "--trust-remote-code",
        "--mem-fraction-static", str(mem_frac),
        "--dsa-topk-backend", dsa_topk,
        "--dsa-prefill-backend", dsa_prefill,
        "--dsa-decode-backend", dsa_decode,
        "--moe-runner-backend", moe_runner,
        "--max-total-tokens", str(kv_tokens),
        "--cuda-graph-max-bs", str(max(batch_size, 1)),
        "--disable-cuda-graph",   # bf16 KV on gfx942 hits float8_e4m3fnuz codegen bug under graphs
        "--result-filename", result_filename,
    ]
    return argv


def _run_one_scenario(
    *, label: str, argv: list[str], env: dict, out_root: Path,
) -> dict:
    """Launch one sglang.bench_one_batch, parse RESULT_JSON, return summary."""
    py = _venv_python()
    result_path = out_root / f"{label}.jsonl"
    log_path = out_root / f"{label}.log"

    entry_source = _ENTRY_TEMPLATE.format(
        shim_dir=str(_SHIM),
        benchmarks_dir=str(_HERE),
        argv=argv,
    )

    # Write the auto-entry to a temp file so tracebacks name a real file.
    entry_path = out_root / f"{label}.entry.py"
    entry_path.write_text(entry_source)

    print(f"── {label} ──  running (log: {log_path.relative_to(out_root.parent)})", flush=True)
    started = time.time()
    with open(log_path, "wb") as logf:
        rc = subprocess.call(
            [py, str(entry_path)],
            env=env,
            stdout=logf,
            stderr=subprocess.STDOUT,
        )
    dur = time.time() - started

    # Parse the last RESULT_JSON line if present.
    metrics: dict = {}
    if result_path.exists():
        with open(result_path) as rf:
            lines = [line.strip() for line in rf if line.strip()]
        if lines:
            try:
                metrics = json.loads(lines[-1])
            except json.JSONDecodeError:
                pass

    summary = {
        "label": label,
        "argv": argv,
        "exit_code": rc,
        "wall_seconds": round(dur, 2),
        "result_jsonl": str(result_path.relative_to(out_root.parent)),
        "log": str(log_path.relative_to(out_root.parent)),
        "metrics": metrics,
    }
    if metrics:
        prefill = metrics.get("prefill_latency")
        prefill_tp = metrics.get("prefill_throughput")
        dec_ms = metrics.get("median_decode_latency")
        dec_tp = metrics.get("median_decode_throughput")
        pieces = []
        if prefill is not None:
            pieces.append(f"TTFT={prefill*1000:.1f}ms")
        if prefill_tp is not None:
            pieces.append(f"prefill={prefill_tp:.0f} tok/s")
        if dec_ms is not None:
            pieces.append(f"decode/tok={dec_ms:.2f}ms")
        if dec_tp is not None:
            pieces.append(f"decode_thpt={dec_tp:.1f} tok/s")
        summary["headline"] = " · ".join(pieces) if pieces else "(no metrics)"
    else:
        summary["headline"] = "(no result — see log)"

    tag = "✓" if rc == 0 else "✗"
    print(f"  {tag} exit={rc}  wall={dur:.1f}s  {summary['headline']}", flush=True)
    return summary


def _run_prefill(args, env: dict, out_root: Path) -> list[dict]:
    """llm_flops prefill scenario: KV=64k, output_len=1, input_len sweep, measure TTFT."""
    lens = args.input_len if args.input_len else PREFILL_INPUT_LENS
    results = []
    for ilen in lens:
        label = f"prefill_ttft_M{ilen}_kv{args.kv_tokens}_tp{args.tp}"
        argv = _bench_argv(
            model_path=args.model_path, tp=args.tp,
            input_len=ilen, output_len=1, batch_size=1,
            kv_tokens=args.kv_tokens, mem_frac=args.mem_fraction_static,
            dsa_prefill=args.dsa_prefill_backend, dsa_decode=args.dsa_decode_backend,
            dsa_topk=args.dsa_topk_backend, moe_runner=args.moe_runner_backend,
            result_filename=str(out_root / f"{label}.jsonl"),
        )
        results.append(_run_one_scenario(label=label, argv=argv, env=env, out_root=out_root))
    return results


def _run_decode(args, env: dict, out_root: Path) -> list[dict]:
    """llm_flops decode scenario: batch_size ∈ {128, 256}, measure decode throughput."""
    batches = args.batch_size if args.batch_size else DECODE_BATCH_SIZES
    results = []
    for bs in batches:
        label = f"decode_thpt_bs{bs}_in{args.decode_input_len}_out{args.decode_output_len}_tp{args.tp}"
        # For decode, keep input_len small so decode dominates TTFT.
        argv = _bench_argv(
            model_path=args.model_path, tp=args.tp,
            input_len=args.decode_input_len, output_len=args.decode_output_len,
            batch_size=bs,
            kv_tokens=max(args.kv_tokens, bs * args.decode_input_len * 2),
            mem_frac=args.mem_fraction_static,
            dsa_prefill=args.dsa_prefill_backend, dsa_decode=args.dsa_decode_backend,
            dsa_topk=args.dsa_topk_backend, moe_runner=args.moe_runner_backend,
            result_filename=str(out_root / f"{label}.jsonl"),
        )
        results.append(_run_one_scenario(label=label, argv=argv, env=env, out_root=out_root))
    return results


def _add_common_flags(ap: argparse.ArgumentParser) -> None:
    ap.add_argument("--model-path", default=os.environ.get("KDA_E2E_MODEL", DEFAULT_MODEL_PATH))
    ap.add_argument("--tp", type=int, default=DEFAULT_TP,
                    help=f"tensor-parallel degree (default {DEFAULT_TP} — GLM-5.2-FP8 needs 8× MI300X)")
    ap.add_argument("--kv-tokens", type=int, default=DEFAULT_KV_CACHE_TOKENS,
                    help=f"max KV cache tokens (default {DEFAULT_KV_CACHE_TOKENS}); prefill uses this verbatim, "
                         "decode floors it to batch × input_len × 2")
    ap.add_argument("--mem-fraction-static", type=float, default=DEFAULT_MEM_FRAC)
    ap.add_argument("--dsa-prefill-backend", default="aiter",
                    choices=("aiter", "flashmla_sparse", "flashmla_kv", "fa3", "trtllm"))
    ap.add_argument("--dsa-decode-backend", default="fa3",
                    choices=("aiter", "flashmla_sparse", "flashmla_kv", "fa3", "trtllm"))
    ap.add_argument("--dsa-topk-backend", default="torch",
                    choices=("torch", "aiter"))
    ap.add_argument("--moe-runner-backend", default="triton",
                    choices=("triton", "deep_gemm", "aiter"))
    ap.add_argument("--overrides", default=None,
                    help="user Python file with a top-level register() applying operator patches")
    ap.add_argument("--out-root", default=os.environ.get("KDA_E2E_OUT", "/tmp/glm5_e2e"),
                    help="results dir (default $KDA_E2E_OUT or /tmp/glm5_e2e)")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="scenario", required=True)

    p1 = sub.add_parser("prefill", help="llm_flops prefill TTFT scenario")
    _add_common_flags(p1)
    p1.add_argument("--input-len", type=int, action="append", default=None,
                    help=f"one or more input lengths (default {PREFILL_INPUT_LENS})")

    p2 = sub.add_parser("decode", help="llm_flops decode throughput scenario")
    _add_common_flags(p2)
    p2.add_argument("--batch-size", type=int, action="append", default=None,
                    help=f"one or more decode batch sizes (default {DECODE_BATCH_SIZES})")
    p2.add_argument("--decode-input-len", type=int, default=DEFAULT_DECODE_INPUT_LEN,
                    help=f"prefill length before decode (default {DEFAULT_DECODE_INPUT_LEN}); "
                         "kept small so decode-throughput number is not TTFT-dominated")
    p2.add_argument("--decode-output-len", type=int, default=64,
                    help="decode tokens per request (default 64)")

    p3 = sub.add_parser("both", help="prefill then decode, all defaults")
    _add_common_flags(p3)
    p3.add_argument("--input-len", type=int, action="append", default=None)
    p3.add_argument("--batch-size", type=int, action="append", default=None)
    p3.add_argument("--decode-input-len", type=int, default=DEFAULT_DECODE_INPUT_LEN)
    p3.add_argument("--decode-output-len", type=int, default=64)

    args = ap.parse_args()

    env = _prepare_env(args.overrides)

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%SZ")
    out_root = Path(args.out_root) / f"run-{stamp}"
    out_root.mkdir(parents=True, exist_ok=True)

    manifest = {
        "started_at": stamp,
        "scenario": args.scenario,
        "model_path": args.model_path,
        "tp": args.tp,
        "kv_tokens": args.kv_tokens,
        "backends": {
            "dsa_prefill": args.dsa_prefill_backend,
            "dsa_decode": args.dsa_decode_backend,
            "dsa_topk": args.dsa_topk_backend,
            "moe_runner": args.moe_runner_backend,
        },
        "overrides": str(Path(args.overrides).resolve()) if args.overrides else None,
        "out_root": str(out_root),
        "env": {k: env[k] for k in sorted(env)
                if k.startswith("SGLANG_") or k.startswith("KDA_") or k.startswith("HIP_")
                or k in ("PYTORCH_ROCM_ARCH", "TORCH_COMPILE_DISABLE")},
    }
    (out_root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    print(f"═════════ GLM-5.2 e2e bench · {args.scenario} · {stamp} ═════════", flush=True)
    print(f"  model={args.model_path}  tp={args.tp}  kv_tokens={args.kv_tokens}", flush=True)
    print(f"  backends: prefill={args.dsa_prefill_backend} decode={args.dsa_decode_backend} "
          f"topk={args.dsa_topk_backend} moe={args.moe_runner_backend}", flush=True)
    print(f"  overrides: {args.overrides or '(none — vanilla sglang)'}", flush=True)
    print(f"  out: {out_root}", flush=True)
    print("═" * 78, flush=True)

    results: list[dict] = []
    if args.scenario in ("prefill", "both"):
        results.extend(_run_prefill(args, env, out_root))
    if args.scenario in ("decode", "both"):
        results.extend(_run_decode(args, env, out_root))

    (out_root / "summary.json").write_text(json.dumps({
        "manifest": manifest, "results": results,
    }, indent=2) + "\n")

    n_ok = sum(1 for r in results if r["exit_code"] == 0)
    n_fail = len(results) - n_ok
    print("═" * 78, flush=True)
    print(f"summary: {n_ok}/{len(results)} scenarios OK, {n_fail} failed", flush=True)
    print(f"detailed: {out_root}/summary.json", flush=True)
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
