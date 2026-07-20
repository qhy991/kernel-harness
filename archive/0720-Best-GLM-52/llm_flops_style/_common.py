"""Drop-in llm_flops-style benches: same tensors for stock and archive candidates.

Comparability contract
----------------------
1. Timing: CUDA Graph (warmup=5, runs=20), identical to llm_flops.
2. Unswapped ops: call llm_flops bench_* verbatim.
3. Swapped ops: build tensors with llm_flops' own quant helpers (same code path
   as stock), freeze them, then time BOTH:
     - stock: deep_gemm.fp8_gemm_nt / fp8_m_grouped_gemm_nt_masked
     - cand:  archive candidate.run(inputs)
   on those same frozen tensors. Speedup is apples-to-apples.
"""
from __future__ import annotations

import importlib.util
import json
import random
import sys
from pathlib import Path
from typing import Callable

import torch
import deep_gemm
from deep_gemm.utils.layout import get_mn_major_tma_aligned_tensor

_HERE = Path(__file__).resolve().parent
_ARCHIVE = _HERE.parent
_BEST = _ARCHIVE / "best"
_REPO = _ARCHIVE.parents[1]
_HARNESS = _REPO / "testbench" / "harness"
_TASKS = _REPO / "testbench" / "tasks" / "glm52"
_LLM_FLOPS = Path("/home/qinhaiyan/llm_flops")

NUM_WARMUP = 5
NUM_RUNS = 20

sys.path[:] = [p for p in sys.path if p and Path(p).resolve() != _HARNESS]


def _load_harness(name: str):
    spec = importlib.util.spec_from_file_location(f"_tb_{name}", _HARNESS / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[f"_tb_{name}"] = mod
    spec.loader.exec_module(mod)
    return mod


def _load_llm_flops_module(filename: str):
    path = _LLM_FLOPS / filename
    if str(_LLM_FLOPS) not in sys.path:
        sys.path.insert(0, str(_LLM_FLOPS))
    name = f"_llm_flops_{path.stem}"
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


candidate_loader = _load_harness("candidate_loader")

# llm_flops name -> (harness_op, archive | None, kind)
# kind: None (stock-only) | "fp8_gemm" | "moe_masked"
DECODE_SWAPS: dict[str, tuple[str | None, str | None, str | None]] = {
    "fused_qkv_a_proj": (None, None, None),
    "q_b_proj": ("q_b", "q_b_decode", "fp8_gemm"),
    "absorbed_W_UK": (None, None, None),
    "absorbed_W_UV": (None, None, None),
    "o_proj": ("o_proj", "o_proj_decode_hbm35", "fp8_gemm"),
    "dsa_decode_attn": (None, None, None),
    "index_k_proj": (None, None, None),
    "index_q_upproj": ("index_q_upproj", "index_q_upproj_decode_hbm15", "fp8_gemm"),
    "index_weights_proj": (None, None, None),
    "index_score": (None, None, None),
    "moe_gate_proj": ("moe_gate", "moe_gate_proj_decode_hbm40", "moe_masked"),
    "moe_up_proj": ("moe_up", "moe_up_proj_decode_hbm40", "moe_masked"),
    "moe_down_proj": ("moe_down", "moe_down_proj_decode_hbm40", "moe_masked"),
}

PREFILL_SWAPS: dict[str, tuple[str | None, str | None, str | None]] = {
    "fused_qkv_a_proj": (None, None, None),
    "q_b_proj": (None, None, None),
    "absorbed_W_UK": (None, None, None),
    "absorbed_W_UV": (None, None, None),
    "o_proj": ("o_proj", "o_proj_prefill", "fp8_gemm"),
    "dsa_prefill_attn": (None, None, None),
    "index_k_proj": ("index_k", "index_k_prefill_bw70", "fp8_gemm"),
    "index_q_upproj": (None, None, None),
    "index_weights_proj": (None, None, None),
    "index_score": (None, None, None),
    "moe_gate_proj": (None, None, None),
    "moe_up_proj": (None, None, None),
    "moe_down_proj": ("moe_down", "moe_down_proj_prefill_mfu65", "moe_masked"),
}


def cuda_graph_bench(run_fn: Callable[[], None]) -> tuple[float, str]:
    torch.cuda.synchronize()
    for _ in range(NUM_WARMUP):
        run_fn()
    torch.cuda.synchronize()
    try:
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            for _ in range(NUM_RUNS):
                run_fn()
        torch.cuda.synchronize()
        for _ in range(NUM_WARMUP):
            graph.replay()
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        graph.replay()
        end.record()
        torch.cuda.synchronize()
        avg = start.elapsed_time(end) / NUM_RUNS
        del graph
        return avg, "cuda_graph"
    except Exception:
        for _ in range(NUM_WARMUP):
            run_fn()
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(NUM_RUNS):
            run_fn()
        end.record()
        torch.cuda.synchronize()
        return start.elapsed_time(end) / NUM_RUNS, "cuda_event"


def resolve_candidate(archive_name: str, harness_op: str, phase: str):
    cand_dir = _BEST / archive_name / "candidate"
    if not cand_dir.is_dir():
        raise FileNotFoundError(cand_dir)
    task_dir = None
    for d in _TASKS.iterdir():
        meta_path = d / "task.json"
        if not meta_path.is_file():
            continue
        meta = json.loads(meta_path.read_text())
        if meta["operator"] == harness_op and meta["phase"] == phase:
            task_dir = d
            break
    if task_dir is None:
        raise FileNotFoundError(f"no task for {harness_op}/{phase}")
    run_fn, source, _ = candidate_loader.resolve(
        task_dir, harness_op, phase, override=str(cand_dir))
    if source == "reference":
        raise RuntimeError(f"candidate fell through to reference: {archive_name}")
    return run_fn, source


# ── llm_flops-identical builders ──────────────────────────────────────────────

def build_fp8_gemm_inputs(lf, rows: int, K: int, N: int, device, seed: int) -> dict:
    """Same construction as llm_flops bench_deepgemm_fp8 (UE8M0 + TMA-aligned x_scale)."""
    g = torch.Generator(device=device)
    g.manual_seed(seed)
    x_bf16 = torch.randn(rows, K, dtype=torch.bfloat16, device=device, generator=g)
    x_fp8, x_scale = lf.per_token_cast_to_fp8(x_bf16)
    x_scale = get_mn_major_tma_aligned_tensor(x_scale)
    del x_bf16
    w_bf16 = torch.randn(N, K, dtype=torch.bfloat16, device=device, generator=g)
    w_fp8, w_scale = lf.per_block_cast_to_fp8(w_bf16)
    del w_bf16
    out = torch.empty(rows, N, dtype=torch.bfloat16, device=device)
    return dict(x_fp8=x_fp8, x_scale=x_scale, w_fp8=w_fp8, w_scale=w_scale, out=out)


def build_moe_masked_inputs(lf, M: int, K: int, N: int, device, seed: int) -> dict:
    """Same construction as llm_flops bench_moe_grouped_masked."""
    E = lf.N_EXPERT
    total_m = M * lf.NUM_EXPERTS_PER_TOK
    expected_m = (total_m + E - 1) // E
    expected_m = ((expected_m + 127) // 128) * 128

    g = torch.Generator(device=device)
    g.manual_seed(seed)
    x_bf16 = torch.randn(E, expected_m, K, dtype=torch.bfloat16, device=device, generator=g)
    x_fp8 = torch.empty_like(x_bf16, dtype=torch.float8_e4m3fn)
    x_scale = torch.empty(E, expected_m, K // 128, dtype=torch.float32, device=device)
    for i in range(E):
        x_fp8[i], x_scale[i] = lf.per_token_cast_to_fp8(x_bf16[i])
    del x_bf16

    w_bf16 = torch.randn(E, N, K, dtype=torch.bfloat16, device=device, generator=g)
    n_ceil = (N + 127) // 128 * 128
    w_fp8 = torch.empty(E, N, K, dtype=torch.float8_e4m3fn, device=device)
    w_scale = torch.empty(E, n_ceil // 128, K // 128, dtype=torch.float32, device=device)
    for i in range(E):
        w_fp8[i], w_scale[i] = lf.per_block_cast_to_fp8(w_bf16[i])
    del w_bf16

    out = torch.empty(E, expected_m, N, dtype=torch.bfloat16, device=device)

    rng = random.Random(seed)
    counts = [0] * E
    for _ in range(total_m):
        counts[rng.randint(0, E - 1)] += 1
    # llm_flops does not clamp; for drop-in safety keep counts <= expected_m
    # by redistributing overflow (rare with expected_m = ceil(total/E) rounded).
    while max(counts) > expected_m:
        e = counts.index(max(counts))
        counts[e] -= 1
        counts[counts.index(min(counts))] += 1
    masked_m = torch.tensor(counts, dtype=torch.int32, device=device)

    return dict(
        x_fp8=x_fp8, x_scale=x_scale, w_fp8=w_fp8, w_scale=w_scale,
        out=out, masked_m=masked_m, expected_m=expected_m,
    )


def stock_fp8_gemm(inputs: dict) -> None:
    deep_gemm.fp8_gemm_nt(
        (inputs["x_fp8"], inputs["x_scale"]),
        (inputs["w_fp8"], inputs["w_scale"]),
        inputs["out"],
    )


def stock_moe_masked(inputs: dict) -> None:
    deep_gemm.fp8_m_grouped_gemm_nt_masked(
        (inputs["x_fp8"], inputs["x_scale"]),
        (inputs["w_fp8"], inputs["w_scale"]),
        inputs["out"], inputs["masked_m"], inputs["expected_m"],
    )


def bench_dropin(
    lf,
    kind: str,
    harness_op: str,
    phase: str,
    archive: str,
    dims: tuple[int, int, int],
    device,
    seed: int = 0,
) -> dict:
    """Build once (llm_flops path), time stock + candidate on the same tensors."""
    rows_or_M, K, N = dims
    if kind == "fp8_gemm":
        inputs = build_fp8_gemm_inputs(lf, rows_or_M, K, N, device, seed)
        stock_fn = lambda: stock_fp8_gemm(inputs)  # noqa: E731
    elif kind == "moe_masked":
        inputs = build_moe_masked_inputs(lf, rows_or_M, K, N, device, seed)
        stock_fn = lambda: stock_moe_masked(inputs)  # noqa: E731
    else:
        raise ValueError(kind)

    cand_run, source = resolve_candidate(archive, harness_op, phase)

    stock_ms, stock_proto = cuda_graph_bench(stock_fn)
    cand_ms, cand_proto = cuda_graph_bench(lambda: cand_run(inputs))

    return {
        "avg_ms": cand_ms,
        "stock_ms": stock_ms,
        "protocol": cand_proto,
        "stock_protocol": stock_proto,
        "source": source,
        "impl": f"dropin:{archive}",
        "archive": archive,
        "same_inputs": True,
    }


def gemm_dims_for(lf, name: str, M: int, S: int, phase: str) -> tuple[int, int, int]:
    """Return (rows, K, N) matching llm_flops get_all_operators*."""
    if name == "q_b_proj":
        return M, lf.Q_LORA_RANK, lf.NUM_HEADS * lf.QK_HEAD_DIM
    if name == "o_proj":
        return M, lf.NUM_HEADS * lf.V_HEAD_DIM, lf.HIDDEN_SIZE
    if name == "index_q_upproj":
        return M, lf.Q_LORA_RANK, lf.INDEX_N_HEADS * lf.INDEX_HEAD_DIM
    if name == "index_k_proj":
        rows = S if phase == "prefill" else M
        return rows, lf.HIDDEN_SIZE, lf.INDEX_HEAD_DIM
    if name == "fused_qkv_a_proj":
        return M, lf.HIDDEN_SIZE, lf.FUSED_QKV_A_OUT
    raise KeyError(name)


def moe_dims_for(lf, name: str, M: int) -> tuple[int, int, int]:
    if name in ("moe_gate_proj", "moe_up_proj"):
        return M, lf.HIDDEN_SIZE, lf.MOE_INTERMEDIATE_SIZE
    if name == "moe_down_proj":
        return M, lf.MOE_INTERMEDIATE_SIZE, lf.HIDDEN_SIZE
    raise KeyError(name)


def print_summary(all_results: list[dict], m_list: list[int], s_list: list[int],
                  title: str) -> None:
    for M in m_list:
        for S in s_list:
            subset = [r for r in all_results
                      if r["M"] == M and r["S"] == S and r.get("avg_ms", 0) > 0]
            subset.sort(key=lambda r: r["avg_ms"], reverse=True)
            used_sum = sum(r["avg_ms"] for r in subset)
            stock_sum = sum(r.get("stock_ms", r["avg_ms"]) for r in subset)

            print(f"\n{'=' * 130}")
            print(f"  {title} Summary (M={M}, S={S})  [drop-in: same llm_flops tensors]")
            print(f"  layer TOTAL stock:   {stock_sum:.4f} ms")
            print(f"  layer TOTAL swapped: {used_sum:.4f} ms"
                  f"  speedup={stock_sum / used_sum:.4f}x" if used_sum else "")
            print(f"{'=' * 130}")
            print(f"  {'rank':<6s} {'name':<24s} {'impl':<28s} {'avg(ms)':>10s} "
                  f"{'stock(ms)':>10s} {'spd':>7s} {'pct':>7s} {'proto':<12s}")
            print(f"  {'-' * 120}")
            for rank, r in enumerate(subset, 1):
                pct = r["avg_ms"] / used_sum * 100 if used_sum else 0
                stock = r.get("stock_ms", r["avg_ms"])
                spd = stock / r["avg_ms"] if r["avg_ms"] else 0
                print(f"  {rank:<6d} {r['name']:<24s} {r.get('impl','stock'):<28s} "
                      f"{r['avg_ms']:>10.4f} {stock:>10.4f} {spd:>6.2f}x "
                      f"{pct:>6.1f}% {r.get('protocol','cuda_graph'):<12s}")
