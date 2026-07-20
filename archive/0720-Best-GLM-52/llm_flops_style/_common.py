"""Drop-in llm_flops-style benches: same tensors for stock and archive candidates.

Supports flat PR#3 `.py` files under archive/0720-Best-GLM-52/ as well as
best/<campaign>/candidate/ directories.
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
from sgl_kernel import bmm_fp8
from sgl_kernel.flash_mla import flash_mla_sparse_fwd

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


def _load_py(path: Path):
    spec = importlib.util.spec_from_file_location(f"cand_{path.stem}_{id(path)}", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


candidate_loader = _load_harness("candidate_loader")

# llm_flops name -> (harness_op | None, archive_ref, kind)
# archive_ref: "best/<dir>" OR flat "<file>.py" under archive root
# kind: fp8_gemm | moe_masked | bmm | dsa | score_mqa | bf16_gemm | None
# After PR#5 vs ours CUPTI bake-off (see PR5_VS_OURS.md):
#   PR5 wins: fused_qkv_a, index_q_upproj, dsa
#   ours wins: q_b (DeepGEMM fork); moe_* tie → keep ours
DECODE_SWAPS: dict[str, tuple[str | None, str | None, str | None]] = {
    "fused_qkv_a_proj": ("fused_qkv_a", "best-hechenxi-0720/fused_qkv_a_decode", "fp8_gemm"),
    "q_b_proj": ("q_b", "best/q_b_decode", "fp8_gemm"),
    "absorbed_W_UK": (None, None, None),
    "absorbed_W_UV": (None, None, None),
    "o_proj": ("o_proj", "best/o_proj_decode_hbm35", "fp8_gemm"),
    "dsa_decode_attn": ("dsa_attn", "best-hechenxi-0720/dsa_decode_attn", "dsa"),
    "index_k_proj": ("index_k", "best/index_k_proj_decode", "fp8_gemm"),
    "index_q_upproj": ("index_q_upproj", "best-hechenxi-0720/index_q_upproj_decode", "fp8_gemm"),
    "index_weights_proj": (None, None, None),  # PR5 fused wk+weights needs different inputs
    "index_score": (None, None, None),
    "moe_gate_proj": ("moe_gate", "best/moe_gate_proj_decode_hbm40", "moe_masked"),
    "moe_up_proj": ("moe_up", "best/moe_up_proj_decode_hbm40", "moe_masked"),
    "moe_down_proj": ("moe_down", "best/moe_down_proj_decode_hbm40", "moe_masked"),
}

# Prefill: PR#3 flat files + prior best/ winners
PREFILL_SWAPS: dict[str, tuple[str | None, str | None, str | None]] = {
    "fused_qkv_a_proj": ("fused_qkv_a", "fused_qkv_a_prefill.py", "fp8_gemm"),
    "q_b_proj": ("q_b", "q_b_prefill.py", "fp8_gemm"),
    "absorbed_W_UK": ("absorbed_W_UK", "absorbed_W_UK_prefill.py", "bmm"),
    "absorbed_W_UV": ("absorbed_W_UV", "absorbed_W_UV_prefill.py", "bmm"),
    "o_proj": ("o_proj", "best/o_proj_prefill", "fp8_gemm"),
    "dsa_prefill_attn": ("dsa_attn", "dsa_prefill_attn.py", "dsa"),
    "index_k_proj": ("index_k", "best/index_k_prefill_bw70", "fp8_gemm"),
    "index_q_upproj": ("index_q_upproj", "index_q_upproj_prefill.py", "fp8_gemm"),
    "index_weights_proj": ("index_weights_proj", "index_weights_proj.py", "bf16_gemm"),
    "index_score": ("index_score", "index_score_prefill.py", "score_mqa"),
    # CUPTI pack+PDL wins (~1.06–1.19×); CUDA Graph drop-in regresses at M=4096
    # (~0.95×) — keep candidate in best/ but do not default-swap for layer tables.
    "moe_gate_proj": (None, None, None),
    "moe_up_proj": ("moe_up", "moe_up_proj_prefill.py", "moe_masked"),
    "moe_down_proj": ("moe_down", "moe_down_proj_prefill.py", "moe_masked"),
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


def _resolve_path(archive_ref: str) -> Path:
    """Campaign dir -> .../candidate ; flat foo.py -> archive/foo.py"""
    if archive_ref.startswith(("best/", "best-hechenxi-0720/")):
        p = _ARCHIVE / archive_ref / "candidate"
        if p.is_dir():
            return p
        cand = _ARCHIVE / archive_ref
        if cand.is_dir() and (cand / "candidate.py").is_file():
            return cand / "candidate.py"
        if (cand / "candidate").is_dir():
            return cand / "candidate"
        raise FileNotFoundError(_ARCHIVE / archive_ref / "candidate")
    p = _ARCHIVE / archive_ref
    if not p.is_file():
        raise FileNotFoundError(p)
    return p


def resolve_candidate(archive_ref: str, harness_op: str | None, phase: str):
    path = _resolve_path(archive_ref)

    # index_weights_proj is not in harness ALL_OPS — load flat + wrap.
    if harness_op == "index_weights_proj":
        mod = _load_py(path if path.is_file() else path / "candidate.py")
        if not hasattr(mod, "run"):
            raise AttributeError(path)

        def run(inputs: dict):
            return mod.run(inputs["x"], inputs["w"], inputs["out"])

        return run, str(path.relative_to(_REPO))

    task_dir = None
    if harness_op is not None:
        for d in _TASKS.iterdir():
            meta_path = d / "task.json"
            if not meta_path.is_file():
                continue
            meta = json.loads(meta_path.read_text())
            if meta["operator"] == harness_op and meta["phase"] == phase:
                task_dir = d
                break
    if task_dir is None:
        # Still load via candidate_loader override with a dummy task if needed —
        # prefer direct bind for flat files with run(inputs).
        mod = _load_py(path if path.is_file() else next(
            (path / n) for n in ("candidate.py", "solution.py", "impl.py")
            if (path / n).is_file()))
        if not hasattr(mod, "run"):
            raise AttributeError(path)
        return mod.run, str(path.relative_to(_REPO))

    run_fn, source, _ = candidate_loader.resolve(
        task_dir, harness_op, phase, override=str(path))
    if source == "reference":
        raise RuntimeError(f"candidate fell through to reference: {archive_ref}")
    return run_fn, source


# ── builders (llm_flops-identical) ────────────────────────────────────────────

def build_fp8_gemm_inputs(lf, rows: int, K: int, N: int, device, seed: int) -> dict:
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
    return dict(x_fp8=x_fp8, x_scale=x_scale, w_fp8=w_fp8, w_scale=w_scale,
                out=out, rows=rows, N=N, device=device)


def build_moe_masked_inputs(lf, M: int, K: int, N: int, device, seed: int) -> dict:
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
    while max(counts) > expected_m:
        e = counts.index(max(counts))
        counts[e] -= 1
        counts[counts.index(min(counts))] += 1
    masked_m = torch.tensor(counts, dtype=torch.int32, device=device)
    return dict(x_fp8=x_fp8, x_scale=x_scale, w_fp8=w_fp8, w_scale=w_scale,
                out=out, masked_m=masked_m, expected_m=expected_m)


def build_bmm_inputs(lf, M: int, K: int, N: int, device, seed: int) -> dict:
    B = lf.NUM_HEADS
    g = torch.Generator(device=device)
    g.manual_seed(seed)
    A_bf16 = torch.randn(B, M, K, dtype=torch.bfloat16, device=device, generator=g)
    B_bf16 = torch.randn(B, K, N, dtype=torch.bfloat16, device=device, generator=g)
    A_fp8, A_scale = lf.cast_to_fp8_per_tensor(A_bf16)
    B_fp8, B_scale = lf.cast_to_fp8_per_tensor(B_bf16)
    A_fp8 = A_fp8.view(B, M, K)
    B_fp8 = B_fp8.view(B, K, N)
    return dict(A_fp8=A_fp8, B_fp8=B_fp8, A_scale=A_scale, B_scale=B_scale)


def build_dsa_inputs(lf, M: int, S: int, device, seed: int) -> dict:
    g = torch.Generator(device=device)
    g.manual_seed(seed)
    q = torch.randn(M, lf.NUM_HEADS, lf.D_QK, dtype=torch.bfloat16, device=device, generator=g)
    kv = torch.randn(S, 1, lf.D_QK, dtype=torch.bfloat16, device=device, generator=g)
    topk = min(lf.TOPK, S)
    # deterministic indices from seeded CPU RNG
    rng = random.Random(seed)
    rows = []
    for _ in range(M):
        idx = list(range(S))
        rng.shuffle(idx)
        rows.append(idx[:topk])
    indices = torch.tensor(rows, dtype=torch.int32, device=device).view(M, 1, topk)
    return dict(q=q, kv=kv, indices=indices, sm_scale=lf.D_QK ** -0.5, d_v=lf.D_V)


def build_score_mqa_inputs(lf, M: int, S: int, device, seed: int) -> dict:
    BLOCK = 128
    g = torch.Generator(device=device)
    g.manual_seed(seed)
    q_bf16 = torch.randn(M, lf.INDEX_N_HEADS, lf.INDEX_HEAD_DIM,
                         dtype=torch.bfloat16, device=device, generator=g)
    q_view = q_bf16.view(M, lf.INDEX_N_HEADS, lf.INDEX_HEAD_DIM // BLOCK, BLOCK)
    q_amax = q_view.abs().float().amax(dim=-1)
    q_scale = (q_amax / torch.finfo(torch.float8_e4m3fn).max).float().clamp(min=1e-12)
    q_fp8 = (q_view.float() / q_scale.unsqueeze(-1)).to(torch.float8_e4m3fn).view(
        M, lf.INDEX_N_HEADS, lf.INDEX_HEAD_DIM)
    del q_bf16, q_view, q_amax, q_scale
    k_bf16 = torch.randn(S, lf.INDEX_HEAD_DIM, dtype=torch.bfloat16, device=device, generator=g)
    k_view = k_bf16.view(S, lf.INDEX_HEAD_DIM // BLOCK, BLOCK)
    k_amax = k_view.abs().float().amax(dim=-1)
    k_scale = (k_amax / torch.finfo(torch.float8_e4m3fn).max).float().clamp(min=1e-12)
    k_fp8 = (k_view.float() / k_scale.unsqueeze(-1)).to(torch.float8_e4m3fn).view(
        S, lf.INDEX_HEAD_DIM)
    k_scale = k_scale.squeeze(-1)
    del k_bf16, k_view, k_amax
    weights = torch.randn(M, lf.INDEX_N_HEADS, dtype=torch.float32, device=device, generator=g)
    ks = torch.zeros(M, dtype=torch.int32, device=device)
    ke = torch.full((M,), S, dtype=torch.int32, device=device)
    return dict(q_fp8=q_fp8, k_fp8=k_fp8, k_scale=k_scale, weights=weights, ks=ks, ke=ke)


def build_bf16_gemm_inputs(lf, M: int, K: int, N: int, device, seed: int) -> dict:
    g = torch.Generator(device=device)
    g.manual_seed(seed)
    x = torch.randn(M, K, dtype=torch.bfloat16, device=device, generator=g)
    w = torch.randn(N, K, dtype=torch.bfloat16, device=device, generator=g)
    out = torch.empty(M, N, dtype=torch.float32, device=device)
    return dict(x=x, w=w, out=out)


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


def stock_bmm(inputs: dict) -> None:
    bmm_fp8(inputs["A_fp8"], inputs["B_fp8"],
            inputs["A_scale"], inputs["B_scale"], torch.bfloat16)


def stock_dsa(inputs: dict) -> None:
    flash_mla_sparse_fwd(inputs["q"], inputs["kv"], inputs["indices"],
                         inputs["sm_scale"], inputs["d_v"])


def stock_score_mqa(inputs: dict) -> None:
    deep_gemm.fp8_mqa_logits(
        inputs["q_fp8"], (inputs["k_fp8"], inputs["k_scale"]), inputs["weights"],
        inputs["ks"], inputs["ke"], clean_logits=False)


def stock_bf16_gemm(inputs: dict) -> None:
    deep_gemm.bf16_gemm_nt(inputs["x"], inputs["w"], inputs["out"])


def bench_dropin(
    lf,
    kind: str,
    harness_op: str | None,
    phase: str,
    archive: str,
    dims: tuple,
    device,
    seed: int = 0,
    S: int | None = None,
) -> dict:
    if kind == "fp8_gemm":
        rows, K, N = dims
        inputs = build_fp8_gemm_inputs(lf, rows, K, N, device, seed)
        stock_fn = lambda: stock_fp8_gemm(inputs)  # noqa: E731
    elif kind == "moe_masked":
        M, K, N = dims
        inputs = build_moe_masked_inputs(lf, M, K, N, device, seed)
        stock_fn = lambda: stock_moe_masked(inputs)  # noqa: E731
    elif kind == "bmm":
        M, K, N = dims
        inputs = build_bmm_inputs(lf, M, K, N, device, seed)
        stock_fn = lambda: stock_bmm(inputs)  # noqa: E731
    elif kind == "dsa":
        M = dims[0]
        assert S is not None
        inputs = build_dsa_inputs(lf, M, S, device, seed)
        stock_fn = lambda: stock_dsa(inputs)  # noqa: E731
    elif kind == "score_mqa":
        M = dims[0]
        assert S is not None
        inputs = build_score_mqa_inputs(lf, M, S, device, seed)
        stock_fn = lambda: stock_score_mqa(inputs)  # noqa: E731
    elif kind == "bf16_gemm":
        M, K, N = dims
        inputs = build_bf16_gemm_inputs(lf, M, K, N, device, seed)
        stock_fn = lambda: stock_bf16_gemm(inputs)  # noqa: E731
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


def bmm_dims_for(lf, name: str, M: int) -> tuple[int, int, int]:
    if name == "absorbed_W_UK":
        return M, lf.QK_NOPE_HEAD_DIM, lf.KV_LORA_RANK
    if name == "absorbed_W_UV":
        return M, lf.KV_LORA_RANK, lf.V_HEAD_DIM
    raise KeyError(name)


def bf16_dims_for(lf, name: str, M: int) -> tuple[int, int, int]:
    if name == "index_weights_proj":
        return M, lf.HIDDEN_SIZE, lf.INDEX_N_HEADS
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
            print(f"  {'rank':<6s} {'name':<24s} {'impl':<36s} {'avg(ms)':>10s} "
                  f"{'stock(ms)':>10s} {'spd':>7s} {'pct':>7s} {'proto':<12s}")
            print(f"  {'-' * 125}")
            for rank, r in enumerate(subset, 1):
                pct = r["avg_ms"] / used_sum * 100 if used_sum else 0
                stock = r.get("stock_ms", r["avg_ms"])
                spd = stock / r["avg_ms"] if r["avg_ms"] else 0
                print(f"  {rank:<6d} {r['name']:<24s} {r.get('impl','stock'):<36s} "
                      f"{r['avg_ms']:>10.4f} {stock:>10.4f} {spd:>6.2f}x "
                      f"{pct:>6.1f}% {r.get('protocol','cuda_graph'):<12s}")
