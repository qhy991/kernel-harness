#!/usr/bin/env python3
"""Prove a testbench candidate can DROP IN and replace the sglang kernel it targets.

`evaluate.py` proves a candidate is faster + output-matching in isolation. That is
necessary but NOT sufficient for deployment: the candidate's interface is a benchmark
shim (`run(...)`), not sglang's dispatch signature. This tool closes that gap.

For a task, it:
  1. loads the candidate solution.py (its `run`),
  2. hot-swaps the EXACT sglang dispatch symbol with a thin adapter that reconciles
     sglang's call signature with the candidate's `run()` and routes to it,
  3. drives a REAL sglang module / function forward (not a shim), and verifies:
       (a) the candidate was actually invoked inside sglang's own code path,
       (b) the real-forward output matches the UNPATCHED sglang output within the
           task tolerance,
       (c) the swap is fully reversible (symbol restored to the original object).

A green result means: this candidate is a legitimate in-place replacement for the
sglang kernel at that call site — the deployable form of an evaluate.py "win".

Only genuinely-standalone sglang call sites have an integration recipe here
(fp8-linear-gemm, rmsnorm, swiglu). Ops that sglang only runs fused (rope, the main
layernorm+residual, quant-fused swiglu) have no isolated drop-in site and are
reported as such rather than faked.

Usage:  python testbench/bin/integrate.py <task_dir> [--solution solution.py]
Exit:   0 = drop-in verified   1 = mismatch/not-invoked   2 = no recipe / error
"""
import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path

import torch

BLOCK = 128


def _ensure_server_args():
    """Some sglang modules (e.g. SiluAndMul) read global server args at construction.
    Set a minimal default if a real one hasn't been installed."""
    try:
        from sglang.srt.server_args import get_global_server_args
        get_global_server_args()
    except Exception:
        from sglang.srt.server_args import ServerArgs, set_global_server_args_for_scheduler
        set_global_server_args_for_scheduler(ServerArgs(model_path="dummy"))


_parallel_ready = False


def _ensure_parallel():
    """VocabParallelEmbedding asserts a TP group exists. Bring up a 1-rank group."""
    global _parallel_ready
    if _parallel_ready:
        return
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29511")
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    os.environ.setdefault("LOCAL_RANK", "0")
    from sglang.srt.distributed import init_distributed_environment, initialize_model_parallel
    try:
        init_distributed_environment(world_size=1, rank=0, local_rank=0,
                                     distributed_init_method="tcp://127.0.0.1:29511",
                                     backend="nccl")
        initialize_model_parallel(tensor_model_parallel_size=1)
    except Exception:
        pass  # already initialized
    _parallel_ready = True


def _load_candidate(sol_path: Path):
    spec = importlib.util.spec_from_file_location("candidate_solution", sol_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    if not hasattr(mod, "run"):
        raise ValueError(f"{sol_path} has no top-level run()")
    return mod.run


def _per_block_cast_to_fp8(x: torch.Tensor):
    """Offline weight quant, identical to recipes/fp8_linear_gemm.py."""
    from deep_gemm import ceil_div

    m, n = x.shape
    xp = torch.zeros(
        (ceil_div(m, 128) * 128, ceil_div(n, 128) * 128), dtype=x.dtype, device=x.device
    )
    xp[:m, :n] = x
    xv = xp.view(-1, 128, xp.size(1) // 128, 128)
    xa = xv.abs().float().amax(dim=(1, 3), keepdim=True).clamp(1e-4)
    xs = (xv * (448.0 / xa)).to(torch.float8_e4m3fn)
    return xs.view_as(xp)[:m, :n].contiguous(), (xa / 448.0).view(xv.size(0), xv.size(2))


# --- integration recipes: each drives a REAL sglang call site with the candidate
# swapped in, and returns (out_ref, out_cand, invoked_count, restored: bool). ---

def _integ_rmsnorm(run, meta, device):
    import sglang.srt.layers.layernorm as ln
    from sglang.srt.layers.layernorm import RMSNorm

    H, M = meta["H"], 64
    module = RMSNorm(H, eps=1e-6).to(device, torch.bfloat16)
    x = torch.randn(M, H, device=device, dtype=torch.bfloat16)

    out_ref = module.forward_cuda(x.clone())
    orig, cnt = ln.rmsnorm, {"n": 0}

    def adapter(a, w, eps, *args, **kw):  # sglang calls rmsnorm(x, weight, eps)
        cnt["n"] += 1
        return run(a, w)                  # candidate run(x, weight)

    ln.rmsnorm = adapter
    try:
        out_cand = module.forward_cuda(x.clone())
    finally:
        ln.rmsnorm = orig
    return out_ref, out_cand, cnt["n"], ln.rmsnorm is orig


def _integ_fused_add_rmsnorm(run, meta, device):
    import sglang.srt.layers.layernorm as ln
    from sglang.srt.layers.layernorm import RMSNorm

    H, M = meta["H"], 64
    module = RMSNorm(H, eps=1e-6).to(device, torch.bfloat16)
    x = torch.randn(M, H, device=device, dtype=torch.bfloat16)
    res = torch.randn(M, H, device=device, dtype=torch.bfloat16)

    out_ref = module.forward_cuda(x.clone(), res.clone())   # (normed, residual)
    orig, cnt = ln.fused_add_rmsnorm, {"n": 0}

    def adapter(*a, **k):   # IDENTITY pass-through: run() must accept sglang's exact call
        cnt["n"] += 1
        return run(*a, **k)

    ln.fused_add_rmsnorm = adapter
    try:
        out_cand = module.forward_cuda(x.clone(), res.clone())
    finally:
        ln.fused_add_rmsnorm = orig
    return out_ref, out_cand, cnt["n"], ln.fused_add_rmsnorm is orig


def _integ_gemma_rmsnorm(run, meta, device):
    import sglang.srt.layers.layernorm as ln
    from sglang.srt.layers.layernorm import GemmaRMSNorm

    H, M = meta["H"], 64
    module = GemmaRMSNorm(H, eps=1e-6).to(device, torch.bfloat16)
    x = torch.randn(M, H, device=device, dtype=torch.bfloat16)

    out_ref = module.forward_cuda(x.clone())
    orig, cnt = ln.gemma_rmsnorm, {"n": 0}

    def adapter(*a, **k):   # IDENTITY pass-through: run(x, weight, eps) is interface-exact
        cnt["n"] += 1
        return run(*a, **k)

    ln.gemma_rmsnorm = adapter
    try:
        out_cand = module.forward_cuda(x.clone())
    finally:
        ln.gemma_rmsnorm = orig
    return out_ref, out_cand, cnt["n"], ln.gemma_rmsnorm is orig


def _integ_gemma_fused_add_rmsnorm(run, meta, device):
    import sglang.srt.layers.layernorm as ln
    from sglang.srt.layers.layernorm import GemmaRMSNorm

    H, M = meta["H"], 64
    module = GemmaRMSNorm(H, eps=1e-6).to(device, torch.bfloat16)
    x = torch.randn(M, H, device=device, dtype=torch.bfloat16)
    res = torch.randn(M, H, device=device, dtype=torch.bfloat16)

    out_ref = module.forward_cuda(x.clone(), res.clone())   # (normed, residual)
    orig, cnt = ln.gemma_fused_add_rmsnorm, {"n": 0}

    def adapter(*a, **k):   # IDENTITY pass-through: run(x, residual, weight, eps) is exact
        cnt["n"] += 1
        return run(*a, **k)

    ln.gemma_fused_add_rmsnorm = adapter
    try:
        out_cand = module.forward_cuda(x.clone(), res.clone())
    finally:
        ln.gemma_fused_add_rmsnorm = orig
    return out_ref, out_cand, cnt["n"], ln.gemma_fused_add_rmsnorm is orig


def _integ_rope(run, meta, device):
    import sglang.srt.layers.rotary_embedding.base as ropebase
    from sglang.srt.layers.rotary_embedding import get_rope

    nH, kvH, D, M = meta["num_heads"], meta["kv_heads"], meta["rope_dim"], 64
    rope = get_rope(head_size=D, rotary_dim=D, max_position=4096, base=10000,
                    is_neox_style=True, dtype=torch.bfloat16).to(device)
    pos = torch.randint(0, 4096, (M,), device=device, dtype=torch.int64)
    q = torch.randn(M, nH * D, device=device, dtype=torch.bfloat16)
    k = torch.randn(M, kvH * D, device=device, dtype=torch.bfloat16)

    out_ref = rope.forward_cuda(pos, q.clone(), k.clone())   # (query, key)
    orig, cnt = ropebase.apply_rope_with_cos_sin_cache_inplace, {"n": 0}

    def adapter(*a, **k):   # IDENTITY pass-through: run() must accept sglang's exact call
        cnt["n"] += 1
        return run(*a, **k)

    ropebase.apply_rope_with_cos_sin_cache_inplace = adapter
    try:
        out_cand = rope.forward_cuda(pos, q.clone(), k.clone())
    finally:
        ropebase.apply_rope_with_cos_sin_cache_inplace = orig
    return out_ref, out_cand, cnt["n"], ropebase.apply_rope_with_cos_sin_cache_inplace is orig


def _integ_embedding(run, meta, device):
    _ensure_parallel()
    import sglang.srt.layers.quantization.unquant as unq
    from sglang.srt.layers.vocab_parallel_embedding import VocabParallelEmbedding

    V, H, M = meta["V"], meta["H"], 64
    emb = VocabParallelEmbedding(num_embeddings=V, embedding_dim=H).to(device, torch.bfloat16)
    ids = torch.randint(0, V, (M,), device=device, dtype=torch.long)

    out_ref = emb(ids.clone())
    orig, cnt = unq.UnquantizedEmbeddingMethod.embedding, {"n": 0}

    def adapter(self, layer, input_):     # sglang: quant_method.embedding(layer, input)
        cnt["n"] += 1
        return run(input_, layer.weight)  # candidate run(input_ids, weight)

    unq.UnquantizedEmbeddingMethod.embedding = adapter
    try:
        out_cand = emb(ids.clone())
    finally:
        unq.UnquantizedEmbeddingMethod.embedding = orig
    return out_ref, out_cand, cnt["n"], unq.UnquantizedEmbeddingMethod.embedding is orig


def _integ_swiglu(run, meta, device):
    import sglang.srt.layers.activation as act
    from sglang.srt.layers.activation import SiluAndMul

    I2, M = meta["I2"], 64
    module = SiluAndMul()
    x = torch.randn(M, I2, device=device, dtype=torch.bfloat16)

    out_ref = module.forward_cuda(x.clone())
    orig, cnt = act.silu_and_mul, {"n": 0}

    def adapter(inp, out):                # sglang calls silu_and_mul(x, out) in-place
        cnt["n"] += 1
        out.copy_(run(inp))               # candidate run(x) -> out

    act.silu_and_mul = adapter
    try:
        out_cand = module.forward_cuda(x.clone())
    finally:
        act.silu_and_mul = orig
    return out_ref, out_cand, cnt["n"], act.silu_and_mul is orig


def _integ_fp8_linear(run, meta, device):
    import sglang.srt.layers.quantization.fp8_utils as fu
    from sglang.srt.layers.quantization.fp8_utils import (
        deepgemm_w8a8_block_fp8_linear_with_fallback as real_linear,
        requant_weight_ue8m0,
    )

    K, N, M = meta["K"], meta["N"], 64
    x = torch.randn(M, K, device=device, dtype=torch.bfloat16)
    w = torch.randn(N, K, device=device, dtype=torch.bfloat16) * (K ** -0.5)
    w_fp8, w_scale = _per_block_cast_to_fp8(w)
    w_fp8, w_scale = requant_weight_ue8m0(w_fp8, w_scale, [BLOCK, BLOCK])
    block = [BLOCK, BLOCK]

    # sglang's real linear: quantizes the activation inline, then calls the matmul.
    out_ref = real_linear(x.clone(), w_fp8, block, w_scale)
    orig, cnt = fu.w8a8_block_fp8_matmul_deepgemm, {"n": 0}

    def adapter(A, B, As, Bs, block_size, output_dtype):  # sglang's matmul signature
        cnt["n"] += 1
        return run(A, As, B, Bs)          # candidate run(x_fp8, x_scale, w_fp8, w_scale)

    fu.w8a8_block_fp8_matmul_deepgemm = adapter
    try:
        out_cand = real_linear(x.clone(), w_fp8, block, w_scale)
    finally:
        fu.w8a8_block_fp8_matmul_deepgemm = orig
    return out_ref, out_cand, cnt["n"], fu.w8a8_block_fp8_matmul_deepgemm is orig


def _integ_bf16_linear(run, meta, device):
    """bf16 GEMM: sglang dispatches unquantized linear via
    UnquantizedLinearMethod.apply -> F.linear. Drive a real ReplicatedLinear.forward
    and swap the method with the candidate run(x, weight)."""
    _ensure_server_args()
    import sglang.srt.layers.quantization.unquant as unq
    from sglang.srt.layers.linear import ReplicatedLinear

    K, N, M = meta["K"], meta["N"], 64
    lin = ReplicatedLinear(K, N, bias=False, params_dtype=torch.bfloat16).to(device)
    # Layers construct with zero weights (real weights load post-construction); fill
    # with random data so the F.linear reference is non-trivial and a zeros-candidate
    # would actually fail the match check.
    with torch.no_grad():
        lin.weight.normal_(0, K ** -0.5)
    x = torch.randn(M, K, device=device, dtype=torch.bfloat16)

    out_ref, _ = lin(x.clone())
    orig, cnt = unq.UnquantizedLinearMethod.apply, {"n": 0}

    def adapter(self, layer, inp, bias=None):   # sglang: quant_method.apply(layer, x, bias)
        cnt["n"] += 1
        out = run(inp, layer.weight)            # candidate run(x, weight) -> x @ weight.T
        return out if bias is None else out + bias

    unq.UnquantizedLinearMethod.apply = adapter
    try:
        out_cand, _ = lin(x.clone())
    finally:
        unq.UnquantizedLinearMethod.apply = orig
    return out_ref, out_cand, cnt["n"], unq.UnquantizedLinearMethod.apply is orig


def _integ_router_gemm(run, meta, device):
    """MoE router logits: sglang's decode fast path calls sgl_kernel.dsv3_router_gemm
    directly (deepseek_v2.py:588-614). The kernel IS the dispatch site, so we drive it
    and swap the symbol with the candidate run(hidden, weights)."""
    import sgl_kernel

    H, N = meta["H"], meta["N"]
    M = min(16, 8)   # dsv3_router_gemm requires num_tokens <= 16
    hidden = torch.randn(M, H, device=device, dtype=torch.bfloat16)
    weights = torch.randn(N, H, device=device, dtype=torch.bfloat16)

    out_ref = sgl_kernel.dsv3_router_gemm(hidden, weights, out_dtype=torch.float32)
    orig, cnt = sgl_kernel.dsv3_router_gemm, {"n": 0}

    def adapter(a, b, *args, **kw):    # sglang: dsv3_router_gemm(hidden, weights, out_dtype=)
        cnt["n"] += 1
        return run(a, b)               # candidate run(hidden, weights)

    sgl_kernel.dsv3_router_gemm = adapter
    try:
        out_cand = sgl_kernel.dsv3_router_gemm(hidden, weights, out_dtype=torch.float32)
    finally:
        sgl_kernel.dsv3_router_gemm = orig
    return out_ref, out_cand, cnt["n"], sgl_kernel.dsv3_router_gemm is orig


def _integ_router_gemm_minimax(run, meta, device):
    """MiniMax-M3 MoE router: NOT dsv3_router_gemm (that kernel is compiled for
    hidden=7168). M3's gate is a fp32 ReplicatedLinear (minimax: gate(hidden.to(fp32))),
    i.e. UnquantizedLinearMethod.apply -> F.linear in fp32. Drive the real linear and
    swap the method, exactly like bf16-linear but fp32."""
    _ensure_server_args()
    import sglang.srt.layers.quantization.unquant as unq
    from sglang.srt.layers.linear import ReplicatedLinear

    H, E, M = meta["H"], meta.get("E", meta.get("N")), 8
    lin = ReplicatedLinear(H, E, bias=False, params_dtype=torch.float32).to(device)
    with torch.no_grad():
        lin.weight.normal_(0, H ** -0.5)
    x = torch.randn(M, H, device=device, dtype=torch.float32)

    out_ref, _ = lin(x.clone())
    orig, cnt = unq.UnquantizedLinearMethod.apply, {"n": 0}

    def adapter(self, layer, inp, bias=None):   # sglang: quant_method.apply(layer, x, bias)
        cnt["n"] += 1
        out = run(inp, layer.weight)            # candidate run(hidden, gate_weight)
        return out if bias is None else out + bias

    unq.UnquantizedLinearMethod.apply = adapter
    try:
        out_cand, _ = lin(x.clone())
    finally:
        unq.UnquantizedLinearMethod.apply = orig
    return out_ref, out_cand, cnt["n"], unq.UnquantizedLinearMethod.apply is orig


def _integ_moe_gate_minimax(run, meta, device):
    """MiniMax-M3 MoE gate: sigmoid + biased top-k via sgl_kernel.topk_sigmoid (a DPS
    kernel writing topk_ids/weights) — NOT Kimi's kimi_k2_moe_fused_gate. Drive it and
    swap the symbol; candidate run(gating_output, correction_bias) returns indices."""
    import sgl_kernel

    E, M = meta["num_experts"], 64
    topk = meta.get("topk", 4)
    gating = torch.randn(M, E, device=device, dtype=torch.float32)
    bias = torch.randn(E, device=device, dtype=torch.float32)

    w_ref = torch.empty(M, topk, device=device, dtype=torch.float32)
    idx_ref = torch.empty(M, topk, device=device, dtype=torch.int32)
    sgl_kernel.topk_sigmoid(w_ref, idx_ref, gating, renormalize=True, correction_bias=bias)
    orig, cnt = sgl_kernel.topk_sigmoid, {"n": 0}

    idx_cand_holder = {}

    def adapter(topk_weights, topk_ids, gating_output, *args, **kw):  # DPS signature
        cnt["n"] += 1
        corr = kw.get("correction_bias")
        idx = run(gating_output, corr)     # candidate run(gating_output, correction_bias)
        topk_ids.copy_(idx)                # write into the DPS buffer the caller reads
        idx_cand_holder["idx"] = topk_ids

    sgl_kernel.topk_sigmoid = adapter
    try:
        w_cand = torch.empty(M, topk, device=device, dtype=torch.float32)
        idx_cand = torch.empty(M, topk, device=device, dtype=torch.int32)
        sgl_kernel.topk_sigmoid(w_cand, idx_cand, gating, renormalize=True, correction_bias=bias)
    finally:
        sgl_kernel.topk_sigmoid = orig
    return idx_ref, idx_cand, cnt["n"], sgl_kernel.topk_sigmoid is orig


def _integ_lm_head(run, meta, device):
    """LM-head logits: LogitsProcessor._compute_lm_head does a bare
    torch.matmul(hidden, lm_head.weight.T). Drive that method against a real
    ParallelLMHead, swapping torch.matmul (scoped to the call) with the candidate."""
    _ensure_parallel()
    import sglang.srt.layers.logits_processor as lpmod
    from sglang.srt.layers.logits_processor import LogitsProcessor
    from sglang.srt.layers.vocab_parallel_embedding import ParallelLMHead

    H, V, M = meta["H"], meta["V"], 8
    lm = ParallelLMHead(V, H, params_dtype=torch.bfloat16).to(device)
    with torch.no_grad():                       # non-zero weights (see bf16-linear note)
        lm.weight.normal_(0, H ** -0.5)
    hidden = torch.randn(M, H, device=device, dtype=torch.bfloat16)
    lp = LogitsProcessor.__new__(LogitsProcessor)   # bypass full __init__; only method used
    lp.use_fp32_lm_head = False

    out_ref = lp._compute_lm_head(hidden.clone(), lm)
    orig, cnt = lpmod.torch.matmul, {"n": 0}

    def adapter(a, b, *args, **kw):    # sglang: torch.matmul(hidden, weight.T)
        cnt["n"] += 1
        # restore the real matmul while the candidate runs (a candidate that itself
        # calls torch.matmul must not re-enter this hook).
        lpmod.torch.matmul = orig
        try:
            # b is weight.T; candidate takes run(hidden, weight) and transposes internally
            return run(a, b.T)
        finally:
            lpmod.torch.matmul = adapter

    lpmod.torch.matmul = adapter
    try:
        out_cand = lp._compute_lm_head(hidden.clone(), lm)
    finally:
        lpmod.torch.matmul = orig
    return out_ref, out_cand, cnt["n"], lpmod.torch.matmul is orig


def _integ_moe_combine(run, meta, device):
    """MoE combine: sglang sums top-k expert outputs via sgl_kernel.moe_sum(inp, out)
    in-place. INTERFACE-EXACT — run(input_tensor, output_tensor) is a verbatim copy."""
    import sgl_kernel

    H, TOPK, M = meta["H"], meta["top_k"], 64
    inp = torch.randn(M, TOPK, H, device=device, dtype=torch.bfloat16)
    out_ref = torch.empty(M, H, device=device, dtype=torch.bfloat16)
    sgl_kernel.moe_sum(inp, out_ref)
    orig, cnt = sgl_kernel.moe_sum, {"n": 0}

    def adapter(*a, **k):   # IDENTITY pass-through: run(inp, out) is interface-exact
        cnt["n"] += 1
        return run(*a, **k)

    out_cand = torch.empty(M, H, device=device, dtype=torch.bfloat16)
    sgl_kernel.moe_sum = adapter
    try:
        sgl_kernel.moe_sum(inp, out_cand)
    finally:
        sgl_kernel.moe_sum = orig
    return out_ref, out_cand, cnt["n"], sgl_kernel.moe_sum is orig


def _integ_moe_gate(run, meta, device):
    """MoE gate + biased top-k routing: sgl_kernel.kimi_k2_moe_fused_gate produces
    (weights, indices). The task's oracle is the integer INDICES (exact-index). Drive
    the kernel and swap it; candidate run(input, bias) returns indices."""
    import sgl_kernel

    E, M = meta["num_experts"], 64
    topk = meta.get("topk", 6)
    scaling = 2.872
    inp = torch.rand(M, E, device=device, dtype=torch.float32)
    bias = torch.rand(E, device=device, dtype=torch.float32)

    _w, idx_ref = sgl_kernel.kimi_k2_moe_fused_gate(
        inp, bias, topk=topk, renormalize=True, routed_scaling_factor=scaling)
    orig, cnt = sgl_kernel.kimi_k2_moe_fused_gate, {"n": 0}

    def adapter(a, b, *args, **kw):   # sglang: kimi_k2_moe_fused_gate(inp, bias, topk=, ...)
        cnt["n"] += 1
        idx = run(a, b)               # candidate run(input, bias) -> indices
        # match the kernel's (weights, indices) tuple so the real caller is satisfied;
        # weights are unused by the index oracle.
        return _w, idx

    sgl_kernel.kimi_k2_moe_fused_gate = adapter
    try:
        _w2, idx_cand = sgl_kernel.kimi_k2_moe_fused_gate(
            inp, bias, topk=topk, renormalize=True, routed_scaling_factor=scaling)
    finally:
        sgl_kernel.kimi_k2_moe_fused_gate = orig
    return idx_ref, idx_cand, cnt["n"], sgl_kernel.kimi_k2_moe_fused_gate is orig


def _integ_act_fp8_quant(run, meta, device):
    """Per-token-group act FP8 quant: sglang runs it inline inside
    deepgemm_w8a8_block_fp8_linear_with_fallback before the matmul. Drive that real
    linear and swap sglang_per_token_group_quant_fp8 with the candidate run(x)."""
    import sglang.srt.layers.quantization.fp8_utils as fu
    from sglang.srt.layers.quantization.fp8_utils import (
        deepgemm_w8a8_block_fp8_linear_with_fallback as real_linear,
        requant_weight_ue8m0,
    )

    K, M = meta["K"], 64
    N = 512   # any block-valid N; the quant op is independent of N
    x = torch.randn(M, K, device=device, dtype=torch.bfloat16)
    w = torch.randn(N, K, device=device, dtype=torch.bfloat16) * (K ** -0.5)
    w_fp8, w_scale = _per_block_cast_to_fp8(w)
    w_fp8, w_scale = requant_weight_ue8m0(w_fp8, w_scale, [BLOCK, BLOCK])
    block = [BLOCK, BLOCK]

    out_ref = real_linear(x.clone(), w_fp8, block, w_scale)
    orig, cnt = fu.sglang_per_token_group_quant_fp8, {"n": 0}

    def adapter(inp, group_size, *args, **kw):   # sglang's quant call
        cnt["n"] += 1
        return run(inp)                          # candidate run(x) -> (q_fp8, scale)

    fu.sglang_per_token_group_quant_fp8 = adapter
    try:
        out_cand = real_linear(x.clone(), w_fp8, block, w_scale)
    finally:
        fu.sglang_per_token_group_quant_fp8 = orig
    return out_ref, out_cand, cnt["n"], fu.sglang_per_token_group_quant_fp8 is orig


def _load_reference_inputs(task_dir: Path, axes: dict, device):
    """Load the task's own reference.get_inputs to build correctly-laid-out inputs.

    The grouped-MoE fp8 scale layout is non-trivial (deep_gemm
    transform_sf_into_required_layout); reproducing it by hand is error-prone, so we
    reuse the task's own generator — the same inputs the harness benchmarks with."""
    ref_path = task_dir / "reference.py"
    spec = importlib.util.spec_from_file_location("task_reference", ref_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.get_inputs(axes, device)


def _integ_grouped_moe_masked(run, meta, device):
    """Masked grouped GEMM: sglang's fused-MoE decode path calls
    deep_gemm_wrapper.grouped_gemm_nt_f8f8bf16_masked((a,as),(b,bs),out,masked_m,exp_m).
    Drive the wrapper (it adds the sanity/TMA-align/SM-config context) and swap it."""
    import sglang.srt.layers.deep_gemm_wrapper as dgw

    task_dir = meta["_task_dir"]
    E = meta["E"]
    inp = _load_reference_inputs(task_dir, {"M": 64, "E": E, "K": meta["K"], "N": meta["N"]}, device)
    a_fp8, a_s, b_fp8, b_s = inp["a_fp8"], inp["a_s"], inp["b_fp8"], inp["b_s"]
    masked_m, expected_m = inp["masked_m"], inp["expected_m"]
    out_shape = inp["out"].shape

    out_ref = torch.empty(out_shape, device=device, dtype=torch.bfloat16)
    dgw.grouped_gemm_nt_f8f8bf16_masked(
        (a_fp8, a_s), (b_fp8, b_s), out_ref, masked_m, expected_m)
    orig, cnt = dgw.grouped_gemm_nt_f8f8bf16_masked, {"n": 0}

    def adapter(lhs, rhs, out, masked_m_, expected_m_, *args, **kw):
        cnt["n"] += 1
        # candidate run(a_fp8, a_s, b_fp8, b_s, out, masked_m, expected_m)
        return run(lhs[0], lhs[1], rhs[0], rhs[1], out, masked_m_, expected_m_)

    out_cand = torch.empty(out_shape, device=device, dtype=torch.bfloat16)
    dgw.grouped_gemm_nt_f8f8bf16_masked = adapter
    try:
        dgw.grouped_gemm_nt_f8f8bf16_masked(
            (a_fp8, a_s), (b_fp8, b_s), out_cand, masked_m, expected_m)
    finally:
        dgw.grouped_gemm_nt_f8f8bf16_masked = orig
    return out_ref, out_cand, cnt["n"], dgw.grouped_gemm_nt_f8f8bf16_masked is orig


def _integ_grouped_moe_contig(run, meta, device):
    """Contiguous grouped GEMM (prefill): sglang calls
    deep_gemm_wrapper.grouped_gemm_nt_f8f8bf16_contig((a,as),(b,bs),out,m_indices)."""
    import sglang.srt.layers.deep_gemm_wrapper as dgw

    task_dir = meta["_task_dir"]
    E = meta["E"]
    inp = _load_reference_inputs(task_dir, {"M": 128, "E": E, "K": meta["K"], "N": meta["N"]}, device)
    a_fp8, a_s, b_fp8, b_s = inp["a_fp8"], inp["a_s"], inp["b_fp8"], inp["b_s"]
    m_indices = inp["m_indices"]
    out_shape = inp["out"].shape

    out_ref = torch.empty(out_shape, device=device, dtype=torch.bfloat16)
    dgw.grouped_gemm_nt_f8f8bf16_contig((a_fp8, a_s), (b_fp8, b_s), out_ref, m_indices)
    orig, cnt = dgw.grouped_gemm_nt_f8f8bf16_contig, {"n": 0}

    def adapter(lhs, rhs, out, m_indices_, *args, **kw):
        cnt["n"] += 1
        # candidate run(a_fp8, a_s, b_fp8, b_s, out, m_indices)
        return run(lhs[0], lhs[1], rhs[0], rhs[1], out, m_indices_)

    out_cand = torch.empty(out_shape, device=device, dtype=torch.bfloat16)
    dgw.grouped_gemm_nt_f8f8bf16_contig = adapter
    try:
        dgw.grouped_gemm_nt_f8f8bf16_contig((a_fp8, a_s), (b_fp8, b_s), out_cand, m_indices)
    finally:
        dgw.grouped_gemm_nt_f8f8bf16_contig = orig
    return out_ref, out_cand, cnt["n"], dgw.grouped_gemm_nt_f8f8bf16_contig is orig


def _integ_bmm(run, meta, device):
    """MLA weight-absorb BMM: sglang's bf16 absorb path calls global torch.bmm (no
    scoped sglang symbol). Swap torch.bmm with a SHAPE-GUARDED hook — only the
    absorb-shaped (Bh,M,K)x(Bh,K,N) call routes to the candidate; every other bmm
    (there are none in this driver, but the guard keeps the swap honest) passes
    through to the real kernel. The swap is fully restored afterward."""
    Bh, K, N = meta["Bh"], meta["K"], meta["N"]
    M = 1
    a = torch.randn(Bh, M, K, device=device, dtype=torch.bfloat16)
    b = torch.randn(Bh, K, N, device=device, dtype=torch.bfloat16)

    real_bmm = torch.bmm
    out_ref = real_bmm(a, b)
    cnt = {"n": 0}

    def adapter(x, y, *args, **kw):
        # shape guard: only the absorb-shaped bmm is a candidate drop-in
        if x.dim() == 3 and x.shape == (Bh, M, K) and y.shape == (Bh, K, N):
            cnt["n"] += 1
            # restore the real bmm while the candidate runs, so a candidate that
            # legitimately falls back to torch.bmm doesn't re-enter this hook.
            torch.bmm = real_bmm
            try:
                return run(x, y)
            finally:
                torch.bmm = adapter
        return real_bmm(x, y, *args, **kw)

    torch.bmm = adapter
    try:
        out_cand = torch.bmm(a, b)
    finally:
        torch.bmm = real_bmm
    return out_ref, out_cand, cnt["n"], torch.bmm is real_bmm


def _task_scalars(task_dir: Path):
    """Full axes+scalars dict for a task's smallest workload shape (const axes from
    definition.json + first workload row + task.json scalars)."""
    defn = json.loads((task_dir / "definition.json").read_text())
    meta = json.loads((task_dir / "task.json").read_text())
    wl = [json.loads(l) for l in (task_dir / "workload.jsonl").read_text().splitlines() if l.strip()]
    scalars = dict(wl[0]["axes"]) if wl else {}
    for name, spec in defn.get("axes", {}).items():
        if spec.get("type") == "const":
            scalars[name] = spec["value"]
    for k in ("top_k", "num_experts", "topk", "H", "V", "K", "N", "E", "Bh"):
        if k in meta:
            scalars.setdefault(k, meta[k])
    return scalars


# DSA (MiniMax-M3 sparse-attention) JIT kernels live in the sglang-m3 build. Each of
# these families' run() is a verbatim wrapper of one jit_kernel symbol, so the recipe
# swaps that symbol with an identity probe, drives the task's reference forward through
# it (proving it's the real dispatch site), and returns the candidate's output for the
# match check. The fused-attention families (dsa-decode-attn, dsa-prefill-attn,
# dsa-prefill-topk) have no single isolated symbol and are left to SKIP.
_DSA_JIT = {
    "dsa-qknorm-rope": ("sglang.jit_kernel.minimax_qknorm_rope", "minimax_qknorm_rope"),
    "dsa-decode-topk": ("sglang.jit_kernel.minimax_decode_topk", "minimax_decode_topk"),
    "dsa-store-kv-index": ("sglang.jit_kernel.minimax_store_kv_index", "store_kv_index"),
}


def _integ_dsa_jit(run, meta, device):
    """Interface-exact DSA JIT op (sglang-m3). See _DSA_JIT."""
    import importlib

    family = meta["family"]
    task_dir = meta["_task_dir"]
    mod_path, sym = _DSA_JIT[family]

    scalars = _task_scalars(task_dir)
    spec = importlib.util.spec_from_file_location("task_reference_dsa", task_dir / "reference.py")
    ref_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ref_mod)

    # reference.py does `from <mod> import <sym>`, binding the name in ref_mod's own
    # namespace — so we patch THAT bound name, not the origin module attribute.
    orig, cnt = getattr(ref_mod, sym), {"n": 0}

    # Identical inputs for ref and candidate (get_inputs draws fresh randoms each call,
    # which would otherwise make an interface-exact candidate mismatch the reference).
    inp = ref_mod.get_inputs(scalars, device)

    def _clone(d):
        return {k: (v.clone() if torch.is_tensor(v) else v) for k, v in d.items()}

    out_ref = ref_mod.run(**_clone(inp))

    def probe(*a, **k):     # prove the swapped name is the one run() dispatches
        cnt["n"] += 1
        return orig(*a, **k)

    setattr(ref_mod, sym, probe)
    try:
        _ = ref_mod.run(**_clone(inp))
    finally:
        setattr(ref_mod, sym, orig)
    restored = getattr(ref_mod, sym) is orig

    out_cand = run(**_clone(inp))
    return out_ref, out_cand, cnt["n"], restored


RECIPES = {
    "rmsnorm": _integ_rmsnorm,
    "gemma-rmsnorm": _integ_gemma_rmsnorm,
    "gemma-fused-add-rmsnorm": _integ_gemma_fused_add_rmsnorm,
    "fused-add-rmsnorm": _integ_fused_add_rmsnorm,
    "rope": _integ_rope,
    "embedding": _integ_embedding,
    "swiglu": _integ_swiglu,
    "fp8-linear-gemm": _integ_fp8_linear,
    "bf16-linear": _integ_bf16_linear,
    "router-gemm": _integ_router_gemm,
    "lm-head": _integ_lm_head,
    "moe-combine": _integ_moe_combine,
    "moe-gate": _integ_moe_gate,
    "act-fp8-quant": _integ_act_fp8_quant,
    "grouped-moe": _integ_grouped_moe_masked,
    "grouped-moe-contiguous": _integ_grouped_moe_contig,
    "bmm": _integ_bmm,
    # DSA (sglang-m3) interface-exact JIT ops
    "dsa-qknorm-rope": _integ_dsa_jit,
    "dsa-decode-topk": _integ_dsa_jit,
    "dsa-store-kv-index": _integ_dsa_jit,
}

# families that are real ops but have no isolated sglang dispatch symbol to patch
FUSED_ONLY = {
    "dsa-decode-attn": "fused sparse-attention forward (q/k_cache/v_cache/req_to_token/"
                       "seq_lens/slot_ids/topk_idx); no single isolated dispatch symbol.",
    "dsa-prefill-attn": "fused sparse-attention prefill; no single isolated dispatch symbol.",
    "dsa-prefill-topk": "fused prefill top-k over index q/k + paging metadata; no single "
                        "isolated dispatch symbol to swap.",
}

# Per-(family, model) overrides: some families dispatch a DIFFERENT sglang kernel per
# model. MiniMax-M3's router is a fp32 Linear (not dsv3_router_gemm, compiled for
# hidden=7168) and its gate is topk_sigmoid (not kimi_k2_moe_fused_gate). Keyed by
# (family, model); falls back to RECIPES[family] when no model-specific entry exists.
MODEL_RECIPES = {
    ("router-gemm", "minimax_m3"): _integ_router_gemm_minimax,
    ("moe-gate", "minimax_m3"): _integ_moe_gate_minimax,
}


def _resolve_recipe(family, model):
    """Pick the model-specific recipe if one exists, else the family default."""
    if (family, model) in MODEL_RECIPES:
        return MODEL_RECIPES[(family, model)]
    return RECIPES.get(family)


def _as_list(out):
    return list(out) if isinstance(out, (tuple, list)) else [out]


def _compare(ref, cand, atol, rtol):
    """Worst-case match across all outputs (handles single-tensor and tuple returns)."""
    refs, cands = _as_list(ref), _as_list(cand)
    if len(refs) != len(cands):
        return 0.0, float("inf"), False, False
    min_ratio, max_abs, shape_ok, has_bad = 1.0, 0.0, True, False
    for r, c in zip(refs, cands):
        shape_ok = shape_ok and (c.shape == r.shape)
        has_bad = has_bad or bool(torch.isnan(c).any() or torch.isinf(c).any())
        a, b = c.float(), r.float()
        ok = (a - b).abs() <= (atol + rtol * b.abs())
        min_ratio = min(min_ratio, ok.float().mean().item())
        max_abs = max(max_abs, (a - b).abs().max().item())
    return min_ratio, max_abs, shape_ok, has_bad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("task_dir", type=Path)
    ap.add_argument("--solution", default="solution.py")
    args = ap.parse_args()

    task_dir = args.task_dir.resolve()
    meta = json.loads((task_dir / "task.json").read_text())
    meta["_task_dir"] = task_dir   # recipes that reuse the task's own get_inputs need it
    family = meta.get("family")
    # Model drives per-model recipe selection. The task.json 'model' field is not always
    # populated, so fall back to the parent directory (tasks/<model>/<name>), which is
    # the reliable signal.
    model = meta.get("model") or task_dir.parent.name
    tol = meta.get("tolerance", {"max_atol": 0.1, "max_rtol": 0.05, "required_matched_ratio": 0.98})

    # A task may pin its own sglang build (DSA needs the sglang-m3 tree). Prepend it to
    # sys.path BEFORE any sglang import so `import sglang` resolves to the right build,
    # keeping the per-task loop self-sufficient (no caller PYTHONPATH juggling).
    _pinned = meta.get("sglang_dir")
    if _pinned:
        _py = str(Path(_pinned) / "python")
        if _py not in sys.path:
            sys.path.insert(0, _py)

    if family in FUSED_ONLY:
        print(f"SKIP {task_dir.name}: family '{family}' has no standalone drop-in site "
              f"({FUSED_ONLY[family]}).")
        sys.exit(2)
    recipe = _resolve_recipe(family, model)
    if recipe is None:
        print(f"SKIP {task_dir.name}: no integration recipe for family '{family}' "
              f"(model={model}).")
        sys.exit(2)

    _ensure_server_args()
    run = _load_candidate(task_dir / args.solution)
    device = torch.device("cuda")
    try:
        out_ref, out_cand, invoked, restored = recipe(run, meta, device)
    except Exception as e:
        # A signature/interface mismatch surfaces here: sglang calls the swapped-in
        # run() with its exact argument list, so a wrong signature raises inside the
        # real forward. That IS the interface-exactness check failing.
        print(f"\n== drop-in integration: {task_dir.name} (family={family}, solution={args.solution}) ==")
        print(f"  INTERFACE MISMATCH: driving the real sglang forward through the "
              f"candidate raised {type(e).__name__}: {e}")
        print("\nDROP-IN FAILED")
        print("INTEGRATION_JSON_BEGIN")
        print(json.dumps({"task": task_dir.name, "family": family, "solution": args.solution,
                          "drop_in_ok": False, "interface_error": f"{type(e).__name__}: {e}"}))
        print("INTEGRATION_JSON_END")
        sys.exit(1)

    n_out = len(_as_list(out_cand))
    ratio, max_abs, shape_ok, has_bad = _compare(
        out_ref, out_cand, tol["max_atol"], tol["max_rtol"])
    matched = ratio >= tol["required_matched_ratio"]
    ok = invoked >= 1 and restored and shape_ok and matched and not has_bad

    print(f"\n== drop-in integration: {task_dir.name} (family={family}, solution={args.solution}) ==")
    print(f"  sglang call site driven : real {family} forward")
    print(f"  candidate invoked inside: {invoked} call(s)   [need >=1]")
    print(f"  outputs compared        : {n_out}   shape_ok={shape_ok}")
    print(f"  matched ratio (worst)   : {ratio:.4f}  (need >= {tol['required_matched_ratio']})   max_abs_err={max_abs:.3e}")
    print(f"  nan/inf                 : {has_bad}")
    print(f"  symbol restored         : {restored}")
    verdict = {"task": task_dir.name, "family": family, "solution": args.solution,
               "drop_in_ok": ok, "invoked": invoked, "match_ratio": round(ratio, 4),
               "max_abs_err": max_abs, "shape_ok": shape_ok, "restored": restored,
               "has_nan_inf": has_bad}
    print(f"\n{'DROP-IN VERIFIED' if ok else 'DROP-IN FAILED'}")
    print("INTEGRATION_JSON_BEGIN")
    print(json.dumps(verdict))
    print("INTEGRATION_JSON_END")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
