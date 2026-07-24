"""GLM-5.2 sglang gfx942 compatibility shim.

Every patch below is REQUIRED for sglang GLM-5.2 to boot on MI300X (gfx942)
under bf16 KV + aiter production dispatch. Extracted from the proven-working
`sglang-exp/tasks/dense-fp8-gemm/runs/run_glm52_no_offload.py`.

Import this module BEFORE `sglang.bench_one_batch` is loaded, so its
sys.modules registration + monkeypatches land in the master interpreter and
propagate to TP worker forks via PYTHONPATH.

The seven patches:
  1. sys.modules['fast_hadamard_transform'] fake (module missing on ROCm)
  2. dsa_indexer.rotate_activation -> torch Hadamard
  3. DeepseekMLAForwardMixin.forward_absorb_prepare -> device fix
  4. tilelang_kernel.act_quant -> pure torch fp8 quant
  5. dsa_indexer.Indexer._store_index_k_cache -> bypass gfx942 aiter memory fault
  6. sgl_kernel.rotary_embedding -> pure torch rotary
  7. moe_align_block_size -> graph-friendly vectorised version

None of these change semantics — every replacement is a pure-torch equivalent
of a kernel that either doesn't exist on ROCm or crashes on gfx942. They must
be applied together; skipping any produces a boot failure or a runtime crash.

Do not add operator-optimisation overrides here — this shim is only about
making sglang bootable on gfx942. User-supplied optimisation overrides
belong in `operator_overrides.py`.
"""
from __future__ import annotations

import os
import sys
import types

# Environment defaults every gfx942 sglang boot needs; harmless if already set.
os.environ.setdefault("PYTORCH_ROCM_ARCH", "gfx942")
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")
os.environ.setdefault("SGLANG_DSA_FUSE_TOPK", "0")
os.environ.setdefault("SGLANG_OPT_USE_AITER_SILU_MUL", "1")
os.environ.setdefault("SGLANG_USE_AITER", "1")
os.environ.setdefault("SGLANG_DISABLE_GFX942_BPRESHUFFLE", "1")
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

# Make aiter + sglang importable if the shared checkouts are in the standard spot.
for _cand in ("/root/repos/aiter", "/root/repos/sglang/python"):
    if os.path.isdir(_cand) and _cand not in sys.path:
        sys.path.insert(0, _cand)

import torch  # noqa: E402


# ── Patch 1: fast_hadamard_transform (pure torch) ────────────────────────────
def _hadamard_transform_pytorch(x, scale=1.0):
    orig_shape = x.shape
    n = x.shape[-1]
    x = x.reshape(-1, n)
    h = 1
    while h < n:
        x = x.view(-1, n // (2 * h), 2, h)
        a = x[:, :, 0, :]
        b = x[:, :, 1, :]
        x = torch.stack([a + b, a - b], dim=2)
        x = x.view(-1, n)
        h *= 2
    return (x * scale).reshape(orig_shape)


import importlib.util  # noqa: E402

_fake_fht = types.ModuleType("fast_hadamard_transform")
# Python 3.11's import machinery inspects __spec__ on every sys.modules entry
# during startup; a bare types.ModuleType has __spec__ = None and triggers a
# ValueError from sitecustomize's early import chain. Give the fake a real spec.
_fake_fht.__spec__ = importlib.util.spec_from_loader("fast_hadamard_transform", loader=None)
_fake_fht.hadamard_transform = _hadamard_transform_pytorch
sys.modules["fast_hadamard_transform"] = _fake_fht


def _apply_sglang_patches() -> None:
    """Second-phase patches — need sglang / sgl_kernel already importable."""
    import sglang.srt.layers.attention.dsa.dsa_indexer as dsa_indexer

    # Patch 2: dsa_indexer.rotate_activation → torch Hadamard
    def _patched_rotate_activation(x):
        hidden_size = x.size(-1)
        assert (hidden_size & (hidden_size - 1)) == 0
        return _hadamard_transform_pytorch(x, scale=hidden_size ** -0.5)

    dsa_indexer.rotate_activation = _patched_rotate_activation

    # Patch 3: MLA forward_absorb_prepare — device fix for w_kc / w_vc / scales
    import sglang.srt.models.deepseek_common.attention_forward_methods.forward_mla as forward_mla

    _orig_forward_absorb_prepare = forward_mla.DeepseekMLAForwardMixin.forward_absorb_prepare

    def _patched_forward_absorb_prepare(self, *args, **kwargs):
        target_device = torch.device("cuda")
        if args and isinstance(args[0], torch.Tensor):
            target_device = args[0].device
        for attr in ("w_kc", "w_vc", "w_scale", "w_scale_k", "w_scale_v"):
            t = getattr(self, attr, None)
            if t is not None and isinstance(t, torch.Tensor) and t.device != target_device:
                setattr(self, attr, t.to(target_device, non_blocking=True))
        return _orig_forward_absorb_prepare(self, *args, **kwargs)

    forward_mla.DeepseekMLAForwardMixin.forward_absorb_prepare = _patched_forward_absorb_prepare

    # Patch 4: tilelang act_quant → pure torch (per-block FP8 quant)
    import sglang.srt.layers.attention.dsa.tilelang_kernel as tilelang_kernel

    def _act_quant_pytorch(x, block_size=128, scale_fmt=None):
        assert x.is_contiguous()
        N = x.size(-1)
        assert N % block_size == 0
        FP8_MAX_VAL = 224.0
        fp8_dtype = torch.float8_e4m3fnuz
        orig_shape = x.shape
        x_flat = x.view(-1, N)
        M = x_flat.shape[0]
        K_blocks = N // block_size
        x_blocks = x_flat.reshape(M, K_blocks, block_size)
        block_max = x_blocks.abs().amax(dim=-1).clamp(min=1e-12)
        scales = block_max / FP8_MAX_VAL
        x_scaled = x_blocks / scales.unsqueeze(-1)
        x_fp8 = x_scaled.reshape(M, N).clamp(-FP8_MAX_VAL, FP8_MAX_VAL).to(fp8_dtype)
        scale_inv = scales.to(torch.float32)
        return x_fp8.view(orig_shape), scale_inv.view(*orig_shape[:-1], K_blocks)

    tilelang_kernel.act_quant = _act_quant_pytorch

    # Patch 5: bypass AITER indexer_k_quant_and_cache (crashes on gfx942 with memory fault)
    _orig_store_index_k = dsa_indexer.Indexer._store_index_k_cache

    def _patched_store_index_k_cache(self, forward_batch, layer_id, key, *,
                                     act_quant=None, out_cache_loc=None):
        from sglang.srt.model_executor.forward_context import get_token_to_kv_pool
        if out_cache_loc is None:
            out_cache_loc = forward_batch.out_cache_loc
        assert act_quant is not None, "act_quant required for gfx942 fallback path"
        k_fp8, k_scale = act_quant(key, self.block_size, self.scale_fmt)
        if not out_cache_loc.is_contiguous():
            out_cache_loc = out_cache_loc.contiguous()
        get_token_to_kv_pool().set_index_k_scale_buffer(
            layer_id=layer_id, loc=out_cache_loc, index_k=k_fp8, index_k_scale=k_scale,
        )

    dsa_indexer.Indexer._store_index_k_cache = _patched_store_index_k_cache

    # Patch 6: sgl_kernel.rotary_embedding → pure torch
    def _rotary_embedding_pytorch(positions, query, key, head_size, cos_sin_cache, is_neox=True):
        cos_sin = cos_sin_cache[positions]
        rot_dim = cos_sin.shape[-1]
        cos = cos_sin[:, : rot_dim // 2]
        sin = cos_sin[:, rot_dim // 2:]

        def _apply(x):
            if x.dim() == 2:
                num_tokens = x.shape[0]
                num_heads = x.shape[1] // head_size
                x_3d = x.view(num_tokens, num_heads, head_size)
            else:
                x_3d = x
            x_rot = x_3d[..., :rot_dim]
            x_pass = x_3d[..., rot_dim:]
            cos_e = cos.unsqueeze(-2).to(x_3d.dtype)
            sin_e = sin.unsqueeze(-2).to(x_3d.dtype)
            if is_neox:
                x1, x2 = x_rot.chunk(2, dim=-1)
                o1 = x1 * cos_e - x2 * sin_e
                o2 = x2 * cos_e + x1 * sin_e
                new_rot = torch.cat([o1, o2], dim=-1)
            else:
                x1 = x_rot[..., ::2]
                x2 = x_rot[..., 1::2]
                o1 = x1 * cos_e - x2 * sin_e
                o2 = x2 * cos_e + x1 * sin_e
                new_rot = torch.stack([o1, o2], dim=-1).flatten(-2)
            result = torch.cat([new_rot, x_pass], dim=-1)
            if x.dim() == 2:
                x.copy_(result.view(num_tokens, -1))
            else:
                x.copy_(result)

        _apply(query)
        _apply(key)

    import sgl_kernel.elementwise as _sgl_elem
    import sgl_kernel
    _sgl_elem.rotary_embedding = _rotary_embedding_pytorch
    sgl_kernel.rotary_embedding = _rotary_embedding_pytorch

    # Patch 7: graph-friendly moe_align_block_size (no .item() calls)
    def _moe_align_block_size_graph_friendly(
        topk_ids, num_experts_with_pad, block_size,
        sorted_token_ids, experts_ids, num_tokens_post_pad, cumsum_buffer,
        pad_sorted_token_ids=False,
    ):
        num_experts = num_experts_with_pad - 1
        M_topk = topk_ids.numel()
        topk_flat = topk_ids.view(-1)

        token_counts = torch.zeros(num_experts, dtype=torch.int32, device=topk_ids.device)
        ones = torch.ones(M_topk, dtype=torch.int32, device=topk_ids.device)
        token_counts.scatter_add_(0, topk_flat.clamp(0, num_experts - 1).long(), ones)

        padded_counts = ((token_counts + block_size - 1) // block_size) * block_size

        cumsum = torch.zeros(num_experts + 2, dtype=torch.int32, device=topk_ids.device)
        cumsum[1:num_experts + 1] = torch.cumsum(padded_counts, dim=0).int()
        cumsum[num_experts + 1] = cumsum[num_experts]
        cumsum_buffer.copy_(cumsum)

        num_tokens_post_pad[0] = cumsum[num_experts]

        pad_val = M_topk if pad_sorted_token_ids else 0
        sorted_token_ids.fill_(pad_val)

        sort_order = torch.argsort(topk_flat.long(), stable=True)
        sorted_experts = topk_flat[sort_order].long()

        token_cumsum_raw = torch.zeros(num_experts, dtype=torch.int64, device=topk_ids.device)
        if num_experts > 1:
            token_cumsum_raw[1:] = torch.cumsum(token_counts[:-1], dim=0).long()

        positions_in_sort = torch.arange(M_topk, dtype=torch.int64, device=topk_ids.device)
        within_rank = positions_in_sort - token_cumsum_raw[sorted_experts]

        output_positions = cumsum[sorted_experts].long() + within_rank
        output_positions = output_positions.clamp(0, sorted_token_ids.shape[0] - 1)
        sorted_token_ids.scatter_(0, output_positions, sort_order.to(sorted_token_ids.dtype))

        max_blocks = experts_ids.shape[0]
        block_starts = torch.arange(max_blocks, device=topk_ids.device, dtype=torch.int64) * block_size
        boundaries = cumsum[:num_experts + 1].long()
        expert_for_block = torch.searchsorted(boundaries, block_starts, right=True) - 1
        expert_for_block = expert_for_block.clamp(0, num_experts - 1).to(torch.int32)
        experts_ids.copy_(expert_for_block)

    import sgl_kernel.moe as _sgl_moe
    _sgl_moe._moe_align_block_size_pytorch = _moe_align_block_size_graph_friendly
    _sgl_moe.moe_align_block_size = _moe_align_block_size_graph_friendly
    sgl_kernel.moe_align_block_size = _moe_align_block_size_graph_friendly

    import sglang.srt.layers.moe.moe_runner.triton_utils.moe_align_block_size as _sglang_moe_align
    _sglang_moe_align.sgl_moe_align_block_size = _moe_align_block_size_graph_friendly

    print("[shim] applied 7 GLM-5.2 gfx942 compatibility patches", flush=True)


def apply() -> None:
    """Idempotent boot-time entrypoint. Safe to call multiple times."""
    global _APPLIED
    if getattr(sys.modules[__name__], "_APPLIED", False):
        return
    _apply_sglang_patches()
    sys.modules[__name__]._APPLIED = True


# Trigger on import — makes `import glm52_gfx942_shim` sufficient in user shims.
try:
    apply()
except ImportError as _e:
    # If sglang/sgl_kernel are not yet importable (e.g. running inside sync tool),
    # let the caller re-apply after the imports are available. Not fatal.
    print(f"[shim] deferred apply (missing dep: {_e}); call `glm52_gfx942_shim.apply()` after sglang is on sys.path",
          file=sys.stderr, flush=True)
