"""Runtime operator-replacement mechanism for the GLM-5.2 e2e bench.

Lets a user swap sglang's dispatch of a named operator for a custom kernel
BEFORE sglang boots, so the change lands in every TP worker fork. Modelled
after how sglang itself gates on env vars (SGLANG_USE_AITER,
SGLANG_USE_AITER_AR, SGLANG_DSA_USE_AITER_SPARSE_MLA, …) and how our
proven-working `run_glm52_no_offload.py` shim monkeypatches the same modules.

## User-facing contract

An "overrides file" is any `.py` on disk that defines a top-level
`register()` function:

    # my_overrides.py
    def register():
        # Called AFTER sglang / aiter are importable, BEFORE bench_one_batch
        # is invoked. Patch anything you want.
        import sglang.srt.layers.quantization.fp8_utils as fp8u
        from my_kernels import faster_fp8_gemm
        fp8u.aiter_w8a8_block_fp8_linear = faster_fp8_gemm

The e2e bench takes this file via `--overrides my_overrides.py`, imports it,
and calls `register()` after the gfx942 shim has landed. Patch order:

  1. `glm52_gfx942_shim.apply()`  — compatibility patches (required to boot)
  2. `<your file>.register()`      — your optimisation patches
  3. `sglang.bench_one_batch.main` — runs the benchmark

## Known override points on the current sglang GLM-5.2 gfx942 dispatch

Names + module paths, alongside the environment gate SGLang uses (if any):

| Op                        | Module.attribute                                                                       | Env gate                             |
|---------------------------|----------------------------------------------------------------------------------------|--------------------------------------|
| FP8 GEMM (all dense)      | `sglang.srt.layers.quantization.fp8_utils.aiter_w8a8_block_fp8_linear`                 | `SGLANG_USE_AITER=1` (default on hip)|
| AllReduce                 | `sglang.srt.distributed.device_communicators.custom_all_reduce._all_reduce_impl_factory` | `SGLANG_USE_AITER_AR=1` (default)  |
| MLA sparse decode         | `sglang.srt.layers.attention.dsa_backend.DSABackend._run_aiter_mla_decode_fwd`         | `--dsa-decode-backend aiter`         |
| MLA prefill               | `sglang.srt.layers.attention.dsa_backend.DSABackend._forward_aiter_extend`             | `--dsa-prefill-backend aiter`        |
| DSA index_k store         | `sglang.srt.layers.attention.dsa.dsa_indexer.Indexer._store_index_k_cache`             | (compat-patched)                     |
| DSA index_score           | `aiter.ops.triton.fp8_mqa_logits.fp8_mqa_logits`                                       | —                                    |
| Hadamard (index_q rotate) | `sglang.srt.layers.attention.dsa.dsa_indexer.rotate_activation`                        | (compat-patched)                     |
| MoE runner                | `sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe`                              | `--moe-runner-backend triton`        |
| Rotary embedding          | `sgl_kernel.rotary_embedding` / `sgl_kernel.elementwise.rotary_embedding`              | (compat-patched)                     |

Any of these are fair game for `register()`. Convention: monkeypatch the
attribute on its owning module; every place sglang looks it up will see your
version because sglang holds the module reference, not a captured local.

## Convenience: named patch registry

If you'd rather declare your patches as a dict instead of writing code,
`register_from_dict({"fp8_gemm": my_fn})` is provided; it maps well-known
short names to the module.attribute paths above and applies the swap. See
`examples/example_overrides.py` for both idioms side by side.

## Verifying a patch took effect

After `register()` runs, the harness prints the resolved id() of each
patched attribute so it's obvious in the log whether your kernel actually
landed. If sglang re-imports the module later (rare, but happens with lazy
imports), a warning is emitted so you can debug quickly.
"""
from __future__ import annotations

import importlib
import importlib.util
import sys
import traceback
from pathlib import Path
from typing import Callable, Mapping


# ── Well-known patch points, keyed by short name ────────────────────────────
# Extend cautiously — each entry must correspond to a real sglang attribute
# path that agents can safely monkeypatch. Adding a name here documents that
# the harness supports overriding it.
KNOWN_OVERRIDE_TARGETS: dict[str, str] = {
    "fp8_gemm":            "sglang.srt.layers.quantization.fp8_utils.aiter_w8a8_block_fp8_linear",
    "hadamard":            "sglang.srt.layers.attention.dsa.dsa_indexer.rotate_activation",
    "moe_align_block":     "sglang.srt.layers.moe.moe_runner.triton_utils.moe_align_block_size.sgl_moe_align_block_size",
    "index_k_store":       "sglang.srt.layers.attention.dsa.dsa_indexer.Indexer._store_index_k_cache",
    "mla_absorb_prepare":  "sglang.srt.models.deepseek_common.attention_forward_methods.forward_mla.DeepseekMLAForwardMixin.forward_absorb_prepare",
    "tilelang_act_quant":  "sglang.srt.layers.attention.dsa.tilelang_kernel.act_quant",
    "rotary_embedding":    "sgl_kernel.elementwise.rotary_embedding",
    "custom_ar":           "sglang.srt.distributed.device_communicators.custom_all_reduce.CustomAllreduce",
    # aiter-side ops (fair to patch if the candidate re-implements them)
    "aiter_mla_decode":    "aiter.mla.mla_decode_fwd",
    "aiter_gemm_asm":      "aiter.gemm_a8w8_blockscale_bpreshuffle_asm",
    "aiter_gemm_ck":       "aiter.gemm_a8w8_blockscale",
    "aiter_fp8_mqa_logits": "aiter.ops.triton.fp8_mqa_logits.fp8_mqa_logits",
    "aiter_fused_moe":     "aiter.fused_moe.fused_moe",
}


def _resolve(dotted: str):
    """Split 'a.b.c.d' into (module 'a.b.c', attribute chain ['c','d'] chased on
    the module). Returns (owner_object, attr_name, current_value)."""
    parts = dotted.split(".")
    for split_at in range(len(parts) - 1, 0, -1):
        mod_name = ".".join(parts[:split_at])
        try:
            mod = importlib.import_module(mod_name)
        except ImportError:
            continue
        owner = mod
        for p in parts[split_at:-1]:
            if not hasattr(owner, p):
                owner = None
                break
            owner = getattr(owner, p)
        if owner is None:
            continue
        attr = parts[-1]
        if not hasattr(owner, attr):
            continue
        return owner, attr, getattr(owner, attr)
    raise ImportError(f"cannot resolve {dotted!r}: no module prefix loadable")


def register_from_dict(mapping: Mapping[str, Callable]) -> dict:
    """Apply a {short_name: callable} dict via KNOWN_OVERRIDE_TARGETS.

    Returns {short_name: {'target': dotted, 'old_id': int, 'new_id': int}}
    so a caller can log what changed.
    """
    changes = {}
    for name, new in mapping.items():
        if name not in KNOWN_OVERRIDE_TARGETS:
            raise KeyError(
                f"unknown override name {name!r}; known: "
                f"{', '.join(sorted(KNOWN_OVERRIDE_TARGETS))}"
            )
        dotted = KNOWN_OVERRIDE_TARGETS[name]
        owner, attr, old = _resolve(dotted)
        setattr(owner, attr, new)
        changes[name] = {"target": dotted, "old_id": id(old), "new_id": id(new)}
    return changes


def patch(dotted: str, new_value) -> dict:
    """Monkey-patch a fully-qualified attribute; used from `register()` bodies."""
    owner, attr, old = _resolve(dotted)
    setattr(owner, attr, new_value)
    return {"target": dotted, "old_id": id(old), "new_id": id(new_value)}


def load_overrides(path: str | None) -> Callable | None:
    """Import a user's overrides .py and return its `register()` callable.
    Returns None if `path` is empty / None (i.e. run without any overrides)."""
    if not path:
        return None
    p = Path(path).resolve()
    if not p.is_file():
        raise FileNotFoundError(f"overrides file not found: {p}")
    spec = importlib.util.spec_from_file_location(f"_user_overrides_{p.stem}", p)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    try:
        spec.loader.exec_module(mod)
    except Exception:
        traceback.print_exc()
        raise
    reg = getattr(mod, "register", None)
    if reg is None or not callable(reg):
        raise AttributeError(
            f"{p} must define a top-level callable `register()` that applies "
            "the operator patches."
        )
    return reg


def apply_overrides(register_callable: Callable | None) -> list[dict]:
    """Call `register()` and echo what changed to stdout. Returns the change list
    so bench_glm5_e2e can serialise it into the run manifest."""
    if register_callable is None:
        print("[overrides] no user overrides — running with sglang defaults", flush=True)
        return []
    print(f"[overrides] applying user overrides via {register_callable.__module__}.register()",
          flush=True)
    result = register_callable()
    # `register()` may return None, a dict, or a list of dicts. Normalise.
    if result is None:
        changes = []
    elif isinstance(result, dict):
        # Assume {name: change_dict} or {name: callable} — best effort log.
        changes = [{"name": k, **(v if isinstance(v, dict) else {"value_id": id(v)})}
                   for k, v in result.items()]
    elif isinstance(result, list):
        changes = result
    else:
        changes = [{"return_value_id": id(result)}]
    for c in changes:
        print(f"[overrides]   → {c}", flush=True)
    return changes
