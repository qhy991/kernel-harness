# The 7 sglang gfx942 compatibility patches

`glm52_gfx942_shim.py` applies seven monkey-patches at Python import time
so `sglang.bench_one_batch` boots GLM-5.2-FP8 on AMD MI300X (gfx942)
without crashing and without silently producing garbage. Every patch is
required. **None of them change math semantics** — each replaces a kernel
that either doesn't exist on ROCm or crashes on gfx942 with a pure-torch
equivalent that produces identical output.

**Provenance**: extracted verbatim from
`sglang-exp/tasks/dense-fp8-gemm/runs/run_glm52_no_offload.py`, which is
the earliest known-working GLM-5.2 gfx942 shim. Behaviour has been
verified in three sglang forks + our checkout.

**How to think about this list**: these are not sglang bugs and they're
not deployment problems — they're a bridge that makes an nvidia-first
sglang boot on AMD hardware. When sglang or aiter release proper ROCm
versions of the affected kernels, individual patches will retire.

## The patches

### 1. `fast_hadamard_transform` — pure-torch fake

sglang's DSA indexer imports `fast_hadamard_transform`, a CUDA-only
extension that doesn't exist on ROCm. Without a fake, `import sglang`
raises `ModuleNotFoundError` before boot.

The shim installs a fake module with `__spec__` set (Python 3.11's import
machinery rejects `types.ModuleType(...)` entries with `__spec__ = None`
during sitecustomize), whose `hadamard_transform(x, scale)` is a pure
torch Hadamard butterfly. Mathematically identical to the CUDA extension
up to floating-point rounding; measurably slower but bootability first.

**Code**: `sys.modules["fast_hadamard_transform"] = <fake>` in
`glm52_gfx942_shim.py`.

**Retires when**: aiter or sglang ships a ROCm-native
`fast_hadamard_transform`. Track upstream aiter for `aiter.hadamard`.

### 2. `dsa_indexer.rotate_activation` — pure-torch Hadamard

sglang wires `rotate_activation` to the same missing CUDA extension. Even
with the fake module, `rotate_activation` on ROCm still calls a slow path
that in some sglang versions triggers a shape assertion. The shim replaces
the sglang-side wrapper directly:

```python
def _patched_rotate_activation(x):
    hidden = x.size(-1); assert (hidden & (hidden-1)) == 0  # power of 2
    return _hadamard_transform_pytorch(x, scale=hidden**-0.5)
```

**Code**:
`sglang.srt.layers.attention.dsa.dsa_indexer.rotate_activation = _patched_rotate_activation`.

**Retires when**: patch 1 retires.

### 3. `DeepseekMLAForwardMixin.forward_absorb_prepare` — device fix for w_kc / w_vc / scales

MLA's absorbed-weight prep expects `self.w_kc`, `self.w_vc`, and their
scales to be on the same device as the input. On some sglang versions
these tensors are constructed on cpu / a different rank's cuda device
during model load, and the forward-time assertion fires. The shim adds a
one-shot device migration guard before the original method runs.

**Code**:
`sglang.srt.models.deepseek_common.attention_forward_methods.forward_mla.DeepseekMLAForwardMixin.forward_absorb_prepare`

**Retires when**: sglang's MLA weight loading correctly places tensors on
the owning rank's device. Reported upstream.

### 4. `tilelang_kernel.act_quant` — pure-torch FP8 per-block quant

The tilelang kernel that quantises indexer input activations to
`float8_e4m3fnuz` uses a code path with a ROCm compile bug (segfaults on
gfx942). The shim replaces it with a pure-torch per-block quantiser:

```python
def _act_quant_pytorch(x, block_size=128, scale_fmt=None):
    # per-block absmax → scale = amax / 224
    # x_fp8 = clamp(x/scale, ±224).to(float8_e4m3fnuz)
    ...
    return x_fp8, scale_inv
```

FP8_MAX = 224.0 (the fnuz safe max per aiter). Note that fnuz's
`torch.finfo` reports 240 but 224 keeps a margin below the NaN-adjacent
code point; sglang production also uses 224.

**Code**:
`sglang.srt.layers.attention.dsa.tilelang_kernel.act_quant = _act_quant_pytorch`

**Retires when**: tilelang / aiter ships a working act_quant for gfx942.

### 5. `Indexer._store_index_k_cache` — bypass aiter indexer_k_quant_and_cache

`aiter.indexer_k_quant_and_cache` reliably GPU-faults on gfx942 (memory
access at a stride the kernel gets wrong for certain KV block counts).
The shim replaces the sglang wrapper with the equivalent split:

```python
def _patched_store_index_k_cache(self, forward_batch, layer_id, key, *,
                                  act_quant=None, out_cache_loc=None):
    # 1. quantise key with act_quant (patch 4)
    # 2. write (index_k, index_k_scale) directly into the KV pool
```

No fused kernel, two calls instead — measurably slower on the
indexer_k_store hot path but doesn't crash. This is one of the biggest
correctness risks in the shim: if this patch is missing, sglang appears
to run for a few tokens and then GPU memory-faults, occasionally leaking
175GB of HBM per rank.

**Code**:
`sglang.srt.layers.attention.dsa.dsa_indexer.Indexer._store_index_k_cache = _patched_store_index_k_cache`

**Retires when**: aiter fixes `indexer_k_quant_and_cache` for gfx942.

### 6. `sgl_kernel.rotary_embedding` — pure-torch RoPE

`sgl_kernel` ships a CUDA-only rotary embedding. Its ROCm build stub
raises at first call; sglang's model construction calls it during forward
and boots fine but crashes on the first prefill step. The shim replaces
both aliases:

```python
sgl_kernel.elementwise.rotary_embedding = _rotary_embedding_pytorch
sgl_kernel.rotary_embedding             = _rotary_embedding_pytorch
```

Neox-style and non-neox both supported (via the `is_neox` argument
sglang already threads through). Pure-torch with cos_sin table lookup;
in-place update matching the CUDA extension's contract.

**Code**: `sgl_kernel.elementwise.rotary_embedding` +
`sgl_kernel.rotary_embedding`.

**Retires when**: sgl_kernel builds against ROCm with a real rotary
kernel. Likely landing in aiter first.

### 7. `moe_align_block_size` — graph-friendly vectorised replacement

sglang's `sgl_moe_align_block_size` uses `.item()` calls that block HIP
graph capture (sglang uses graph capture for prefill on gfx942 when the
env allows). The CUDA version doesn't have this issue because CUDA graphs
are more permissive about host reads. The shim replaces every entry-point
alias (there are four) with a fully-vectorised torch implementation:

```python
def _moe_align_block_size_graph_friendly(topk_ids, ..., cumsum_buffer, ...):
    # 1. count tokens per expert via scatter_add
    # 2. pad each expert to block_size
    # 3. cumsum → per-expert offsets
    # 4. argsort + scatter to build sorted_token_ids
    # 5. searchsorted to build experts_ids for each block
    # zero .item() calls; every step is a graph-safe torch op
```

Also handles the `pad_sorted_token_ids` flag that some sglang versions
require. This patch is the one that most often breaks silently — if
`moe_align_block_size` returns garbage, the MoE reads land in adjacent
expert's rows, no error is raised, finite garbage propagates, and the
model produces coherent-looking but wrong tokens.

**Code**: four aliases:
- `sgl_kernel.moe._moe_align_block_size_pytorch`
- `sgl_kernel.moe.moe_align_block_size`
- `sgl_kernel.moe_align_block_size`
- `sglang.srt.layers.moe.moe_runner.triton_utils.moe_align_block_size.sgl_moe_align_block_size`

**Retires when**: sglang refactors so the same
`moe_align_block_size` symbol is used everywhere (or fixes the .item()
calls in the CUDA version so ROCm graph capture works too).

## Environment defaults the shim also sets

Not patches per se, but the same file exports env vars every gfx942 sglang
boot needs. `setdefault` — the caller can override:

| var | default | reason |
|---|---|---|
| `PYTORCH_ROCM_ARCH` | `gfx942` | aiter+triton codegen path |
| `TORCHDYNAMO_DISABLE` | `1` | dynamo hits fnuz codegen bug |
| `TORCH_COMPILE_DISABLE` | `1` | same |
| `SGLANG_DSA_FUSE_TOPK` | `0` | fused DSA topk kernel is CUDA-only |
| `SGLANG_OPT_USE_AITER_SILU_MUL` | `1` | aiter silu_mul is production path |
| `SGLANG_USE_AITER` | `1` | required for aiter dispatch |
| `SGLANG_DISABLE_GFX942_BPRESHUFFLE` | `1` | saves 5GB weight_original; costs 30% GEMM speed |
| `PYTORCH_ALLOC_CONF` | `expandable_segments:True` | large-arena to reduce fragmentation |

## How to add a new patch

If a future sglang release introduces a new gfx942 incompatibility:

1. Isolate it — a fresh sglang tree, boot GLM-5.2, capture the traceback.
2. Add a pure-torch replacement to `glm52_gfx942_shim.py`, following the
   pattern:

    ```python
    import sglang.srt.<module> as _tgt
    _orig_fn = _tgt.<attr>

    def _patched_fn(...):
        # pure-torch equivalent
        return ...

    _tgt.<attr> = _patched_fn
    print("[shim] applied N GLM-5.2 gfx942 compatibility patches", flush=True)
    ```

3. Document it here with:
   - What sglang symbol
   - What CUDA-only feature it needs
   - Concrete failure mode without the patch
   - When it can retire

4. If the patch changes math semantics (not just performance), STOP —
   that's not a compatibility patch, it's a semantic change and belongs
   in the operator override layer, not the shim.

## Correctness verification

Every patch has been verified end-to-end by three independent boots of
sglang GLM-5.2-FP8 on 8× MI300X:

- Boot succeeds; no traceback in the first 100 prefill tokens.
- `sglang.bench_one_batch --output-len 1` produces bit-close outputs
  across two runs with the same seed.
- Op-level tests (`archive/replay-20260723-pr12-verify/`) using the
  post-PR#12 math oracle pass at `calc_diff ≤ 5e-9` for all shapes.
- E2E prefill runs (`archive/e2e-prefill-20260723-pr12/`) produce
  sensible TTFT numbers matching manual instrumentation.

If any patch is skipped, the failure mode is one of:

- **1 skipped**: `ImportError: No module named 'fast_hadamard_transform'`
- **2 skipped**: prefill of the first token crashes in dsa_indexer
- **3 skipped**: `AssertionError: expected w_kc on cuda:X, got cpu`
- **4 skipped**: prefill's first indexer call segfaults in tilelang
- **5 skipped**: GPU memory fault after a few tokens; sglang leaks 175GB HBM per rank
- **6 skipped**: prefill's first attention step crashes in sgl_kernel
- **7 skipped**: silent — MoE reads adjacent-expert rows and produces
  wrong tokens with no error signal

If your boot fails in a way not listed above, it's probably a **new**
gfx942 incompatibility — file a follow-up.
