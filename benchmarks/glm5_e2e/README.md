# GLM-5.2 end-to-end benchmark

Real end-to-end inference on GLM-5.2-FP8 (~700GB) running under sglang on
8× MI300X, with a flexible operator-replacement layer so you can swap in a
custom kernel and measure its impact on TTFT / decode throughput.

Scenarios are **aligned with `llm_flops`** (the NVIDIA-side operator-level
benchmark):

|              | Prefill (TTFT)                       | Decode (throughput)         |
|--------------|--------------------------------------|-----------------------------|
| llm_flops    | M ∈ {1024, 2048, 4096}, S=65536      | batch_size ∈ {1,4,8,16,32,64} (operator-level) |
| **This bench** | input_len ∈ {1024, 2048, 4096}, KV=64k, output_len=1 → TTFT | **batch_size ∈ {128, 256}** → decode tok/s |
| Parallelism  | TP=8 (single node)                    | TP=8                        |

Prefill scenarios match llm_flops exactly. The decode scenario aims higher
than llm_flops (llm_flops tests up to bs=64 at the *operator* level, we test
bs=128/256 at *end-to-end* — this is the deployment regime users actually run).

## Layout

```
benchmarks/glm5_e2e/
  README.md                          ← you are here
  bench_glm5_e2e.py                  ← Python entrypoint (wraps sglang.bench_one_batch)
  operator_overrides.py              ← runtime patch mechanism (importable in your overrides file)
  run_prefill_ttft.sh                ← thin bash wrapper for the prefill scenario
  run_decode_throughput.sh           ← thin bash wrapper for the decode scenario
  shim/
    glm52_gfx942_shim.py             ← gfx942 compat patches (7 fixes; required for sglang to boot)
  examples/
    example_overrides.py             ← two-idiom template
    huyan_o_proj_prefill.py          ← install huyan's tuned o_proj kernel (worked example)
```

## Prerequisites

- **Hardware**: 8× MI300X on one node (visible via `HIP_VISIBLE_DEVICES=0,...,7`)
- **Model**: GLM-5.2-FP8 at `$KDA_E2E_MODEL` (default `/mnt/public/qinhaiyan/models/GLM-5.2-FP8`)
- **Python**: `$ROCM_TORCH_VENV/bin/python` (default `/root/venvs/rocm-torch`)
- **sglang + aiter** checkouts on PYTHONPATH; the shim assumes:
  - `/root/repos/sglang/python`
  - `/root/repos/aiter`
- **Disk**: `/root/repos/kernel-harness/runs/` (or wherever `KDA_E2E_OUT` points)
- The compat shim disables coredumps (`HSA_ENABLE_COREDUMP_ON_EXCEPTION=0`) —
  necessary because a crash mid-run previously filled the local disk with a
  40GB GPU-memory dump.

## Quick start

### 1. Baseline (no overrides — vanilla sglang production dispatch)

```bash
cd /root/repos/kernel-harness/benchmarks/glm5_e2e

# Prefill TTFT: KV=64k, input_len ∈ {1024, 2048, 4096}
./run_prefill_ttft.sh

# Decode throughput: batch_size ∈ {128, 256}
./run_decode_throughput.sh

# Both, defaults
./bench_glm5_e2e.py both
```

Output goes to `$KDA_E2E_OUT/run-<utcstamp>/` (default `/tmp/glm5_e2e/`):

```
run-20260723_130000Z/
  manifest.json                            ← what was configured
  summary.json                             ← per-scenario headline metrics
  prefill_ttft_M1024_kv65536_tp8.log       ← full sglang.bench_one_batch stdout
  prefill_ttft_M1024_kv65536_tp8.jsonl     ← sglang.bench_one_batch RESULT_JSON stream
  prefill_ttft_M1024_kv65536_tp8.entry.py  ← the auto-generated launcher (kept for repro)
  decode_thpt_bs128_in128_out64_tp8.log
  decode_thpt_bs128_in128_out64_tp8.jsonl
  … (one triplet per scenario point)
```

Every RESULT_JSON line from sglang is preserved verbatim in `<label>.jsonl`
— that's the source-of-truth. `summary.json` is a compacted headline.

### 2. With operator overrides

Write a Python file with a top-level `register()` function:

```python
# my_overrides.py
def register():
    """Called after sglang/aiter are importable, before bench_one_batch starts."""
    from operator_overrides import patch
    from my_kernel_pkg import faster_ar_gemm

    # Replace sglang's FP8 GEMM dispatcher wholesale.
    return [patch(
        "sglang.srt.layers.quantization.fp8_utils.aiter_w8a8_block_fp8_linear",
        faster_ar_gemm,
    )]
```

Then:

```bash
./run_prefill_ttft.sh --overrides my_overrides.py
./run_decode_throughput.sh --overrides my_overrides.py
```

The bench prints exactly which attributes it swapped (with `id()` before/after)
so it's obvious in the log whether your kernel landed.

### 3. Worked example — install huyan's tuned o_proj kernel

`examples/huyan_o_proj_prefill.py` pulls in the archived candidate for
`o_proj_prefill` (see `archive/0723-amd-glm52/o_proj_prefill_m*.py`) and
gates on M so:

- **M=1024**: use huyan's kernel (measured 1.315x vs baseline)
- **M=2048**: use huyan's kernel (measured 1.196x)
- **M=4096**: fall back to sglang default (huyan's M=4096 wrapper has a correctness bug)

Bench:

```bash
# baseline
./run_prefill_ttft.sh
# with the override; compare TTFT column against the baseline run
./run_prefill_ttft.sh --overrides examples/huyan_o_proj_prefill.py
```

At M=1024, sglang's o_proj call accounts for a few percent of prefill total,
so end-to-end TTFT delta will be smaller than the isolated-kernel 1.315x —
that's the whole point of e2e testing: it names the Amdahl ceiling before
you spend more time on the same op.

## What operators can I override?

Named short cuts (via `register_from_dict`), see
[`operator_overrides.py`](operator_overrides.py) for full paths:

| Short name             | Op family                | Env gate SGLang uses                     |
|------------------------|--------------------------|------------------------------------------|
| `fp8_gemm`             | all dense FP8 GEMM       | `SGLANG_USE_AITER=1` (default on ROCm)   |
| `hadamard`             | dsa_indexer rotate       | (compat-patched)                         |
| `moe_align_block`      | MoE token-to-expert bin  | —                                        |
| `index_k_store`        | DSA index_k KV store     | (compat-patched)                         |
| `mla_absorb_prepare`   | MLA weight-device fix    | (compat-patched)                         |
| `tilelang_act_quant`   | DSA act quant            | (compat-patched)                         |
| `rotary_embedding`     | RoPE application         | (compat-patched)                         |
| `custom_ar`            | TP AllReduce class       | `SGLANG_USE_AITER_AR=1`                  |
| `aiter_mla_decode`     | sparse-MLA decode kernel | `--dsa-decode-backend aiter`             |
| `aiter_gemm_asm`       | ASM FP8 GEMM (large M)   | `_use_aiter_bpreshuffle_gfx942` internal |
| `aiter_gemm_ck`        | CK FP8 GEMM (small M)    | ↑                                        |
| `aiter_fp8_mqa_logits` | DSA index_score kernel   | —                                        |
| `aiter_fused_moe`      | fused MoE runner         | `--moe-runner-backend triton`/etc.       |

For anything not in the short-name table, use `patch("full.dotted.path", fn)`
— every module.attribute in the sglang / aiter / sgl_kernel tree is fair
game. Verify the swap actually took by looking for the `[overrides] →`
lines in the log; every patched attribute prints its `old_id` and `new_id`.

## Backend flags you might want to tune

All of these forward straight to `sglang.bench_one_batch`:

- `--dsa-prefill-backend` ∈ `aiter | flashmla_sparse | flashmla_kv | fa3 | trtllm`
- `--dsa-decode-backend`  ∈ (same set)
- `--dsa-topk-backend`    ∈ `torch | aiter` (SGLANG_DSA_FUSE_TOPK=0 → `torch`)
- `--moe-runner-backend`  ∈ `triton | deep_gemm | aiter`
- `--kv-tokens`, `--mem-fraction-static`

Default backends chosen for this bench are the ones we proved boot cleanly
on gfx942 (`aiter` prefill + `fa3` decode + `torch` topk + `triton` MoE);
change them if your patched kernel needs a different upstream backend.

## Interpreting the output

Each scenario line in the log prints, e.g.

```
── prefill_ttft_M1024_kv65536_tp8 ──  running (log: run-…/prefill_ttft_M1024_kv65536_tp8.log)
  ✓ exit=0  wall=32.4s  TTFT=393.8ms · prefill=2600 tok/s
```

Full metrics per shape sit in `<label>.jsonl` (one JSON object per
bench_one_batch replay; last object is the winning median). Standard keys:

| key                          | meaning                                          |
|------------------------------|--------------------------------------------------|
| `prefill_latency`            | TTFT in seconds (one prefill of `input_len` tokens) |
| `prefill_throughput`         | tokens / sec of prefill                          |
| `median_decode_latency`      | median seconds per decode step                   |
| `median_decode_throughput`   | median tokens / sec of decode                    |

To compare two runs (baseline vs overrides) it's usually enough to `jq`
these keys across the two `summary.json` files.

## Reproducibility notes

- The compat shim (`shim/glm52_gfx942_shim.py`) applies **seven** patches
  required to boot GLM-5.2 on gfx942. These are pure-torch equivalents of
  kernels that either don't exist on ROCm (`fast_hadamard_transform`) or
  crash on gfx942 (`indexer_k_quant_and_cache` memory fault). They do NOT
  change math semantics — they're the smallest set that makes sglang boot.
- The shim also sets `SGLANG_DISABLE_GFX942_BPRESHUFFLE=1`, which forces the
  Triton path for dense FP8 GEMM instead of ASM bpreshuffle. This trades
  ~30% GEMM speed for ~5GB less HBM (no `weight_original` copy) — required
  to fit the model with a working KV pool. **Consequence**: your baseline
  is *not* peak sglang production speed; if you disable the shim your
  baseline shifts and any speedup number you measure shifts with it.
- `--disable-cuda-graph` is on by default: HIP graph capture hits a
  `float8_e4m3fnuz` codegen bug on bf16 KV under sglang's forward pass at
  the moment. Timings are still comparable across candidates as long as
  every candidate runs with the same flag.
- The benchmark uses **one** replay per shape (whatever `bench_one_batch`
  measures) — small variance is expected. Run the sweep 3× if you need
  tighter medians.

## When something fails to boot

Most likely causes:

1. **`_use_aiter_bpreshuffle_gfx942` disagreements** — some code path checks
   the flag before the shim sets `SGLANG_DISABLE_GFX942_BPRESHUFFLE`. Fix by
   exporting the env var in your shell first.
2. **Leaked GPU memory from a previous crash** — the previous sglang run's
   TP workers can leave 175GB per rank pinned; `rocm-smi --showpids` shows
   dead PIDs still owning memory. Reboot or `rocm-smi --gpureset -d N`.
3. **`aiter` C++ module not built** — the wrappers under `aiter/jit/` are
   expected pre-compiled. If they're missing, `pip install -e .` in the
   aiter repo. Note: bpreshuffle_asm's C++ JIT needs a writable aiter
   package dir; if it's read-only, sglang falls back to CK.
4. **Model path** — `KDA_E2E_MODEL` must point at a valid GLM-5.2-FP8
   directory (with `config.json`, `.safetensors`, tokenizer files, etc.).
