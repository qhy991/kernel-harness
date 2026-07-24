# GLM-5.2 end-to-end benchmark

**Thin wrappers around `python -m sglang.bench_one_batch`**, aligned with
`llm_flops` scenarios but running the real GLM-5.2-FP8 model on 8× MI300X
end-to-end instead of just at the operator level.

**Start here**: [`METHODOLOGY.md`](METHODOLOGY.md) — the two-layer test
model (op-level + e2e), operator replacement mechanism, timing standard,
what to report, known gaps. Written after two audits + one e2e run.

**Also read**: [`shim/PATCHES.md`](shim/PATCHES.md) — the seven gfx942
compat patches the shim applies to sglang at import time, each explained
with sglang code pointers and failure modes if skipped.

The two shell scripts here forward everything to `sglang.bench_one_batch`
(which natively sweeps `--input-len 1024 2048 4096` and
`--batch-size 128 256`, and writes RESULT_JSON per shape into a `.jsonl`
via `--result-filename`). This directory only owns:

- the **gfx942 compat shim** (7 pure-torch patches required for sglang to
  boot on MI300X — extracted from the proven-working `run_glm52_no_offload.py`)
- the **operator replacement mechanism** (monkey-patch registry + a
  `sitecustomize.py` that runs it before sglang imports)
- two shell wrappers pinning the scenario args
- example overrides files

## Scenarios (aligned with llm_flops)

|          | Prefill (TTFT)                              | Decode (throughput)             |
|----------|---------------------------------------------|---------------------------------|
| Inputs   | `input_len ∈ {1024, 2048, 4096}`, `output_len=1` | `batch_size ∈ {128, 256}`, small prefill |
| KV pool  | 65536 tokens                                | auto-sized to fit batch × in+out |
| TP       | 8 (single node)                             | 8                               |
| Measures | prefill latency (TTFT) + prefill tok/s      | median decode tok/s per batch   |

Prefill matches `llm_flops/bench_glm5_prefill.py` M sweep exactly. Decode is
larger than llm_flops's operator-level range (up to bs=64) because the point
of e2e testing is to reach the deployment regime.

## Layout

```
benchmarks/glm5_e2e/
├── README.md                        ← you are here (quick start + reference)
├── METHODOLOGY.md                   ← full testing methodology + operator replacement design
├── operator_overrides.py            ← runtime patch mechanism (KNOWN_OVERRIDE_TARGETS registry)
├── run_prefill_ttft.sh              ← wrapper: python -m sglang.bench_one_batch (prefill args)
├── run_decode_throughput.sh         ← wrapper: python -m sglang.bench_one_batch (decode args)
├── shim/
│   ├── glm52_gfx942_shim.py         ← 7 gfx942 compat patches (bootability)
│   ├── PATCHES.md                   ← each patch documented: what/why/when-it-retires
│   └── sitecustomize.py             ← auto-runs shim + user overrides on Python start
└── examples/
    ├── example_overrides.py         ← two-idiom template
    ├── huyan_o_proj_prefill.py      ← worked example: archive/0723-amd-glm52 kernel
    └── huyan_pr12_o_proj_prefill.py ← worked example: PR #12 kernel (from task tree)
```

## Quick start

```bash
cd /root/repos/kernel-harness

# 1. Baseline — no overrides, vanilla sglang production dispatch
benchmarks/glm5_e2e/run_prefill_ttft.sh
benchmarks/glm5_e2e/run_decode_throughput.sh

# 2. With operator swap
benchmarks/glm5_e2e/run_prefill_ttft.sh --overrides my_overrides.py

# 3. Single-shape probe
benchmarks/glm5_e2e/run_prefill_ttft.sh --input-len 4096
benchmarks/glm5_e2e/run_decode_throughput.sh --batch-size 256

# 4. Extra sglang args — pass after --
benchmarks/glm5_e2e/run_prefill_ttft.sh -- --profile --profile-stage prefill
```

Every result goes to `$KDA_E2E_OUT/prefill_ttft-<stamp>/` (default
`/tmp/glm5_e2e/…`):

```
prefill_ttft-20260723_130000Z/
├── manifest.json      ← config + resolved env
└── results.jsonl      ← one JSON per shape, verbatim from sglang.bench_one_batch
```

Standard `sglang.bench_one_batch` keys in `results.jsonl` per shape:

| key                        | meaning                                       |
|----------------------------|-----------------------------------------------|
| `prefill_latency`          | TTFT in seconds (this is what the prefill run measures) |
| `prefill_throughput`       | tokens/sec of prefill                         |
| `median_decode_latency`    | seconds/token (this is what the decode run measures) |
| `median_decode_throughput` | tokens/sec of decode                          |

## How overrides plumb in (no wrapper, no argv rewriting)

`run_prefill_ttft.sh` / `run_decode_throughput.sh` do exactly this:

1. Prepend `benchmarks/glm5_e2e/shim` to `PYTHONPATH`. That directory
   contains `sitecustomize.py`, which Python auto-imports on every
   interpreter start.
2. Prepend `benchmarks/glm5_e2e/` to `PYTHONPATH` so
   `operator_overrides` is importable from inside a user's overrides file.
3. If `--overrides <path>` was passed, export `KDA_E2E_OVERRIDES=<abspath>`.
4. `exec python -m sglang.bench_one_batch --model-path … --input-len 1024 2048 4096 …`.

When the sglang process starts, `sitecustomize.py`:

1. `import glm52_gfx942_shim` → seven pure-torch patches land (fast_hadamard
   fake, rotate_activation, MLA absorb device fix, tilelang act_quant,
   indexer_k_store bypass, rotary_embedding, graph-friendly moe_align).
2. If `$KDA_E2E_OVERRIDES` is set, import the file and call `register()`.

**Both the master process AND every TP worker fork** inherit `PYTHONPATH` and
`KDA_E2E_OVERRIDES`, so the shim + your patches land on every rank without
argv gymnastics. That's why nothing here needs a custom launcher.

## Writing an overrides file

Any `.py` with a top-level `register()`:

```python
# my_overrides.py
def register():
    from operator_overrides import patch
    from my_kernel_pkg import faster_ar_gemm

    return [patch(
        "sglang.srt.layers.quantization.fp8_utils.aiter_w8a8_block_fp8_linear",
        faster_ar_gemm,
    )]
```

Two idioms available (see `examples/example_overrides.py`):

- **`patch("dotted.attribute.path", new_fn)`** — one-off, fully-qualified
- **`register_from_dict({short_name: fn, ...})`** — well-known targets:

| Short name             | Op family                | SGLang env gate                          |
|------------------------|--------------------------|------------------------------------------|
| `fp8_gemm`             | all dense FP8 GEMM       | `SGLANG_USE_AITER=1` (default on ROCm)   |
| `hadamard`             | dsa_indexer rotate       | (compat-patched)                         |
| `moe_align_block`      | MoE token→expert bin     | —                                        |
| `index_k_store`        | DSA index_k KV store     | (compat-patched)                         |
| `mla_absorb_prepare`   | MLA weight-device fix    | (compat-patched)                         |
| `tilelang_act_quant`   | DSA act quant            | (compat-patched)                         |
| `rotary_embedding`     | RoPE                     | (compat-patched)                         |
| `custom_ar`            | TP AllReduce class       | `SGLANG_USE_AITER_AR=1`                  |
| `aiter_mla_decode`     | sparse-MLA decode kernel | `--dsa-decode-backend aiter`             |
| `aiter_gemm_asm`       | ASM FP8 GEMM (M ≥ 4096)  | `_use_aiter_bpreshuffle_gfx942` internal |
| `aiter_gemm_ck`        | CK FP8 GEMM (M < 4096)   | ↑                                        |
| `aiter_fp8_mqa_logits` | DSA index_score kernel   | —                                        |
| `aiter_fused_moe`      | fused MoE runner         | `--moe-runner-backend triton\|aiter`     |

Every applied patch prints its `old_id → new_id` to the log so you can
verify the swap actually landed.

## Worked example — install huyan's tuned `o_proj` kernel

`examples/huyan_o_proj_prefill.py` pulls the archived candidate for
`o_proj_prefill` (see `archive/0723-amd-glm52/o_proj_prefill_m*.py`) and
gates on M so:

- **M=1024**: use huyan's kernel (isolated: 1.315x vs baseline, calc_diff 2e-10)
- **M=2048**: use huyan's kernel (isolated: 1.196x)
- **M=4096**: fall back to sglang default (huyan's M=4096 wrapper has a correctness bug — see `archive/replay-20260723/results.csv` for the audit)

Compare:

```bash
# Baseline TTFT sweep
benchmarks/glm5_e2e/run_prefill_ttft.sh
# → /tmp/glm5_e2e/prefill_ttft-<stamp1>/results.jsonl

# Same sweep, with the huyan o_proj override
benchmarks/glm5_e2e/run_prefill_ttft.sh \
  --overrides benchmarks/glm5_e2e/examples/huyan_o_proj_prefill.py
# → /tmp/glm5_e2e/prefill_ttft-<stamp2>/results.jsonl

# Diff the TTFT column
jq -r '.batch_size as $bs | "M=\(.input_len)  TTFT=\(.prefill_latency*1000|round)ms"' \
  /tmp/glm5_e2e/prefill_ttft-<stamp1>/results.jsonl \
  /tmp/glm5_e2e/prefill_ttft-<stamp2>/results.jsonl
```

The **isolated** kernel win (1.315x on M=1024) is bounded above by the
o_proj share of total prefill time. `archive/replay-20260723/README.md`
puts o_proj at a few percent of prefill, so end-to-end delta will be
correspondingly small — this is exactly the amdahl reality check that
motivates running the e2e benchmark at all.

## Prerequisites

- **Hardware**: 8× MI300X on one node (`HIP_VISIBLE_DEVICES=0,...,7`)
- **Model**: `$KDA_E2E_MODEL` = a GLM-5.2-FP8 directory (default
  `/mnt/public/qinhaiyan/models/GLM-5.2-FP8`)
- **Python**: `$ROCM_TORCH_VENV/bin/python` (default `/root/venvs/rocm-torch`)
- **sglang + aiter** checkouts importable; the shim assumes
  `/root/repos/sglang/python` and `/root/repos/aiter` are on `sys.path`
- Disk for `$KDA_E2E_OUT` (default `/tmp/glm5_e2e/`)

## Reproducibility notes

- `SGLANG_DISABLE_GFX942_BPRESHUFFLE=1` is exported by the shim. This trades
  ~30% dense FP8 GEMM speed for ~5GB less HBM (no `weight_original` copy) —
  required to fit the model with a usable KV pool. **Consequence**: your
  baseline is not peak sglang production; every candidate is measured
  against the same reduced baseline. Un-set the env var to compare against
  the bpreshuffle path (may OOM depending on `--mem-fraction-static`).
- `--disable-cuda-graph` is on because HIP graph capture hits a
  `float8_e4m3fnuz` codegen bug on bf16 KV under sglang's forward pass right
  now. Every candidate runs under the same flag, so speedups compare fairly.
- The wrapper script is idempotent about env vars — anything you export
  before running takes precedence over the shim defaults.

## When boot fails

Most likely causes (in this order):

1. **Leaked GPU memory from a previous sglang crash** —
   `rocm-smi --showpids` shows dead PIDs still owning HBM. Reboot or
   `sudo rocm-smi --gpureset -d N`.
2. **`_use_aiter_bpreshuffle_gfx942` disagreement** — a code path checks the
   env var before `sitecustomize.py` sets it. Fix by exporting
   `SGLANG_DISABLE_GFX942_BPRESHUFFLE=1` in your shell before invoking.
3. **aiter or sglang not on `sys.path`** — the shim tries `/root/repos/aiter`
   and `/root/repos/sglang/python`; adjust `sys.path` in
   `shim/glm52_gfx942_shim.py` if your checkouts live elsewhere.
4. **Model path** — `$KDA_E2E_MODEL` must be a real GLM-5.2-FP8 directory
   (config.json + safetensors + tokenizer files).
