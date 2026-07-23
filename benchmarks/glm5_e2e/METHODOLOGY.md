# E2E testing methodology

How this repo tests GLM-5.2 kernel replacements against the real sglang
serving path. Written after two rounds of audits
(`archive/replay-20260723/`, `archive/replay-20260723-pr12-verify/`) and
one e2e run (`archive/e2e-prefill-20260723-pr12/`) — the choices below
are calibrated to what those runs proved worked or broke.

## Two-layer test model

We don't have one benchmark, we have two — because a single op-level
number is not a deployment decision, and a single e2e number can't
diagnose a kernel bug. Each layer answers a different question, and each
layer has different failure modes.

```
┌────────────────────────────────────────────────────────────────────┐
│ Layer 1 · Op-level  ─ "does the kernel win against production?"    │
│   testbench/tasks/glm52_amd/<task>/run.sh --candidate PATH         │
│   · single-process, one operator, frozen inputs from glm52_ops     │
│   · reference = provider's aiter production dispatch               │
│   · correctness = MATH oracle (post-PR#12), not the reference      │
│   · timing = HIP graph capture+replay median, cold-L2 flush        │
│   · verdict = per-shape WIN / REGRESS / neutral (p10/p90 gate)     │
│   · runs the kernel ONCE per sweep iteration                       │
├────────────────────────────────────────────────────────────────────┤
│ Layer 2 · End-to-end  ─ "does the kernel move deployment TTFT?"    │
│   benchmarks/glm5_e2e/run_prefill_ttft.sh [--overrides ov.py]      │
│   · 8× MI300X, TP=8, real GLM-5.2-FP8 (~700GB) via sglang          │
│   · reference = whatever sglang dispatches on this hardware        │
│   · timing = sglang.bench_one_batch's own prefill latency          │
│   · verdict = TTFT delta at the user's input_len                   │
│   · runs the kernel PER LAYER (~78×) per prefill                   │
└────────────────────────────────────────────────────────────────────┘
```

You need both because they disagree. In
`archive/e2e-prefill-20260723-pr12/README.md` we measured huyan's PR #12
o_proj kernel at op-level 0.636× (REGRESS) but e2e 1.356× (SPEEDUP) —
same kernel, same shape, opposite verdict. Op-level dominates the wrapper
per-call overhead (input quant, output alloc) that amortises across 78
layer calls in e2e. **Neither number is wrong. They measure different
things.** Use op-level to tune kernels, use e2e to decide deployments.

## Op-level test (Layer 1)

Fully documented at `testbench/tasks/glm52_amd/<task>/run.sh --describe`.
Short version:

```bash
T=testbench/tasks/glm52_amd/o_proj_prefill    # or index_k_prefill, etc.

$T/run.sh --describe                          # what is this problem?
$T/run.sh                                     # default candidate, full sweep
$T/run.sh --candidate ~/my_kernel.py          # test any .py defining run(inputs)
$T/run.sh --candidate ~/dir/                  # or a dir holding candidate.py
$T/run.sh --M 4096                            # one shape only
$T/run.sh --repeat 10 --iterations 30         # more samples for tight bounds
```

Exit codes: `0` = correct + WIN, `1` = correct + no WIN, `2` = incorrect,
`3` = infrastructure.

The **correctness gate** post-PR#12 uses a deterministic math oracle
(dequant → f32 matmul for GEMM, fully-f32 sparse oracle for MLA), NOT the
production kernel that's also the latency baseline. That decoupling was
huyan's fix (`archive/replay-20260723/ROOTCAUSE_AND_FIX.md`), and it's
required — the pre-PR#12 gate flagged correct candidates as INCORRECT at
M≥4096 because the reference itself was garbage in that regime.

The **latency baseline** is `glm52_ops.reference()` → provider dispatch
(`aiter-torch-reference` on MI300X), which tries in order:

1. sglang's `aiter_w8a8_block_fp8_linear` full dispatcher (if sglang is on
   sys.path — this is the actual production path SGLang runs at)
2. aiter's Triton `gemm_a8w8_blockscale` (fallback if bpreshuffle
   unavailable)
3. `aiter.ops.gemm_op_a8w8.gemm_a8w8_blockscale` CK (fallback)
4. `torch._scaled_mm` hipBLASLt (last-resort)

Whichever succeeds is what timing uses — the RESULT_JSON records the
concrete backend name so you can tell which fallback ran.

## E2E test (Layer 2)

Located under [`benchmarks/glm5_e2e/`](.). See [`README.md`](README.md)
for full usage; the important knobs:

```bash
# Baseline — no overrides, vanilla sglang production dispatch
./run_prefill_ttft.sh                             # input_len ∈ {1024,2048,4096}
./run_decode_throughput.sh                        # batch_size ∈ {128,256}

# With operator replacement
./run_prefill_ttft.sh --overrides my_overrides.py
./run_decode_throughput.sh --overrides my_overrides.py

# Warm each shape by passing it multiple times
./run_prefill_ttft.sh --input-len "1024 1024 1024 2048 2048 2048 4096 4096 4096"
```

**Fresh prefill, not incremental.** `sglang.bench_one_batch` starts with
an empty KV pool and prefills `--input-len N` tokens from scratch. To
measure "extend by N tokens on top of a 64k prefix" you need
`sglang.bench_serving` with prefix caching, which is not yet wrapped here
(see "Known gaps" below).

**Repeats.** `sglang.bench_one_batch` runs each `(batch_size, input_len,
output_len)` triple ONCE. To get n>1 samples per shape, pass the same
value multiple times in the sweep — first sample is warmup (drop it),
median the rest. n=3 gives a directional signal; n=5–10 gives sp_cons.

## Operator replacement mechanism

The mechanism plugs into sglang **before boot** so every TP worker sees
the patch. Nothing about sglang's argv or startup path changes — the
plumbing is entirely at Python-import time.

### For the user

1. Write a `.py` with a top-level `register()`:

    ```python
    # my_overrides.py
    def register():
        from operator_overrides import patch
        from my_kernel_pkg import faster_fp8_gemm
        return [patch(
            "sglang.srt.layers.quantization.fp8_utils.aiter_w8a8_block_fp8_linear",
            faster_fp8_gemm,
        )]
    ```

2. Pass it to the runner:

    ```bash
    ./run_prefill_ttft.sh --overrides my_overrides.py
    ```

3. Check the log for the swap confirmation (one line per TP worker + master):

    ```
    [overrides]   → {'target': '…aiter_w8a8_block_fp8_linear',
                     'old_id': 139...42, 'new_id': 139...16}
    ```

### How it plumbs

```
bash run_prefill_ttft.sh                        ┐
  export PYTHONPATH=shim:benchmarks:$PYTHONPATH │  bash wrapper only sets
  export KDA_E2E_OVERRIDES=/abs/my_overrides.py │  env + PYTHONPATH; then
  exec python -m sglang.bench_one_batch …       ┘  execs sglang as-is
                                                
  ┌─ every python interpreter that starts (master, 8 TP workers) ─┐
  │  sitecustomize.py auto-imports because it's on PYTHONPATH:    │
  │    1. import glm52_gfx942_shim   → 7 gfx942 patches applied   │
  │    2. import KDA_E2E_OVERRIDES   → user register() runs       │
  │  Both write to sys.modules and to module attributes, so       │
  │  every sglang code path that later resolves them sees them.   │
  └───────────────────────────────────────────────────────────────┘
```

The patch site is always **the module that owns the attribute**, not the
call site. This is critical: sglang holds a module reference, not a
captured local, so `setattr(fp8_utils, "aiter_w8a8_block_fp8_linear", ...)`
propagates to every place that resolves `fp8_utils.aiter_w8a8_block_fp8_linear`
— including sglang's own layer construction that's about to happen.

### Two ways to declare patches

Same effect, pick whichever reads better:

- **`patch("dotted.attr.path", new_fn)`** — one-off, fully qualified
- **`register_from_dict({short: fn, …})`** — well-known targets (short table below)

```python
def register():
    from operator_overrides import patch, register_from_dict
    from my_kernels import faster_ar_gemm, mla_v2, moe_v2

    return [
        # method A
        patch("sglang.srt.layers.quantization.fp8_utils.aiter_w8a8_block_fp8_linear",
              faster_ar_gemm),
        # method B — validates short names against a registry
        *register_from_dict({
            "aiter_mla_decode": mla_v2,
            "aiter_fused_moe":  moe_v2,
        }).values(),
    ]
```

The full short-name registry (`KNOWN_OVERRIDE_TARGETS` in
`operator_overrides.py`) covers 13 named targets across FP8 GEMM, hadamard,
MoE align, MLA, index_k store, rotary, custom AllReduce, aiter dense
paths, fp8 MQA logits, fused MoE. See
[`operator_overrides.py`](operator_overrides.py) for the mapping table.

### When the swap doesn't take

Grep for `[overrides]` in the log. If:

- **No `[overrides]` line at all** — sitecustomize.py didn't run. Check
  `PYTHONPATH` includes `benchmarks/glm5_e2e/shim` and `benchmarks/glm5_e2e`.
- **Only 1 line (from master)** — TP workers didn't pick up the patch.
  Check that env vars are inherited across `multiprocessing.spawn`
  (usually yes; use `spawn` explicitly if you built a custom sglang).
- **Line present but perf identical to baseline** — patch landed but the
  attribute is resolved via a different path than you patched. Some sglang
  paths import a symbol at module load time (`from foo import bar`) rather
  than looking it up on the module — those need patching at both the
  original site and every import site. Grep the sglang tree for the
  attribute name to find all import lines.

## Timing standard

- **Op-level** — sglang timing knobs go through `evaluate_task.py`. The
  default `--repeat 3 --iterations 10` is fast (~30s per task) but n=3 is
  too small to compute `sp_cons`; use `--repeat 10 --iterations 30` (huyan's
  `run_flow.py` default) for the conservative bound. HIP graph
  capture+replay median with cold-L2 flush between iterations.
- **E2E** — `sglang.bench_one_batch`'s own prefill/decode timing. One
  measurement per (batch_size, input_len, output_len) triple. Repeat by
  passing the same shape multiple times in the sweep; first sample is
  warmup, median the rest.

`SGLANG_DISABLE_GFX942_BPRESHUFFLE=1` is set by the shim so the model fits
in HBM (no `weight_original` copy — saves ~5GB). This costs ~30% dense
FP8 GEMM speed vs the ASM bpreshuffle path, and **your baseline reflects
that cost**. If you disable the shim's env var, both baseline and
candidate move; comparisons stay valid, absolute numbers shift.

`--disable-cuda-graph` is passed by the wrappers because HIP graph capture
hit a `float8_e4m3fnuz` codegen bug under sglang's bf16-KV forward pass
at the time of writing. All candidates run under the same flag, so the
comparison is fair.

## What to report

For any speedup claim, include:

1. **Op-level `--repeat 10 --iterations 30`** median and `sp_cons`. A
   median > 1 with `sp_cons < 1` means "faster on average, slower on
   p90" — that's a real regression in production even though the median
   line looks good. Don't ship median-only wins.
2. **E2E prefill or decode TTFT/throughput** for at least 3 runs per
   shape. Drop the first sample as warmup. Report the medians + all
   raw runs.
3. **Correctness `calc_diff`** against the math oracle. Post-PR#12 you
   should see 1e-9 range for FP8 GEMM candidates. `calc_diff ≈ 1` under
   the OLD gate was almost always the harness's fault, not the
   candidate's — re-run under the new gate before rejecting.
4. **Which patches landed**, from the log's `[overrides] →` lines.
5. **Ops that were NOT patched** — a swap that only patches `o_proj` but
   leaves `index_k`, `fused_qkv_a`, etc. on aiter production still bottles
   on those; report which fraction of e2e time is theoretically improvable
   under your patch set.

## Known gaps

- **Incremental prefill** (extend by N tokens on top of a 64k prefix) is
  not yet wrapped. `sglang.bench_serving` with prefix caching would fit;
  scaffolding is a follow-up.
- **`sp_cons` at n=3 per shape** in e2e is not meaningful. To compute a
  conservative bound at e2e level you need n≥5 per shape, which triples
  wall time.
- **Multi-node TP or EP>1** paths are untested here. The compat shim
  targets 8× MI300X on one node.

## Related documents

- [`README.md`](README.md) — user-facing quickstart, environment,
  troubleshooting
- [`operator_overrides.py`](operator_overrides.py) — patch registry
  reference and how to add new short names
- [`shim/PATCHES.md`](shim/PATCHES.md) — the 7 gfx942 compatibility
  patches, each explained with sglang code pointers
- [`archive/replay-20260723-pr12-verify/README.md`](../../archive/replay-20260723-pr12-verify/README.md)
  — worked op-level example (huyan PR #12 candidates)
- [`archive/e2e-prefill-20260723-pr12/README.md`](../../archive/e2e-prefill-20260723-pr12/README.md)
  — worked e2e example (same kernels)
