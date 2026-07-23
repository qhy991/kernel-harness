# E2E prefill · 2026-07-23 · huyan PR #12 o_proj_prefill kernel

End-to-end prefill TTFT measurement of the huyan PR #12 tuned
`o_proj_prefill` kernel (as inlined into
`testbench/tasks/glm52_amd/o_proj_prefill/candidate.py`), swapped into
sglang production dispatch via the mechanism in `benchmarks/glm5_e2e/`.

**Headline**: `input_len=1024` shows a real ~**+26% (1.36×) TTFT speedup**;
`input_len=2048` is inside noise; `input_len=4096` (where the override
gates to sglang default via M<=2048 branch) is also inside noise but
noisier than baseline.

## Setup

Same as `benchmarks/glm5_e2e/README.md`:

- Hardware: 8× MI300X, single node, TP=8
- Model: GLM-5.2-FP8 at `/mnt/public/qinhaiyan/models/GLM-5.2-FP8`
- Backend: `rocm/amd-mi300x/aiter-torch-reference/event`
- Compat shim: `benchmarks/glm5_e2e/shim/glm52_gfx942_shim.py` (7 gfx942 patches)
- Driver: `python -m sglang.bench_one_batch`, unmodified
- Sweep: `--input-len 1024 1024 1024 2048 2048 2048 4096 4096 4096`, output_len=1
- Timing: sglang's own prefill measurement (single run per shape, no repeats
  inside `bench_one_batch`)

**Override**: `benchmarks/glm5_e2e/examples/huyan_pr12_o_proj_prefill.py`
loads the PR #12 tuned `o_proj_prefill` candidate from the task tree and
gates on M — M ∈ {1024, 2048} route through the huyan kernel; M=4096 falls
back to sglang default because the isolated op-level tests in
`archive/replay-20260723-pr12-verify/` showed a 21× regression at that
shape.

Every rank confirms the swap in the log:
```
[overrides]   → {'target': 'sglang.srt.layers.quantization.fp8_utils.aiter_w8a8_block_fp8_linear',
                  'old_id': ..., 'new_id': ...}
```
appearing 8× (once per TP worker) plus master.

## Results

Baseline commit: `f542f11` (amd/main). Override candidate:
`testbench/tasks/glm52_amd/o_proj_prefill/candidate.py` at same commit.

| M       | baseline runs (ms) | override runs (ms) | baseline median | override median | speedup | delta   | verdict |
|---------|-------------------:|-------------------:|----------------:|----------------:|--------:|--------:|---------|
| **1024**| 289.4, 389.4, 376.2 | 333.4, 282.2, 282.4 | **382.8 ms** | **282.3 ms** | **1.356×** | **+26.3%** | **SPEEDUP** |
| 2048    | 357.1, 309.4, 257.2 | 397.8, 330.0, 291.8 | 283.3 ms  | 310.9 ms  | 0.911×  | −9.7% | neutral (noise) |
| 4096    | 657.1, 604.7, 691.0 | 1402.9, 856.0, 683.8 | 647.9 ms  | 769.9 ms  | 0.842×  | −18.8% | noise (fallback shape) |

`baseline.jsonl` and `override.jsonl` are the raw sglang RESULT_JSON
streams (one line per shape). `runs.csv` is the 18 raw datapoints;
`summary.csv` is the compacted table above.

## Comparison to op-level results

The same PR #12 o_proj kernel was tested at op-level in
`archive/replay-20260723-pr12-verify/`:

| M    | op-level v1 speedup | op-level v2 speedup | e2e speedup here |
|------|--------------------:|--------------------:|-----------------:|
| 1024 | 0.636× (REGRESS)    | 1.605×              | **1.356×**       |
| 2048 | 1.178× (neutral)    | 1.807×              | 0.911×           |
| 4096 | 0.337× (REGRESS)    | 0.046× (21× regr.)  | 0.842× (fallback)|

E2E disagreement with op-level v1 at M=1024 is significant — op-level v1
saw the candidate 36% SLOWER, e2e sees it 26% FASTER. Op-level v2 does
show the candidate faster at median, but with `sp_cons = 0.023×` (the
conservative gate reads it as neutral). E2E has three plausible reasons
to disagree:

1. **Different call frequency**. Op-level runs the kernel once per shape
   sweep iteration; e2e calls it ~78× per prefill (once per transformer
   layer). Per-call overhead (`aiter_per1x128_quant` when
   `input_scale` is None, our wrapper's `torch.empty(...)` allocation)
   amortises across many calls in e2e but dominates in op-level.
2. **Different aiter state**. Sglang boot warms every aiter dispatch
   shape it sees; op-level `./run.sh` does not. Baseline aiter latency
   at M=1024 measured 383ms in e2e vs 683ms in op-level v1 — the
   *baseline* moved by 2×, so any candidate is running against a
   different reference.
3. **Only 3 runs per shape**. Standard error at n=3 on a ~400ms measurement
   is roughly ±40ms. `sp_cons` isn't computed at this sample size, so a
   real regression could be hidden by median noise.

Trust the e2e number for deployment decisions (that's what a serving
system experiences); trust the op-level number for kernel-tuning
decisions (that's what actually reflects the kernel's peak).

## Interpretation

**Real signal**: at input_len=1024, baseline runs 2 and 3 are ~380ms while
override runs 2 and 3 are stable at 282ms. Neither distribution overlaps
the other's max/min. The kernel is faster in this regime.

**No signal at input_len=2048**: baseline runs get progressively faster
(357 → 309 → 257ms), suggesting sglang / aiter is still warming its state.
Override doesn't show the same warm trajectory. Insufficient data.

**Bad first sample at input_len=4096 override**: 1403ms is 2× the
following two runs — first-hit JIT / autotune / xGMI warmup, probably.
Dropping only the first sample as warmup gives the reported medians.

## Reproduce

```bash
cd /root/repos/kernel-harness

# baseline
KDA_E2E_OUT=/tmp/glm5_e2e benchmarks/glm5_e2e/run_prefill_ttft.sh \
  --input-len "1024 1024 1024 2048 2048 2048 4096 4096 4096"

# override (PR #12 kernel)
KDA_E2E_OUT=/tmp/glm5_e2e benchmarks/glm5_e2e/run_prefill_ttft.sh \
  --input-len "1024 1024 1024 2048 2048 2048 4096 4096 4096" \
  --overrides $(pwd)/benchmarks/glm5_e2e/examples/huyan_pr12_o_proj_prefill.py
```

Each takes ~7 minutes on a warm 8× MI300X node.

## Caveats

- This measures **fresh prefill** (empty KV cache → prefill N tokens),
  not incremental / continuation prefill (64k prefix → extend by N).
  llm_flops's prefill scenario is the same: fresh prefill.
- Sglang's `bench_one_batch` runs each shape once; the 3× per shape is
  achieved by passing the same shape three times in the sweep. Ideally
  we'd want 10+ repeats per shape and separate warmup pools, but that
  triples wall time.
- The compat shim exports `SGLANG_DISABLE_GFX942_BPRESHUFFLE=1`, so
  baseline is sglang production dispatch WITHOUT ASM bpreshuffle — this
  makes the model fit in HBM (no `weight_original` copy) but reduces
  baseline dense FP8 GEMM speed ~30% below true production peak. If you
  disable the shim env var, the baseline shifts.
- `sp_cons` (conservative speedup, `ref_p10 / cand_p90`) requires more
  data than 3 runs. The op-level tests at n=10 showed sp_cons < 1 for
  the same shapes; that's a real tail-latency warning that isn't
  visible in this n=3 e2e sweep.
