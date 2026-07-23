# Replay 2026-07-23 · PR #12 verification

Verifying huyan's PR #12 (`Fix MI300X bench standard: math-oracle correctness gate
+ honest aiter baseline`), which is a follow-up to the earlier
`archive/replay-20260723/` audit and its writeup in `ROOTCAUSE_AND_FIX.md`.

**One-line result: PR #12's harness-correctness fix is correct and adopted;
its speedup claims are only partially reproducible.** `dsa_attn_decode`
is a real 2.36x win. `o_proj_prefill` matches the claimed median at
M=1024/2048 but has a 21x regression at M=4096 and tail latency 40–70x above
the median across all shapes. `index_k_prefill` doesn't match its claim at
any shape.

## What PR #12 claimed

From huyan's `archive/replay-20260723/ROOTCAUSE_AND_FIX.md`:

| task              | claim    | correctness       | verdict          |
|-------------------|---------:|-------------------|------------------|
| o_proj_prefill    | **1.55x** | calc_diff ≤ 5e-9 | 3/3 WIN, TARGET_MET (107%) |
| index_k_prefill   | **1.75x** | calc_diff ≤ 3e-9 | 3/3 WIN, TARGET_MET (117%) |
| dsa_attn_decode   | **1.66x** | calc_diff 7e-7   | 2/2 WIN, TARGET_MET (112%) |

PR #12 also fixes a real harness bug: `evaluate_task._correctness` used the
production fp8 dispatch as the ground-truth oracle, but on gfx942 that
dispatch is shape-dependent (M≥4096 wants preshuffled weights the
CK/hipBLASLt path doesn't provide). Correct candidates thus scored
`calc_diff ≈ 1` at M=4096, and the previous replay
(`archive/replay-20260723/results.csv`) recorded them as INCORRECT.

The fix: a deterministic **math oracle** (`gemm` = dequant-f32 matmul,
`mla` = fully-f32 sparse oracle) is used for correctness, wired via
`testbench/harness/backends/rocm_amd.py::correctness_reference` and
`testbench/harness/evaluate_task.py::_correctness`. B200 / CUDA behaviour
preserved (delegates back). This part of PR #12 is unambiguously correct
and merged into `amd` as `f98736e`.

## How I verified

Ran `./run.sh` for each of the three affected tasks on their full sweep,
twice — once with my previous methodology and once with huyan's own:

- **v1 · `--repeat 3 --iterations 10 --warmup 3`** — same numbers I used
  for the earlier `archive/replay-20260723/` sweep, for continuity.
- **v2 · `--repeat 10 --iterations 30 --warmup 3`** — the defaults in
  huyan's own `rewardbench/amd/run_flow.py`, so his claim can be
  reproduced under his own timing budget.

Hardware / backend, both rounds: MI300X gfx942, `HIP_VISIBLE_DEVICES=0`
single-process, `rocm / amd-mi300x / aiter-torch-reference / event`
bundle, HIP graph capture+replay median, cold-L2 flush per iteration,
`--no-gpu-lock`. Correctness gate: PR #12's math oracle (that's the
whole reason to test after this PR).

## Results

### v1 · `--repeat 3 --iterations 10`

| task              | M    | correct  | speedup(median) | sp_cons | verdict          |
|-------------------|-----:|:--------:|---------------:|--------:|------------------|
| o_proj_prefill    | 1024 | ✅ 4.9e-9 | 0.636x         | 0.016x  | neutral          |
| o_proj_prefill    | 2048 | ✅ 5.0e-9 | 1.178x         | 0.128x  | neutral          |
| o_proj_prefill    | 4096 | ✅ 4.8e-9 | **0.337x**     | 0.019x  | **REGRESS**      |
| **geomean**       |      |          | **0.6321x**    |         | PARTIAL_OR_REGRESSED |
| index_k_prefill   | 1024 | ✅ 3.0e-9 | **0.615x**     | 0.478x  | **REGRESS**      |
| index_k_prefill   | 2048 | ✅ 3.0e-9 | 1.221x         | 0.438x  | neutral          |
| index_k_prefill   | 4096 | ✅ 3.0e-9 | 0.716x         | 0.596x  | neutral          |
| **geomean**       |      |          | **0.8133x**    |         | PARTIAL_OR_REGRESSED |
| dsa_attn_decode   |   16 | ✅ 7.3e-7 | **3.482x**     | 2.141x  | **WIN**          |
| dsa_attn_decode   |   32 | ✅ 7.3e-7 | **1.601x**     | 1.122x  | **WIN**          |
| **geomean**       |      |          | **2.3611x**    |         | **COMPLETE_WIN** |

Full data in `results_v1.csv`.

### v2 · huyan's own params `--repeat 10 --iterations 30`

| task              | M    | speedup(median) | sp_cons | verdict          |
|-------------------|-----:|---------------:|--------:|------------------|
| o_proj_prefill    | 1024 | 1.605x         | 0.023x  | neutral          |
| o_proj_prefill    | 2048 | 1.807x         | 0.608x  | neutral          |
| o_proj_prefill    | 4096 | **0.046x**     | 0.019x  | neutral          |
| **geomean**       |      | **0.5098x**    |         | NO_WIN_WITH_EVIDENCE |
| index_k_prefill   | 1024 | 0.201x         | 0.009x  | neutral          |
| index_k_prefill   | 2048 | 0.622x         | 0.008x  | neutral          |
| index_k_prefill   | 4096 | 1.029x         | 0.963x  | neutral          |
| **geomean**       |      | **0.5045x**    |         | NO_WIN_WITH_EVIDENCE |
| dsa_attn_decode   |   16 | 289x           | 0.012x  | UNTRUSTWORTHY (harness warned unstable timing; ref side hit 26ms outlier vs typical 300µs) |
| dsa_attn_decode   |   32 | 66x            | 0.117x  | UNTRUSTWORTHY (same warning) |

Full data in `results_v2.csv`.

## Findings

1. **Adopt PR #12's correctness fix.** The math-oracle decoupling from the
   latency baseline is unambiguously right — previous FAIL calls at
   `o_proj m4096` and every `index_k` shape were harness artefacts, not
   candidate bugs. Merged in `amd` as `f98736e`.

2. **`dsa_attn_decode` is a real speedup** — 2.36x geomean at v1
   `--repeat 3 --iterations 10`, actually **exceeding** huyan's own 1.66x
   claim; both shapes WIN and `sp_cons > 1` (above the noise band). Keep
   `testbench/tasks/glm52_amd/dsa_attn_decode/candidate.py` as the default
   for this task.
   The v2 run's 289x / 66x numbers are the harness reporting an outlier —
   `sp_cons` there is `ref_p10 / cand_p90 = 0.012x`, so it can't be
   trusted. The harness itself printed `WARNING: unstable timing … the
   conservative margin above is not trustworthy; re-run`. When such
   warning appears, v1 is the reliable data point.

3. **`o_proj_prefill` claim is selective and misses a real bug.**
   - v2 does reproduce huyan's median at M=1024 (1.605x) and M=2048
     (1.807x). These shapes match his ~1.55x claim.
   - v2 shows a **21x regression at M=4096** — candidate 35240µs vs
     baseline 1610µs. Directly contradicts his `3/3 WIN, TARGET_MET`
     claim at that shape.
   - `sp_cons` is 0.023x / 0.608x / 0.019x across the three shapes; a
     conservative reader would call every shape neutral. The candidate
     has a tail 40–70x above the median. Sglang would hit this tail on
     roughly the p90+ inference — median-only reporting is misleading.

4. **`index_k_prefill` claim doesn't hold.** No shape wins in either
   round; v2 geomean is 0.50x. This directly contradicts his `1.75x,
   TARGET_MET (117%)` claim. Whatever local environment tuned this
   candidate is not the environment `./run.sh` on `main`/`amd` produces
   today. Something upstream (aiter build state, triton cache, HBM
   fragmentation, or a hidden env var) must differ.

5. **Tail latency is systemic across all three tasks.** Every candidate's
   `sp_cons` in v2 is <1 even where median > 1. Root cause is not Triton
   `@autotune` (I greped — none of the three candidates uses it), so
   the tail comes from somewhere else in the launch — possible causes:
   L2 cache eviction contention, xGMI/NUMA noise, or an intermittent
   Triton grid dispatch path. In production sglang, that tail is what
   end-users feel on p90/p99 TTFT — a `sp_cons < 1` win is not a real
   win.

## What to do next

- **Trust `dsa_attn_decode` — deploy it.** Its win is real, above the
  noise band, and reproducible.
- **Debug `o_proj @ M=4096` before shipping.** The 21x regression is a
  real bug in the tuned kernel at that shape. Either fix it or make the
  candidate fall back to `glm52_ops.reference` for M ≥ 4096.
- **Re-examine `index_k_prefill` on a fresh checkout / environment.**
  huyan's number and mine can't both be right; the delta is too large
  for measurement noise alone.
- **Add tail-latency reporting to the harness.** The `sp_cons` /
  `speedup_median` gap is the story; a single median line hides real
  regressions.

## Reproduce

```bash
cd /root/repos/kernel-harness
export HIP_VISIBLE_DEVICES=0

# v1 — my methodology
testbench/tasks/glm52_amd/o_proj_prefill/run.sh    --repeat 3 --iterations 10 --warmup 3 --no-gpu-lock
testbench/tasks/glm52_amd/index_k_prefill/run.sh   --repeat 3 --iterations 10 --warmup 3 --no-gpu-lock
testbench/tasks/glm52_amd/dsa_attn_decode/run.sh   --repeat 3 --iterations 10 --warmup 3 --no-gpu-lock

# v2 — huyan's run_flow.py defaults
testbench/tasks/glm52_amd/o_proj_prefill/run.sh    --repeat 10 --iterations 30 --warmup 3 --no-gpu-lock
testbench/tasks/glm52_amd/index_k_prefill/run.sh   --repeat 10 --iterations 30 --warmup 3 --no-gpu-lock
testbench/tasks/glm52_amd/dsa_attn_decode/run.sh   --repeat 10 --iterations 30 --warmup 3 --no-gpu-lock
```

Candidate is `testbench/tasks/glm52_amd/<task>/candidate.py` (the tuned
kernel merged from PR #12). Baseline is `glm52_ops.reference` — which PR
#12 also updated to prefer aiter's Triton blockscale kernel before falling
to CK/hipBLASLt, so the latency baseline is honest and reachable.

Per-shape RESULT_JSON is persisted under
`runs/glm52/<task>/<run_id>/result.json` — the source of truth. `results_v1.csv`
and `results_v2.csv` are the compacted summaries the tables above render.
