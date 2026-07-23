# E2E decode · 2026-07-23 · partial baseline only

**Partial** e2e decode baseline for GLM-5.2-FP8 on 8× MI300X, TP=8.
`bench_serving-style` end-to-end via `python -m sglang.bench_one_batch`,
`--dsa-decode-backend aiter` (the only decode backend that boots on
gfx942 in this sglang build; `fa3` needs `flash_attn_with_kvcache`
which isn't compiled, and `flashmla_sparse` needs
`sgl_kernel.flashmla_ops` which needs NVRTC).

**Not archived here yet**: override run (huyan PR #12 dsa_attn_decode
kernel), because the aiter decode kernel GPU-faults at bs=256 mid-run
and leaves the 8-rank sglang process stuck in `D` (uninterruptible)
state, with ~125 GB HBM allocated per rank and unreleasable without
`sudo rocm-smi --gpureset` or a reboot. **User is expected to reset
GPUs manually; override run will land in a follow-up.**

## Baseline data (from `bench_one_batch` log)

| bs   | input_len | output_len | prefill (s) | total (s) | overall tok/s | decode-only tok/s | status |
|-----:|----------:|-----------:|------------:|----------:|--------------:|------------------:|--------|
|   16 | 128       | 64         | 0.40        | 27.3      |         112.4 |              38.1 | ok     |
|   32 | 128       | 64         | 0.72        | 19.6      |         312.7 |             108.3 | ok     |
|   64 | 128       | 64         | 7.02        | 27.6      |         445.6 |             198.6 | ok (prefill spike; aiter shape recompile) |
| **128** | **128** | **64**   | **2.45**    | **21.4**  | **1150.0**    | **433.5**         | **ok — this is the user-requested batch** |
|  256 | 128       | 64         | 4.10        | crashed   |             — |                 — | **CRASH** — aiter illegal memory access; GPUs stuck in D-state |

`decode-only tok/s` = `bs × output_len / (total_lat − prefill_lat)`. It's
what serving-mode consumers actually see for decoding after prefill is
done. `overall tok/s` includes input tokens in the numerator, which is
sglang's default reporting.

## The bs=256 crash (context for user)

Not caused by our shim, our overrides, or PR #12. Boots into
`_run_aiter_mla_decode_fwd` (compat-patched but only for `_store_index_k_cache`),
then somewhere inside aiter's MLA decode kernel at bs=256 hits an
illegal memory access (`hipErrorLaunchFailure`). Every rank throws, NCCL
aborts, but the master python process ends up in kernel D-state trying
to release ROCm resources — `kill -9` doesn't work because it's in
uninterruptible I/O.

Effect: 8/8 GPUs pinned with ~125 GB HBM each. Cannot launch another
sglang run until:

- The ROCm driver eventually times out (can be minutes to hours),
- Someone runs `sudo rocm-smi --gpureset -d <N>` per rank,
- Or the node reboots.

**bs=128 works fine.** So the user-requested "global batch=128" case has
a real baseline number. The "global batch=256" case is aiter's
responsibility to fix; this bench faithfully surfaced the crash.

## Files

- `baseline_bs_sweep.csv` — the numbers above, machine-readable
- `baseline_bs16-256_crash.log` — empty placeholder (the raw log was
  overwritten by a subsequent bs=128-only rerun before it could be
  copied; the values in the CSV were extracted before overwrite via
  `re.findall` and are the source-of-truth for what to reproduce)

## Reproduce

After the user resets GPUs:

```bash
cd /root/repos/kernel-harness

# baseline (bs=128 works; bs=256 will crash — split the runs)
KDA_E2E_OUT=/tmp/glm5_e2e benchmarks/glm5_e2e/run_decode_throughput.sh \
  --batch-size "128 128 128"

# with override (huyan PR #12 dsa_attn_decode candidate) — TODO:
# needs a new example_overrides that patches
#   sglang.srt.layers.attention.dsa_backend.DSABackend._run_aiter_mla_decode_fwd
# to route through the archive/0723-amd-glm52/dsa_attn_decode wrappers
# (currently only o_proj example exists).
```

## What's missing (follow-up)

1. **Reset the GPUs** — `sudo rocm-smi --gpureset` per rank.
2. **Write `examples/huyan_pr12_dsa_attn_decode.py`** — an override that
   installs the PR #12 tuned dsa_attn_decode kernel via
   `patch("sglang.srt.layers.attention.dsa_backend.DSABackend._run_aiter_mla_decode_fwd", ...)`.
   Same pattern as `huyan_pr12_o_proj_prefill.py` but for the MLA
   dispatcher. Op-level tests recorded 2.36x geomean speedup for that
   kernel (`archive/replay-20260723-pr12-verify/results_v1.csv`); e2e
   will show how much of that survives layer-wise amortisation.
3. **Then run**: baseline bs=128×3, override bs=128×3, compare medians.
4. **Skip bs=256** until aiter fixes it upstream, or run it separately
   under a supervisor that force-resets GPUs on crash.
