# GLM-5.2 FlashMLA sparse-decode PTX/SASS follow-up

Retained source and evidence for the exact SGLang `flashmla_kv` FP8 sparse-index
decode ABI at local M16 and M32, continuing from the completed
[`glm52-flashmla-sparse-decode-hotspot-20260729`](../glm52-flashmla-sparse-decode-hotspot-20260729)
campaign without reopening its rejected claim.

Terminal disposition: **`no-replacement`**. One material candidate was built. It
is bitwise exact and passes both mandatory CUDA Graph lanes at ~1.06x at both
buckets on a null-validated instrument, but the eager containing SGLang DSA
region measures 0.66–0.76 and that lane is enumerated by the plan. An identity
control proves the failing lane is an API-v1 integration property (+17.4 µs of
host Python) rather than a kernel property.

Start here:

- [`FINAL_REPORT.md`](FINAL_REPORT.md)
- [`evidence/attempt_ledger.json`](evidence/attempt_ledger.json)
- [`evidence/timing_gate_audit.json`](evidence/timing_gate_audit.json) — every
  gate decision recomputed from raw ordered pairs
- [`evidence/binary_manifest.json`](evidence/binary_manifest.json)
- [`evidence/p0_mechanism_m16.json`](evidence/p0_mechanism_m16.json) — the single
  NCU capture and the shared-memory headroom that bounded the search
- [`EXTERNAL_ACCEPTANCE.md`](EXTERNAL_ACCEPTANCE.md)

## Two results worth carrying forward

1. **The measurement instrument mattered more than any candidate.** A
   stock-versus-stock null — the identical installed binary in both arms — spreads
   ±1.23% at one call per CUDA-event observation, and the prior campaign's
   control spread ±2.9%. That is the same size as the effect under test, so that
   instrument cannot resolve a 1.03 gate at ~30 µs. Twenty calls per observation
   tightens the null to ±0.28% and removes 2.86 µs of per-observation instrument
   cost that had been diluting both arms toward 1.0.

2. **Deleting the scattered scale chain is worth ~1.00x; relocating it is worth
   ~1.06x.** Its cost is its position in the dependency graph, not its
   instruction count or memory traffic — and deleting it also destroys an
   accidental L2 prefetch, because the FP32 scales at token offset `[512,528)`
   share 128-byte sectors with the RoPE bytes at `[528,656)` that the RoPE gather
   then reads. This is why seven prior candidates that only reduced instruction
   counts all measured ~1.00.

## Harness

Scripts under [`harness/`](harness/) refuse to overwrite evidence. Builds refuse
to run inside a GPU lease; every CUDA command requires the shared lease via
`with_hotspot_gpu.sh`.

```bash
export SGLANG_ROOT=/home/qinhaiyan/glm52-hotspot-goal-runs/worktrees/flashmla-sparse-decode/sglang

# CPU-only, no lease
python3 harness/cpu_audit.py --output evidence/cpu_audit.json
.venv/bin/python harness/build_variant.py --variant p1_consumer_scale --output /tmp/build.json

# Requires the shared GPU lease
GLM52_FLASHMLA_VARIANT=p1_consumer_scale \
  /home/qinhaiyan/glm52-hotspot-goal-runs/with_hotspot_gpu.sh -- \
  .venv/bin/python harness/measure_paired.py --m 16 --comparison stock_provider \
    --lanes graph_gate --series 3 --pairs 100 --replays-per-observation 20 \
    --output /tmp/out.json
```

`measure_paired.py --lanes graph_gate` runs the plan's first-screen lanes alone so
a rejected candidate never consumes the eager lanes. `--comparison stock_stock`
runs the null control.

## Variants

The provider is
[`serving_native/candidates/flashmla_sparse_decode_provider.py`](../../serving_native/candidates/flashmla_sparse_decode_provider.py)
and requires an explicit `GLM52_FLASHMLA_VARIANT`. All compiled globals begin
with `infini_kernel_glm52_flashmla_sparse_decode`. No variant is enabled by
default in SGLang, and every new FlashMLA code path is behind a `GLM52_*` macro
so the default upstream instantiations are unchanged.

| Variant | Role |
|---|---|
| `identity` | control, source-identical to upstream V32; never promotable |
| `b3_b5_native_exact` | prior campaign's rejected composite, rebuilt to bind its terminal result |
| `p1_consumer_scale` | this campaign's candidate |
| `ablate_scale_chain` | diagnostic ablation, numerically wrong by construction; never promotable and never used for a correctness run |
