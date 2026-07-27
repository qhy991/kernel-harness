# Attempt ledger

## Baseline collection

- `baseline_20260723a`: stopped during initial evidence plumbing; preserved with
  its failure note.
- `baseline_20260723b`: reached paired measurements but the runtime JSON was
  polluted by compiler diagnostics; preserved.
- `baseline_20260723c`: compiler diagnostics disabled, malformed raw output
  retained, deterministic JSON prefix normalized and hashed; this is the
  baseline used by the report.

## Kernel attempts

1. **Packed-native warp staging (v1).**
   Hypothesis: remove SFA/SFB TMA issue work from producer warp 0 by loading
   already-packed int32 words in transposer warp 2. Correctness passed and no
   adapter kernel appeared. Rejected because the isolated module retained a
   128-SM default and, even aside from launch-wrapper timing, NCU showed a
   slower kernel with additional global/shared instructions.

2. **Fixed N/K specialization.**
   Hypothesis: compile `N=16384,K=2048` into the kernel to remove dynamic
   scheduler/address work. Three paired series per bucket did not improve the
   exact-shape variant, and NCU remained slower. Rejected.

3. **Prefetch plus vectorized SFB stores (v2).**
   Hypothesis: issue packed reads before the A/B barrier, retain them in
   registers, replace four scalar SFB stores per lane with one `uint4`, and
   restore stock's 148-SM launch. This corrected the comparison and improved
   v1, but NCU remained slower and graph replay regressed at both buckets.
   Rejected.

4. **Lane-0 SFB load plus warp broadcast (v3).**
   Hypothesis: the 128-row weight block has one repeated scale word, so one
   global load followed by a shuffle can replace 32 redundant loads. The change
   improved the M32 device kernel relative to v2 but increased registers and
   did not beat stock; M16 became slightly slower. Production layer and graph
   replay both regressed. Rejected and selected as the final documented
   no-replacement experiment.

## Interpretation

The apparently large `serving_native` leaf speedups are launch-submission
effects: CUDA events bracket the Python/custom-op call, so an idle device sees
the CPU launch gap. NCU proves the experimental device kernels are slower, and
graph replay removes the host gap and also regresses. The binding limit is the
stock one-wave, shared-memory-limited TMA pipeline; replacing its scale TMAs
with lane instructions adds work without increasing useful overlap.

## Rollback

No production bucket is promoted. The rollback/reference is always
`SGLANG_GLM52_OPT=0`; the default `serving_safe` profile with no explicit q_b
allowlist also calls stock immediately.
