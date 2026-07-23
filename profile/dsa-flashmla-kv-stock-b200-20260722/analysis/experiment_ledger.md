# Experiment ledger

The production gate is bucket-local: at least 3% paired-p50 improvement in a
repeated alternating session, exact correctness, graph/stream semantics, then a
containing-region and complete-server win. An isolated profiler delta is not a
promotion result.

The original Experiment 0/1 measurements below are preserved as historical
fixed-GPU evidence. The scheduler-corrected authority is the later
`flex-20260723T160729Z` revalidation, which keeps the whole alternating campaign
and profiler collection on one dynamically selected physical B200.

## Ranked hypotheses

1. **Reduce the M16 combine compile-time split bound.** Runtime metadata shows
   eight splits/request at M16, while stock selects the bound-160 combine body
   from 148 scheduler parts. This is a small, exact-guarded host dispatch change
   with no ABI or workspace change, so it was tested first.
2. **Change main-kernel split/scheduler balance.** The main kernel occupies all
   148 SMs for one wave, and the simulator can enumerate alternative splits,
   but changing the scheduler affects work ownership, PDL dependencies, and
   combine layout. The measured main kernel is already a one-wave sparse gather;
   this higher-risk idea was not implemented after hypothesis 1 failed the full
   graph chain.
3. **Fuse or persist main plus combine.** Nsight Systems proves PDL overlap and
   zero launch gap. Fusion would have to eliminate the 16.8-MB combine gather
   without sacrificing the 148-block sparse main, not merely save a launch.
   This requires a materially different algorithm and was not justified by the
   rejected low-risk experiment.
4. **Rewrite sparse index/cache movement or FP8 dequantization.** Nsight Compute
   identifies low L2 hit rate, long scoreboards, sector under-use, and shared
   conflicts, so this is the strongest future device-code direction. It is also
   the largest correctness and maintenance risk and needs complete-model/TP8
   validation unavailable on this host.

The CPU-only [scheduler simulation](scheduler_simulation.json) is a hypothesis
aid, not dynamic performance evidence.

## Build/ABI bring-up (not a performance experiment)

The base-source `stock-control`, optional-caster `stock-pybind`, and initial
`combine32-m16` builds are retained in `build_*.json`. They established that the
source and SM100 translation units compile, but the current PyTorch 2.11 pybind
call needed both optional and Tensor caster registration before the production
Tensor ABI could be invoked reliably. The compatibility changes were applied
symmetrically to the control (`0657fff`) and candidate (`d18ff63`); only those
final `*_tensor` artifacts were timed. The earlier artifacts are rejected as
ABI bring-up, carry no speed claim, and roll back simply by selecting neither
overlay. This distinction prevents a build/import fix from being counted as a
kernel optimization.

## Experiment 0: pinned-source rebuild control

- **Hypothesis:** rebuilding the pinned FlashMLA source as an import overlay
  should be performance-neutral; its paired distribution bounds compiler,
  loader, and session noise before interpreting a source change.
- **Baseline evidence:** the installed `sgl-kernel 0.4.4` extension is SHA-256
  `d8d97150...`; three unpaired repeat-50 stock sessions drift from 0.049456 to
  0.044544 ms at M16 and 0.049920 to 0.047728 ms at M32, so cold/warm sessions
  cannot be compared as a promotion claim.
- **Exact delta:** FlashMLA base `05e26647` plus only the PyTorch tensor-caster
  binding needed by this environment, committed as `0657fff`; build artifact
  SHA-256 `b1afc294...`. Device dispatch and kernels are stock.
- **Expected effect:** identical production outputs, scheduler metadata, device
  code, and latency centered at unity; no PTX/SASS optimization is claimed.
- **Correctness:** exact output checks, invalid-index handling, current-stream
  behavior, and native graph replay passed at M16 and M32.
- **Paired p50 and distribution:** eager session speedups were
  `1.011679/0.992644/1.027925` (M16) and
  `1.018621/1.003591/1.010142` (M32). Graph speedups were
  `0.996567/0.999013/0.989950` and
  `0.982680/0.992860/0.991372`. Every session missed 1.03.
- **Profiler delta:** static decoded SASS/resources match the candidate build;
  this experiment exists to calibrate the overlay, not to claim a profiler win.
- **Risk:** tensor-caster compatibility is host binding code and could mask an
  import mismatch. The overlay verifies manifest, source patch, extension hash,
  symbol, shapes, and metadata before measurement.
- **Decision:** retain as the compiler/build control; never promote it as an
  optimization.
- **Rollback point:** installed stock extension, which was never overwritten;
  isolated control source commit `0657fffdfd1c981517647e043e4ef30ffdc1480f`.

## Experiment 1: M16 combine bound-32 dispatch

- **Hypothesis:** the fixed M16 runtime needs eight splits/request, so dispatching
  the already-compiled combine `max_num_splits=32` body instead of bound 160
  should reduce shared state and combine instructions without changing the main
  kernel or ABI.
- **Baseline evidence:** stock Nsight Systems measures M16 main/combine at
  17.568/13.632 us with 4.160 us PDL overlap and a 27.040 us chain. The stock
  combine uses 128 blocks, 48 registers/thread, and 6,144 bytes/block as reported
  by cuobjdump. It is a material part of the chain despite overlapping main.
- **Exact code/config delta:** FlashMLA commit `d18ff63` adds
  `CombineParams.max_num_splits` and an exact SM100 guard for M16, Q head64,
  V32, sparse top-k 2048, FP8 paged KV64, scale 0.0625, scheduler shape `[148,8]`,
  and the production contiguous layout. The guard selects 32 only there. M32,
  main launch, scheduler metadata, buffers, PDL, stream, and unsupported ABIs use
  the existing stock fields and dispatch.
- **Expected PTX/SASS/runtime effect:** select the existing bound-32 BF16 combine
  body, reducing ptxas static shared memory from 5 KiB to 1 KiB and its static
  body from 344 to 280 SASS records. The sparse main executable must remain
  unchanged. A useful result must reduce the complete captured chain by at least
  3%, not merely the isolated combine.
- **Correctness:** installed stock, rebuilt control, and candidate agree at both
  buckets. Invalid `-1` slots, nondefault stream, output non-aliasing, graph
  capture/replay, and mutated-input replay checks all pass.
- **Paired p50 and distribution:** eager candidate sessions were
  `1.036845/1.016081/1.020847` at M16 and
  `0.989919/0.993737/1.010906` at M32. Only the first M16 session crossed 1.03.
  Graph sessions were `0.998068/0.987407/0.983109` at M16 and
  `0.993963/0.994055/0.999149` at M32; all six failed. See the
  [generated paired summary](paired_measurements_summary.md) for every raw p50,
  pair distribution, and graph check.
- **Profiler delta:** Nsight Systems records candidate M16 combine 13.216 us and
  chain 26.720 us in one trace, only 1.18% below stock chain. Nsight Compute
  replay records 10.784 us versus 11.008 us for isolated combine (2.03%), shared
  memory 2,048 versus 6,144 bytes/block, but virtually identical 16.817/16.818 MB
  DRAM reads and worse long-scoreboard ratio; 291/400 versus 288/402 PC samples
  are long-scoreboard stalls. Static audit confirms the predicted bound-32 body
  and unchanged main SASS.
- **Risk:** a future broadened guard could under-bound scheduler splits; reducing
  shared resources alone cannot fix the 128-block tail or sparse gather. The
  current exact predicate and stock else-path fail closed, but any deployment
  would still require complete backend, model, and TP8 graph validation.
- **Decision:** reject. The apparent first eager win is not repeatable and every
  graph session misses the gate. Do not integrate or enable any bucket.
- **Rollback point:** operational rollback is the empty enable policy and stock
  installed extension. Source rollback is control commit `0657fff`; the rejected
  attempt remains reproducibly preserved at `d18ff63` and in
  [the source patch](flashmla_d18ff63.patch).

## Scheduler-corrected revalidation: `flex-20260723T160729Z`

- **Scheduler/provenance delta:** no source or build changed. Kernel-Harness
  commit `8c18448` replaced the superseded fixed-GPU assumption with one
  `with_flexible_gpu.sh` lease and immutable campaign artifacts. The wrapper
  allocated physical GPU 1,
  `GPU-5b9be10b-5bfc-b658-9b31-f7ae8516dc54`, as logical GPU 0 for the complete
  paired-plus-profiler command.
- **Correctness:** both runtime ABI/invalid-index/stream/graph checks, both exact
  SGLang tests, and all 24 paired producer checks passed.
- **Paired result:** candidate eager session speedups were
  `1.024761/1.014102/1.016634` at M16 and
  `1.004522/1.021972/0.998688` at M32. Candidate graph speedups were
  `0.989357/0.990007/0.989124` at M16 and
  `0.992105/0.985662/0.991270` at M32. No candidate or rebuild-control session
  reached 1.03.
- **Profiler result:** Nsys measured stock/candidate M16 chains at
  25.888/25.536 us with zero launch gap, a descriptive 1.36% difference. NCU
  measured stock/candidate isolated combine at 10.752/10.912 us; both read about
  16.8 MB with about 0.5% L2 hit rate. The candidate increased long-scoreboard
  PC samples from 271/390 (69.5%) to 298/401 (74.3%).
- **Risk/decision/rollback:** the exact M16 predicate still fails closed, but
  deploying it has no measured benefit. Reject it, enable no bucket, and keep
  the installed stock extension active. The full current report is
  [`../campaigns/flex-20260723T160729Z/REPORT.md`](../campaigns/flex-20260723T160729Z/REPORT.md).

## Final conclusion

This path is not launch-gap bound. The main kernel has a one-wave, low-occupancy
sparse gather with barrier and long-scoreboard pressure; M16 combine is a
grid-underfilled, long-scoreboard-dominated 16.8-MB gather. Selecting a smaller
precompiled combine bound reduces shared bookkeeping but not the binding memory
latency or total graph chain. The only honest disposition is no replacement.
