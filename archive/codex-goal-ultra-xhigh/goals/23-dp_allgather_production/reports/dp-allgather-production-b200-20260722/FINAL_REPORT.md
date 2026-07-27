# GLM-5.2 DP AllGather production assessment

## Disposition

**NO_REPLACEMENT.** No SGLang production dispatch is changed or enabled. The
stock `GroupCoordinator.all_gather_into_tensor` path remains active for every
TP8/DP8/EP8 bucket and every unvalidated tuple. Local evidence is TP4/DP4 only
and cannot promote a production bucket.

Kernel-Harness started at `bcd005409e65786af82c86f621507ebef12b2766`.
The final runner/oracle evidence revision is `522a3b8`; the post-upgrade atomic
launcher hardening is `05d058547495be9c3c4e431fbf86b4415678cf29`.
SGLang stayed clean and unchanged at
`f93f8867b4bc124c9809c9110ec7361ed11b6b4a`; its source diff is empty.

## Reachability and ABI

The balanced GLM-5.2 decode source path is:

`GlmMoeDsaForCausalLM` / `DeepseekV2DecoderLayer.forward`
→ `LayerCommunicator.prepare_mlp`
→ `dp_gather_replicate`
→ `_dp_gather_via_all_gather`
→ `GroupCoordinator.all_gather_into_tensor`
→ graph-enabled PyNCCL `ncclAllGather`.

Sparse layers using the configured DeepEP A2A path can remain scattered and do
not establish this gather as a universal MoE-layer operation. For a dense layer,
the source-mapped immediate consumer is `self.mlp(...)`; the exact live layer ID
and consumer trace were not available and remain external.

The matching serving-native ABI is contiguous BF16 local input `[M, 6144]`, a
distinct caller-owned output `[world_size*M, 6144]`, rank-major ordering, current
CUDA stream, and no timed setup/allocation/adapter copy. Locked TP4 runs prove
ranks `[0,1,2,3]`, ordinary non-symmetric buffers, NCCL 2.28.9, graph-enabled
PyNCCL, exact values/order, output aliasing, unchanged input, poison checks, and
three separate off-timing producer→collective→dependent-consumer ordering
replays. They do not prove the corresponding TP8 runtime identities.

Default balanced eager prefill is not AllGather: source selects SUM_LEN and calls
`_dp_gather_via_all_reduce`. `dp_allgather_prefill` is retained as an explicitly
nondefault MAX_LEN eager alternate. The two paths are never relabeled.

## Measurements

All admitted measurements used physical B200 GPUs 0-3 through the required
four-GPU wrapper, rank-max latency, `SGLANG_GLM52_OPT=0`, NCCL 2.28.9, PyTorch
2.11.0+cu130, CUDA 13.0, and the frozen engine environment.

| Diagnostic bucket | Three reference session p50s (ms) |
|---|---|
| TP4 decode M16 | `0.100928001`, `0.0931039974`, `0.0989919975` |
| TP4 decode M32 | `0.0844639987`, `0.0936479978`, `0.106992003` |

The direct-PyNCCL identity control produced pre-hardening paired speedups
`0.865741065`, `1.13422056`, and `1.06234725`. It demonstrates large session
noise and a losing minimum, but is excluded from the strict oracle because its
JSON predates embedded topology and runner/workload hashes.

Two grouped-broadcast M16 results passed exact correctness, but their containing
bundles detected foreign CUDA processes and are marked `INCOMPLETE`; apparent
speedups `1.01998596` and `1.03203009` are excluded in full. No grouped M32 or
prefill value is reported. A post-hardening combined run was retried repeatedly
through the wrapper in a bounded loop; every observed attempt returned busy status
75 and no result directory was created. The exact count and time window are not
asserted because the loop's raw console log was not preserved. A separate final
attempt is auditable in `scheduler/final-locked-retry.log`: at
`2026-07-22T17:01:50Z` the wrapper returned 75 because another four-GPU request was
pending or running, again before creating `grouped-auto-all-v3/`.

After the scheduler was upgraded to atomically acquire all four GPU locks for one
command, `scheduler/atomic-retry-20260722.log` records a new request at
`2026-07-22T17:05:32Z`. It returned 75 immediately because another four-GPU request
was pending or running. The wrapper acquired no GPU subset and did not invoke the
wrapped command: no ranks, CUDA work, benchmark, or profile ran, and no
`grouped-auto-all-v4-atomic/` directory was created. This is scheduling provenance
only and changes no performance, correctness, TP4-success, or TP8 conclusion. Both
TP4 launchers now require the inherited atomic-intent FD 8 as well as GPU-lock FDs
9-12, covered by a CPU regression test.

A second post-upgrade receipt, `scheduler/atomic-retry2-20260722.log`, records a
clean-tree request from `5e8a81a` at `2026-07-22T17:10:09Z`. The wrapper returned
75 immediately because an external or scheduled CUDA process was active. It did
not invoke the wrapped command and created no `grouped-auto-all-v5-atomic/`
directory, so it likewise contributes no measurement evidence.

After the intervening CPU validation and evidence commit,
`scheduler/atomic-retry3-20260722.log` records another clean-tree request from
`4df0157` at `2026-07-22T17:11:38Z`. It received the same immediate external-or-
scheduled-CUDA exit 75, did not invoke the wrapped command, and created no
`grouped-auto-all-v6-atomic/` directory.

The strict topology × M × backend oracle consequently has zero eligible
candidate sessions and enables nothing. See `paired_summary.md` and
`paired_summary.csv` for the complete concise ledger and exclusions.

## Profiler result

The complete stock TP4 M16 Nsight Systems/NCCL TRACE bundle identifies:

- BF16 count 98,304, or 196,608 bytes contributed per rank;
- auto-selected 32-channel Ring/LL;
- `ncclDevKernel_AllGather_RING_LL`, grid 32, block 512, 96 registers/thread;
- twelve timed per-rank `cudaGraphLaunch` API calls at 24.697-37.479 µs
  (mean 30.866 µs), without interpreting CUPTI-distorted queue/kernel durations;
- NVLink utilization samples on GPUs 0-2 consistent with low utilization and no
  timed-window sample on GPU3.

The timed graph contains the collective. A separate graph validates ordering;
it is not a live model consumer or overlap trace. The logical TP4 ring send
volume is 589,824 bytes per rank and 2,359,296 bytes aggregate, with equal
receive volume; physical wire bytes were not measured. Profile timing inflated
the microbenchmark to roughly 4.16 ms and is excluded from performance claims.

This supports a small-message launch/coordination limit, not a link-bandwidth
kernel rewrite. Stock already selects Ring/LL. The NVLink query is process-scoped
as well as correlation-scoped and has a regression test. No device code changed,
so Nsight Compute, PTX, SASS, and ptxas evidence are not applicable.

## Attempts and rollback

- c10d writes the caller output in eager mode but SGLang disables it for graph
  capture; the candidate fails closed for decode graph buckets.
- Direct PyNCCL is the same native collective and only a noise control.
- Grouped rank broadcasts are ABI-compatible and graph-safe, but no clean
  post-hardening performance bundle was admitted.
- Ring/LL forcing is identical to NCCL auto selection and cannot be an
  auto-versus-forced paired claim within one communicator.
- Symmetric/registered storage is disabled by the audited recipe; adding a timed
  registration, allocation, or copy would violate the live ABI.
- The available SM100 multimem path consumes a different hidden-shard/symmetric
  storage ABI. A new custom kernel is unjustified without a reproducible dispatch
  win.
- The gathered tensor's dense MLP consumer is data-dependent. The correctness
  sentinel offers no independent work to overlap; live scheduling remains an
  external trace question.

Rollback is therefore complete by construction: omit every external candidate.
There is no SGLang integration diff to revert.

## Acceptance boundary and exact next command

This four-GPU host cannot execute the required one-node TP8/DP8/EP8 gate. No
TP8 correctness, paired ≥3% timing, live layer trace, containing-region result,
or end-to-end SGLang request result exists. The model artifact and request-client
fixture were not provided, so they are not invented.

On an exclusively reserved eight-B200 node, print the exact per-op rerun manifest
with:

```bash
bash profile/dp-allgather-production-b200-20260722/harness/print_external_tp8_acceptance.sh \
  /home/qinhaiyan/glm52-goal-runs/23-dp_allgather_production/kernel-harness/serving_native/candidates/allgather_grouped_broadcast.py \
  UNIQUE_SESSION
```

That helper emits three independent paired sessions for M16, M32, and the
explicit MAX_LEN prefill alternate plus three default SUM_LEN reference runs,
with clean-status, topology, GPU, environment, source, import, and runtime
provenance. Those microbenchmarks are still insufficient: the exact live GLM
launch, containing dense-layer/logits region, and frozen end-to-end request
commands must be supplied and pass before any enable decision.

## Final enable/fallback policy

`promotion_allowed` is false. A future dispatch may key without host/device
synchronization only on a fully validated
`world_size × topology × phase × local_M × hidden × dtype × layout × graph_mode × backend`
tuple. Until the external TP8, containing-region, and end-to-end gates all pass,
every tuple calls stock SGLang.
