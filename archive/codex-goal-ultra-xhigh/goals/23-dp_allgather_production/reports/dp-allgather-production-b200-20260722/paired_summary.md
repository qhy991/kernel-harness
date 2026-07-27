# DP AllGather paired summary

Disposition boundary: every row is TP4/DP4 diagnostic evidence. No row is TP8,
containing-region, end-to-end, or production-promotion evidence.

## Admitted reference baseline

| Workload | Session p50 rank-max latency (ms) | Correctness | Artifact status |
|---|---:|---|---|
| `tp4_allgather_decode_m16` | `0.100928001`, `0.0931039974`, `0.0989919975` | Exact rank-major values, caller-output alias, input unchanged, poison check, three ordering replays | Three complete, clean-start reference bundles |
| `tp4_allgather_decode_m32` | `0.0844639987`, `0.0936479978`, `0.106992003` | Same | Three complete, clean-start reference bundles |

These are standalone reference baselines, not candidate oracle sessions. The
strict candidate oracle requires three complete post-hardening alternating
sessions with embedded candidate/source/topology/runtime provenance.

## Candidate evidence and exclusions

| Candidate / bucket | Session paired speedups | Strict-oracle sessions | Decision |
|---|---|---:|---|
| Direct PyNCCL identity control, M16 | `0.865741065`, `1.13422056`, `1.06234725` | 0 | Pre-hardening control lacks embedded topology and runner/workload hashes; its minimum session loses and it is not an optimization. |
| Grouped PyNCCL broadcasts, M16 | `1.01998596`; contaminated retry `1.03203009` | 0 | Both bundles carry `INCOMPLETE.txt`; all timing is excluded. |
| Grouped PyNCCL broadcasts, M32 | none | 0 | Local rerun blocked by active scheduler/foreign CUDA work. |
| Default SUM_LEN prefill reference | none | N/A | Local characterization blocked; source proves this is AllReduce, not AllGather. |
| MAX_LEN prefill alternate | none | 0 | Local characterization blocked; it is not default eager prefill. |

The topology × M × backend oracle therefore contains zero eligible candidate
sessions and no enabled bucket. `analysis/summarize.py` recomputes every paired
speedup from raw rank-max samples and rejects malformed ordering, rank sets,
rank maxima, provenance, runtime stacks, imports, or correctness contracts.

## Profiler binding limit

The valid stock M16 trace identifies a 196,608-byte contribution and auto-selected
32-channel `ncclDevKernel_AllGather_RING_LL`. Timed-window NVLink samples on GPUs
0-2 are consistent with low utilization; GPU3 had no sample. CUPTI inflated the
profiled latency to milliseconds, so those durations are never mixed into this
table. The evidence supports a small-message launch/coordination diagnosis, while
stock already selects Ring/LL. No device code changed, so NCU/PTX/SASS is not
applicable.

## Local scheduler blocker

A clean-source combined rerun was attempted only through
`with_all_gpus_lock.sh`. Repeated nonblocking attempts in a bounded retry loop
returned exit 75 because another four-rank request or a foreign CUDA process was
active. No `grouped-auto-all-v3` directory was created. The exact count and time
window are not asserted because the loop's raw console log was not preserved;
the absence is not converted into favorable candidate evidence. The final
standalone retry has an exact receipt in `scheduler/final-locked-retry.log`: it
started and ended at `2026-07-22T17:01:50Z`, returned 75 because another four-GPU
request was pending or running, and created no diagnostic directory.

The upgraded atomic scheduler was then retried after the closeout commit.
`scheduler/atomic-retry-20260722.log` records start/end
`2026-07-22T17:05:32Z` and immediate exit 75. The wrapper acquired no GPU subset
and never invoked the wrapped command; no ranks, CUDA work, benchmark, profile, or
`grouped-auto-all-v4-atomic/` directory exists. It is scheduler evidence only.

`scheduler/atomic-retry2-20260722.log` records a second clean-tree request from
`5e8a81a` at `2026-07-22T17:10:09Z`. It returned 75 immediately because an
external or scheduled CUDA process was active; the wrapped command did not run
and no `grouped-auto-all-v5-atomic/` directory exists.

After CPU validation and committing that receipt,
`scheduler/atomic-retry3-20260722.log` records the same immediate exit 75 from a
clean `4df0157` tree at `2026-07-22T17:11:38Z`. No wrapped command, ranks, CUDA
work, or `grouped-auto-all-v6-atomic/` directory exists.

## Gate result

- Local conservative three-session candidate gate: not passed.
- TP8 correctness and paired ≥3% gate: not run.
- Containing-region and end-to-end gates: not run.
- Production promotion: false.
- Fallback: stock `GroupCoordinator.all_gather_into_tensor` for every bucket.
