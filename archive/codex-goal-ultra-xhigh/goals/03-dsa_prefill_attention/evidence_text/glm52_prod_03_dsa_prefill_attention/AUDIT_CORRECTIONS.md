# Audit corrections and supersession record

Date: 2026-07-22

This file records corrections discovered while continuing the same goal. Raw
artifacts and historical commits are preserved; only the interpretation is
superseded.

## 1. The earlier `source_trial_pdl_off_*` files did not time SGLang integration

All three JSON files name
`serving_native/candidates/dsa_prefill_pdl_off.py` in `candidate.path`. The
runner pins `SGLANG_GLM52_OPT=0`, and the candidate directly calls the
FlashInfer leaf with `enable_pdl=False`. Therefore the pooled 0.996817x result
is valid negative evidence for the direct external leaf candidate, but it is
not a measurement of the temporary SGLang dispatch code.

SGLang commit `b03db3f648f9db5b9264638716d20adacc510d6e` did add a guarded
PDL policy and tests, and commit
`5a444f66cf5764d2d76003a3a4c4631af152a253` reverted it. No persisted result
timed that integration. The current SGLang tree is clean, and the reached
backend source is byte-identical to stock base
`f93f8867b4bc124c9809c9110ec7361ed11b6b4a`.

## 2. The inherited direct leaf omitted SGLang's dummy page

The old `dsa_trtllm_prefill_m4096_ctx32768` workload allocates 512 pages. A
runtime hit counter around the real backend class observed a raw cache shaped
`[513,1,64,576]`: SGLang reserves a leading 64-slot dummy page and presents 512
usable pages after it. The old workload remains available and is relabeled a
legacy compact-pool replay so its historical results stay reproducible.

The separately named
`dsa_trtllm_prefill_m4096_ctx32768_rawpool` workload carries the corrected
physical ABI: 513 pages and sparse physical indices offset by 64. The source
tactic decision was repeated on this workload in one paired/profiler bundle.

## 3. The old runtime trace was a direct leaf, not the backend class

`runtime_abi_trace.json` remains valid for its direct FlashInfer call. It does
not prove execution of `DeepseekSparseAttnBackend.forward_extend`. The new
`profile/dsa-backend-prefill-m4096-fixture-20260722/results/
backend_hit_trace_zeroed_v2.json` installs a scoped hit counter around one real
backend-class call and records the Python stack, entry BF16 tensors, metadata,
FP8 cache write, exact 513-page leaf ABI, eager mode, and finite BF16 output.

The fixture is still checkpoint-free. It excludes model projections, live
indexer scoring/top-k, collectives, scheduling, and server end-to-end behavior.

## 4. Historical GPU commands are not new measurement evidence

Older scripts and reports hard-code `CUDA_VISIBLE_DEVICES=3` and sometimes
collected reference/candidate profiles in separate sessions. They are retained
as historical advisory evidence only. All decision-bearing continuation work
used the scheduler wrappers:

- one complete alternating series plus profiles per
  `with_flexible_gpu.sh` allocation;
- the DP4 diagnostic under one `with_all_gpus_lock.sh` allocation.

No performance claim compares unpaired results from different physical GPUs.

## 5. Four-GPU availability is no longer a blocker

The old report said only one GPU was authorized. The current instructions made
all four host B200s available through the all-GPU scheduler. The new DP4
rank-local diagnostic completed and is reported separately. It does not weaken
or satisfy the unchanged TP8/DP8/EP8 eight-rank gate.

The real-request and eight-rank blockers remain: the configured GLM-5.2 model
directory is empty, there is no other cached checkpoint, and the host has four
physical GPUs.

## 6. Historical verifier status is not reused

The prior report's verifier statement is not treated as current evidence. The
final validation log records the exact closeout commands and their actual exit
status. Serving-native result JSON has a separate schema, so
`audit_result.py` does not apply to it.
