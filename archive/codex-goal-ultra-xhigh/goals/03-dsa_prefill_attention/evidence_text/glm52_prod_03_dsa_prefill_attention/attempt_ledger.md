# DSA prefill attempt ledger (audited)

Date: 2026-07-22

Decision-bearing single-GPU measurements put each complete alternating series
and its profiler collection inside one `with_flexible_gpu.sh` allocation. The
four-GPU diagnostic used one `with_all_gpus_lock.sh` allocation. Absolute times
from different physical GPUs are not used as comparison claims.

## Attempt 0 — reject the frozen FlashMLA task

- **Hypothesis:** the frozen `dsa_prefill_attn` task could represent current
  production prefill.
- **Evidence:** it calls `sgl_kernel.flash_mla.flash_mla_sparse_fwd` on flat BF16
  tensors. Current B200/FP8 source resolution reaches FlashInfer TRTLLM-gen on a
  raw paged FP8 pool.
- **Decision:** reject before optimization. It is mathematically related but
  operationally mismatched.
- **Rollback:** no frozen task or oracle file was changed.

## Attempt 1 — inherited compact-pool leaf

- **Purpose:** retain the earlier M4096/context32768 direct FlashInfer replay and
  its scattered/trailing/eight-request diagnostic distributions.
- **Audit finding:** the original `[512,1,64,576]` KV pool omitted SGLang's
  leading dummy page, so it is no longer labeled the exact physical backend ABI.
- **Use:** historical compact-pool results remain reproducible and corroborate
  kernel behavior. They do not carry the final source-tactic decision.
- **Rollback:** workload remains available under its existing name; its notes
  now direct exact-ABI work to the separately named raw-pool workload.

## Attempt 2 — disable PDL at the direct leaf

- **Hypothesis:** programmatic dependent launch policy might explain the
  synchronization/polling SASS in the generated kernel.
- **Delta:** external candidate
  `serving_native/candidates/dsa_prefill_pdl_off.py` passes
  `enable_pdl=False`; nonmatching inputs fall back.
- **Expected effect:** change launch dependency policy while leaving the device
  instruction body unchanged.
- **Correctness:** mandatory runner comparison passed for scattered 32K,
  trailing 32K, and eight-request fallback artifacts.
- **Paired result:** the authoritative three direct-candidate files named
  `source_trial_pdl_off_01..03.json` pooled to 0.996817x over 90 pairs. This is a
  0.318% regression, not a win.
- **Profiler delta:** stock and PDL-off have the same kernel symbol, launch
  resources, and normalized 4888-instruction SASS sequence. NSYS mean differs
  by -0.019%; NCU duration differs by +0.294%.
- **Risk:** a neutral leaf cannot prove predecessor overlap in the complete
  region.
- **Decision:** reject.
- **Rollback:** omit the explicit `enable_pdl` argument.

### Correction to inherited labeling

The three JSON files above explicitly name the external candidate path and pin
`SGLANG_GLM52_OPT=0`. They did **not** time the temporary SGLang dispatch.
Commit `b03db3f648f9db5b9264638716d20adacc510d6e` added that guarded policy and
tests, but no persisted performance run exercised it. Commit
`5a444f66cf5764d2d76003a3a4c4631af152a253` reverted it. The source experiment
is preserved as unmeasured implementation work, not performance evidence.

## Attempt 3 — real backend-class checkpoint-free fixture

- **Hypothesis:** a backend-class fixture can close the direct-leaf reachability
  gap without claiming a real request.
- **Delta:** add `dsa_backend_prefill_m4096_ctx32768_fixture`, constructing the
  real `DeepseekSparseAttnBackend`, metadata, raw FP8 cache, fused physical top-k,
  and GLM dimensions through SGLang's test-only model-runner seam.
- **Runtime proof:** a scoped hit counter records exactly one
  `trtllm_batch_decode_with_kv_cache_mla(backend="trtllm-gen")` call from
  `forward_extend`, FP8 query `[4096,1,64,576]`, FP8 KV
  `[513,1,64,576]`, clipped lengths 2048, eager mode, and finite BF16 output.
- **API correction:** FlashInfer documents the workspace as zero-required on
  first use. The fixture zeros it once during construction, outside timing.
- **Paired controls:** three 20-pair stock-vs-stock runs have reference p50s
  0.990112/0.979568/0.993392 ms and paired speedups
  0.998075x/0.999608x/0.998658x.
- **Region profile:** 0.947472 ms mean over 20 iterations: attention 0.833439 ms,
  fused RoPE/quantize 0.102480 ms, KV write 0.003240 ms, residual 0.008314 ms.
- **Risk:** identity-candidate correctness is self-reference; the fixture uses
  generated top-k and excludes live projections/indexer/scheduler/collectives.
  Its test seam is process-scoped.
- **Decision:** retain as exact backend-boundary evidence, not server E2E.

## Attempt 4 — add the observed 513-page raw-pool leaf

- **Hypothesis:** the source tactic must be judged with the physical cache shape
  observed from the backend fixture rather than silently changing the inherited
  workload.
- **Delta:** add
  `dsa_trtllm_prefill_m4096_ctx32768_rawpool`, one dummy page plus 512 usable
  pages, physical token offset 64. Update structural tests and document all fixed
  parameters.
- **Correctness:** stock and both source tactics pass the mandatory output
  comparison on the exact scattered workload.
- **Decision:** retain as the decision-bearing direct leaf. The inherited name
  remains a compact replay.

## Attempt 5 — FlashInfer Q32/Q16 Swaps source overlay

- **Hypothesis:** stock Q64 could be under-filled at M4096; splitting each query
  token's 64 heads across Q32 or Q16 CTAs might expose more parallel work.
- **Baseline evidence:** stock launches one 512-thread CTA with 128
  registers/thread and 220672 bytes shared memory. Resource use limits it to one
  CTA/SM, while the profile is dominated by long-scoreboard/cross-warp waits and
  uses little DRAM bandwidth.
- **Source delta:** copy FlashInfer 0.6.12 headers from tag `v0.6.12`, peeled
  commit `d768c14e7cf5dd5df45a8a1de78ae815879f108a`, into a repo-local overlay.
  Relocate relative includes mechanically for the overlay, then add one exact
  dtype/shape/top-k selector predicate for shipped Q32 or Q16
  `PersistentSwapsAb` cubins. Build unique JIT libraries at import; do not
  overwrite the installed package.
- **Expected low-level effect:** 2x/4x more head-splitting CTAs, possibly hiding
  sparse-gather or producer/consumer stalls.
- **Correctness:** both tactics pass the exact raw-pool scattered case and the
  compact trailing case. The eight-request mismatch calls stock.
- **Paired p50, exact raw pool:** three 30-pair runs per row:

  | Tactic | Candidate run p50s (ms) | Median paired speedup |
  |---|---|---:|
  | stock control | 0.882448 / 0.887040 / 0.968560 | 1.000624x |
  | Q32 Swaps | 1.608112 / 1.612208 / 1.600384 | 0.541023x |
  | Q16 Swaps | 2.924160 / 3.022368 / 2.923248 | 0.296514x |

  Two allocation-local excursions raised both arms; no decision uses unpaired
  absolute time, and every experimental ratio remains decisively negative.
- **Profiler delta:** tensor-op count and ~204 MB DRAM reads are unchanged.
  Q32/Q16 double/quadruple grid and global loads, raise SM instructions
  1.956x/3.378x, raise shared-memory instructions 3.062x/5.063x, reduce TMEM
  active from 69.48% to 34.84%/22.07%, and create 1.851M/4.111M shared-store
  bank conflicts. All remain at 25% occupancy; Q32 increases spills 8.15x.
- **Risk:** the selector is a goal-scoped oracle over shipped cubins, not a
  general dynamic tactic policy. No live distribution or eight-rank accuracy
  result exists.
- **Decision:** reject both tactics. More waves amplify the measured pipeline
  work without improving residency.
- **Rollback:** SGLang never imports the overlay; stock Q64 selector remains
  active for every production input.

Raw source-attempt artifacts:

- exact raw pool:
  `profile/dsa-prefill-trtllm-m4096-rawpool-tactic-oracle-20260722/`;
- compact corroboration and source counters:
  `profile/dsa-prefill-trtllm-m4096-tactic-oracle-20260722/`.

## Attempt 6 — four-GPU rank-local diagnostic

- **Purpose:** use all host hardware without weakening the production gate.
- **Delta:** add a separately named world-size-4 workload that replays the
  corrected 513-page leaf independently on each rank and gates timing on the
  maximum rank.
- **Correctness:** all three stock-vs-stock paired runs completed.
- **Result:** reference rank-max p50s are 0.911072, 0.940944, and 0.941648 ms;
  paired speedups are 1.003564x, 1.004960x, and 0.986111x, consistent with a
  neutral stock control.
- **Profiler:** bundled NSYS report retained under
  `profile/dsa-prefill-trtllm-m4096-dp4-diagnostic-20260722/`.
- **Risk:** this has no attention collective and is DP4, not TP8/DP8/EP8.
- **Decision:** retain as topology diagnostic only.

## Final policy

No DSA prefill optimization bucket is enabled. The final SGLang tree is clean
at rollback head `5a444f66cf5764d2d76003a3a4c4631af152a253`, and the reached
backend file is byte-identical to stock base
`f93f8867b4bc124c9809c9110ec7361ed11b6b4a`. Stock FlashInfer/TRTLLM-gen Q64
behavior remains active for every shape, ABI, topology, and graph mode.
