# Related-goal warm start (not Goal 22 acceptance evidence)

Read-only audit date: 2026-07-22.

The sibling Goal 01 worktree contains the only matching FlashMLA-KV evidence on
this host.  It used the same pinned FlashMLA commit and the same installed
extension SHA-256 (`d8d97150bd86381c73406603cb7d6b682767535e0526053f04e3acefadb13316`),
but ran on physical GPU 1 from a global conda environment, has an uncommitted
harness, and did not retain raw paired samples.  It is therefore hypothesis
evidence only, never a Goal 22 baseline or acceptance result.

Matching corrected ABI facts from
`/home/qinhaiyan/glm52-goal-runs/01-dsa_decode_value_path/kernel-harness/profile/dsa-flashmla-kv-stock/analysis/`:

- Q/output are `[M,1,64,576]` BF16 and `[M,1,64,512]` BF16.
- indices are `[M,1,2048]` int32, selected cache lengths are 2048, scheduler
  metadata is `[148,8]`, and the model scale is 0.0625.
- `num_splits` advances by 8 per request at M16 and 4 per request at M32.
- Nsight Systems steady medians were 17.600/12.704/26.080 microseconds for
  M16 main/combine/overlapped chain, and 24.960/9.696/30.624 microseconds for
  M32.
- Nsight Compute reported a 148-CTA, 384-thread main grid with 168 registers
  per thread and 232,656 bytes shared memory per CTA.  Main SM throughput was
  14.63% (M16) and 20.03% (M32); long-scoreboard and barrier stalls dominated.
- The combine kernels were material tails (128/256 CTAs and 10.656/8.768
  microseconds under NCU) and were long-scoreboard dominated.

The source scheduler launches one main CTA per B200 SM (148), while the equal
2048-token production buckets use only 128 useful partitions.  This ranks a
128-part scheduler experiment ahead of a broad kernel rewrite: it tests whether
removing empty work and reducing M16 from 8 to 7 splits offsets the larger
per-part payload.  Stock remains the rollback point.

Goal 02 targets FlashInfer TRTLLM-gen, not FlashMLA-KV.  Its split selector
experiment is conceptual support for testing scheduling only; none of its code
or timings are transferable acceptance evidence.

Rejected sibling evidence under
`rejected-preflight-context8192-scale576/` used scheduler lengths 8192 and
`1/sqrt(576)`.  Those results are invalid and must not be cited.
