# Source and build disposition

## Source attempt

The reached backend was not copied or replaced. The source experiment is the
external serving-native candidate
`serving_native/candidates/indexer_score_balanced_chunks.py`, followed by the
fail-closed
`serving_native/candidates/indexer_score_balanced_mixed_bucket.py`.
Both invoke the real SGLang `Indexer._get_topk_ragged` method and alter only
the already-cached logits budget supplied to its existing chunk scheduler.
The enabled schedule has the same two launches and a smaller maximum
allocation than stock, so the production OOM guard is preserved.

The focused dispatch predicate is entirely host-resident: local query count,
K count, request count, extend lengths, sequence lengths, and stock chunk
rows. It performs no CUDA tensor read, memory query, copy, adapter kernel, or
synchronization. Unsupported signatures immediately call the stock method.

## Build and import resolution

No native build is required for the schedule candidate. It uses the installed
DeepGEMM 0.1.4, SGL-Kernel 0.4.4, FlashInfer 0.6.12, and Triton 3.6.0
artifacts recorded by the campaign. SGLang imports from the isolated worktree
at commit `f93f8867b4bc124c9809c9110ec7361ed11b6b4a`; no site-package was
overwritten.

The gather launch experiments call the existing production Triton kernel and
also require no build. NCU showed that gather is too small a fraction of the
containing region to justify promotion or a library fork.

The only post-campaign runner change is diagnostic hardening: untimed
distributed correctness failures are reduced across ranks before timing, and
barriers name each logical local device. It neither changes the production
callable nor enters the CUDA-event interval. The fresh TP4 diagnostic proved
that it surfaces rank-local top-k mismatches promptly instead of timing out.

## Integration state

There is deliberately no SGLang runtime patch: a candidate is not integrated
until its score, complete-indexer, graph-split, selected-DSA, and unchanged
eight-rank model-server gates all pass. Stock SGLang therefore remains the
only active production policy. The candidate files are reproducible source
artifacts, not an enabled runtime dispatch.

The focused mixed-context candidate passed the pooled score-only threshold
but missed the complete-indexer, exact graph-split, and selected-DSA
containing-region thresholds. Its final enable policy is therefore the empty
set: every production signature uses stock SGLang. No build artifact,
environment flag, import override, or runtime monkey patch is required to
restore that state because the experiment stayed external to SGLang.
