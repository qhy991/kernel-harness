# Prior-art routing

Date: 2026-07-22

The source experiment was chosen after consulting the Blackwell kernel
knowledge base and the GLM-5/5.1 SGLang PR history. These sources were used to
select an experiment and validation surface; they are not benchmark evidence.

## Kernel prior art

- KernelWiki's sparse-MLA note identifies the two-stage indexer -> sparse-MLA
  pipeline, fixed 64-token paging, top-k 2048, FP8 gather, and Blackwell
  warp-specialized/TMEM implementation pressure. Its generic 656-byte cache
  description did not match the reached TRTLLM raw 576-byte cache, reinforcing
  the need to freeze the runtime ABI rather than copy a neighboring backend.
- The preserved FlashInfer PR #2836 history locates sparse-MLA generated-kernel
  selection in `fmhaKernels.cuh`. The installed 0.6.12 selector already shipped
  Q64 Keeps and Q32/Q16 Swaps cubins, so the lowest-risk device-code hypothesis
  was an exact-shape tactic oracle before attempting a new CUDA implementation.
- The profiling guidance made occupancy, registers, shared memory, waves,
  tensor/TMEM activity, spills, bank conflicts, and synchronization stalls the
  decision metrics. The measured lack of occupancy gain plus multiplied waves
  and shared-memory synchronization directly rejected finer head splitting.

## SGLang model history

- SGLang PR #20062 moved GLM-5 on Blackwell toward sparse MLA prefill through a
  model-specific dense-attention threshold. This supported tracing the actual
  prefill backend independently from decode and rejecting the frozen FlashMLA
  surrogate.
- PR #22850 fused indexer projection/cache work. It reinforced that the complete
  production DSA region includes live indexer score/top-k and cache preparation,
  so a direct attention-leaf result cannot be promoted as an end-to-end win.
- PR #27053 added GLM-5 prefill/piecewise-CUDA-graph work, while the GLM-5.2
  deployment updates in #28437, #28448, #28460, #29380, and #29466 show that
  launch configuration and quantization recipes remain model/hardware specific.
  The final gate therefore retains graph/overlap, exact launch flags, and the
  checkpoint-backed eight-rank lane.

## Resulting decision

The prior art narrowed the experiment to the reached FlashInfer selector seam,
but the persisted paired and profiler evidence—not the prior art—decides the
outcome. Q32 and Q16 both regress, no SGLang integration is enabled, and stock
Q64 remains the fail-closed path.
