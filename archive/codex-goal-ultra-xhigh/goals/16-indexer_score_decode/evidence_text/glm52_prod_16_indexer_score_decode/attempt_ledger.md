# Attempt ledger

## Reference and hypothesis

The reference is the current SGLang CUDA decode path with
`SGLANG_GLM52_OPT=0`: `Indexer._get_topk_paged` resolves
`DSAPagedMQALogitsBackend.AUTO` to DeepGEMM, runs the normal `next_n=1` split
wrapper, materializes FP32 logits, and calls the shipped top-k-v2 transform.
The exact rank-local buckets are M16 and M32 at context 8192, page size 64,
32 index heads, head dimension 128, and top-k 2048.

KernelWiki and the SGLang/vLLM model PR histories directed the experiment
toward the already shipped SM100 CuTe-DSL backend and the score-to-top-k
handoff. Prior SGLang work also ruled out treating the old independent K
projection or a dense score kernel as production evidence.

## Attempt 0: first campaign infrastructure

- **Delta:** external CuTe-DSL backend candidate plus the first bundled
  score campaign.
- **Outcome:** paired correctness completed, but profiler collection failed
  and graph capture order was not balanced.
- **Decision:** superseded, not used for final performance claims.
- **Rollback point:** commit
  `7cf0f27e99772306fd46037e40e4a3627df62c96`; repairs are in `9029b37` and
  `6c86a8d`.
- **Evidence:** `runs/20260723T112717Z/` and `SUPERSEDED_CAMPAIGN.md`.

## Attempt 1: shipped CuTe-DSL score backend

- **Hypothesis:** CuTe-DSL's lower register and shared-memory footprint could
  reduce the latency floor of the 148-CTA score kernel at M16/M32.
- **Exact delta:** the external candidate changes only the backend argument
  from `deepgemm` to `cutedsl`; it keeps the fused FP8 Q/K ABI, FP32 gates,
  compact page table, `clean_logits=False`, top-k-v2, and output contract.
- **Expected device effect:** reduce 168 to 80 registers/thread and 221,696 to
  115,328 shared bytes/CTA, with no adapter, copy, or synchronization.
- **Correctness:** graph replay passes; logits differ by at most
  `2.384185791015625e-7`; every row has exactly the same 2048 physical slots.
- **Paired result:** graph M16 is `1.019197x, 1.003703x, 1.011231x`; graph
  M32 is `0.977756x, 1.040790x, 1.040749x`. Eager is a stable regression at
  median series ratios `0.805730x` and `0.791694x`.
- **Profiler delta:** NCU reports the expected resource reduction, but the
  kernels remain a single 148-CTA launch with low SM and DRAM utilization.
  Nsys reduces the isolated M16 score kernel from 5.120 to 4.928 us and M32
  from 6.464 to 6.240 us; top-k plus fill remains about 7.1-7.5 us.
- **Risk:** shape-specific dispatch would add policy complexity for a result
  that reverses within the same M32 graph bucket.
- **Decision:** reject; no bucket is reproducibly at least 3% faster.
- **Rollback point:** external candidate only; SGLang was never modified.

## Attempt 2: propagate through the complete indexer

- **Hypothesis:** a small score win might survive the packed-FP8 `wq_b`, fused
  BF16 `wk_weights_proj`, LayerNorm/RoPE/Q quantization, and current-token
  K-cache store.
- **Exact delta:** use CuTe-DSL only for score/top-k inside the otherwise
  unchanged reconstructed `Indexer._fused_q_prepare_and_store ->
  Indexer._get_topk_paged` region.
- **Correctness:** all bidirectional graph, pre-timing, and post-timing checks
  pass, including cache mutation and exact top-k sets.
- **Paired result:** M16 is `1.014806x, 1.031918x, 1.056648x`; M32 is
  `1.028806x, 1.032633x, 1.015328x`.
- **Profiler delta:** the region contains nine GPU operations. Its largest
  kernels are packed-FP8 `wq_b`, top-k, paged score, and the BF16
  projection/reduction, so the score change is only one small term.
- **Decision:** reject; neither bucket clears the gate in every series.

## Attempt 3: propagate through selected TRT-LLM DSA

- **Hypothesis:** eliminating score latency might remain visible after the
  exact physical slots feed the resolved `trtllm-gen` sparse DSA decode.
- **Exact delta:** the same score backend change inside the complete indexer,
  followed by unchanged
  `flashinfer.decode.trtllm_batch_decode_with_kv_cache_mla`.
- **Correctness:** all graph and output checks pass.
- **Paired result:** M16 is `1.011807x, 1.014385x, 1.011058x`; M32 is
  `1.021669x, 1.019186x, 1.020334x`.
- **Profiler delta:** the TRT-LLM main/reduction kernels add about
  16.1 us at M16 and 21.7 us at M32 in the one-iteration Nsys capture.
- **Decision:** reject; the full locally reproducible DSA region is below 3%.

## Attempt 4: four-rank diagnostic

- **Hypothesis:** maximum-rank timing might expose a topology-dependent
  advantage hidden by one-rank measurements.
- **Exact delta:** run the complete indexer plus selected DSA independently on
  four ranks, preserve local M16/M32, and reduce latency by maximum rank.
- **Correctness:** every rank and every series passes.
- **Paired result:** M16 is `1.004793x, 1.002506x, 1.018696x`; M32 is
  `1.023810x, 1.030381x, 1.017586x`.
- **Decision:** reject. This is TP4/DP4 diagnostic evidence only, never TP8
  acceptance.

## Final decision

The alternative has no reproducible three-series microbenchmark win and loses
the containing-region threshold before a model-level promotion gate is
relevant. Stock DeepGEMM remains enabled for both buckets and every topology.
