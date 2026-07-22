# Goal 22 model-PR history notes

Source consulted:
`AI-Infra-Auto-Driven-SKILLS/model-pr-optimization-history/sglang/glm5-glm51/README.en.md`.

Relevant prior changes and how they constrain this goal:

- SGLang PR #18521 introduced GLM/DeepSeek-style DSA support. The production
  backend, rather than the retired standalone score/value proxies, is the
  required optimization boundary.
- PR #20062 made attention backend and prefill threshold selection explicit.
  This goal therefore forces `flashmla_kv`; it does not infer reachability from
  device family or benchmark a no-flag Blackwell default.
- PR #22850 fused indexer work. That optimization is upstream of the measured
  FlashMLA region and cannot be claimed as a split-KV/combine win.
- PR #25821 performed the NSA-to-DSA refactor and moved the active integration
  into `dsa_backend.py`. Source tracing and tests use the post-refactor path.
- PRs #28437, #28448, and #28460 cover GLM-5.2 deployment enablement. They
  reinforce TP8/DP8/EP8 production validation and the fixed local M16/M32
  contract; a smaller-rank diagnostic cannot replace that gate.
- PR #28607 covers HiSparse integration. It is relevant to reachability but
  does not justify changing the explicitly selected `flashmla_kv` backend.

No reviewed history entry supplied evidence for a pre-existing static M16/M32
FlashMLA scheduler override. Scheduler experiments in this goal must therefore
be justified by its own Nsight and paired-timing evidence and must fail closed
to the stock operator.
