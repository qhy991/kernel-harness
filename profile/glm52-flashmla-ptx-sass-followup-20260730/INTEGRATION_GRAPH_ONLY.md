# FlashMLA graph-only integration follow-up

## Why the prior campaign could not submit as L2

`p1_consumer_scale` already clears CUDA Graph containing/leaf gates at about
**1.06×** on M16/M32. It cannot clear the **eager containing** lane under the
API-v1 Python provider: a fixed ~17 µs host tax makes the lane arithmetically
unreachable (ceiling ≈1.018 even with a zero-cost guard). Plan §6 therefore
forced `no-replacement` even though production decode is CUDA-graph-bound.

## What this change does

In the FlashMLA hotspot SGLang worktree:

1. **`KernelSpec.graph_only=True`** for `dsa_decode_attn` hotspot registration.
2. **Default `SGLANG_GLM52_FLASHMLA_GRAPH_ONLY=1`**: `try_dispatch_flashmla_sparse_decode`
   returns `None` immediately when the current stream is **not** capturing, so
   eager decode stays on stock with no provider launch.
3. Hot-path reductions from the campaign ledger:
   - skip empty NVTX context-manager enter/exit when NVTX is off;
   - cache `(op, phase, m, profile)` lookups;
   - precompute ABI shape/stride tuples per M bucket.

Diagnostic eager leaf timing can still force selection with
`SGLANG_GLM52_FLASHMLA_GRAPH_ONLY=0`.

## Registration implication

With graph-only selection, the promotion policy matches the audit note used for
router graph-selective candidates:

- **Required:** graph containing + graph leaf ≥1.03 (already measured for P1).
- **Eager containing:** must fall back to stock (no candidate hit); do **not**
  require 1.03× speedup.
- Ceiling remains **L2 external E2E** until checkpoint TP8/DP8/EP8 acceptance;
  still not L3 production-default.

Kernel binary remains FlashMLA commit `b5af443` (`p1_consumer_scale`).
