# GLM-5.2 decode `index_q_upproj` FP8 GEMM — graph-only fixed-N/K, round 1

**Disposition: `external-acceptance-candidate`** (default **OFF**; production-on
requires checkpoint-backed TP8/DP8/EP8 acceptance, which is unreachable on this
4-GPU host with no GLM-5.2 FP8 checkpoint). Not `production-win`, not
`no-replacement`.

## Objective

Unlock decode `index_q_upproj` (DSA `Indexer.wq_b`, the q-up projection) with the
same graph-only + fixed-N/K playbook that promoted decode `o_proj`: a
`compiled_dims="nk"` DeepGEMM candidate registered graph-only, cleared against
production stock DeepGEMM at the locked packed-UE8M0 ABI, with eager
stock-fallback.

## Reachability & frozen ABI (CPU audit)

- Symbol: `C4Indexer.wq_b` (`sglang/srt/layers/attention/dsv4/indexer.py`), a
  **replicated** `[K=2048 → N=4096]` FP8 block linear. Distinct from the
  TP-sharded attention `q_b_proj`; `prefix_to_op_name` maps `wq_b →
  index_q_upproj`.
- Call path (decode): `Indexer.compute_q` → `wq_b(q_lora)` →
  `Fp8LinearMethod.apply` → `op_context("index_q_upproj")` →
  `apply_w8a8_block_fp8_linear` → `try_dispatch_fp8_gemm` →
  `lookup("index_q_upproj","decode",m)` → `fixed_nk` →
  `deep_gemm.fp8_gemm_nt(compiled_dims="nk")`.
- Locked ABI: local `M ∈ {16, 32}` (never divided by DP degree), `N=4096`,
  `K=2048`, `128×128` block, contiguous CUDA FP8 E4M3 activation+weight, packed
  **int32 UE8M0** scales (activation `[M,4]` stride `[1,M]`, weight `[4096,4]`
  stride `[1,4096]`), BF16 output, no bias. `DEEPGEMM_SCALE_UE8M0=True`
  confirmed on device. Selection fails closed on any mismatch (`speculative`,
  `mixed`, `split-prefill`, `target-verify`, wrong `M`/`N`/`K`/dtype/layout).

## Candidate identity (two identities, no third spent)

1. **Control = production stock** `deep_gemm.fp8_gemm_nt` (default compile,
   dynamic N/K).
2. **Candidate = same DeepGEMM family with `compiled_dims="nk"`** (N=4096,
   K=2048 baked as compile-time immediates). No hand-written PTX/SASS, no scale
   adapter, no archive wrapper. Proven distinct from stock: captured leaf
   mangled names differ **only** in the N/K template immediates
   (`…Lj4096ELj2048E…` vs `…Lj0ELj0E…`), identical grid `[148,1,1]`, block
   `[256,1,1]`, smem `230188 B`; outputs bit-identical. Profiler symbol
   `infini_kernel_glm52_index_q_upproj_decode_nk`.

The optional second epilogue/schedule identity (plan hypothesis 2) was **not
spent**: hypothesis 1 cleared both the leaf and the containing-region gate with
wide headroom, so it was not region-limited.

## Graph-only mechanism

`KernelSpec.graph_only=True` for the `index_q_upproj` `_E2E_DECODE` entry.
`try_dispatch_fp8_gemm` calls `_graph_only_declines(spec)` immediately after the
registry lookup — **before** the ABI check and the hit/miss lock — and returns
the stock path (`None`) whenever the spec is graph-only, its env allows it, and
the current CUDA stream is **not** capturing. Eager decode therefore stays on
stock with zero provider launch; CUDA-graph capture selects the candidate and
bakes it into the replayed graph. `SGLANG_GLM52_INDEX_Q_UPPROJ_GRAPH_ONLY=0`
forces eager selection for a diagnostic leaf.

## Gate results — 3 paired AB/BA series, one lease, single B200 `GPU-30b619de-87f2-1862-0d07-a595da8fe417`

All performance gates evaluated on **every one of the 3 series × 4 estimators**
(pooled, order-balanced, AB-median, BA-median); the table shows the
order-balanced median and the worst estimator over all series.

| Gate (floor) | M16 | M32 | Verdict |
|---|---|---|---|
| Correctness max-abs-err — eager + graph + region (= 0) | 0.0 | 0.0 | **PASS** |
| CUDA-graph **leaf** speedup vs stock (≥1.03) | 1.218× (worst 1.196) | 1.188× (worst 1.172) | **PASS** |
| CUDA-graph **containing-region** speedup vs stock (≥1.03) | 1.173× (worst 1.136) | 1.159× (worst 1.126) | **PASS** |
| Dispatch A (eager graph-only declines, no hit) | ✓ | ✓ | **PASS** |
| Dispatch B (diagnostic eager selects, 1 hit) | ✓ | ✓ | **PASS** |
| Dispatch C (leaf = 1 clean GEMM node, differs from stock) | ✓ | ✓ | **PASS** |
| Dispatch D (capture identical graph-only on/off) | ✓ | ✓ | **PASS** |

Absolute paired p50 (µs): M16 leaf stock 11.4 → cand 9.4; M16 region stock 13.2
→ cand 11.4. M32 leaf stock 11.5 → cand 9.5; M32 region stock 13.1 → cand 11.4.

**Independent confirmation lease** (fresh process, same physical B200) reproduced
the win — M16 leaf 1.238× / region 1.153×; M32 leaf 1.189× / region 1.161×,
status `pass`. Two independent leases with tight spread rule out the
graph-scheduler noise that sank the prior indexer no-go
(`glm52-goal-runs/15-indexer_wq_b_decode`, which used a numerically-divergent
Triton kernel and an SM-budget grid change — neither the `compiled_dims="nk"`
graph-only path measured here).

## Eager identity lane (informational, NOT a gate)

Both arms execute the same stock kernel; the candidate arm additionally pays the
graph-only **decline** host tax. Worst ratio ~**0.91×** (M16) / **0.91×** (M32)
on a ~100 µs eager region. This is the exact eager tax that graph-only exists to
avoid: in production decode the region is CUDA-graph-replayed (~13 µs, no Python
dispatch), where the candidate **wins** ~16%. The plan explicitly permits eager
stock-fallback under a graph-only policy, so this lane is reported, not gated.

## Enable / fallback policy

The candidate stays **default off**: `index_q_upproj` is an explicit-only
`_E2E_EXPLICIT_OPS` entry, so `serving_safe` (and an empty `OPT_OPS`) never
enable it. Enabling requires, per worker:

```
SGLANG_GLM52_OPT=1
SGLANG_GLM52_OPT_PROFILE=e2e_candidates
SGLANG_GLM52_OPT_OPS=index_q_upproj
SGLANG_GLM52_OPT_M_BUCKETS=index_q_upproj:16|32
# graph-only defaults on; do not set _GRAPH_ONLY=0 in production
```

No SGLang default is changed. Every unsupported bucket / mode / ABI and every
pure-eager decode fails closed to stock DeepGEMM. Promotion to a production-on
default additionally requires a stable ≥3% gain in a checkpoint-backed
TP8/DP8/EP8 end-to-end serving metric with no correctness, other-bucket, or SLA
regression — unreachable on this host (4× B200, no complete GLM-5.2 FP8
checkpoint). The four-rank lane is diagnostic only and must never be cited as
TP8 acceptance.

## Scope note — containing region

The containing region measured is the production `wq_b` linear apply (dynamic
FP8 quant + dispatched GEMM), identical in scope to the accepted decode `o_proj`
region. The broader indexer subsystem (fused RoPE/hadamard/quant →
`fp8_paged_mqa_logits` score → `topk_transform_512`) is dominated by the score
and top-k, which would dilute any wq_b leaf win and requires a checkpoint-backed
end-to-end run to measure faithfully. Bit-exactness (max-abs-err = 0 at both M)
is the guard that a candidate cannot corrupt the downstream top-k selection —
the failure mode that rejected the prior Triton attempt.

## Environment

- `SGLANG_GLM52_OPT=0` for stock arms; single B200/SM100
  `GPU-30b619de-87f2-1862-0d07-a595da8fe417`; `torch 2.11.0+cu130`.
- sglang worktree `goal/glm52-hotspot-index-q-upproj`; kernel-harness same
  branch; deepgemm **unchanged** (candidate is a pure `compiled_dims="nk"`
  argument through installed DeepGEMM).
- Task caches only; free disk kept > 8 GiB throughout.

## Deliverables

- Source: `sglang/python/sglang/srt/layers/glm52_opt/{registry,config,dispatch}.py`.
- Tests: `sglang/test/registered/kernels/test_glm52_infini_fixed_nk.py`
  (extended for graph-only) — 5 GPU subtests pass;
  `serving_native/test_index_q_upproj_graph_only.py` — 6 CPU checks pass.
- Contract: `serving_native/index_q_upproj_graph_only_gpu_contract.py`.
- Raw evidence: `gpu_contract_r1.json`, `gpu_contract_r1_confirm.json` (this dir).
- `attempt_ledger.md`, `reachability_and_abi.md` (this dir).
