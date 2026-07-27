# Validation matrix and fallback policy

| Contract | Evidence | Status |
|---|---|---|
| Reached callable | Static SGLang trace identifies eager `Indexer.forward_cuda -> Indexer._fused_q_prepare_and_store`; corrected runtime traces call that real unbound method | Partial: production-shaped rank-local reconstruction only; no live model route |
| Fixed checkpoint | Pinned config and ranged safetensors headers at `nvidia/GLM-5.2-NVFP4@aec724e8...` | Pass (`fixed_model_contract_cpu.json`) |
| Fixed shape/ABI | BF16 `x[4096,6144]`, BF16 `q_lora[4096,2048]`, BF16 `wq_b[4096,2048]`, BF16 fused weight `[160,6144]`, FP8 Q, FP32 gates, page-64 uint8 cache | Pass (`exact_bf16_wq/runtime_abi_trace_*.json`) |
| Quantization resolution | ModelOpt ignore list covers `self_attn`; actual SGLang config resolves `indexer.wq_b` to `UnquantizedLinearMethod`; checkpoint header is BF16 | Pass |
| RoPE/norm | Official default interleaved RoPE uses max position 1048576/base 8000000; absent norm field selects FP32 LayerNorm eps 1e-6 | Pass |
| Numerical Q/gates | Immutable results compare before timing and after the series on a fresh deterministic seed; dtype/shape exact, floating rtol/atol 2e-2 | Pass (`hardened_runs/20260722T174049Z-immutable/validation.json`) |
| Full cache mutation/content | Immutable reference caches use independent A5/5A poison and candidates use 3C/C3; the full page-64 uint8 cache is compared byte-exact, including scale bytes | Pass: dual-poison write-coverage replay before and after timing |
| Checkpoint fused wk loader | CPU test covers BF16 K/gate row placement, the block-FP8 dequantization call and weight/scale rendezvous, BF16 output dtype, pending pair handling, and fail-closed fallback | Pass (`loader_contract_cpu.json`); the dequantizer itself is mocked |
| Eager stream semantics | Real alternate stream, production `enable_dual_stream=True`, final current-stream wait; immutable Nsys identifies BF16 wq, BF16 wk, Q, K | Pass for source/reconstruction mapping; Nsys overlap/gaps are capture-local because event latency is perturbed 6.20x-7.56x |
| Backend candidates | TGV and direct ATen measured in repeated paired corrected fused prepare/store runs | No stable >=1.03x win |
| Stream configuration | Exact stock-linear single-stream path measured three times at 1.012753x, 0.985414x, and 0.978400x with a matched profile | Reject; no stable >=1.03x gain and all four captured kernels serialize on one stream |
| CUDA graph behavior | Static config/source mapping resolves the fixed lane eager; no candidate is enabled in graph replay and the source trial is reverted | No source regression; no live graph/eager replay obtained |
| Added synchronization | Final SGLang source equals stock; external candidates add no production synchronization | Pass |
| LoRA | Fixed recipe disables LoRA. Static audit found a pre-existing stale fusion-flag import, so no LoRA compatibility claim is made | Out of scope; deployment warning |
| Score/top-k + selected DSA attention containing path | Static source mapping only; the named benchmark ends at fused prepare/store | Not obtained; no candidate reached this acceptance lane |
| Four-rank live request | Invalid TP4/DP1/EP1 attempt preserved; corrected allocation failed closed before launch on a logical/canonical venv check, fixed by `95060f3`; fresh retry made 180 wrapper attempts | Not obtained: fresh run never executed and ended exit 75 under shared four-GPU contention; not TP8 evidence |
| Eight-rank production request | Fixed acceptance is TP8/DP8/EP8; host exposes four B200s | External validation blocker; never weakened or relabeled |
| Profiler fidelity | Immutable Nsys in-capture event 1.085568 ms vs unprofiled stock 0.143616-0.175168 ms | 6.20x-7.56x perturbation: absolute gaps/idle fraction excluded; identity/order/grids/streams and instrumented-relative facts only |
| Source provenance | Immutable runner/candidates/SGLang/JIT inputs and outputs are SHA-bound; module origins, raw samples, final HEADs, and untracked allowlist are validated | Pass for `hardened_runs/20260722T174049Z-immutable/`; earlier campaigns remain provisional |
| Superseded evidence | Root-level prepare/store/K-before-Q/post-revert and old Q/K NCU used wrong FP8 wq/generic RoPE | Excluded from final fixed-model claims (`SUPERSEDED_CAMPAIGN.md`) |

## Exact enable/fallback policy

- `SGLANG_GLM52_OPT=0` remains the reference and final policy.
- No new production environment variable, registry entry, shape guard, or
  call-site enables a candidate.
- Trial commit `a75a772a2` is undone by `2fbd443a1`; final SGLang source is
  stock.
- Candidate files are reproduction tools only.
- Every shape, dtype, graph mode, and topology therefore falls back to stock
  SGLang.

The missing containing and TP8 gates prevent an end-to-end acceptance claim.
They do not hide a promotable local win: every corrected candidate already
fails the necessary stable rank-local fused prepare/store inner gate.
