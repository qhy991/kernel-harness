# Root cause & fix — why the 0723 replay showed wrong-correctness / negative-optimization

Follow-up to `README.md` + `results.csv` in this directory. The replay ran the archived
`0723-amd-glm52` candidates against the current harness and reported: o_proj **m4096** and
all **index_k** shapes FAIL (calc_diff ≈ 1), **dsa_attn** ERROR, and several ops at
low/negative speedup. Investigation (scripts under `/opt/mizar/huyan/reopt/`) showed the
**kernels were correct** — the harness bench standard and baseline were wrong.

## Root cause

1. **Correctness gate used a production fp8 kernel as the ground-truth oracle.**
   `evaluate_task._correctness` fed the candidate *and* the reference through
   `glm52_ops_amd.reference` → `AmdRewardbenchProvider.reference(gemm)`, whose chain is
   sglang `aiter_w8a8_block_fp8_linear` → aiter CK → hipBLASLt → dequant. On gfx942 that
   dispatch is **shape-dependent** (sglang routes M≥4096 to
   `gemm_a8w8_blockscale_bpreshuffle_asm`, which needs *preshuffled* weights the harness
   never supplies → wrong output) and **environment-dependent** (CK/ASM not built here;
   hipBLASLt rejects blockwise scales). The `calc_diff < 5e-6` + abs_tol gate is calibrated
   for a dequant-f32 oracle ("same fp8 bytes, only accumulation order differs"), not for
   another kernel. So at M≥4096 (o_proj m4096; index_k projects S=65536 ≥ 4096 for every M)
   the *oracle* was garbage and correct candidates scored calc_diff ≈ 1.
   Proof: our candidates AND aiter's Triton blockscale both match a dequant oracle to
   ~5e-9 at **every** shape.

2. **Same defect in MLA.** The dsa correctness oracle (`_sparse_mla_reference`) rounds
   `probs → bf16` and does the PV product in bf16 — itself ~6e-6 (calc_diff) off the true
   f32 math, enough to FAIL a correct flash-decode candidate on a few near-zero outputs.

3. **dsa_attn ERROR = schema drift.** `_build_mla` yields 2-D `kv (S,576)` + 2-D int64
   `indices (M,2048)`, but the archived `dsa_factory` wrapper indexed `[:,0,:]` (3-D).

4. **Low/negative speedup = stale baseline.** The archived 2.22× geomean was vs aiter's
   slow raw-Triton fallback; `amd_glm5_targets.csv` and the runtime baseline had drifted.

## Fix (bench standard + baseline)

- **Gate → deterministic math oracle, decoupled from the latency baseline.** New
  `correctness_reference` (`backends/rocm_amd.py`, `glm52_ops_amd.py`, `glm52_ops_cuda.py`,
  wired in `evaluate_task._correctness`): gemm → dequant-f32 matmul; mla → fully-f32 sparse
  oracle; other families unchanged. B200/CUDA behaviour is preserved (delegates).
- **Baseline → runnable + honest.** `reference(gemm)` now falls to aiter's Triton
  blockscale (correct, no CK/ASM build) before CK/hipBLASLt, so latency is measured against
  a real aiter kernel instead of erroring or dropping to the slow dequant.
- **dsa wrapper** accepts 2-D and 3-D kv/indices; PV product lifted to f32.
- **Targets regenerated** (`emit_targets.py` → aiter provider) against the honest baseline.
- **run_flow.py** points at `glm52_amd`, uses the `aiter-torch-reference` bundle, and honors
  `KH_PYTHON` (the tmpfs venv is rebuilt per session).

## Re-validated (MI300X gfx942, HIP-event, math-oracle gate, aiter-Triton baseline)

| op | shapes | correctness | geomean speedup | vs 1.5× target |
|---|---|---|---|---|
| o_proj_prefill  | 3/3 WIN | calc_diff ≤ 5e-9 | 1.55× | TARGET_MET (107%) |
| index_k_prefill | 3/3 WIN | calc_diff ≤ 3e-9 | 1.75× | TARGET_MET (117%, re-optimized) |
| dsa_attn_decode | 2/2 WIN | calc_diff 7e-7 | 1.66× | TARGET_MET (112%) |

No spurious FAILs, no negative optimization. The re-launched optimize step (config sweep)
improved index_k from 108% → 117% of target (322.9µs → 297.9µs). The tuned winners are
ported into `testbench/tasks/glm52_amd/*/candidate.py` as the default seeds.
