# gate_v1_fusednq

Imported from the 2026-07-31 fusion campaign as a preserved, non-default
candidate for `norm_quant_gate_prefill`.

Run it from the task directory:

```bash
./run.sh --candidate variants/gate_v1_fusednq
```

The CUDA kernel fuses residual-add, RMSNorm, and packed-UE8M0 per-token-group
quantization; the Python wrapper then calls the same packed-scale
`deep_gemm.fp8_gemm_nt` as the production-region reference. Its default knobs are
`GATE_NQ_ROWS=2` and `GATE_NQ_SMEM=0`. JIT artifacts default to
`/tmp/kernel-harness-<uid>/gate_v1_fusednq`; set `GATE_NQ_BUILD_DIR` to override.

Campaign evidence was reproduced on two physical B200s for M=1024/2048/4096,
but remains external-acceptance evidence: no checkpoint-backed TP8/DP8/EP8 or
end-to-end GLM-5.2 serving result was available. The candidate is therefore
saved under `variants/` and is not production-defaulted.

Main-worktree revalidation on 2026-08-01 passed the schema-1.4 full gate:
`COMPLETE_WIN`, 3/3 shapes won, 0 regressed, `calc_diff=0`, with conservative
paired speedups 1.157x / 1.148x / 1.180x at M=1024/2048/4096. The persisted run
is `runs/glm52/norm_quant_gate_prefill/20260801T032655Z-bdb223/result.json`.
It audits as provisional only because the integration worktree and candidate
were uncommitted during measurement; it is still not end-to-end promotion proof.

Do not reuse it unchanged for `norm_quant_qkv_*`. It deliberately does not write
or return normalized BF16, while the QKV/DSA production seam requires that third
output. The corrected QKV harness contract will reject it.
