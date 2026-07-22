# testbench/tasks/glm52 — legacy bridge tree

**Status: retained for compatibility, prefer the per-platform trees for new work.**

The single-tree, IS_ROCM-branching layout has been superseded by two independent
trees, one per hardware platform. Same 26 task names, but the CUDA and AMD data
flows, dtypes, and reference kernels are now separated at the module level:

- **CUDA / B200**: [`../glm52_cuda/`](../glm52_cuda/) — `float8_e4m3fn`,
  `deep_gemm.fp8_gemm_nt` / `sgl_kernel.bmm_fp8` / `flash_mla_sparse_fwd`,
  TMA/UE8M0 scale format, `logits_paged` for `index_score_decode`
- **AMD / MI300X**: [`../glm52_amd/`](../glm52_amd/) — `float8_e4m3fnuz` (FP8_MAX=224.0),
  `aiter.gemm_a8w8_blockscale` / `aiter.mla.mla_decode_fwd` / `aiter.ops.triton.fp8_mqa_logits`,
  no UE8M0, `logits_ksrange` for `index_score_decode`

Why this tree still exists:

- Existing archive tarballs, log entries under `archive/0720-Best-GLM-52/`, and
  older `kda/*` branches reference paths like
  `testbench/tasks/glm52/o_proj_decode/candidate.py`. Removing those paths would
  break reproducing prior runs.
- The `testbench/harness/glm52_ops.py` shim still routes here transparently, so
  every `./run.sh` in this tree still works — it just resolves to
  `glm52_ops_cuda` or `glm52_ops_amd` based on the active backend.

## For agents

If you are starting a new session, use the platform-specific tree:

```bash
# CUDA agent
T=testbench/tasks/glm52_cuda/o_proj_prefill
$T/run.sh --describe

# AMD agent
T=testbench/tasks/glm52_amd/o_proj_prefill
$T/run.sh --describe
```

If you are auditing or reproducing a historical run that names a path under
this directory, the shim keeps it working — no migration needed.

## For maintainers

- `testbench/bin/sync_glm52_tasks.py` targets `glm52_{cuda,amd}/` only. It does
  NOT regenerate this legacy tree. If a task file here drifts from
  `glm52_ops`, `evaluate_task.py` will reject it with exit 3.
- Delete this tree once no external references remain (`git log --all -- ...` shows
  it is truly unused). Not before.
