# Fork and build provenance

## Source identity

- Repository: `https://github.com/sgl-project/DeepGEMM`
- Vendored release: `v0.1.4`
- Recorded upstream commit:
  `731e7c7a97d269e4b9f482ea18d0e709a948f293`
- SGLang experiment commit:
  `68e047c9a9a19f70ff10e62457ca642863f84d53`
- Changed source:
  `third_party/DeepGEMM-GLM52/csrc/jit_kernels/heuristics/sm100.hpp`
- Exact diff: `source_experiment.patch`
- Final rollback commit: `8f450dbdf`

The vendored DeepGEMM directory has no nested Git metadata. The SGLang commit,
preserved patch, source hashes, upstream identity, and generated cubin hashes
are therefore the reproducibility anchors.

## Build command

Run from the SGLang worktree:

```bash
DEEPGEMM_GLM52_COMMIT=68e047c9a9a19f70ff10e62457ca642863f84d53 \
HARNESS_PYTHON=/home/qinhaiyan/glm52-goal-runs/13-attn_o_prefill/kernel-harness/.venv/bin/python \
third_party/deepgemm_glm52/build_overlay.sh
```

The build stages a separately named `deep_gemm_experimental` package and does
not install over the harness venv's stock `deep_gemm`.

## Artifact and import identity

- Overlay:
  `/home/qinhaiyan/glm52-goal-runs/13-attn_o_prefill/sglang/third_party/DeepGEMM-GLM52/overlays/68e047c9a9a19f70ff10e62457ca642863f84d53`
- Package:
  `site/deep_gemm_experimental`
- Dedicated JIT cache: `jit_cache`
- Overlay `_C.so` SHA-256:
  `1dfef5bc2f443af684b560e38c26b775c65d9284ae7ecf2c6c17d2dc81121a69`
- Python: repo-local Kernel-Harness `.venv`, Python 3.12.13
- Torch/CUDA: 2.11.0+cu130 / 13.0
- ABI: C++11 ABI enabled

`overlay_provenance.json` and `overlay_manifest.json` contain the complete
machine-readable paths. `import_identity.txt` proves stock and experimental
modules were distinct objects and records their resolved files.

The exact five-stage line-info cubin was copied to the profiler bundle with
its generated source cache. `profile/.../analysis/stage5_cubin_resources.txt`
records 42 registers, zero stack/local memory, and the cubin symbol with
`num_stages=5`. The NVCC command line is preserved in `bundle.log`.
