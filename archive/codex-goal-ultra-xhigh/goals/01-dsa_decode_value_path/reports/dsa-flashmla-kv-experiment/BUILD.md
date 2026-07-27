# Isolated build provenance

The installed `sglang-kernel` package was never overwritten.  The experiment
was compiled into a separate operator namespace and loaded with
`torch.ops.load_library`, allowing stock and candidate calls to alternate in a
single process.

## Source pins

- SGLang base: `f93f8867b4bc124c9809c9110ec7361ed11b6b4a`
- SGLang local build-hook commit: `d33ad5bf4aa06c7aaf6d6c5dc66770a521719921`
- FlashMLA base: `05e26647fe840b8baedae486c2d86d5ce4efeb7c`
- FlashMLA local experiment: `cccb46c93dd3470c021201628623c7e010616d3d`
- FlashMLA CUTLASS submodule: `147f5673d0c1c3dcf66f78d677fd647e4a020219`
- PyTorch: `2.11.0+cu130`; CUDA runtime: `13.0`; build nvcc: `13.2.78`

The portable source delta is
`source/0001-experiment-bound-sparse-decode-combine-splits.patch`.

## Configure and build

Run from the isolated Kernel-Harness root:

```bash
.venv/bin/cmake \
  -S /home/qinhaiyan/glm52-goal-runs/01-dsa_decode_value_path/sglang/sgl-kernel \
  -B profile/dsa-flashmla-kv-experiment/build \
  -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_PREFIX_PATH=/home/qinhaiyan/glm52-goal-runs/01-dsa_decode_value_path/kernel-harness/.venv/lib/python3.12/site-packages/torch/share/cmake \
  -DPython_EXECUTABLE=/home/qinhaiyan/glm52-goal-runs/01-dsa_decode_value_path/kernel-harness/.venv/bin/python \
  -DPython3_EXECUTABLE=/home/qinhaiyan/glm52-goal-runs/01-dsa_decode_value_path/kernel-harness/.venv/bin/python \
  -DSKBUILD_SABI_COMPONENT=Development.SABIModule \
  -DSKBUILD_SABI_VERSION=3.10 \
  -DFETCHCONTENT_SOURCE_DIR_REPO-FLASHMLA=/home/qinhaiyan/glm52-goal-runs/01-dsa_decode_value_path/sglang/third_party/FlashMLA-goal01 \
  -DENABLE_BELOW_SM90=OFF \
  -DSGL_KERNEL_ENABLE_FA3=OFF \
  -DSGL_KERNEL_COMPILE_THREADS=4 \
  '-DCMAKE_CXX_FLAGS=-DSGL_FLASHMLA_TORCH_LIBRARY=sgl_kernel_goal01 -DSGL_FLASHMLA_EXTENSION_NAME=flashmla_goal01_ops'

.venv/bin/cmake --build \
  profile/dsa-flashmla-kv-experiment/build \
  --target flashmla_ops -j 4
```

The final artifact is `artifacts/flashmla_goal01_ops.so`, SHA-256
`091b5d42408e0a2def6086cc1d44e0ec12db6e936481ac019c7d982d90518435`.
The installed stock extension remains
`d8d97150bd86381c73406603cb7d6b682767535e0526053f04e3acefadb13316`.
Both namespaces were loaded and observed in one process before paired timing.

The ignored `build/` directory is preserved locally for reproduction; the
source patch and final binary are committed evidence.
