# KernelWiki prior-art note

The session warm-start queried the Blackwell/Hopper KernelWiki for FlashMLA,
SM100 sparse decode, and SGLang DSA integration.

Useful prior art:

- The FlashMLA decode path stores each FP8 KV token as 656 bytes: 512 FP8
  latent bytes, four FP32 inverse scales (16 bytes), and 64 BF16 RoPE values
  (128 bytes). The serving-native builder uses SGLang's production
  `quantize_k_cache` helper rather than fabricating random FP8 bytes.
- SGLang PR #22372 added Q-head padding for FlashMLA compatibility. The
  containing-path tests preserve this behavior, while the production
  TP8/DP8 symbol workload reaches the head64 instantiation directly.
- SGLang PR #15242 updated the FlashMLA dependency. This goal records and builds
  from the newer exact CMake pin in the isolated checkout rather than assuming
  a repository-level FlashMLA submodule.
- Contest/prior sparse-MLA work emphasizes split scheduling and sparse gather
  locality. Those are hypotheses only until the run-local Nsight reports show
  their magnitude on the fixed M16/M32 workload.

The pinned source and compiled extension, not the prior-art summary, remain the
authority for exact dispatch, scheduler metadata, and generated code.
