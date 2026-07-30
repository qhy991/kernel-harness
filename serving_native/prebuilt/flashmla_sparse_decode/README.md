# Prebuilt FlashMLA sparse-decode extensions

## `p1_consumer_scale` (`a39236323dc57a97`)

CUDA **13.2** / B200 build of tip FlashMLA `b5af443` P1. Use this when the
host toolkit cannot JIT the same sources (CUDA **13.1** `ptxas` rejects
`cvt.rn.bf16x2.e4m3x2`).

```bash
export GLM52_FLASHMLA_VARIANT=p1_consumer_scale
export GLM52_FLASHMLA_USE_PREBUILT=1
# optional override:
# export GLM52_FLASHMLA_PREBUILT_SO=/abs/path/to/….so
```

Verify:

```bash
sha256sum serving_native/prebuilt/flashmla_sparse_decode/infini_kernel_glm52_flashmla_sparse_decode_p1_consumer_scale_a39236323dc57a97.so
# expect a368b1451733140d9801d4519ba265e5d85252d83ab7ac25e8f570059b68e4e1
```

Arch is `sm_100` / `sm_100f`. On B300 (`sm_103`), confirm the cubin loads
before treating e2e numbers as tip-equivalent.
