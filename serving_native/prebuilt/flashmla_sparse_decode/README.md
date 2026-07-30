# Prebuilt FlashMLA sparse-decode extensions

CUDA **13.2** / B200 builds. Use when the host toolkit cannot JIT tip PTX
(CUDA **13.1** `ptxas` rejects `cvt.rn.bf16x2.e4m3x2`).

## Preferred: P1 + combine_c2 stack

```bash
export GLM52_FLASHMLA_COMBINE_VARIANT=combine_c2_bucket_stages
export GLM52_FLASHMLA_USE_PREBUILT=1
# provider: serving_native/candidates/flashmla_combine_decode_provider.py
```

sha256 `combine_c2…ea8c72aac9631a91.so`:
`e509dff9cfb2ecf19febf4ba979608608588b173e8d14e71a448e74cd324180f`

## P1 main only

```bash
export GLM52_FLASHMLA_VARIANT=p1_consumer_scale
export GLM52_FLASHMLA_USE_PREBUILT=1
```

sha256 `p1_consumer_scale…a39236323dc57a97.so`:
`a368b1451733140d9801d4519ba265e5d85252d83ab7ac25e8f570059b68e4e1`

Arch is `sm_100` / `sm_100f`. Confirm load on B300/`sm_103` before treating
e2e numbers as tip-equivalent.
