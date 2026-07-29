# Goal 24 P0 fairness audit

Status: **CPU contract closed; clean same-source build completed; leased
JIT/GPU validation unblocked after the mandatory immediate disk recheck.**

No timing from Goal 19 is inherited. Its denominator used PDL disabled and an
installed-stock versus v0.1.4-overlay comparison, so it is not a same-source,
same-runtime-state control. Its BM32 result is retained only as a hypothesis
for the two-CTA configuration `(32,128,128,11,2)`; it is not evidence for the
one-CTA configuration.

## Locked source identity

- Kernel-Harness base:
  `c1c48c3d1e826c243727ed45d52ef8dbfeb3f701`
- SGLang base:
  `0a723222cf653758dcf5ad677453b226f1981444`
- DeepGEMM source commit materialized for both arms:
  `731e7c7a97d269e4b9f482ea18d0e709a948f293`
- CUTLASS:
  `f3fde58372d33e9a5650ba7b80fc48b3b49d40c8`
- fmt:
  `553ec11ec06fbe0beebfbb45f9dc3c9eabd83d28`
- tracked complete candidate patch:
  `997348b6498aa18a7d70a5b1d36249b356b508cdc71e2f514a979818c48490a5`
- reconstructed stock tree:
  `917592ab68ea0608c9be33208c2c609bc7f20bd9b1603f32743dd0d1ae03d0ed`
- reconstructed candidate tree:
  `d38d8bf9a2118a2506be0fd71827568e70a20839505238a36a9c0325415332ef`

`third_party/deepgemm_w13/build_variants.py --audit-materialization`
reconstructed each tree twice from pinned `git archive` inputs and the tracked
patch, then compared byte/mode/path tree hashes. It neither imported Torch nor
compiled or initialized CUDA. The old ignored
`third_party/deepgemm_w13/build/{base,candidate}` copies were removed after
their ignore status and exact 4.5 MiB scope were verified; no evidence depends
on them.

The final build command used `--force`, removed both prior object directories
and both JIT roots, and used one build function and one recorded
flag/include/link template for stock and candidate. Copied archive trees never
ran a git-submodule update.

The completed build manifest is
`/home/qinhaiyan/glm52-v2-goal-runs/cache/24-moe_w13_decode/deepgemm/w13_variants/manifest.json`
with SHA-256
`afa7063a860cd045138e51abcb5b8b44c226db13c08e212ae30da077f5655621`.
It records:

- equal normalized stock/candidate Ninja plan:
  `9c3896ef36e436ee9de3a5808cad755f58e00adc34e0440498bca2973b109c46`;
- GCC binary:
  `1353e9bdd29a7295c7226bf6c63abccce056d8cac31f112e5cdbecc3f28c2769`;
- NVCC binary:
  `02afd6a20bd29fae33bf278fa847a4e9711db25afec4a1bf648be81a3b210af0`;
- stock DSO:
  `085ded6f88cb6f1f0cf7542d1362d9a732772052996aca5e849d732be1ef45d8`;
- candidate DSO:
  `d2bf446348ea1c4e285b3862240990d5369c7c13d7aea4a5d75673a98a75efd7`;
- stock/candidate generated Ninja files:
  `1de4bc4f24442b63360801fa947da05ea991f9bffe76a228ef9e29045bea3fb2`
  and
  `74b79153b54f14fbc83e099dbfb97f06f2e50f66b2370b00b1e0c24120276be6`.

Both JIT roots remained empty after the CPU build. The immediate post-build
`df -BG /` check reported 17 GiB available, above the 8 GiB stop threshold.

## Closed CPU contracts

- Dedicated W13 dispatch is default-off and import-time CPU-only.
- GPU/DSO/runtime setup is reachable only from
  `update_deep_gemm_config` after worker assignment.
- Startup binds and warms exact stock before importing `compile_utils`,
  disables broad precompile, then binds and warms candidate.
- Stock and candidate have distinct package, DSO, and JIT roots. Runtime
  evidence must include package/DSO hashes and every frozen JIT artifact hash.
- Both lazy compilers are fixed to NVCC (not NVRTC) and the caller's JIT-backend
  environment is restored after startup.
- Both modules explicitly set/read `PDL=true`, `num_sms=148`, and
  `tc_util=100`. Both mutation directions for all three fields are read back
  and restored. Failed opt-in setup restores the installed stock runtime state.
- The selected route accepts only E32/slab1024/K6144/N4096, exact FP8/BF16
  tensors, exact packed-int32 scale shapes/strides, expected-M 4/5/8/9 paired
  to token buckets 16/32, no overlap or recipes, and `max_block_n=256`.
- A selected call executes exactly one low-level candidate launch, propagates
  errors, bypasses generic hooks/precompile/locks/NVTX/statistics, and preserves
  the successful no-overlap return value `None`.
- The production eager and decode-graph model callsites enter a private
  token-reset W13 context. Generic GLM52 phase ContextVars are not used.
- Leaf and containing-region graph contracts independently name all four
  expected-M points. Replay mutates activation and device `masked_m`, restores
  full-output poison, checks pointer stability and deterministic output, and
  validates untouched masked rows from CPU-known metadata without a device
  `.tolist()`.
- The result auditor rejects source/build/manifest drift, runtime-to-manifest
  path/hash mismatches, aliased caches, missing state-independence proof, broad
  precompile, and incomplete W13 graph observation.

## CPU gate results

All commands ran with `CUDA_VISIBLE_DEVICES=''`.

- SGLang W13 contract: 12 passed.
- Existing GLM52 registry contract: 11 passed.
- SGLang overlap/startup contract: 9 passed.
- serving-native structural inventory: 46 workloads, including four W13 leaf
  and four containing-region eager/graph workloads.
- serving-native adversarial schema audit: 33 passed.
- W13 graph/cache-lifecycle tests in the Torch environment: 7 passed.
- `verify_harness.py --skip-task-projection`: passed. The omitted generated
  task projection requires its GPU-derived tensor tables and remains a leased
  GPU gate; frozen task files were not changed.
- Ruff fatal/import checks passed. `git diff --check` passes outside the
  byte-locked patch payload; its whitespace-only added lines are intentionally
  preserved as part of the attested `997348b6...` patch hash.

No JIT, CUDA Graph, kernel launch, timing, or profiler capture is claimed by
this report.
