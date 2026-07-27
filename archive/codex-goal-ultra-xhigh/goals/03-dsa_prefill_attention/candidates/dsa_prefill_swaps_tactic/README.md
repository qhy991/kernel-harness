# GLM-5.2 DSA prefill FlashInfer tactic oracle

This candidate is a narrow source experiment, not a production replacement.
It builds FlashInfer 0.6.12's packaged TRTLLM-gen launcher against two
repo-local Apache-licensed headers and reuses the package's signed cubins. The
only selector delta is guarded by the GLM-5.2 rank-local FP8 prefill ABI. The
candidate accepts both the inherited 512-page compact replay and the observed
513-page raw SGLang pool (one dummy page plus 512 usable pages); its production
decision is based on the separately named 513-page workload.

`GLM52_DSA_SWAPS_TILE_Q=32` (the default) selects the shipped Q32
Persistent/Swaps tactic; `16` selects Q16. All non-matching shapes call the
stock reference. Compilation and cubin-loader initialization occur at import
time, outside the timed region.

The reproducible build/measurement entry point is
`GLM52_DSA_SWAPS_TILE_Q={16|32} serving_native/run.sh <workload> --candidate
serving_native/candidates/dsa_prefill_swaps_tactic`. FlashInfer's JIT emitted
`fmha_gen_glm52_dsa_swaps_q{16,32}_v2.so` under
`~/.cache/flashinfer/0.6.12/100a/cached_ops/`; each trace JSON records the exact
resolved library. Python resolves FlashInfer 0.6.12 from the repo-local venv and
SGLang from the isolated goal worktree; no installed package is overwritten.

Upstream source fingerprints before relocation:

- repository: `https://github.com/flashinfer-ai/flashinfer`;
- annotated tag: `v0.6.12` (`dd86b5bd7bac37cc9a0cc537c29c15a61a4aca99`),
  peeled source commit `d768c14e7cf5dd5df45a8a1de78ae815879f108a`;
- the four package fingerprints below match the same files fetched from that
  peeled commit byte-for-byte;

- `fmhaKernels.cuh`: `728a29df0f717990ed8b7987600c84e2f2c836200c882b733bdf79e29416d0b0`
- `fmhaRunner.cuh`: `5198b71149a96ebec048483029c7c15a92446cac0e7b0292554afa79daff6141`
- packaged launcher: `866129de2b56b165be5d4dd1bbb1b92729ddc13d31020aa1c378e1639cdac3f3`
- packaged reduction source: `2c6e4721742de7d1e21ee429c9ec429c8d3f0e5539cdba463d3f84af97d511fd`

The TP8/DP8/EP8 production gate remains external. A four-GPU result is only a
diagnostic and must never be relabeled as eight-rank acceptance.
