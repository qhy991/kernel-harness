# Final report: no-replacement

## Outcome

No FlashMLA replacement is enabled for either M16 or M32. Seven material
candidates were tested after an identity control. Every candidate missed at
least one required local performance lane, so the plan's terminal rule selects
`no-replacement`. Installed SGLang/sgl-kernel FlashMLA remains the production
path, and the default-off hotspot profile remains off.

The exact target was SGLang API-v1 `flashmla_kv` at base
`83d313104d089bcd2af26b28453ff880f1e6a80b`, not the frozen BF16 DSA task or a
score/value proxy:

- M16: Q `[16,1,64,576]` BF16, KV `[2049,64,1,656]` FP8 E4M3FN.
- M32: Q `[32,1,64,576]` BF16, KV `[4097,64,1,656]` FP8 E4M3FN.
- Sparse top-k 2048, page size 64, V dimension 512, scale 0.0625.
- Complete prefixed V32 main plus the unchanged stock BF16 combine.

The installed control extension is SHA-256
`d8d97150bd86381c73406603cb7d6b682767535e0526053f04e3acefadb13316`.
The final experimental binary is build
`24c522c90bc8583e2aa98a1e926d0bf853d1ed0eb01b59dd735f642fb68fa331`,
SHA-256
`8929f9c76419e06b9349cc6ee858dfd32b3072fb42e4c5cdbc6e8c57b2b36862`.
The final local source commits are SGLang
`c52f23b567e5061b71c28a8971459224c057ada1` and FlashMLA
`65293ac6553d5119504be1f0ffcb229a0dc1fe42`.
The immutable kernel-harness campaign source and raw evidence are committed at
`db8e1bd755fdc37940c4b1ecf5585ace6f3048fa`; the append-only recipe is
[`testbench/knowledge/entries/glm52--dsa_flashmla_kv_decode--b200--20260729c.json`](../../testbench/knowledge/entries/glm52--dsa_flashmla_kv_decode--b200--20260729c.json).

## What changed

SGLang now fails closed on the exact page count for each bucket and validates
the returned LSE shape, stride, dtype, device, and storage offset before
recording a hit. CPU contract tests cover non-promotional M, phase/speculative
mode, dtype, layout, page count, scale, backend kind, and device cases. A
selected provider error propagates; stock is never called after candidate
invocation.

FlashMLA contains goal-local, macro-gated V32 instantiations. Default upstream
instantiations are unchanged. All compiled candidate globals begin with
`infini_kernel_glm52_flashmla_sparse_decode`. The provider compiles and
allocates fixed buffers before measurement, then performs one allocation-free
extension call on the current PyTorch stream.

The seven bounded hypotheses were coordinate/scale handoff, NoPE shared-load
staggering, native packed FP8-to-BF16 conversion, two cache/producer ordering
variants, exact-contract specialization, and one justified composite. The full
ledger, including the first B5 build failure and evidence-gated non-attempts,
is [`evidence/attempt_ledger.json`](evidence/attempt_ledger.json).

## Correctness and topology

The final B3+B5 composite is bitwise exact against installed stock for M16 and
M32 at:

- leaf eager and non-default stream;
- independently captured leaf graphs before and after Q/index mutation;
- deterministic repeated graph replay with poisoned outputs;
- containing `DeepseekSparseAttnBackend._forward_flashmla_kv` eager and graph;
- exact provider launch count and wrong-page zero-launch rejection.

It also passes 17 adversarial cases per bucket: random, zero, signed ramp,
extreme finite, exponent-boundary, two repeated changed Q/KV inputs,
duplicate/interleaved/sorted/reverse/boundary/`-1` indices, and minimum,
mixed-edge, and maximum scheduler/split cases. See
[`evidence/b3_b5_correctness_m16.json`](evidence/b3_b5_correctness_m16.json),
[`evidence/b3_b5_correctness_m32.json`](evidence/b3_b5_correctness_m32.json),
[`evidence/b3_b5_matrix_m16.json`](evidence/b3_b5_matrix_m16.json), and
[`evidence/b3_b5_matrix_m32.json`](evidence/b3_b5_matrix_m32.json).

Nsys reports for both buckets show exactly one main followed by one stock BF16
combine in eager and in each of five graph replays, with no other device kernel
inside the marked ranges. Candidate graphs differ at the main symbol only.
The audited summaries are
[`evidence/b3_b5_chain_m16_summary.json`](evidence/b3_b5_chain_m16_summary.json)
and
[`evidence/b3_b5_chain_m32_summary.json`](evidence/b3_b5_chain_m32_summary.json);
the raw `.nsys-rep` and SQLite exports are retained beside them.

## Generated binary

The composite reduces the main from 4,128 to 3,336 static SASS instructions.
It removes generic-contract address/control work and uses direct
`F2FP.BF16.E4M3` conversion, but stays at 168 registers, 16 allocated barriers,
one 148-CTA wave, and zero stack/local/spill traffic. All eight control/variant
binaries have the same entire extracted combine SASS SHA-256
`98a76d6f578e29ba175dc2718ecdfc1dfed12d95eff1308fb896eb077e870e4c`.

The machine-readable resource/opcode audit is
[`evidence/binary_manifest.json`](evidence/binary_manifest.json). Raw control
and composite main SASS is under
[`evidence/disassembly/`](evidence/disassembly/). The fair build intentionally
contains native `sm_100f` cubins and no embedded PTX fallback; the inline PTX
source and resulting SASS instruction are both recorded.

## Fair performance result

Each cell below is the minimum–maximum over all four required estimators in
three independent series, with 100 alternating pairs per series (50 AB and 50
BA). A lane passes only if its minimum is at least 1.03.

| Bucket | Leaf eager | Leaf graph | Containing eager | Containing graph |
|---|---:|---:|---:|---:|
| M16 | 1.1435–1.2636 pass | 1.0056–1.0633 fail | 0.7610–0.8634 fail | 1.0189–1.0989 fail |
| M32 | 1.1402–1.2428 pass | 1.0082–1.0578 fail | 0.7929–0.8694 fail | 1.0131–1.0851 fail |

Leaf eager benefits from the experimental provider's preallocated output
ownership, whereas graph replay isolates the stable device chain. The graph
gain is variable and below the gate. The current API-v1 containing eager route
also includes Python dispatch and hit bookkeeping and is materially slower.
Neither favorable leaf eager measurements nor individual graph estimators can
promote a bucket.

The immutable audit of all 14 candidate/bucket timing files is
[`evidence/timing_gate_audit.json`](evidence/timing_gate_audit.json); all raw
ordered observations remain in the corresponding `*_paired_m*.json` files.
There are zero promotable buckets.

## Profiler decision

KernelWiki guidance was followed: preserve the one-SM tcgen05/TMEM/TMA and
mbarrier phase structure, keep 128-byte TMA alignment, and avoid an unproven
2-SM rewrite or TMEM-as-random-scratch design. Prior NCU evidence already showed
one CTA per SM, low eligible warps, long-scoreboard/barrier pressure, and zero
spills. No candidate survived fair timing, so there was no concrete survivor
question that justified a new NCU capture. This is recorded as a deliberate
non-invocation, not missing evidence.

## Repository validation

The goal-scoped CPU checks pass: all nine SGLang registry tests, campaign
Python compilation, all retained JSON parsing, the 24-task structural
selftest, knowledge lint, and `git diff --check` in all three worktrees.

Two broader base checks remain nonzero for inherited contracts outside this
campaign:

- `python3 serving_native/selftest.py` expects
  `PRODUCTION_FLASHMLA_KV_DECODE_CASES` in
  `test/registered/attention/unittests/dsa/test_dsa.py`. The expectation is
  already present at kernel-harness base
  `660f88ef6d551cffc89b7fc1bd8fe3817fadbc3a`, while the marker is absent from
  the mandated, unmodified SGLang base
  `83d313104d089bcd2af26b28453ff880f1e6a80b`.
- `python3 testbench/bin/verify_harness.py` reaches its generated-projection
  check and reports 48 pre-existing stale `problem.json`/`README.md`
  projections plus advisory pointer drift. The relevant task, harness, and
  generated files are unmodified from kernel-harness HEAD; this goal forbids
  regenerating them.

Neither failure touches the provider, dispatch guard, FlashMLA source,
generated binary, correctness harness, or paired measurements. They are
reported rather than repaired by changing unrelated or forbidden files.

## Provenance and limitations

The host has four B200 GPUs, so checkpoint-backed TP8/DP8/EP8 acceptance was
not locally runnable. That does not affect this disposition because mandatory
local gates already failed. The exact external commands are retained in
[`EXTERNAL_ACCEPTANCE.md`](EXTERNAL_ACCEPTANCE.md), but this rejected candidate
must not be enabled to seek an external override.

The paired runs were made while the three dedicated worktrees contained the
uncommitted session changes. Every result binds the exact FlashMLA source
closure and binary SHA, and the final local commits contain those same source
bytes. This dirty-tree provenance is therefore disclosed rather than relabeled
as an official frozen-task `result.json`. There is no applicable
`audit_result.py` artifact because this production serving-native ABI is not a
frozen `testbench/tasks/glm52` task.

No remote state was modified.
