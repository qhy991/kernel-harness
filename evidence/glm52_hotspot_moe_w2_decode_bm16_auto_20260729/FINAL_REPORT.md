# GLM-5.2 MoE W2 decode hotspot terminal report

## Disposition

**No replacement.** The final correct per-call BM16 candidate failed the first
mandatory fair-performance lane, expected-M 4 W2 leaf eager. The plan requires
every estimator in every independent series to be at least 1.03x. All four
aggregate estimators were below 1.0:

| estimator | aggregate speedup |
| --- | ---: |
| pooled | 0.801642x |
| order-balanced | 0.809902x |
| AB median | 0.795571x |
| BA median | 0.824491x |

The run contains three independent alternating series with 50 AB/BA pairs
each. Their pooled speedups were 0.799674x, 0.801754x, and 0.801013x. The
candidate median was 0.116960 ms and the stock median was 0.093760 ms.

This is a terminal local failure, not an external-acceptance candidate.
Production remains default-off. No checkpoint-backed TP8/DP8/EP8 acceptance
command was run, and there is no valid candidate command to hand to that lane.

## Scope and early stop

The exact target was the current SGLang API-v1 `moe_w2` callback at base
`83d313104d089bcd2af26b28453ff880f1e6a80b`, for E32, slab M1024, K2048,
N6144, packed int32 UE8M0 scales, PDL, 148 SMs, and expected-M 4/5/8/9.

The expected-M 4 eager leaf completed the full eager edge-mask and ownership
suite and then failed its required 3x50 timing gate. Per the plan's terminal
rule, the remaining expected-M 5/8/9 timing lanes and all graph and
containing-region promotion lanes were not run. They are neither claimed nor
needed to classify this candidate as no-replacement.

The final result self-audits as:

```text
VALID VALID_NON_WIN: moe_w2_hotspot_decode_em4 mode=eager
```

## Correctness and routing evidence

The final eager leaf result passed:

- zero experts; single-row and maximum-expert cases;
- 15/16/17, 31/32/33, 127/128/129, and 1024 boundaries;
- deterministic ramp, random sparse, extreme finite, exponent-boundary, and
  skewed masks;
- poisoned caller-owned output and untouched candidate tail rows;
- exact return-`None`, shape, dtype, stride, storage offset, stream, packed
  scale bytes, input pointer, and input immutability checks.

The candidate selected exactly 174 times. Provider attempts and completions
were both 174, with four import-time expected-M warmups. Dispatch misses,
candidate fallbacks, and reference delegations were all zero. Stock executed
only as the separately measured denominator. The selected callback contains
one candidate launch and raises if dispatch declines; it cannot invoke stock
after selection.

The stock BM128 kernel writes padded rows for positive counts that are not
multiples of 128. That behavior is recorded as the exact denominator behavior,
not misreported as satisfying the candidate's stronger caller-owned-tail
contract. The BM16 candidate preserves every invalid row inside the same
kernel.

## Generated identity and binary evidence

Both arms came from one task-local DSO with SHA-256
`395655ab609ef5037fe3bb93f4a2d813c047f1265fd5f30554948dd8d0e51780`.
The DeepGEMM, CUTLASS, and fmt bases were respectively
`edcf77b276965de8f03cdc47c23f01b08bf7c7ab`,
`f3fde58372d33e9a5650ba7b80fc48b3b49d40c8`, and
`553ec11ec06fbe0beebfbb45f9dc3c9eabd83d28`.

| property | stock | candidate |
| --- | --- | --- |
| kernel | current-source grouped masked GEMM | `infini_kernel_glm52_moe_w2_decode_bm16_auto` |
| tile/stages | BM128/BN128/BK128, 8 stages | BM16/BN128/BK128, auto-selected 12 stages |
| JIT key | `394687d565c010ed0cc18272659871a9` | `e8a6deeb7f7a319bbe485bfc6c351cae` |
| registers | 36 | 43 |
| stack/local/spills | 0/0/0 | 0/0/0 |
| dynamic shared memory | 213804 bytes | 230188 bytes |
| launch | grid 148, block 256, cluster 2, PDL | grid 148, block 256, cluster 2, PDL |
| TMA load/store | 10/16 | 10/2 |
| two-SM instructions | 25 | 25 |

The candidate uses a vectorized 16-byte valid-row store for partial BM16
tiles after the TMEM-to-shared drain and named-barrier handoff. Full tiles
retain TMA stores. This removes padded global writes without a helper kernel,
host mask read, allocation, synchronization, or transformed scale.

The CPU-only signed identity is
`17a5e23c0bd3cac16d88a1054047c4488fd9ca46b34fcca25560f83fdca7858b`.
The stock and candidate manifest hashes are
`5cbda917dfc2be33362a3fcf9a0a7a7a240edc03db352b03a6beba20a2df4fb4`
and
`4b2eac068b797b3d798c3bc24ecdf1b4c72284868a1ef3a565137bcb6ce32734`.

## Profiler interpretation

The in-process CUDA profiler captured exactly one W2 kernel in each arm:

- stock kernel: 77.659 us;
- candidate kernel: 68.223 us.

The generated-binary hypothesis therefore worked at device-kernel scope:
fewer output stores make the BM16 kernel faster. It did not survive the
current API-v1 enqueue interval. The complete selected path measured about
0.117 ms versus about 0.094 ms for stock. Removing two per-call Python counter
dictionary snapshots recovered roughly 8 us versus the preceding candidate,
but the final lane still lost decisively.

No NCU run was made because there was no surviving end-to-end candidate and
therefore no concrete survivor question. No Nsys promotion trace was made
because the first required performance lane failed before graph or containing
region promotion.

## Attempt portfolio

The bounded portfolio used five material kernel binaries including stock,
below the plan's limit of eight:

1. CPU identity/setup audits found and fixed a wrong pre-imported DeepGEMM
   namespace and missing runtime ptxas-verbose flag before valid timing.
2. Raw BM16-auto was rejected as incorrect because it overwrote invalid rows
   in the caller-owned slab.
3. Restoring invalid tails inside the kernel was correct but measured
   0.698625x pooled.
4. Scalar direct valid-row stores were correct and measured 0.799973x pooled.
5. Vector direct valid-row stores were correct; deferring Python counter
   materialization produced the final 0.801642x pooled result.

The full machine-readable ledger is in `attempt_ledger.json`. Routes B1, B3,
B4, and B5 were not expanded after the focused B2 survivor failed the first
end-to-end gate. In particular, there was no evidence to justify an adjacent
stage experiment, barrier rewrite, alternate GEMM, inline PTX, or NCU capture.

## Evidence locations

The raw final result is:

```text
/home/qinhaiyan/glm52-hotspot-goal-runs/cache/moe_w2_decode/results/b2_vector_postrun_counters/leaf_eager_em4.json
SHA-256 0f6506edd485d40dc360e717744c31ace1bcfc964445468148391640635603f0
```

The immutable task-local identity root is:

```text
/home/qinhaiyan/glm52-hotspot-goal-runs/cache/moe_w2_decode/identity_v1
```

The result records dirty in-session task branches. This is disclosed rather
than presented as a clean-tree measurement. The measured runner, workload,
candidate, provider, DSO, generated sources, cubins, PTX, SASS, and manifests
are individually hashed in the result. No measured source changed after the
final run; only this terminal evidence and the append-only knowledge entry
were added before local commits.

## Enable and fallback policy

The optimized path is opt-in only and exact-descriptor fail-closed. Unsupported
operator, phase, M bucket, expected-M, tensor ABI, recipe, topology, or runtime
state declines before candidate invocation and leaves stock selected by
SGLang. Once the candidate is selected, decline or failure is fatal and no
stock fallback is permitted.

For this terminal disposition, omit the hotspot environment variables and use
stock SGLang. The candidate is not eligible for external acceptance, and
production default remains `false`.
