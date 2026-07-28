# Task 26 single-B200 attempt: fail-closed no-replacement

Status: **FAILED / IN_PROGRESS; independent raw review pending**.

This was the one explicitly released production-mode attempt. It used:

- Kernel-Harness `7e27974573a49025c020c01541e2abb28da044a5`
- SGLang `6bb6f99a392354314dcb37210fa8ef2829868750`
- physical GPU 1, `GPU-5b9be10b-5bfc-b658-9b31-f7ae8516dc54`
- exact post1 stock versus exact-post1-plus-BM16 candidate
- warmup 3, repeat 10, three alternating series
- the inherited `gpu1.lock` FD and the task-wide campaign lock

The untouched raw root is:

```text
runs/glm52_prod_26_moe_w2_decode_scoped_bm16/20260728T182705Z
```

It has `FAILED` and `IN_PROGRESS` markers and no `COMPLETE` marker. The driver
stopped immediately after the strict audit gate rejected the fifth lane. It
was not retried or promoted.

## Audited results

Each completed result independently re-audits as structurally valid with no
warnings or contract errors. Series values are paired median speedups.

| Lane | Series 1 | Series 2 | Series 3 | Gate |
|---|---:|---:|---:|---|
| M16 eager, expected-M 4 | 1.115753 | 1.048058 | 1.072064 | pass |
| M16 graph, expected-M 4 | 1.111381 | 1.103330 | 1.105005 | pass |
| M16 eager, current expected-M 5 | 1.149619 | 1.086777 | 1.071711 | pass |
| M16 graph, current expected-M 5 | 1.097645 | 1.096102 | 1.110980 | pass |
| M32 eager, expected-M 8 | 1.140842 | **1.023989** | 1.114165 | **fail** |

The M32 result has `valid=true` and `performance_gate_passed=false`. The plan
requires every required series to reach 1.03x, so this is the specified
`no-replacement` stop condition. The later M32-current and containing-region
lanes did not run and must not be inferred from historical evidence.

The failing series is order-sensitive in the raw samples: its alternating
paired speedups were `0.621618, 0.928215, 1.105580, 0.998714, 1.125757,
1.012290, 1.092540, 1.021696, 1.100819, 1.026281`. This observation does not
relax or override the gate.

## Contract closure

The released source already includes and tests the resumed fail-closed P0s:

- leaf-only exact edge phases and explicit edge graph-validation accounting;
- raw edge graph node/count/type/identity/forbidden-node recomputation;
- exact edge capture IDs and positive distinct graph/stream bindings;
- exactly one W2 CUDA event for each leaf eager profile;
- exactly one CUDA KERNEL node for every main and edge leaf graph;
- equal containing-region event counts with exactly one W2 substitution and
  identical ordered/multiset non-W2 events;
- adversarial BM128, duplicate-BM16, forged metadata, boolean metadata, and
  unrelated-extra-kernel fixtures.

The post-attempt CPU suite passed 60 tests. No source or audit semantics were
changed in response to the performance failure. Raw file hashes are recorded
in `FAILED_ATTEMPT_20260728T182705Z.sha256`.
