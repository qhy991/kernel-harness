# Prior Knowledge — dsa_attn decode >40% HBM

## Frozen baseline

Kernel-Harness `7d79e5e`, idle B200 GPU 3, 2026-07-19.
Seed SHA-256:
`9960b32616270535c916a2e110f5f57c0ee1dd05f0f5aa4e4be986be031533c4`.

| M | candidate us | reference us | HBM utilization | correctness |
|---:|---:|---:|---:|---|
| 16 | 45.488 | 45.607 | 11.02% | PASS, calc_diff 0 |
| 32 | 46.952 | 46.992 | 21.36% | PASS, calc_diff 0 |

Strict >40% requires M16 `<12.533760 us`, M32 `<25.067520 us`.

## Required BitLessons

- `BL-20260716-mla-occupancy-wall`: pure Triton sparse MLA variants hit an
  occupancy/architecture wall; do not repeat without new evidence.
- `BL-20260719-single-kernel-span-floor`: measure main-kernel/span floors before
  broad source work.
- `BL-20260716-retarget-overwrites-solution`: ensure build/retarget steps do not
  replace the independent candidate implementation.

## Source-level options

If NCU shows recoverable headroom, the plan explicitly allows:

1. Task-local CUDA/CuTe sparse MLA using tcgen05/TMEM, indexed gather, online
   softmax, persistent scheduling and warp specialization.
2. Isolated FlashMLA / `sgl-kernel` fork loaded only by candidate.

Never overwrite Harness stock packages. Keep frozen reference on stock
`flash_mla_sparse_fwd`.
