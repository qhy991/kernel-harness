# Shared vocabulary — internal recipes ↔ KernelWiki

The experience bank stays useful only if two sessions describe the same bottleneck and
the same technique with the same words. This file is the controlled vocabulary the
`knowledge.py` schema and the KernelWiki bridge share, so an internal entry's tokens
index straight into KernelWiki prior-art.

## `bottleneck.kind` (enforced by `knowledge.py`)

`memory-bandwidth` · `compute` · `launch-overhead` · `kernel-count` ·
`quantization-overhead` · `occupancy` · `synchronization` · `none-identified` · `other`

Each maps to a KernelWiki `--symptom` for the prior-art bridge
(`testbench/bin/kwiki_bridge.py::_SYMPTOM`):

| internal `bottleneck.kind` | KernelWiki `--symptom` |
|---|---|
| memory-bandwidth | memory-bound |
| compute | compute-bound |
| occupancy | low-occupancy |
| launch-overhead / kernel-count | tail-effect |
| synchronization | pipeline-stall |

Unmapped kinds fall back to a natural-language query (still useful, just less precise).

## `approaches[].technique` (free text, but converge on KernelWiki's tags)

`technique` is free text so a novel idea is never rejected, but prefer KernelWiki's
canonical technique names when one fits, so `knowledge.py brief` can cross-link:
`warp-specialization`, `ping-pong-scheduling`, `persistent-kernel`, `epilogue-fusion`,
`pipeline-stages`, `vectorized-loads`, `swizzling`, `split-k`, `2sm-cooperative`,
`tcgen05-mma`/`tmem`, `nvfp4-block-scale`, `tma`, `cuda-graph`, `pdl`.
Canonical list + aliases: `KernelWiki/data/tags.yaml`, `KernelWiki/data/aliases.yaml`
(query them with `python3 KernelWiki/scripts/query.py --tag <t>`).

## Why not hard-reject off-vocabulary techniques
Capture must stay cheap (a rejected close-out is a lost lesson). The vocabulary is a
convergence target and a bridge key, not a gate. `knowledge.py brief` degrades to a
natural-language KernelWiki query when a technique isn't a known tag.
