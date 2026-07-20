# Attempt DAG — glm52/index_score_decode HBM >=82%

```
[baseline: stock fp8_paged_mqa_logits]  M16 54.3% / M32 61.3%  calc_diff=0  (CORRECT, fastest-correct)
        |
        v
[NCU profile] --> bytes actual≈requested (±1%)  --> coalescing/row-load mechanism: RULED OUT (no waste)
        |
        +--> in-kernel DRAM ceiling 64.3% / 75.1%
        |         limited by occupancy 18.75% (1 block/SM; regs 168 + smem 220.67KB + launch_bounds(384,1))
        |         KV pipeline already 5 stages (smem-capped) --> deeper-pipeline mechanism: RULED OUT (no smem)
        |
        +--> [roofline] best case kernel@100% realistic peak + span:
                  M16 80.5% , M32 78.3%   ==> both < 82%  ==> TARGET ABOVE PHYSICAL CEILING
```

## Candidate mechanisms vs evidence

| Mechanism (plan §4) | Status | Evidence |
|---|---|---|
| 132B row load / coalescing | ruled out | actual/requested DRAM = 1.011 / 1.003 — already optimal |
| page / address arithmetic | ruled out | subsumed by "bytes optimal"; no extra traffic to remove |
| split_kv / splits_per_chunk | ruled out (as HBM% lever) | grid already 148 = #SMs, 1 wave; more splits cannot raise concurrent blocks past 1/SM (smem cap) and would add a combine kernel (forbidden host gap) |
| persistent scheduling / SM count | ruled out | already persistent over all 148 SMs, 1 wave |
| deeper TMA pipeline (more KV stages) | ruled out | smem at 220.67/228 KB; no room for a 6th stage |
| output writeback / epilogue | negligible | output is M×65536×logits; writeback tiny vs 142–285 MB KV reads |
| reduce span floor | insufficient | span 3.55/8.36 us; even zero-span + 100% realistic peak = 81.x% best (still needs kernel far above its 64/75% ceiling) |

## Terminal node

No source mechanism closes the gap to 82% on the **binding M=16 shape**, whose
82% span-inclusive limit demands ~98% of *spec* HBM peak — above the ~95.8%
realistically achievable peak and far above the kernel's 64% in-kernel ceiling.

Outcome: **reviewed NO-GO** (pending independent Codex confirmation), fastest
correct candidate = stock frozen call, preserved in `candidate/candidate.py`.
