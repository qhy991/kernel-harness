# FlashMLA split-scheduler experiment

Disposition: rejected.  Reducing useful split partitions lowers combine traffic,
but lengthens the dominant split-KV main kernel and loses all three eager series
for both production buckets.  Stock scheduler metadata remains active.

All measurements below are from immutable run `runs/screen3`, collected in one
wrapper invocation on physical GPU 1,
`GPU-5b9be10b-5bfc-b658-9b31-f7ae8516dc54`.  The reference and candidate were
alternated in a single process; eager rows contain 200 pairs each and identity
controls contain 100 pairs each.

## Paired timing

| Bucket / configuration | Run 1 | Run 2 | Run 3 | Decision |
|---|---:|---:|---:|---|
| M16 identity control | 0.9949x | 1.0017x | 0.9997x | Neutral reference noise |
| M16, 120 useful partitions | 0.9582x | 0.8780x | 0.9502x | Reject |
| M32 identity control | 0.9923x | 1.0000x | 0.9963x | Neutral reference noise |
| M32, 112 useful partitions | 0.8706x | 0.8852x | 0.8747x | Reject |

One exploratory graph series gave M16 1.0354x (p10 0.9810x, p90 1.1041x),
but that isolated favorable median contradicts all three eager series and is not
a promotion result.  M32 graph replay was 0.8952x and clearly regressed.
Fresh-input graph correctness passed in both buckets.

## Profiler explanation

| Bucket | Stock main | Candidate main | Stock combine | Candidate combine |
|---|---:|---:|---:|---:|
| M16 | 22.336 us | 24.096 us | 11.456 us | 10.208 us |
| M32 | 30.432 us | 35.488 us | 8.736 us | 8.064 us |

The combine benefit is real: DRAM reads fell from 16,818,176 to 14,717,184
bytes at M16 and from 16,818,432 to 12,615,936 bytes at M32.  It is smaller than
the main-kernel regression caused by less parallel split work.  Main-kernel DRAM
reads are unchanged (23,043,072 bytes at M16 and 46,012,928 bytes at M32), while
duration grows.  Registers stay at 168/thread, launch shared memory remains
232,656 bytes, and the launch remains one CTA per SM.

The stock NCU evidence also motivates the next source attempt: M16 has 987,080
shared-memory wavefronts versus 724,936 ideal, with 152,955 load and 151,122
store bank conflicts; M32 has 1,709,512 wavefronts, 393,216 excessive, with
325,705 load and 191,740 store conflicts.  Long-scoreboard and barrier stalls,
low eligible-warps/cycle, and low tensor-pipe activity show that reducing those
conflicts is more promising than reducing split count.

## Rollback and preserved artifacts

The attempt is an external candidate only:
`serving_native/candidates/dsa_flashmla_kv_sched_override.py`.  No production
dispatch changed.  Raw JSON, Nsight Systems reports, eight full NCU reports,
parsed metric tables, environment identity, and failed preliminary screens are
preserved under `runs/`.
