# GLM-5.2 FlashMLA combine PTX round 1

Optimize the **stock BF16 split-combine** that runs after the sparse FP8 MLA
decode main kernel, holding the main kernel at the round-2 survivor P1.

**Disposition: `external-acceptance-candidate`.** `combine_c2_bucket_stages` is
selected for both M16 and M32. Production default stays off; registration ceiling
is L2 external E2E. Start with [`FINAL_REPORT.md`](FINAL_REPORT.md).

| Bucket | Graph containing vs stock | Graph leaf vs stock | vs P1 + stock combine | combine kernel |
|---|---:|---:|---:|---|
| M16 | 1.2705–1.2753 | 1.2702–1.2748 | 1.1819–1.2155 | 11.81 → 7.97 µs |
| M32 | 1.1376–1.1426 | 1.1352–1.1425 | 1.0641–1.0715 | 9.57 → 7.97 µs |

## Documents

| File | What it holds |
|---|---|
| [`FINAL_REPORT.md`](FINAL_REPORT.md) | the disposition, the mechanism, every gate, every disclosed hazard |
| [`EXTERNAL_ACCEPTANCE.md`](EXTERNAL_ACCEPTANCE.md) | enable block, eight-GPU commands, acceptance criteria, rollback |
| [`evidence/attempt_ledger.json`](evidence/attempt_ledger.json) | both identities, the rejected M32 result, and what was not attempted with the evidence that closed it |

## Source

In the FlashMLA worktree, all new files — the concurrent round-3 main-kernel goal's
`kernel.cuh` and provider were never edited:

| File | Role |
|---|---|
| `csrc/glm52_hotspot/combine.cuh` | combine variants; upstream body plus macro-guarded batched path |
| `csrc/glm52_hotspot/combine.h` | launcher declaration for `api_combine.cpp` |
| `csrc/glm52_hotspot/v32_combine_identity.cu` | control: upstream algorithm, renamed only |
| `csrc/glm52_hotspot/v32_combine_c1_stage8.cu` | identity 1: eight-stage register-batched gather |
| `csrc/glm52_hotspot/v32_combine_c2_bucket_stages.cu` | identity 2 (**selected**): bucket-specialized depth |
| `csrc/glm52_hotspot/api_combine.cpp` | copy of `api.cpp`, 4 lines changed, calls the prefixed combine |

In the Kernel-Harness worktree:
`serving_native/candidates/flashmla_combine_decode_provider.py` — a separate
API-v1 provider so this campaign and the concurrent main-kernel campaign have
disjoint source sets and therefore disjoint build ids.

## Harness

Reused from round 2/3 with the provider path retargeted: `preflight.py`,
`cpu_audit.py`, `build_variant.py`, `validate_a0.py`, `validate_matrix.py`,
`measure_paired.py`, `trace_chain.py`, `probe_graph_only.py`, `summarize_nsys.py`
(now takes `--candidate-combine-symbol`, since this campaign replaces the second
kernel of the chain).

New for this campaign:

| Script | What it does |
|---|---|
| `measure_provider_pair.py` | directly paired provider-versus-provider CUDA Graph lanes — the P1-relative denominator, immune to the stock arm's workspace-placement term |
| `audit_combine_binaries.py` | proves the main SASS is unchanged P1, and that the identity control's combine SASS is the stock combine's instruction stream up to one dead assert-line immediate |
| `audit_sass_rebinding.py` | proves a re-measured variant's committed machine code is the measured machine code |
| `audit_timings_combine_r1.py` | recomputes every estimator from the raw ordered pairs, applies both denominators, reports per-bucket null bands, refuses a decision dataset spanning more than one physical GPU |

## Reproducing

```bash
# CPU only, no GPU lease
python harness/cpu_audit.py --output evidence/cpu_audit.json
python harness/build_variant.py --variant combine_c2_bucket_stages \
  --output evidence/build_combine_c2_bucket_stages.json

# every CUDA command goes through the shared scheduler; exit 75 means retry
W=/home/qinhaiyan/glm52-hotspot-goal-runs/with_hotspot_gpu.sh
$W -- bash harness/lease_r1_correctness.sh combine_c2_bucket_stages
$W -- bash harness/lease_r1_evidence.sh    combine_c2_bucket_stages
$W -- bash harness/lease_r1_final.sh          # the single-GPU decision dataset

python harness/audit_timings_combine_r1.py --evidence-dir evidence \
  --name-filter final_ --require-single-gpu --output evidence/timing_gate_audit.json
```

`lease_r1_final.sh` must stay one lease: it is the only dataset whose ratios are
mutually comparable, because the two B200s in this host differ about 7% on the
installed-stock arm at identical nominal clocks.

## Known limitations

- checkpoint-backed TP8/DP8/EP8 acceptance is not runnable here (four-GPU host);
- the shared root volume finished below the plan's 8 GiB floor because of
  concurrent goals — see [`evidence/disk_pressure_disclosure.json`](evidence/disk_pressure_disclosure.json);
- `combine_c1_stage8` is retained as evidence and is not promotable: it fails the
  P1-relative condition at M32.
