# Results — glm52 / dsa_attn_decode >40% HBM

## VERDICT: NO-GO-CONFIRMED (reviewed, evidence-backed)

The joint >40% HBM target (M16 **and** M32) is a **physical no-go**. **M16 is the
blocking shape**: the irreducible gather-read floor of any correct sparse-MLA kernel
(12.42 us, reading the per-query topk rows) is essentially equal to the strict-40%
ceiling (12.53 us), leaving no slack for QK/softmax/AV compute or the cross-split
reduction. Full derivation in `docs/floor_evidence.md`. Kept candidate = stock seed
(correct, calc_diff 0); no faster CORRECT candidate exists or is physically reachable.

### Reachability floor table (gate protocol: CUPTI device-span, cold-L2 253 MB flush)

| shape | strict-40% ceiling | realistic gather floor (per-query topk) | distinct-once floor | headroom for compute+reduce | verdict |
|---|---:|---:|---:|---|---|
| **M16** | 12.534 us | **12.42 us / 40.4%** | 10.88 us / 46.1% | ≈0 us (floor ≈ ceiling) | **NO-GO (blocking)** |
| **M32** | 25.068 us | 20.94 us / 47.9% | 15.58 us / 64.3% | ~4 us | reachable in principle |

- Gather floor = pure read of the required KV rows at max memory parallelism; no
  compute, no combine, no O write. It is a *lower bound* on any correct kernel.
- M16's only sub-ceiling read (10.88 us, distinct-once) needs a pre-deduplicated row
  set a per-query kernel cannot form without first materializing a gather.
- A cross-split FP32 partial round-trip busts the byte budget (M16 +17 MB at S=4
  alone), so the reduction cannot be done via DRAM partials — confirming no viable
  Python/2-kernel path either.

Denominator 8.0 TB/s. Strict target: M16 median `<12.533760 us`, M32 `<25.067520 us`
(equality misses). Stretch: >=45% both.

## Baseline (frozen reference == candidate seed)

| M | median us | HBM util | vs strict-40% ceiling | speedup needed |
|---:|---:|---:|---|---:|
| 16 | 45.927 | 10.92% | 45.927 > 12.534 | ~3.66x |
| 32 | 47.151 | 21.27% | 47.151 > 25.068 | ~1.88x |

Correctness: PASS both shapes, calc_diff 0. Seed sha256 confirmed.

## Best correct candidate so far

Seed (stock `flash_mla_sparse_fwd`) — correct (calc_diff 0), HBM 10.92% / 21.27%.
It is the active `candidate/candidate.py` (restored; sha256 9960b326…). No faster
CORRECT candidate yet.

## Attempts

| Attempt | Mechanism | Correct? | M16 | M32 | Outcome |
|---|---|:--:|---|---|---|
| A0 | stock sparse_prefill_fwd (baseline) | yes (calc_diff 0) | 45.927us / 10.92% | 47.151us / 21.27% | ROOT, kept (active candidate) |
| A1/P1 | multi-stream Python split-KV + lse merge | **no** | — | — | REJECTED: elementwise layer fails (35,105 elems, bf16-partial precision); see BL-20260719-bf16-partial-split-fails-elementwise |
| A2 | split-heads reshape (fold 64-head axis into query axis, free occupancy) | **no** | — | — | BLOCKED: stock head64 kernel dispatches only h_q∈{64,128} (tcgen05 MMA atom B_H=64); cannot form smaller head-groups |

P1 details: aggregate calc_diff 3.77e-6 (< 5e-6) but elementwise `abs OR rel`
layer fails on small outputs (max_abs_err 1.95e-3 vs abs_tol 2.85e-5; max_rel_err
105x). Stock kernel returns BF16 partials → merging loses precision the single-pass
reference keeps in FP32. Also, eager multi-stream span is host-gap-dominated. Both
failure modes require a single-launch in-kernel FP32 reduction (A1/P2 fork). Kept as
`attempts/A1_P1_multistream_split.py` (not the active candidate).

## Profiler evidence (Round 0)

Stock main kernel `sm100::fwd::head64::sparse_attn_fwd_kernel`, one launch/call:

| M | Duration us | Grid | Waves/SM | DRAM % | SM % | Occ % |
|---:|---:|---:|---:|---:|---:|---:|
| 16 | 44.06 | 16 | 0.11 | 15.20 | 6.26 | 17.67 |
| 32 | 42.78 | 32 | 0.22 | 15.52 | 13.18 | 17.78 |

Root cause: **grid == M** → occupancy wall (~89% of SMs idle). Not bandwidth-bound;
under-parallelized. Raw: `docs/ncu/ncu_m16.ncu-rep`, `ncu_m32.ncu-rep`, `nsys_m16.nsys-rep`.

## Reachability (FINAL — NO-GO-CONFIRMED)

Superseded the preliminary "plausibly clears 40%" note. The occupancy-fill intuition
(grid==M → fill to 148 CTAs via split-KV) is real for M32 but does **not** rescue M16:
raising occupancy cannot beat a pure-read lower bound (12.42 us) that is already ≈ the
strict-40% ceiling (12.53 us). Every occupancy mechanism was chased to its floor:
split-KV (A1/P1) fails on bf16-partial precision + partial-traffic budget; split-heads
(A2) is blocked by the fixed B_H=64 MMA atom; the only design that avoids partial
traffic (in-kernel cluster/DSM reduction fork) could plausibly clear M32 but still
cannot beat M16's gather floor. Since the target demands **both** shapes, **M16 is the
decisive no-go**. See `docs/floor_evidence.md` for the full lower-bound derivation.
