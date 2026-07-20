# Attempt DAG — GLM-5.2 o_proj_decode, 40% HBM extreme

```
A0  SEED (byte-for-byte from hbm35; candidate.py sha 054cf91a7cf5, scale_pack.cu sha c8106d4227a3)
│     Fused single-launch CUDA scale pack (1568x128) + packed-UE8M0 deep_gemm.fp8_gemm_nt(compiled_dims="nk")
│     AUTHORITATIVE 3-gate median: M=16 33.184us/38.10%  M=32 34.063us/37.29%   -> MISS both (ceilings 31.611/31.757)
│     correct diff 0, is_reference_fallback=false, compliance PASS.  PRESERVED pristine at variants/seed/.
│
├─ (measure) GEMM floor — deep_gemm with scales prepacked OUTSIDE timed window (non-accepted oracle)
│     M=16 30.720us/41.16%   M=32 31.184us/40.73%   -> clears 40% => the GEMM is NOT the blocker
│
├─ (measure) single-kernel floor — hand-written noop kernel (writes 1 int), same cold-L2 device-span timer
│     2.07us  => irreducible floor of ANY legal separate pack kernel (budget is only 0.573-0.891us)
│
├─ A1  SEED + PDL (deep_gemm.set_pdl(True)) — DEC-2 gap-removal lever
│     M=16 32.505us/38.90%   M=32 33.809us/37.57%   -> best achievable; recovers ~0.6us gap; STILL MISS both
│     Not baked into candidate (keeps preserved candidate byte-identical to the proven seed).
│
├─ A3  BEST-KNOB — DeepGEMM public-knob search. Round 1: isolated controls (compiled_dims x PDL at
│     defaults; num_sms, tc_util each at 'nk'). Round 2: the actual CROSS-PRODUCT for K-compiled
│     compiled_dims{nk,k,mk,mnk} x PDL{F,T} x num_sms{74,96,128,148} x tc_util{50,80,100} = 96 rows/
│     shape, + non-K compiled_dims measured at most-favorable knob (dominance: +2.36/+2.90us over
│     ceiling). 200 rows. ANY_CONFIG_CLEARS_40% = False.
│     Best (PDL + per-shape optima: M=16 mnk/nsms148/tc50, M=32 mk/nsms74/tc80).
│     Round 3: PROMOTED into the submitted candidate/ (knobs saved+restored in finally; packed bytes
│     byte-identical to seed) and re-gated via run.sh: M=16 32.849us/38.49%  M=32 33.304us/38.14%
│     -> best correct candidate, beats seed on both shapes, STILL MISS both. clean reference (~52.6/53.8us).
│     compiled_dims K-compile is the main lever (seed 'nk' near-optimal); PDL ~0.6us; num_sms
│     interacts with PDL at M=32 (best num_sms=74); tc_util negligible (memory-bound).
│
├─ A2  (reject) optimized broadcast CUDA pack — 1536 distinct values, uint4 coalesced broadcast, one-wave grid
│     byte-IDENTICAL to seed (losslessness ok), but pack_only 7.2-12.1us -> e2e M=16 38.1us / M=32 43.6us
│     WORSE than the seed: the seed's high-parallelism 1568x128 hides latency better. REJECTED.
│
└─ (control) Triton weight-pack — pack_only 2.66us  => same ~2us launch floor as CUDA; no separate-kernel
      approach (CUDA seed / CUDA broadcast / Triton) can beat the floor. Confirms the bound.

REJECTED-DIRECTION SUMMARY
| Attempt | Result | Reason dropped |
|---|---|---|
| A2 optimized broadcast pack | e2e 38.1 / 43.6 us | slower than seed; fewer warps in flight -> worse latency hiding |
| GEMM-side rewrite (t11) | not attempted | GEMM already memory-bound (DRAM 49-52%); no roofline case; BL-dont-handroll; and GEMM floor already <40% ceiling so a rewrite is unnecessary AND out of scope |
| reused scratch (DEC-1) | no-op | host-side alloc is outside the device-kernel span; does not change the gated metric |
| compact per-block weight-scale layout | closed | per-row expansion is API-forced (DeepGEMM SM100 TMA descriptor shape_mn=N) |

FINAL SELECTION: submitted candidate/ = A3 (best-knob: seed pack + per-shape public knobs, saved+
restored); pristine seed A0 preserved byte-identical at variants/seed/. Terminal disposition =
evidence-backed NO-GO on both shapes: GEMM_floor (30.72 / 31.18us) + irreducible single-kernel pack
floor (~2.07us) > 40% ceilings on both shapes; required residual 0.573-0.891us < 2.07us floor; and no
DeepGEMM public-knob combination clears 40% (200-row cross-product). Best correct candidate
32.849/33.304us still MISS. See docs/no_go_disposition.md.
```
