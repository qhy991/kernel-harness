# Run Log — glm52/index_score_decode HBM >=82%

## Round 0 (resumed after mid-task2 API disconnect)

### task1 — baselines (COMPLETE, AC-1)
- Seed hash verified against prompt.md.
- 3 idle runs (GPU-3, Harness `7d79e5e`): M16 32.80us/54.3%, M32 58.16us/61.3%,
  calc_diff 0.0, timing stable. Raw: `work/baseline_trial{1,2,3}.{json,log}`.
- Reconciliation: user 73% ≈ in-kernel/warm metric; frozen cold-L2 gate = 54/61%.
  See `docs/results.md` AC-1.

### task2 — NCU / roofline / span floor (COMPLETE, AC-3)
- `work/ncu_driver.py` isolates the stock paged-MQA kernel (warmup outside the
  profiler range; only paged-MQA launches profiled).
- Full reports: `work/ncu_m16_full.ncu-rep`, `work/ncu_m32_full.ncu-rep`;
  section dumps `work/ncu_m{16,32}_details.txt`.
- Actual vs requested DRAM: M16 144.2/142.67 MB (1.011x), M32 286.3/285.35 MB
  (1.003x) — bytes optimal, no coalescing waste.
- In-kernel DRAM throughput ceiling: M16 64.3%, M32 75.1%.
- Occupancy hard-capped 18.75% (1 block/SM) by regs(168)+smem(220.67KB)+
  `__launch_bounds__(384,1)`; KV pipeline already 5 stages (smem-capped).
- Span floor: `work/time_probe.py` warm/cold-L2 event timing on idle GPU-1
  (M16 warm 31.73us/cold 37.89us; M32 warm 52.34us/cold 64.51us).
- Roofline: even kernel@100% realistic peak + span → M16 80.5%, M32 78.3% < 82%.
  Passing needs 98–101.5% of spec peak span-inclusive → physically impossible.

### task3 — independent adversarial review (COMPLETE, AC-1/AC-3)
- Codex gpt-5.5:xhigh, 149s, transcript `.humanize/skill/2026-07-20_01-25-48-*/output.md`.
- **CONFIRMED NO-GO.** Codex read the fork source and independently ruled out every
  mechanism (KV stages fixed at 5 / smem-capped; 2-CTA/SM variant has no defensible
  number; no TMA multicast reuse since tokens_per_request=1; grid already 1 CTA/SM
  over 148 SMs; bytes already ~1.00–1.01x). Refinement adopted: M32 is the strict
  absolute-roofline proof (>100% spec peak with span); M16 the weakest efficiency.

### task8 — finalize reviewed no-go (COMPLETE, AC-6)
- `docs/results.md`, `docs/attempt_dag.md` written; fastest correct candidate
  (stock, calc_diff 0) preserved in `candidate/candidate.py`.
- Fresh idle GPU-1 gate reproduction saved: `work/gpu1_gate_confirm.json`
  (M16 53.9%, M32 61.2%, PASS correctness).

### Outcome
**Reviewed NO-GO.** 82% span-inclusive HBM on both M=16 and M=32 is above the
physical ceiling of this frozen single-kernel memory path. Stock reference stays
frozen; DeepGEMM-GLM52 fork untouched (no source edit had a credible pass path).

### Provenance
- DeepGEMM-GLM52 fork present at `/home/qinhaiyan/DeepGEMM-GLM52` (loaded as
  `deep_gemm_experimental`); prior campaign overlays under `overlays/`. Stock
  Harness `deep_gemm` untouched and frozen.
