# Goal 24 W13 decode attempt ledger

No failed or superseded attempt is used as timing evidence for another arm.
The terminal performance failure is retained rather than averaged away.

## 1. Historical Goal 19 timings

- Hypothesis: inherit the clean historical BM32 speedups.
- Audit: Goal 19 used DeviceRuntime's default PDL=false and compared a
  v0.1.4 candidate overlay against the separately installed stock package.
- Outcome: rejected as a timing denominator.
- Why: stock and candidate were neither independent same-source builds nor
  explicitly equal at the production PDL state.
- Reuse retained: BM32 and cluster-N 2 were accepted only as a bounded
  hypothesis.

## 2. Exact same-source build

- Hypothesis: reconstruct stock and candidate from one immutable DeepGEMM
  source and expose the complete BM/BN/BK/stage/cluster tuple in the JIT key.
- Delta: tracked patch
  `third_party/deepgemm_w13/patches/0001-explicit-w13-config.patch`.
- Outcome: passed.
- Evidence: base commit `731e7c7a97d269e4b9f482ea18d0e709a948f293`,
  deterministic stock tree `917592ab...`, candidate tree `d38d8bf9...`,
  complete patch `997348b6...`, common normalized Ninja plan `9c3896ef...`,
  distinct DSOs `085ded6f...` and `d2bf4463...`.
- Why it remains valid: the materializer uses pinned git archives for
  DeepGEMM/CUTLASS/fmt, does not run `git submodule update` in archive trees,
  and reconstructs each tree byte-for-byte from tracked inputs.

## 3. Genuine BM32 one-SM

- Hypothesis: `(32,128,128,10,1)` removes cooperative-cluster overhead for
  tiny expert rows.
- Correctness: passed every adversarial and production bucket with exact
  numerical equality and exact `None` return semantics.
- Code generation: 33 registers, no spills, PTX
  `tcgen05.mma.cta_group::1`, plain one-CTA `UTCQMMA`, no `.2CTA`/`UCGABAR`.
- Fair eager leaf result: pooled 1.031906x and order-balanced 1.031817x, but
  series 2 AB was 1.026873x.
- Outcome: valid non-win; stopped before graph/region promotion.
- Why: every finite estimator in every independent series must reach 1.03x.

## 4. BM32 two-SM anchor

- Hypothesis: `(32,128,128,11,2)` preserves the mature cooperative tcgen05
  pipeline while reducing BM128 padded output work.
- Correctness: passed the same complete suite with exact equality and exact
  ABI/return behavior.
- Code generation: 36 registers, no spills,
  `tcgen05.mma.cta_group::2`, `UTCQMMA.2CTA`, cluster multicast and
  `UCGABAR`.
- Promoted lanes:
  - eager leaf: gate pass, order-balanced 1.045652x;
  - graph leaf: gate pass, order-balanced 1.043579x;
  - eager containing region: gate pass, order-balanced 1.038210x.
- Terminal lane: graph-containing region pooled 1.036620x and
  order-balanced 1.035978x, but series 2 BA was 1.028125x.
- Outcome: `no-replacement`.
- Why: a containing-region per-series estimator below 1.03x is an explicit
  terminal condition even when pooled summaries look positive.

## 5. Stock-versus-stock graph identity

- Hypothesis: establish the local graph-region noise floor with the same
  alternating protocol.
- Result: pooled 1.000000x, order-balanced 1.000138x, minimum estimator
  0.999447x.
- Outcome: passed as a control and forced to non-win.
- Why it matters: the candidate failure is a contract threshold miss, not an
  identity arm spuriously passing the promotion gate.

## 6. Graph-observation repair

- Problem: the old grouped-masked reference read device `masked_m` to the host
  after each wrapper call, which is illegal during graph capture.
- Delta: CPU-known masks, pre-poisoned full outputs, device-mask plus
  activation mutation, poison restoration, pointer/determinism checks and
  untouched-region validation.
- Outcome: passed eager and independent stock/candidate graph capture/replay.
- Why: the graph result now observes fresh work without a capture-time D2H
  synchronization or a packed-GEMM-only mutation shortcut.

## 7. CLC scheduler rewrite

- Hypothesis: a dynamic CLC schedule could improve a partially occupied final
  persistent wave.
- Current profile: both exact specializations launch 148 CTAs on 148 B200 SMs
  at one wave/SM.
- Outcome: abandoned without implementation.
- Why: there is no current tail-wave underfill evidence; the historical
  seven-wave geometry is not this exact compiled launch.

## 8. NCU capture

- First attempt: an NVTX include filter matched no kernel and produced
  retained failed logs.
- Correction: captured full+PM and source stock/candidate reports in one
  wrapper invocation on the same physical GPU.
- Outcome: complete.
- Finding: BM32 two-SM reduces NCU replay duration 136.58→128.32 us and DRAM
  writes 31.45→10.08 MB without register/spill regression; persistent wait and
  cluster-barrier PCs remain dominant.

## Stop discipline

The remaining expected-M 5/8/9 lane matrix, TP4 diagnostic and TP8 acceptance
were not run after the mandatory M16/expected-M4 graph-containing region
failed. Repeating or widening the sweep would violate the plan's terminal
rule. TP8 commands are retained for a future candidate; they cannot rescue
this one.
