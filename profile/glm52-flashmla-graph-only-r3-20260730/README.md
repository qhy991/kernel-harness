# GLM-5.2 FlashMLA sparse decode — round 3 (graph-only gates)

Plan: `/home/qinhaiyan/glm52-hotspot-goal-runs/tasks/flashmla_ptx_graph_only_r3/plan.md`

Disposition: **external-acceptance-candidate** (`p1_consumer_scale`, M16 and M32).
Production default off; ceiling L2 external E2E pending TP8/DP8/EP8 acceptance.

Read [`FINAL_REPORT.md`](FINAL_REPORT.md) first.

## Layout

| Path | Contents |
|---|---|
| `FINAL_REPORT.md` | disposition, gates, negative results, limitations |
| `evidence/attempt_ledger.json` | every attempt, closed hypothesis, and rollback |
| `evidence/timing_gate_audit_r3.json` | all estimators recomputed from raw pairs |
| `evidence/binary_manifest.json` | resources, opcodes, combine invariance |
| `evidence/disassembly/` | full SASS for identity, P1, P1 rebuild, R3-A |
| `harness/` | the scripts that produced everything above |

## Reproducing

CPU-only audits and builds hold no GPU:

```bash
harness/cpu_audit.py     --output evidence/cpu_audit.json
harness/build_variant.py --variant p1_consumer_scale --output evidence/build_p1.json
harness/audit_binaries.py --extensions-dir "$TORCH_EXTENSIONS_DIR" --output evidence/binary_manifest.json
harness/audit_timings_r3.py --evidence-dir evidence --output evidence/timing_gate_audit_r3.json
```

Every CUDA-initializing command goes through the shared scheduler; exit 75 means
retry later rather than bypass the lock:

```bash
../../../../with_hotspot_gpu.sh -- bash harness/lease_correctness.sh p1_consumer_scale --with-preflight
../../../../with_hotspot_gpu.sh -- bash harness/lease_graph_gate.sh   p1_consumer_scale
../../../../with_hotspot_gpu.sh -- bash harness/lease_evidence.sh     p1_consumer_scale
../../../../with_hotspot_gpu.sh -- bash harness/lease_null_control.sh
```

One lease holds one complete set of alternating AB/BA series so paired
measurements stay on the same physical GPU.

## Round-3-specific notes

- `lease_correctness.sh` sets `SGLANG_GLM52_FLASHMLA_GRAPH_ONLY=0` on purpose.
  Under the production default the eager containing region falls back to stock,
  which would silently turn that correctness comparison into stock-versus-stock.
  Performance runs use the production default, where the fallback is the
  contract being verified.
- `audit_timings_r3.py` is a new script rather than an edit of round-2's
  `audit_timings.py`, which is kept unmodified as the record of what round 2
  did. Round 2 gated four lanes and asserted `no-replacement`; neither applies
  here.
- The provider build id hashes `kernel.cuh`, so adding a macro-gated branch
  rebuilds every earlier variant under a new id. `audit_binaries.py` proves all
  generations of a variant emit identical main SASS instead of picking one.
