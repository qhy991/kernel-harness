# Final report: external-acceptance-candidate

## Outcome

`p1_consumer_scale` is promoted to **external-acceptance-candidate** for both
M16 and M32. It is bitwise exact against installed stock, and it clears both
mandatory CUDA Graph lanes at both buckets by a margin more than four times the
instrument's own noise floor:

| Bucket | Graph containing DSA region | Graph leaf (main + stock combine) |
|---|---:|---:|
| M16 | **1.0694–1.0728** pass | **1.0716–1.0760** pass |
| M32 | **1.0636–1.0669** pass | **1.0628–1.0664** pass |

Each cell is the minimum and maximum over all four required estimators in three
independent alternating series of 100 pairs each, recomputed from the raw
ordered pairs rather than from the stored summaries.

The eager containing region is a **stock fallback** with zero provider launches
and bitwise-stock output, which is what round-3 policy requires of that lane in
place of round-2's unreachable speedup gate.

The production default remains **off**. The registration ceiling is **L2
external E2E**, pending checkpoint-backed TP8/DP8/EP8 acceptance that this
four-GPU host cannot run. This is not a `production-win` claim.

One new material candidate was built and rejected on its merits; the second
permitted identity was deliberately not spent. Both decisions are evidenced
below.

Target: SGLang API-v1 `flashmla_kv` FP8 sparse decode at bases kernel-harness
`c6a5802`, SGLang `d7fe89a71`, FlashMLA `b5af443` — prefixed V32 main plus the
unchanged stock BF16 combine, M16 Q `[16,1,64,576]` BF16 with KV
`[2049,64,1,656]` FP8 E4M3FN, M32 Q `[32,1,64,576]` with KV `[4097,64,1,656]`,
sparse top-k 2048, page size 64, V dim 512, scale 0.0625.

## 1. Why the disposition changed without the kernel changing

Round-2 measured the same candidate at essentially the same graph numbers and
still had to return `no-replacement`, because plan §6 enumerated an eager
containing lane that the API-v1 Python provider made arithmetically
unreachable — a ceiling of about 1.018 even with a zero-cost guard.

SGLang `d7fe89a71` then made the FlashMLA hotspot **graph-only**: outside CUDA
graph capture `try_dispatch_flashmla_sparse_decode` returns before the ABI
guard, so eager decode stays on stock. Round-3 policy follows that integration
and requires the eager containing lane to *fall back*, not to win.

So this round's job was to re-establish the candidate against the new
integration and then search for more. The re-establishment is not a formality,
and it is recorded as such: the round-2 build id
`a39236323dc57a97…160649` recomputes exactly from the clean committed tree, the
identity control reproduces 4128 static main instructions and P1 reproduces
3368, both matching round-2's record.

## 2. The graph-only contract, measured directly

Round-3 drops a timing gate and replaces it with a behavioural one, so that
behaviour gets direct evidence rather than an inference from timings
([`evidence/p1_consumer_scale_graph_only_contract_m16.json`](evidence/p1_consumer_scale_graph_only_contract_m16.json),
[`…_m32.json`](evidence/p1_consumer_scale_graph_only_contract_m32.json)):

| Probe | M16 | M32 |
|---|---:|---:|
| eager containing, provider launches | **0** | **0** |
| eager containing output equals stock bitwise | yes | yes |
| graph capture, provider launches | 1 | 1 |
| graph replay, host launches | 0 | 0 |
| eager with `SGLANG_GLM52_FLASHMLA_GRAPH_ONLY=0` | 1 | 1 |

The counter increments in the provider's host code, which runs at capture and
not at replay — replay re-executes captured device nodes without re-entering
host code. Selection under graph is therefore established at capture time, and
the captured node identity is established separately by Nsys in §5. An earlier
version of this probe asserted a non-zero counter on replay and was wrong; it
was corrected rather than reinterpreted.

The last row matters for §4: because the production default would make an eager
containing correctness comparison silently degrade into stock-versus-stock, all
correctness runs force selection with `GRAPH_ONLY=0` so the kernel is validated
everywhere it can execute.

**Honest cost of the fallback.** The eager containing lane measures 0.856–0.871
(M16) and 0.904–0.920 (M32) with **zero provider launches**. That is not a
kernel regression — the candidate never runs. It is the residual host cost of
the graph-only early return: **4.74 µs** (M16) and **3.18 µs** (M32) per call
for the enable check, phase inference, cached spec lookup and capture probe
before returning `None`. Round-3 stops gating this lane, so this is reported to
the integration owner rather than buried.

## 3. R3-A: the one new candidate, and why it lost

After P1 removed the scattered scale round trip from in front of the gather
issue, the only producer work still on the TMA-issue critical path is the
coordinate load itself. Both gather warps run

```
wait(bar_valid_coord_scale_ready)   // acquire: coordinates now visible
wait(bar_raw_free | bar_qk_done)    // guards the *destination* buffer
LDS.128 tma_coord[0]                // first coordinate quad
...address math...
UTMALDG.2D.GATHER4                  // first gather issue
```

The second wait guards a different resource than the coordinates, so the load is
legal to issue between the two waits, where its latency overlaps the wait.
ptxas cannot do this itself: hoisting a shared load above an acquire mbarrier
`try_wait` spin loop is not a legal reordering. The current binary confirmed the
opportunity was unexploited before any timing was run.

The predeclared effect was recorded before measurement and **was delivered**:

| | P1 | R3-A |
|---|---|---|
| raw-NoPE warp | `TRYWAIT` / `TRYWAIT` / `LDS.128` | `TRYWAIT` / **`LDS.128`** / `TRYWAIT` |
| RoPE warp | `TRYWAIT` / `TRYWAIT` / `LDS.128` | `TRYWAIT` / **`LDS.128`** / `TRYWAIT` |
| static main instructions | 3368 | 3368 |
| registers / barriers / spill | 168 / 16 / 0 | 168 / 16 / 0 |
| LDG, STS, UTMALDG, F2FP | 14, 115, 57, 272 | 14, 115, 57, 272 (identical) |

R3-A is bitwise exact at all 10 boundaries per bucket plus 17 adversarial cases
per bucket, and it clears both graph lanes against stock (M16 1.0654–1.0756,
M32 1.0604–1.0673). But against **P1** it measures 1.0031, 0.9983, 0.9948 and
0.9950 — a maximum deviation of 0.52% against an instrument null half-width of
1.42%. It is indistinguishable from P1.

The mechanism was delivered and is simply not worth anything. The latency it
removes is a shared-memory load of roughly thirty cycles; P1's six percent came
from removing a scattered *global* round trip about twenty times longer. Under a
stall profile dominated by long-scoreboard (33.2%) and barrier (23.2%) waits on
global and TMA traffic, hiding a shared load inside a barrier wait does not move
the kernel.

R3-A is therefore promotable on the letter of the policy but **not selected**:
it would add a source delta against frozen upstream for no measured gain. P1
remains the candidate. The source and its evidence are retained as a negative
result, and the rollback is recorded in the ledger.

## 4. Correctness

Both variants are **bitwise exact** against installed stock (`max_abs = 0.0`) at
every required boundary for both buckets: leaf eager, non-default stream,
independently captured leaf graphs before and after Q/index mutation,
deterministic repeated replay with poisoned outputs, containing
`_forward_flashmla_kv` in eager and graph, output/LSE value-shape-stride-
ownership checks, and input immutability. Dispatch records 3 hits with zero
fallback after selection and zero launches on the wrong-page rejection.

Each bucket additionally passes a 17-case adversarial matrix covering random,
zero, signed ramp, extreme finite and exponent-boundary values, and duplicate,
interleaved, sorted, reverse, boundary-page, `-1` and beyond-valid-length
indices.

Correctness runs force `SGLANG_GLM52_FLASHMLA_GRAPH_ONLY=0` so the eager
containing boundary genuinely executes the candidate. Under the production
default that boundary is a stock fallback, which §2 verifies separately.

## 5. Device chain

Nsys confirms exactly one prefixed main followed by one unchanged stock combine
in eager and in each of five graph replays, with no other device kernel in the
marked ranges, and the candidate graph differing from stock at the main symbol
only. Median device spans (ns):

| Bucket | lane | stock main | candidate main | main-only ratio |
|---|---|---:|---:|---:|
| M16 | graph | 18,880 | 16,832 | 1.122 |
| M32 | graph | 26,880 | 24,224 | 1.110 |

Combine is unchanged, as required, and its extracted SASS is byte-identical
across the identity control, P1 and R3-A.

## 6. The instrument, reported as measured

Every conclusion here is a ratio compared against 1.03, and §3 additionally
claims an *absence* of difference, so the same-binary null bounds what may be
concluded. Measured this round with the identical installed stock binary in both
arms:

| Null | spread | half-width |
|---|---|---:|
| M16 | 0.9974–1.0038 | ±0.38% |
| M32 | 0.9907–1.0142 | ±1.42% |

Round-2 validated K=20 at ±0.28% on M16 only. This round's M16 is comparable,
but **M32 is materially looser**, driven by two of six series; the other four sit
within ±0.4%. The worst case is used as the resolution bound rather than
carrying round-2's tighter figure forward.

Consequences, stated plainly: P1's graph margins of 6.3–7.6% exceed the
worst-case null by more than four times, so the pass is robust. R3-A's ≤0.52%
differences from P1 are below it, so they support only "no measurable
difference" — not a claim that R3-A is equal, and not a claim that it is worse.

## 7. Provenance

Adding R3-A's macro-gated branches to `kernel.cuh` changed the provider's build
id for **every** variant, because that id hashes `kernel.cuh`. The P1 actually
measured is therefore the rebuild `38e3b1a7a5d5bf00`, not the original
`a39236323dc57a97`.

This is disclosed and then closed rather than merely noted: the two generations
produce **byte-identical generated main SASS**, verified instruction by
instruction, so the macro-gated branches are provably inert when the macro is
undefined and the measurements bind to the same machine code as the committed
round-2 candidate. `binary_manifest.json` now enforces that equality for every
variant instead of selecting one binary and assuming.

## 8. Why the second identity was not spent

The plan permits at most two new identities. One was used. The second was not,
because all three of its bounded open-search items are closed by measurement or
by the frozen contract, and the shared contract forbids rerunning a mechanism
the evidence rejects merely to fill a budget:

1. **Scoreboard/barrier overlap beyond P1** — closed by R3-A. The coordinate
   load was the last producer work in front of the TMA issue; moving it produced
   the predeclared SASS motion and measured 1.00 within the null band.
2. **Tail / partition imbalance** — closed by the frozen contract.
   `tile_scheduler_metadata` and `num_splits` are caller-supplied inputs
   computed host-side. Round-2's capture measured average SM active at 68.4% of
   elapsed with a 6,892–34,531 cycle spread, which is tail imbalance across the
   148 fixed partitions, but the kernel only reads that schedule; repartitioning
   needs a host metadata rewrite that this plan's item 2 excludes.
3. **Narrow inline PTX for one predeclared effect** — closed by measurement. Of
   the three effects allowed, earlier TMA issue is delivered by P1 and now
   bounded by R3-A; further shortening of the FP8-to-BF16 scale dependency chain
   is bounded at ~1.00 by round-2's ablation, which deleted the entire chain and
   measured 0.997–1.008; and register live range has no headroom, with
   registers/barriers/stack/local/spill at 168/16/0/0/0 across every variant.

Previously closed and not reopened: the P2 bank-conflict remap (shared-memory
stalls are 10.7% against 56.3% for long-scoreboard plus barrier, and round-1's
B2 already lost to its own address arithmetic), the two-SM rewrite (one CTA
already fills all 148 SMs in a single wave, and 816 shared bytes are free
against the 106,496 a third KV buffer needs), and instruction-count-only edits.

No NCU capture was taken this round. Round-2's single capture already answered
the stated critical-path question, and R3-A's outcome was settled by the
generated binary and the paired timings. This is a deliberate non-invocation,
not missing evidence.

## 9. Repository validation

All 10 SGLang GLM-5.2 hotspot registry tests pass, all campaign Python compiles,
all 28 retained JSON artifacts parse, and `git diff --check` is clean in all
three worktrees. The SGLang worktree is unmodified by this campaign.

The two broader base checks that round-1 and round-2 disclosed remain nonzero
for inherited contracts outside this campaign — `serving_native/selftest.py`
expects a marker in a mandated, unmodified SGLang base file, and
`testbench/bin/verify_harness.py` reports pre-existing stale generated
projections. Neither touches the provider, dispatch guard, FlashMLA source,
generated binary, correctness harness or measurements, and this goal forbids
regenerating them.

## 10. Limitations and what acceptance still requires

This host has four B200s, so checkpoint-backed TP8/DP8/EP8 acceptance is not
locally runnable, and that is the only reason this is not a production win. The
authorized external procedure is [`EXTERNAL_ACCEPTANCE.md`](EXTERNAL_ACCEPTANCE.md),
which supersedes round-2's refusal now that every local gate passes.

Scale expectation, so the external result is not misread: the local DSA share of
full-server short-decode GPU kernel time is about 5.1%, so a ~6.5% kernel win
bounds the achievable end-to-end effect at well under 1%. A whole-model result
of 1.00x is a legitimate outcome and must be recorded as such.

Two further caveats belong with the candidate:

- The M32 instrument null was looser this round (±1.42%) than round-2's M16
  figure. The margins clear it comfortably, but a re-measurement on the
  acceptance host should re-validate the null before trusting a tighter bound.
- Graph-only selection leaves a 3–5 µs per-call host cost on eager decode steps
  even though they run stock. That is an integration property, and it is the
  measured item to hand back to the integration owner.

Production default remains off, stock remains the fallback for every bucket, and
no remote state was modified.
