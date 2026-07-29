# GLM-5.2 serving-native harness

This is an isolated replacement-oriented suite for the verified B200
TP8/DP8/DeepEP balanced deployment plus a separate four-GPU diagnostic lane.
It leaves the frozen 24-task synthetic suite under `testbench/tasks/glm52/`
unchanged.

The important difference is the reference contract: every task calls the
production SGLang or DeepEP symbol with the production dtype/layout. FP8 GEMMs
use packed int32 UE8M0 scales; DSA uses FlashInfer TRT-LLM; the indexer uses
`wq_b` and fused `wk_weights_proj`; MoE exposes the real fused W13, SwiGLU+quant,
and W2 stages; and communication includes both DP AllGather and both DeepEP modes.

Schema V2 makes execution mode and promotion evidence explicit. An audited
candidate comparison always contains at least three independent same-process
series, each alternating AB/BA after warmup. In every series, the pooled,
AB-paired, BA-paired, and order-balanced speedup estimates must all reach
`1.03x`; a favorable aggregate median is insufficient, and the bundled
identity control is always a non-win.

## Fixed deployment shapes

| Phase | Fixed local shape | Reason |
|---|---:|---|
| decode | `M=16` and `M=32` on every DP rank | observed production CUDA-graph buckets; DP8 does not divide M |
| prefill | `M=4096` per DP rank | 32768-token balanced chunk split across DP8 |
| O-projection smoke | decode M16/M32 eager + graph; prefill M4096 eager | exact packed-FP8 compute contract consumed by V2 goals |
| DeepEP buffer | max dispatch 128 per rank | current SGLang production default |
| DeepEP-LL W13 | E=32, slab=1024, expected M=4/5/8/9 | four independently named EP8 decode points at M16/M32 |
| topology | 8 ranks | official single-node B200 TP8/DP8/EP8 lane |
| diagnostic topology | 4 ranks | separate TP4/DP4/EP4 lane; not equivalent to TP8/EP8 |
| model | hidden 6144, 256 experts, top-k 8 | GLM-5.2 FP8 |

These are deliberate test shapes, not a claim that every runtime step has the
same M. Add another named workload when a different serving lane is needed.

## Commands

```bash
cd /home/qinhaiyan/Kernel-Harness

serving_native/run.sh --list
serving_native/run.sh --describe dp_allgather_decode_m16

# One-GPU production ABI
serving_native/run.sh linear_indexer_wq_b_decode_m16 \
  --candidate serving_native/candidates/reference.py

# Exact O-projection V2 controls (three series are the default)
serving_native/run.sh linear_attn_o_prefill_m4096 \
  --execution-mode eager --output /tmp/o-prefill-eager.json
serving_native/run.sh linear_attn_o_decode_m16 \
  --execution-mode cuda_graph --output /tmp/o-decode-m16-graph.json
python3 serving_native/audit_result.py /tmp/o-decode-m16-graph.json

# Eight-GPU SGLang GroupCoordinator AllGather
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  serving_native/run.sh dp_allgather_decode_m16 \
  --candidate serving_native/candidates/allgather_torch.py

# Eight-GPU DeepEP normal config tuning
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  serving_native/run.sh deepep_normal_dispatch_prefill \
  --candidate serving_native/candidates/deepep_config.py

# Four-GPU SGLang AllReduce, independently at M16 and M32
CUDA_VISIBLE_DEVICES=0,1,2,3 \
  serving_native/run.sh tp4_allreduce_decode_m16 \
  --candidate serving_native/candidates/allreduce_torch.py
CUDA_VISIBLE_DEVICES=0,1,2,3 \
  serving_native/run.sh tp4_allreduce_decode_m32 \
  --candidate serving_native/candidates/allreduce_torch.py

# Four-GPU DeepEP low-latency dispatch/combine
CUDA_VISIBLE_DEVICES=0,1,2,3 \
  serving_native/run.sh ep4_deepep_ll_dispatch_decode_m16
CUDA_VISIBLE_DEVICES=0,1,2,3 \
  serving_native/run.sh ep4_deepep_ll_combine_decode_m16
```

Candidate comparisons run three independent series by default. Every series
alternates reference/candidate order on every pair, with successive series
starting in AB, BA, AB order. The gate independently recomputes the pooled,
AB-paired, BA-paired, and geometric order-balanced estimates from raw samples
and requires all four in every series to be finite and at least `1.03x`.
Distributed samples use the maximum CUDA-event latency across ranks. Omitting
`--candidate` selects the explicit identity control. The target environment
must provide the same `deep_ep` package as the SGLang image; point
`KERNEL_HARNESS_PYTHON` at that environment when needed.

For the required one-lease smoke matrix:

```bash
/home/qinhaiyan/glm52-goal-runs/with_flexible_gpu.sh -- \
  serving_native/run_v2_identity_smoke.sh /tmp/serving-native-v2-smoke
```

The script keeps all five workload/mode controls and their in-process profiler
or graph-node captures on one physical B200. An exit status of 75 is scheduler
back-pressure; retry later without bypassing the wrapper.

Execution modes are workload allowlists, not an unchecked global switch.
`linear_attn_o_decode_m16/m32` allow eager and `cuda_graph`; the exact
`linear_attn_o_prefill_m4096` workload is eager-only. The four exact W13 leaf
workloads and their four W13→SwiGLU/packed-quant→W2 containing regions also
allow eager and `cuda_graph`; other workloads remain eager until their own
graph semantics are implemented and tested.

The W13 points are explicit rather than formula-derived:
`moe_w13_grouped_decode_m16_em4/em5`,
`moe_w13_grouped_decode_m32_em8/em9`, with matching
`moe_w13_region_*` names. Set
`SGLANG_GLM52_W13_DECODE_MANIFEST` to the schema-2 same-source build manifest
before importing a W13 candidate. The stock and candidate DeepGEMM modules
then load side by side from distinct package, DSO, and JIT-cache paths.

Before any W13 timing, run the leased correctness/ABI matrix. It validates
both bounded variants against same-source stock over production expected-M
4/5/8/9 plus zero, minimum, maximum, skewed, and BM32-boundary masks; random,
ramp, extreme-finite, changed, and invalid-row-poison data; a non-default
stream; exact `None` returns; packed-scale byte hashes; output ownership; and
pre-poisoned rows outside each scheduled full `store_block_m` tile. The
artifact separately reports writes to invalid padding inside a scheduled tile:

```bash
/home/qinhaiyan/glm52-goal-runs/with_flexible_gpu.sh -- \
  /path/to/task/python serving_native/validate_w13.py \
  --manifest /path/to/w13_variants/manifest.json \
  --variant both --output /path/to/evidence/correctness.json
```

Trace the same four points through the real production wrapper, including the
private eager/graph marker, one low-level candidate call per selected wrapper
call, marker reset, exact `None`, marker-free stock fallback, and
candidate-error propagation without a stock retry:

```bash
/home/qinhaiyan/glm52-goal-runs/with_flexible_gpu.sh -- \
  /path/to/task/python serving_native/validate_w13_production.py \
  --manifest /path/to/w13_variants/manifest.json \
  --variant bm32_1sm --output /path/to/evidence/production_trace_1sm.json
```

The runner pins the reference process to `SGLANG_GLM52_OPT=0` after consuming
any worker side-channel env file, so a stale deployment configuration cannot
silently replace the baseline.

## Candidate contract

A candidate exports:

```python
def run(inputs, runtime):
    return runtime.reference(inputs)
```

Candidate import/JIT setup must finish at import time or during the runner's
pre-measurement warmup. Optional `ARTIFACT_PATHS` lists compiled shared objects
whose exact paths and hashes must be bound into the result. Ordinary candidate
code cannot exempt a call to `runtime.reference`: every such delegation is a
fallback and cannot claim a win.

DeepEP normal-mode Config tuning uses the runner-owned declarative API instead:

```python
CANDIDATE_API = "reference_with_config_v1"
CANDIDATE_CONFIG = {
    "num_sms": 24,
    # remaining deep_ep.Config fields...
}
```

Such a module must not export `run()`. The runner validates the API/workload
pair and performs the production call itself; the auditor closes its trusted
delegation counts separately. The API is accepted only for DeepEP normal
dispatch/combine workloads.

`runtime.reference(inputs, config=...)` keeps the exact production call while
allowing a DeepEP `Config` dictionary to be tuned. A replacement collective or
kernel may instead return its own tensor/tree. Correctness is checked before
and after timing and again on fresh inputs.

## CUDA Graph contract

Reference and candidate graphs are captured independently on non-default
streams after eager JIT warmup. Every graph series captures both R→C and C→R
orders and round-robins the two replicas, carrying forward the corrected
capture-order control from the goal-16 campaign.

Before graph timing, the runner mutates captured input storage in place,
poisons captured output storage before replay, verifies deterministic repeated
replay, restores the original inputs, checks approved numeric tolerance, and
requires stable input/output pointers. CUDA graph nodes are enumerated through
the driver API; host, memcpy, memset, allocation, and free nodes fail closed.
For W13, the runner additionally mutates the device-resident `masked_m`, uses
only CPU-known mask metadata fixed before capture, pre-poisons the complete W13
and W2 output slabs, and proves that rows outside each replay mask remain
poisoned. No device-mask read or `.tolist()` occurs inside the captured region.

## Schema-v2 audit

Each result records raw samples in execution order, unique series/capture
identities, the exact canonical `WORKLOADS` entry and source hashes, candidate artifact hashes,
actual Python/shared-object import paths, repository SHAs/status, physical GPU
UUID, clocks, driver/CUDA versions, kernel identities or graph nodes, and
candidate hit/fallback accounting. Cache/import/artifact snapshots surround
every capture and timed series; any late activity is treated as JIT during a
forbidden phase.

Run the standalone auditor with:

```bash
python3 serving_native/audit_result.py RESULT.json
```

It recomputes graph node counts, type counts, kernel identities, forbidden
nodes, and non-default-stream truth from raw nodes and IDs. It also binds every
timed graph sample to the expected independent round-robin capture and requires
requested/completed series, repeats, raw samples, call totals, and per-phase
counts to close exactly. Missing provenance/correctness, wrong hashes, workload
drift, zero hits, silent fallback, execution-mode drift, late JIT activity, and
graph semantic violations all fail closed. Structural tests include complete
valid eager/graph fixtures and adversarial mutations for each rejection class.

Treat any pooled, AB-paired, BA-paired, or order-balanced estimate below 3% in
any required series as a failure. A candidate is not serving-ready until the
complete SGLang request workload and overlap region also improve.

Promotion is evaluated independently for each `operator x M` bucket. A win at
M16 may be deployed only for M16 while M32 keeps the stock SGLang path; both
buckets do not need to win. Configure that policy in SGLang with, for example,
`SGLANG_GLM52_OPT_OPS=q_b_proj` and
`SGLANG_GLM52_OPT_M_BUCKETS=q_b_proj:16`.

## Superseded synthetic tasks

- `index_k` + `index_weights` are replaced by `indexer_wk_weights_decode_m16/m32`.
- indexer Q is `linear_indexer_wq_b_decode_m16/m32`, separate from attention Q-B.
- separate `moe_gate`/`moe_up` assumptions are replaced by
  M16/M32 variants of fused W13, SwiGLU+quant, and W2.
- `flash_mla_sparse_fwd` is replaced by `dsa_trtllm_decode_m16/m32`.
- communication is represented explicitly by M16/M32 AllGather and DeepEP tasks.
- the four-GPU diagnostic lane adds M16/M32 AllReduce, AllGather, and DeepEP
  tasks with `world_size=4`; EP4 has 64 local experts per rank.

Run the GPU-free structural check with:

```bash
python3 serving_native/selftest.py
python3 -m unittest serving_native.test_contract_v2
/path/to/torch/python -m unittest serving_native.test_w13_graph_contract
python3 testbench/bin/verify_harness.py --skip-task-projection
```

The omitted generated-task projection needs the configured venv to build its
GPU-derived tensor tables. The one-lease smoke script runs that exact
`sync_glm52_tasks.py --check` gate inside the scheduler wrapper; frozen task
files are never rewritten.
