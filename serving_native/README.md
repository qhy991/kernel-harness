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

## Fixed deployment shapes

| Phase | Fixed local shape | Reason |
|---|---:|---|
| decode | `M=16` and `M=32` on every DP rank | observed production CUDA-graph buckets; DP8 does not divide M |
| prefill | `M=4096` per DP rank | 32768-token balanced chunk split across DP8 |
| DeepEP buffer | max dispatch 128 per rank | current SGLang production default |
| DeepEP-LL MoE | E=32, slab=1024, expected M=4/8 | EP8 packed receive layout at decode M=16/32 |
| topology | 8 ranks | official single-node B200 TP8/DP8/EP8 lane |
| diagnostic topology | 4 ranks | separate TP4/DP4/EP4 lane; not equivalent to TP8/EP8 |
| model | hidden 6144, 256 experts, top-k 8 | GLM-5.2 FP8 |

These are deliberate test shapes, not a claim that every runtime step has the
same M. Add another named workload when a different serving lane is needed.

The DSA decode workload defaults to a logical KV length of 8192 only for backward
compatibility. It is not a production context distribution. Override it with
`--kv-context N`, or pass exactly M comma-separated lengths with `--kv-contexts`
for a ragged M16/M32 batch. The allocation is paged and each request keeps its own
logical `seq_lens`; the result records the resolved vector under
`kv_context_coverage`.

## Commands

```bash
cd /home/qinhaiyan/Kernel-Harness

serving_native/run.sh --list
serving_native/run.sh --describe dp_allgather_decode_m16

# One-GPU production ABI
serving_native/run.sh linear_indexer_wq_b_decode_m16 \
  --candidate serving_native/candidates/reference.py --execution-mode both

# Fixed and ragged paged-KV decode probes
serving_native/run.sh dsa_trtllm_decode_m16 --kv-context 32768
serving_native/run.sh dsa_trtllm_decode_m16 \
  --kv-contexts 2048,2048,4096,4096,8192,8192,12288,12288,16384,16384,24576,24576,32768,32768,65536,65536

# Gate a candidate across a traced production context profile
.venv/bin/python serving_native/context_matrix.py \
  --profile serving_native/context_profiles/my-production-profile.json \
  --candidate /absolute/path/to/candidate.py

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

Candidate comparisons alternate reference/candidate order on every repeat and
gate on p10 of paired latency ratios. The first pair is discarded; a timing spread
above 1.25× retries once with three times as many pairs, and continued instability
returns no verdict. Correctness is re-checked after timing on inputs generated from
a different seed. Distributed samples use the maximum CUDA-event latency across all
ranks. The target environment must provide the
same `deep_ep` package as the SGLang image; the lightweight Kernel-Harness venv
may not contain it. Point `KERNEL_HARNESS_PYTHON` at that environment when
needed.

`run.sh` supervises the complete launch/torchrun process group with a 30-minute
wall timeout and SIGTERM→SIGKILL cleanup. Set
`KERNEL_HARNESS_TIMEOUT_SECONDS=<seconds>` to change it (`0` disables the timeout).

`--execution-mode eager` is the default diagnostic lane and is deliberately
probe-only for a candidate. Use `--execution-mode both` to measure independent
eager and CUDA-graph lanes; with repeat≥10 and warmup≥8, exit 0 requires paired
p10 speedup ≥1.03 in both. A graph-capture failure is recorded as
`EXECUTION_MODE_UNAVAILABLE`, never silently replaced with eager timing. Even an
`EAGER_GRAPH_LEAF_WIN` has `production_ready: false`: containing-region eager/graph
and end-to-end serving confirmation remain mandatory.

## KV cache and transfer to serving

Only `dsa_trtllm_decode_*` directly consumes paged KV in this leaf suite, and it
times sparse attention after top-k selection. It deliberately spreads the selected
rows over the logical cache, but above `sparse_topk=2048` its attention work does not
include the indexer's context-length-dependent score scan or top-k/page-table
transform. GEMM, MoE, and communication leaf callables do not read KV cache, but
their fraction of a serving step and their overlap with attention/communication
still change with context. A single-card leaf speedup is therefore evidence about
one callable, not an end-to-end prediction.

Incremental prefill is not represented by the fixed `*_prefill` communication
tasks. In SGLang, existing prefix KV and newly extended tokens populate
`ForwardBatch` metadata and can change the DSA prefill backend and top-k transform.
Do not substitute full prefill with prefix=0, a dense contiguous K/V tensor, or the
decode TRT-LLM callable. Record `prefix_tokens x extend_tokens x batch/raggedness`
in a context profile and replay it through the real SGLang containing region and
then end-to-end. See [`context_profiles/README.md`](context_profiles/README.md).
The complete scope analysis and transfer criteria are recorded in
[`testbench/docs/kv-context-transfer-assessment-20260801.md`](../testbench/docs/kv-context-transfer-assessment-20260801.md).

The TP4 communication tasks call the production SGLang coordinator. Their graph
lane is now first-class, but decode AllReduce backend promotion still requires a
containing SGLang graph replay and end-to-end confirmation because the standalone
callable does not model overlap with adjacent communication and compute.

The runner pins the reference process to `SGLANG_GLM52_OPT=0` after consuming
any worker side-channel env file, so a stale deployment configuration cannot
silently replace the baseline.

## Candidate contract

A candidate exports:

```python
def run(inputs, runtime):
    return runtime.reference(inputs)
```

`runtime.reference(inputs, config=...)` keeps the exact production call while
allowing a DeepEP `Config` dictionary to be tuned. A replacement collective or
kernel may instead return its own tensor/tree. Correctness is checked before
timing. The reference candidate is intentionally neutral and is useful for
validating a target node.

Treat wins below 3% as noise unless the target deployment has established a
tighter paired noise floor. A candidate is not serving-ready until the complete
SGLang request workload and overlap region also improve.

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
.venv/bin/python serving_native/selftest.py
```
