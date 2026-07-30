# External acceptance: GLM-5.2 FlashMLA combine candidate

`combine_c2_bucket_stages` passes every locally runnable gate at M16 and M32.
The one remaining gate — checkpoint-backed single-node TP8/DP8/EP8 serving — needs
eight B200s and the GLM-5.2 checkpoint, neither available on this four-GPU host.

**This document must not be used to seek an external override for a candidate
that fails a local gate.** It exists because every local gate passed. If a future
change to the kernel, the provider or the frozen ABI invalidates any local gate,
re-run the local lanes first.

## What has already passed locally

- bitwise-exact output and LSE against installed stock at both buckets, on every
  boundary, plus 17 adversarial cases per bucket;
- CUDA Graph containing region and CUDA Graph leaf at both buckets, at or above
  1.03 on all four estimators in three alternating series, against **installed
  stock** and against **P1 plus the stock combine**;
- eager containing region falls back to stock with zero provider launches;
- fail-closed rejection for the wrong page count, with zero provider launches;
- Nsys proof of exactly one prefixed main followed by exactly one prefixed
  combine, in eager and in each of five graph replays;
- main-kernel SASS identical to the round-2 reference P1 build; zero spill, zero
  local memory, unchanged shared-memory footprint in every combine instantiation.

See [`FINAL_REPORT.md`](FINAL_REPORT.md) and
[`evidence/timing_gate_audit.json`](evidence/timing_gate_audit.json).

## Enable (explicit only; never default-on)

```bash
export SGLANG_GLM52_OPT=1
export SGLANG_GLM52_OPT_PROFILE=hotspot_candidates
export SGLANG_GLM52_OPT_OPS=flashmla_sparse_decode
export SGLANG_GLM52_OPT_M_BUCKETS='dsa_decode_attn:16|32'
export SGLANG_GLM52_HOTSPOT_MODULE=/home/qinhaiyan/glm52-hotspot-goal-runs/worktrees/flashmla-sparse-decode/kernel-harness/serving_native/candidates/flashmla_combine_decode_provider.py
export GLM52_FLASHMLA_COMBINE_VARIANT=combine_c2_bucket_stages
# graph-only selection is the default; leave it on
export SGLANG_GLM52_FLASHMLA_GRAPH_ONLY=1
```

The provider compiles on first import. It must be imported **after** the worker
has its CUDA device, which `initialize_hotspot_provider(gpu_id=...)` guarantees.
Finish JIT before capture or timing.

## Eight-GPU acceptance run

Reference arm first, with the hotspot profile off, then the candidate arm with
the exact same launch, workload and seed. `--dsa-decode-backend flashmla_kv` is
mandatory: the no-flag FP8 default on SM100 may select TRT-LLM, which is a
different ABI and not what this candidate replaces.

```bash
# 1. reference: installed stock
env -u SGLANG_GLM52_OPT -u SGLANG_GLM52_HOTSPOT_MODULE \
    -u GLM52_FLASHMLA_COMBINE_VARIANT \
    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  python -m sglang.launch_server \
    --model-path <GLM-5.2-checkpoint> \
    --trust-remote-code \
    --tp-size 8 --dp-size 8 --enable-dp-attention \
    --enable-ep-moe --ep-size 8 \
    --dsa-decode-backend flashmla_kv \
    --cuda-graph-max-bs 32 \
    --port 30000

# 2. candidate: same launch, with the enable block above exported
```

Workload must hold the promoted decode buckets. Local `M` is 16 and 32 per rank
and must **not** be divided by the data-parallel world size.

```bash
python -m sglang.bench_serving \
  --backend sglang --host 127.0.0.1 --port 30000 \
  --dataset-name random --random-input-len 8192 --random-output-len 512 \
  --max-concurrency <16 or 32 per rank> --num-prompts 512 --seed 0
```

Distributed latency is the **maximum across ranks**, not the mean.

## Acceptance criteria

1. Output quality unchanged: the candidate is bitwise exact locally, so any
   accuracy delta at eight ranks indicates an integration fault, not a numeric
   trade — investigate rather than accept.
2. Decode TPOT / ITL improves, or at minimum does not regress, at both promoted
   buckets under CUDA Graph replay.
3. `dispatch._HIT_COUNTS` shows `hotspot_plugin/flashmla_sparse_decode` for
   `dsa_decode_attn` at `m16` and `m32`, with zero fallback after selection.
4. Eager decode shows **zero** provider launches (graph-only selection).
5. No regression in prefill or in any non-promoted path.

Expected magnitude, from the local graph lanes against installed stock: the
FlashMLA DSA decode leaf improves by at least 27.0% at M16 and at least 13.5% at
M32, those being the minimum of all twelve estimators. Layer share bounds the
end-to-end effect: DSA/FlashMLA is roughly 13.6–13.7% of a
comm-free decode layer and roughly 5.1% of full-server short-decode GPU kernel
time, so expect a low-single-digit end-to-end effect, not 27%.

## If acceptance passes

Registration may move to **L2 (external E2E)**. It does **not** move to L3
production-default from this evidence. Promote only the buckets the eight-GPU run
actually validates, and keep stock as the fallback for everything else.

## If acceptance fails

Roll back by unsetting `SGLANG_GLM52_OPT`. To isolate whether the combine or the
round-2 main kernel is implicated, set
`GLM52_FLASHMLA_COMBINE_VARIANT=combine_identity`, which keeps the same provider,
the same P1 main and the stock combine algorithm.
