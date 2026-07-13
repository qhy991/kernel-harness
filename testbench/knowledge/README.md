# Optimization-recipe knowledge base

One structured entry per **completed optimization session** — the bottleneck diagnosis,
every approach tried (failures included), the measured outcome, and a transferable
lesson. Later sessions (any model family, any GPU) query it before touching
`solution.py`, so the fleet accumulates recipes instead of rediscovering dead ends.

```
testbench/knowledge/
  README.md        # this contract
  entries/         # one JSON file per session, filename = entry id (append-only)
```

Tool (stdlib-only, runs anywhere — like `bin/selftest.py`):

```bash
python3 testbench/bin/knowledge.py query --task kimi_k27/o_proj_decode   # prior recipes
python3 testbench/bin/knowledge.py query --family fp8-linear-gemm --gpu B200
python3 testbench/bin/knowledge.py add my_entry.json                     # validate + install
python3 testbench/bin/knowledge.py lint                                  # validate all entries
```

## Rules (enforced by `knowledge.py`, not just requested)

- **One entry per session, win or not.** A `no-win` or `failed` entry with honest
  "why" lines on each approach is as valuable as a win — it saves the next agent the
  same detour.
- **Every number comes from `evaluate.py`'s final `VERDICT_JSON`.** Never record a
  `profile.py` (advisory) number or an unmeasured estimate as a result. The linter
  rejects `status: "win"` unless `min_speedup_conservative > 1.0` and at least one
  approach has `outcome: "win"`.
- **Append-only.** `add` refuses to overwrite; never edit or delete an existing entry.
  To correct or supersede one, add a new entry (query sorts newest first).
- **Pin the substrate.** `hardware.gpu`/`sm` and `stack.sglang_commit` are required —
  a recipe is a claim about a kernel on a chip at a commit, not a universal truth.
  Copy them from `bin/check_env.py` output.

## Schema (`schema_version: 1`)

| Field | Type | Meaning |
|---|---|---|
| `schema_version` | int | always `1` |
| `id` | str | unique slug, must equal the filename stem; suggested `<model>--<task>--<gpu>--<yyyymmdd><a,b,...>` |
| `date` | str | `YYYY-MM-DD` |
| `model` | str | target model family, e.g. `kimi_k27`, `minimax_m3` |
| `task` | str | `<model>/<task_dir>`, e.g. `kimi_k27/o_proj_decode` |
| `op`, `family`, `phase` | str | copied from the task's `task.json` (`phase` ∈ `prefill`/`decode`) |
| `shapes` | object | the swept regime, e.g. `{"K": 8192, "N": 7168, "M_sweep": [1, 256]}` |
| `hardware` | object | `{"gpu": "NVIDIA B200", "sm": "sm_100"}` |
| `stack` | object | `{"sglang_commit": "<short sha>"}` (+ optional `torch`, `cuda`) |
| `baseline_kernel` | str | what was beaten / not beaten, from `task.json` `backend` |
| `bottleneck` | object | `kind` ∈ `memory-bandwidth` `compute` `launch-overhead` `kernel-count` `quantization-overhead` `occupancy` `synchronization` `none-identified` `other`; `evidence` = the measurement that proves it (roofline numbers, profile output) |
| `approaches` | array | ≥1; each `{technique, summary, outcome, geomean_speedup, why}` — `outcome` ∈ `win` `partial` `slower` `incorrect` `error` `abandoned`; `geomean_speedup` required for `win`/`partial`, else may be null; `why` = one-sentence causal explanation |
| `result` | object | `status` ∈ `win` `no-win` `failed`; `geomean_speedup`, `min_speedup_conservative`, `repeat` from the final `VERDICT_JSON` (null if not applicable); `integrate` ∈ `pass` `fail` `not-run` `no-recipe` |
| `lesson` | str | 1–3 sentences: the transferable rule, stated so an agent on a *different* task can apply it |
| `transfers_to` | array of str | where this likely applies (families/ops/shape regimes/arch) |
| `caveats` | array of str | where it will NOT transfer (may be empty) |
| `agent` | str, optional | which agent/model produced the session |

Unknown top-level keys are rejected (typo protection).

## Example (illustrative numbers, not a real measurement)

```json
{
  "schema_version": 1,
  "id": "kimi_k27--o_proj_decode--b200--20260713a",
  "date": "2026-07-13",
  "model": "kimi_k27",
  "task": "kimi_k27/o_proj_decode",
  "op": "O_proj",
  "family": "fp8-linear-gemm",
  "phase": "decode",
  "shapes": {"K": 8192, "N": 7168, "M_sweep": [1, 256]},
  "hardware": {"gpu": "NVIDIA B200", "sm": "sm_100"},
  "stack": {"sglang_commit": "abc1234", "torch": "2.11.0", "cuda": "13.0"},
  "baseline_kernel": "deep_gemm w8a8_block_fp8 (blackwell)",
  "bottleneck": {
    "kind": "memory-bandwidth",
    "evidence": "profile.py roofline at M=16: AI=32 far below ridge ~280; weight reads dominate, 71% of HBM peak"
  },
  "approaches": [
    {
      "technique": "triton-persistent-gemm",
      "summary": "Persistent Triton kernel, 128x128 tiles, fp8 dot, weight-stationary",
      "outcome": "slower",
      "geomean_speedup": 0.71,
      "why": "deep_gemm's TMA pipelining already saturates HBM at these shapes; Triton adds scheduling overhead without reducing bytes moved"
    },
    {
      "technique": "deep-gemm-config-sweep",
      "summary": "Alternate deep_gemm block config for M<=16",
      "outcome": "partial",
      "geomean_speedup": 1.04,
      "why": "Smaller M-tile cuts wave quantization at M<=8 but loses at M>=64"
    }
  ],
  "result": {
    "status": "no-win",
    "geomean_speedup": 1.01,
    "min_speedup_conservative": 0.93,
    "repeat": 3,
    "integrate": "not-run"
  },
  "lesson": "For weight-memory-bound fp8 decode GEMMs on sm_100, don't re-implement the GEMM: deep_gemm is at the HBM roof. The only headroom is shape-regime config selection at M<=8, and it trades away large-M performance, so it can't win across a full sweep.",
  "transfers_to": ["fp8-linear-gemm decode tasks on sm_100 with M_sweep spanning 1-256"],
  "caveats": ["Prefill (large M, compute-bound) not covered", "May not hold on sm_90 where deep_gemm uses a different pipeline"]
}
```
