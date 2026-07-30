# Reproducing the GLM-5.2 AMD op-level winners (MI300X / gfx942)

Self-contained, portable guide for reproducing the four op-level GATE-1 wins on a fresh
MI300X node. Everything here is version-controlled; nothing depends on a personal profile
or a gitignored path.

- **Scope:** op-level GATE-1 only (single MI300X card). No end-to-end serving.
- **What "reproduces" is the *win*** — the conservative speedup and zero regression vs the
  reference — **not the absolute microseconds**, which are node/thermal/binary dependent
  (see [Why absolute µs vary](#why-absolute-µs-vary)).

---

## 1. Results you should be able to reproduce

All four candidates are A/B'd against the harness reference oracle at S/KV = 32768,
sweeping M ∈ {1024, 2048, 4096} (decode: M ∈ {16, 32}).

| Op | Speedup (geomean) | Shapes | Correctness | Lever |
|---|---|---|---|---|
| `index_score_prefill` | **2.90×** | 3/3 win | **bit-exact** (calc_diff 0.0) | launch-config override of aiter's `_fp8_mqa_logits` Triton kernel — `BLOCK_KV=256`, `num_stages=1` (KV-loop tiling only, never the reduction) |
| `moe_total_prefill` | **~1.12×** | 3/3 win | **bit-exact** (calc_diff 0.0) | `BLOCK_SIZE_M` 128→256 token-tile resize + per-M `GROUP_SIZE_M` (round-2 upgrade; M=4096 flipped parity→win) |
| `moe_total_decode` | **1.0541×** | 2/2 win | **bit-exact** (calc_diff 0.0) | `BLOCK_SIZE_M` shrink on the gated dense-degenerate path |
| `dsa_prefill_attn` | **1.3266×** | 3/3 win, 0 regress | calc_diff 1.87e-6 (gate ≤ 5e-6) | purpose-built native-64-head Triton sparse-MLA flash kernel (half the padded-128 ASM FLOPs; `matrix_instr_nonkdim=16`, bf16 PV, fp32 QK/softmax/accum) |

Three of four are **bit-exact** (output identical to the reference to the last bit). `dsa`
is within the harness correctness gate (≤ 5e-6). None regress any shape.

### Reference baseline (median µs on the origin node — for orientation only, NOT a pass/fail target)

| Op (phase) | M=1024 | M=2048 | M=4096 | Note |
|---|---|---|---|---|
| `dsa_prefill_attn`   | 1667 | 2907 | 5931 | ASM `mla_decode_fwd` path (not the ~664 ms Triton dev placeholder) |
| `index_score_prefill`| 930  | 7520 | 14708 | slow `_fp8_mqa_logits` Triton (this node lacks compiled CK; strongest available) |
| `moe_total_prefill`  | 1365 | 2800 | 3744 | healthy, M-driven |
| `moe_total_decode`   | 325 (M16) | 328 (M32) | — | healthy |

---

## 2. Requirements

**Hardware:** AMD Instinct **MI300X** (`gfx942`). One healthy card is enough.

**Software (pinned to what produced the numbers above):**

| Component | Version / ref | Source |
|---|---|---|
| ROCm | 7.0.0 | system install, targeting `gfx942` |
| Python | 3.11.4 | — |
| torch | 2.10.0+rocm7.0 | ROCm 7.0 PyTorch wheel (carries its own triton 3.6.0) |
| triton | 3.6.0 (ROCm build) | bundled with the torch wheel |
| sgl-kernel | 0.4.3 | pip |
| numpy | 2.4.4 | pip |
| einops | 0.8.2 | pip |
| **aiter** | `2ca7878e2d47f7f96c4e5e6299065d95e0f6489f` (`v0.1.18-6-g2ca7878e2`) | **`https://github.com/ROCm/aiter.git`** — used from source on `PYTHONPATH`, *not* pip-installed |
| **sglang** | `20fc529abdb9c9da36dbf7f2789fa285be3663f4` | **`https://github.com/qhy991/SGLang-DGMK.git`** — used from source (`$SGLANG_ROOT/python` on `PYTHONPATH`) |

> **⚠ sglang is OUR ROCm fork, not upstream.** Clone `qhy991/SGLang-DGMK` at the pinned
> commit — **not** `sgl-project/sglang`. The fork carries the ROCm/aiter dispatch paths
> (`SGLANG_USE_AITER`, the `fused_moe` triton path, the sparse-MLA wiring) that the moe and
> index_score candidates call into; upstream sglang will not reproduce these results.

Nearby versions will likely work, but the bit-exact claims are only guaranteed against the
exact reference kernels above (a different aiter/triton can reorder a reduction — see the
`matrix_instr_nonkdim` note in the `index_score_prefill` candidate header).

---

## 3. One-time setup

1. Install ROCm 7.0.0 and confirm the card enumerates:
   ```bash
   rocminfo | grep -m1 gfx        # expect gfx942
   ```
2. Create the Python 3.11 venv and install the pinned runtime. torch's ROCm-7.0
   wheel carries its own triton 3.6.0:
   ```bash
   python3.11 -m venv rocm-torch && . rocm-torch/bin/activate
   # torch 2.10.0+rocm7.0 from the ROCm 7.0 PyTorch wheel index
   # (adjust the index URL to wherever your ROCm-7.0 wheels live):
   pip install --index-url https://download.pytorch.org/whl/rocm7.0 torch==2.10.0
   pip install sgl-kernel==0.4.3 numpy==2.4.4 einops==0.8.2
   ```
3. Check out the two source trees at the **exact** pinned commits. Both are used
   from source via `PYTHONPATH` (`runenv.sh` wires them in) — neither is pip-installed:
   ```bash
   git clone https://github.com/ROCm/aiter.git
   git -C aiter   checkout 2ca7878e2d47f7f96c4e5e6299065d95e0f6489f

   # sglang = OUR ROCm fork (qhy991/SGLang-DGMK), NOT upstream sgl-project/sglang:
   git clone https://github.com/qhy991/SGLang-DGMK.git sglang
   git -C sglang  checkout 20fc529abdb9c9da36dbf7f2789fa285be3663f4
   ```
4. Point the env at your locations — either edit the vars at the top of
   `repro/runenv.sh`, or export them before sourcing:
   ```bash
   export ROCM_TORCH_VENV=/your/venvs/rocm-torch
   export AITER_ROOT=/your/aiter
   export SGLANG_ROOT=/your/sglang
   ```

---

## 4. Run it

```bash
# from the repo root
source repro/runenv.sh          # loads ROCm/gfx942 env + the FROZEN gate identity
repro/gate.sh smoke             # 1-iteration sanity check (fast; proves the env imports)
repro/gate.sh gate              # authoritative A/B for all four ops (repeat 10, iters 30, warmup 3)
# or one op:
repro/gate.sh gate index_score_prefill
```

`gate.sh` drives each op's own `run.sh`, which A/Bs `candidate.py` against the reference and
prints per-shape speedup + correctness. Exit-code legend (per op):

| code | meaning |
|---|---|
| 0 | correct **and** faster (a win) |
| 1 | correct but not faster |
| 2 | incorrect (failed the correctness gate) |
| 3 | infra / contract error |

Each `gate` run also persists an auditable record under `runs/glm52/<op>/<run_id>/result.json`
(the `runs/` tree is gitignored — per-machine). Audit one with:

```bash
"$ROCM_TORCH_VENV/bin/python" testbench/bin/audit_result.py runs/glm52/<op>/<run_id>/result.json
```

A clean pass reports `OFFICIAL` with `git_dirty=False`, `worst_calc_diff` at/under the op's
tolerance (0.0 for the three bit-exact ops), `shapes_regressed=0`, and
`timing_unstable_shapes=[]`.

---

## 5. Frozen gate identity (do NOT change)

`runenv.sh` sets these; changing any of them means you are no longer reproducing the same
measurement:

- `KERNEL_HARNESS_PLATFORM=rocm`, `KERNEL_HARNESS_PROFILE=amd-mi300x`
- `KERNEL_HARNESS_PROVIDER=aiter-torch-reference`, `KERNEL_HARNESS_TIMER=event`
- `SGLANG_USE_AITER=1`, `AITER_TRITON_ONLY=0`
  (the `dsa` ASM ~1.7 ms baseline and the `fp8_mqa_logits` dispatch both require the aiter path)
- S/KV = 32768 (comes from each task's `problem.json`; the runner only sweeps M)

**GPU selection caveat:** `HIP_VISIBLE_DEVICES` orders by PCI bus, which does **not** match
`rocm-smi` GPU numbering. Verify your pinned card is healthy before trusting timings — a
degraded card silently inflates and destabilises them. Default is `HIP_VISIBLE_DEVICES=0`;
override for your box.

---

## Why absolute µs vary

The reference baselines above are node-specific. On the origin node the aiter source build
dispatched two kernels to slow fallbacks (sparse-MLA fwd → a ~662 ms Triton dev placeholder
vs ~1.7 ms in production; `fp8_mqa_logits` → ~15 ms vs ~4.3 ms), because the CK/ASM
components for `gfx942` were not compiled into that source build. The **harness code is
correct**; the gap is purely runtime/binary. A node with the production binaries will show
different (smaller) absolute µs. What must reproduce is the **relative** result: each
candidate faster than the reference on the same node, with the correctness above.

---

## Troubleshooting

**`import aiter` fails during GPU-arch detection.** aiter probes the arch at import time
(`aiter/jit/utils/chip_info.py` → `rocminfo`). If `rocminfo`/kfd is unhealthy on the node,
the import dies inside `_detect_native()` / `get_gfx_runtime()`. First confirm the card is
actually up — this is the common cause, and a degraded card also invalidates timings:

```bash
rocminfo | grep -m1 gfx     # must print gfx942
"$ROCM_TORCH_VENV/bin/python" -c "import torch; print(torch.cuda.is_available(), torch.cuda.device_count())"
```

If the GPU is healthy this never triggers — fix the driver (`sudo modprobe -r amdgpu &&
sudo modprobe amdgpu`, or reboot the node) rather than trusting numbers from a bad card. If
you *must* import against a node whose `rocminfo` is degraded but the GPU is fine, shadow
`rocminfo` on `PATH` with a script that emits `Name: gfx942`, `Device Type: GPU`,
`Compute Unit: 304`, and export `GPU_ARCHS=gfx942 CU_NUM=304` (aiter resolves the tool via
`shutil.which`, so a `PATH` shim wins).

---

## Security note

`repro/runenv.sh` is deliberately credential-free. The original per-round scripts under
`.humanize/kernel-agent/` (gitignored) sourced a personal profile that exported LLM-gateway
tokens; do **not** copy those into any shared or remote setup.
