# GPU recovery — what happens when aiter crashes and how to restore

## The failure mode

Any HIP `hipErrorLaunchFailure` (illegal memory access, out-of-bounds
write, invalid kernel arg) that happens inside an aiter kernel on gfx942
can leave three things in bad state:

1. **The Python process** — main worker enters `D` (uninterruptible
   disk sleep) in the kernel, waiting on a ROCm resource release that
   never completes. `kill -9` does not work on `D` processes.
2. **The 8 GPUs' HBM** — each rank had ~125 GB allocated for weights +
   KV cache + workspace. Because the process can't finish teardown,
   allocations aren't released. `rocm-smi --showmeminfo vram` reports
   `~125 GB used` per rank indefinitely.
3. **The KFD driver context (`/dev/kfd`)** — the illegal access
   corrupted an internal amdkfd data structure. Subsequent
   `open("/dev/kfd", O_RDWR)` from ANY process returns `-EINVAL`, which
   surfaces as:
    - `rocminfo` → `Unable to open /dev/kfd read-write: Invalid argument`
    - `torch.cuda.is_available()` → `False`
    - `torch.cuda.init()` → `RuntimeError: No HIP GPUs are available`
   even though `rocm-smi` (which uses a different path) still lists
   the 8 devices and reads their state.

**Once (3) happens, no user-space fix can recover.** The kernel driver
context is dead until the driver is reloaded.

## Root cause of the trigger (specifically for GLM-5.2 aiter decode at bs≥256)

`sglang/srt/layers/attention/dsa_backend.py::DSABackend.__init__`
allocates on ROCm:

```python
max_bs = model_runner.req_to_token_pool.size    # ← from sglang.max_running_requests
self.kv_indices = torch.zeros(
    max_bs * self.dsa_index_topk,               # dsa_index_topk = 2048 for GLM-5.2
    dtype=torch.int32, device=self.device,
)
```

At forward time `_run_aiter_mla_decode_fwd` calls:

```python
get_valid_kv_indices(page_table_1, kv_indptr, self.kv_indices, bs)
```

which writes `bs * dsa_index_topk` int32 values into `self.kv_indices`.
**If sglang's inferred `max_running_requests < bs`, the buffer overruns.**

`sglang.bench_one_batch` doesn't pass `--max-running-requests` by
default, so sglang picks a value from HBM headroom. On our runs that
value came out < 256 for the bench sweeps, and the overrun corrupted
KFD.

Fix: pin `--max-running-requests` ≥ largest bench batch size.
Implemented in `run_decode_throughput.sh` as of commit that ships this
document; the env var `KDA_E2E_MAX_RUNNING_REQUESTS` can override.

## What to try before rebooting

None of these worked in our case, but they cost nothing to try in order:

1. **Wait 60-120 seconds.** Some `D`-state processes finish teardown when
   an amdgpu timeout eventually completes. `ps -p <pid>` will show it
   gone. This clears (1) and (2) but not always (3).
2. **`kill -9 <pid>`.** Doesn't work on `D`, but if the process
   transitioned to `Z` (zombie) or `R`, kill it now to free the
   process table entry.
3. **`ipcrm -M / --all`.** Occasionally frees a shm segment sglang left
   pinned. Rarely helps for aiter crashes.
4. **Wait for amdgpu major reset window** (up to hours on some
   configs). Reports vary; we did not observe this actually recover
   `/dev/kfd` on our node.

## What actually works

**In order of user cost:**

| Action                                       | Cost      | Notes |
|----------------------------------------------|-----------|-------|
| `sudo rocm-smi --gpureset -d N` for each N   | least     | Fails on many datacenter OAM 8-way systems ("Not supported on the given system"). Try first anyway. |
| `sudo modprobe -r amdgpu && sudo modprobe amdgpu` | medium | Requires no one else is currently using the GPUs. May fail if any process still holds a KFD fd. Usually works. |
| `sudo reboot`                                | most      | Always works. |

After any of these, verify recovery:

```bash
rocminfo | head -20                                        # should list 8 agents
python -c "import torch; print(torch.cuda.is_available())"  # should print True
```

## How to avoid the crash going forward

- `run_decode_throughput.sh` now pins `--max-running-requests` — the
  known trigger for gfx942 aiter's decode kv_indices overrun.
- If you write a new bench script that dispatches to sglang aiter
  decode at bs≥128, always pass `--max-running-requests <max_bs>`.
- If a new bench hits a HIP crash, immediately capture:
    - the last 30 lines of stdout/stderr (the aiter shape info)
    - `ps -p <pid> -o pid,stat,cmd`
    - `rocm-smi --showmeminfo vram`
  so the failure mode is diagnosable without needing another crash to
  reproduce.

## For our archive: what was measured before the crash

`archive/e2e-decode-20260723-partial/baseline_bs_sweep.csv` has the
bs=16/32/64/128 baseline extracted from the log before the bs=256 aiter
crash. **bs=128 at 1150 tok/s (overall) / 433 tok/s (decode-only)** —
the user's requested "global batch=128, TP=8" scenario is on record.
bs=256 result is pending recovery + a re-run with the fixed
`--max-running-requests` argument.
