# External validation blockers

Rechecked on 2026-07-23 in the isolated goal worktrees.

## Full-model SGLang decode

The expected local checkpoint directory is present but empty:

```text
$ du -sh /mnt/OS-oKqEXySb/models/GLM-5.2-NVFP4
4.0K    /mnt/OS-oKqEXySb/models/GLM-5.2-NVFP4

$ ls -la /mnt/OS-oKqEXySb/models/GLM-5.2-NVFP4
total 8
drwxr-xr-x 2 root root 4096 Jul 14 09:09 .
drwxr-xr-x 4 root root 4096 Jul 14 10:08 ..
```

A bounded search below `/mnt/OS-oKqEXySb/models` found only the unrelated
`Qwen3-4B` checkpoint. Consequently a GLM-5.2 server cannot be started and the
complete-model decode latency/throughput gate cannot be measured on this host.
The direct production-symbol workload, exact SGLang attention fixture, and core
CUDA-graph replay are preserved as lower-level evidence; none is relabeled as a
complete SGLang decode result.

## Production rank count

The host exposes four physical NVIDIA devices (`/dev/nvidia0` through
`/dev/nvidia3`). The production gate is one-node TP8/DP8/EP8, so eight ranks are
unavailable.

Single-GPU work now uses
`/home/qinhaiyan/glm52-goal-runs/with_flexible_gpu.sh`; the complete corrected
campaign ran in one lease on physical GPU 1. Four-rank diagnostics are separately
available through
`/home/qinhaiyan/glm52-goal-runs/with_all_gpus_lock.sh`. This operator's exact
serving-native workload is nevertheless rank-local: attention TP is 1 under
DP4/DP8, it contains no collective, and running four independent copies would
not validate a different FlashMLA ABI or topology. The useful four-rank
diagnostic would be a complete captured model/backend replay, which requires the
missing GLM-5.2 checkpoint. No all-GPU command was invented or relabelled as
that unavailable model test.

The missing eight-rank gate is retained as an external validation requirement;
neither the rank-local one-GPU result nor a hypothetical TP4 diagnostic is
substituted for it.

## Disposition impact

The scheduler-corrected M16 combine-bound campaign fails the repeated 3% paired
gate in every eager and real CUDA-graph session (0/12), so it is not eligible
for promotion even before the unavailable containing-model and TP8 gates. The
exact enable policy is therefore empty: stock FlashMLA remains active for M16,
M32, and every unsupported ABI/topology. A future promotion attempt must rerun
the complete SGLang decode and the true TP8/DP8/EP8 production lane with the
checkpoint and hardware available.
