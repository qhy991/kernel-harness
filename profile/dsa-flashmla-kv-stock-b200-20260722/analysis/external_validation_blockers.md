# External validation blockers

Observed on 2026-07-22 in the isolated goal worktrees.

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

The host exposes four physical NVIDIA devices (`0000:05:00.0` through
`0000:08:00.0`, `/dev/nvidia0` through `/dev/nvidia3`). The production gate is
one-node TP8/DP8/EP8. Eight ranks are therefore unavailable. In addition, this
goal's mandatory command wrapper is
`/home/qinhaiyan/glm52-goal-runs/with_gpu_lock.sh 3`, which intentionally makes
only physical GPU 3 visible to each GPU command. It cannot run even the separate
four-rank diagnostic lane, and the wrapper was not bypassed.

The missing eight-rank gate is retained as an external validation requirement;
no four-rank or one-rank result is substituted for it.

## Disposition impact

The M16 combine-bound specialization already fails the repeated 3% paired gate
in both eager and real CUDA-graph replay, so it is not eligible for promotion
even before the unavailable containing-model and TP8 gates. The exact enable
policy is therefore empty: stock FlashMLA remains active for M16, M32, and every
unsupported ABI/topology. A future promotion attempt must rerun the complete
SGLang decode and the true TP8/DP8/EP8 production lane with the checkpoint and
hardware available.
