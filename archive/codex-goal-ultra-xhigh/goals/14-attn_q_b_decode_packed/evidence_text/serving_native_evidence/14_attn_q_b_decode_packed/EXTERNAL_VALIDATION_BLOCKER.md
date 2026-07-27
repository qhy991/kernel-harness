# External full-decode validation blocker

Observed on 2026-07-23:

```text
$ ls -la /mnt/OS-oKqEXySb/models/GLM-5.2-NVFP4
total 8
drwxr-xr-x 2 root root 4096 Jul 14 09:09 .
drwxr-xr-x 4 root root 4096 Jul 14 10:08 ..
```

A read-only search under `/mnt` found no GLM-5.2 `config.json` or safetensors
index. The host exposes exactly four NVIDIA B200 devices. Therefore:

- neither stock nor candidate can start a GLM-5.2 server locally without the
  checkpoint;
- the independent TP4/DP4/EP4 diagnostic cannot be run;
- the required TP8/DP8/EP8 production acceptance lane cannot be represented by
  this four-GPU host and is not weakened or relabeled.

The stock path remains active. If weights become available on this host, a TP4
diagnostic must run as one command under
`/home/qinhaiyan/glm52-goal-runs/with_all_gpus_lock.sh`. Production acceptance
must run separately on an eight-B200 node, alternating complete stock and
candidate decode workloads with rank latency reduced by maximum and with the
same graph buckets, prompts, clocks, and launch configuration.

This blocker does not hide a potential local win: the single-rank production
layer and CUDA-graph gates already regress at M16 and M32. A future external
run is required only for completeness after a new local candidate clears those
gates.
