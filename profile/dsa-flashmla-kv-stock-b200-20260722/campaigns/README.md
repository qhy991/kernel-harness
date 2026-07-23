# Flexible-GPU campaigns

Each child directory is one immutable campaign run under a single invocation of
`with_flexible_gpu.sh`. The wrapper allocation line is retained in `wrapper.log`;
start/intermediate/end GPU identity and clock snapshots are under `analysis/`.
All alternating eager and CUDA Graph sessions plus Nsys and NCU collection stay
inside that one lease. CPU-only report export and summarization happen after the
lease is released.

The earlier flat `../analysis/` and `../reports/` artifacts were collected under
the superseded fixed-GPU instruction. They are retained unchanged as historical
evidence and are not relabelled as flexible-scheduler campaigns.
