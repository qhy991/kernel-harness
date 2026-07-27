# Baseline preflight `20260723a`

The locked bundle stopped during `capture_reachability.py`, before any paired
timing, Nsys, or NCU collection.

- The B200 environment check passed on the wrapper-selected GPU recorded in
  `gpu_identity.csv`.
- The production-shaped eager and CUDA-graph calls completed, but serializing
  metadata failed because `deep_gemm_fp8_fp8_bf16_nt` is a PyTorch
  `OpOverloadPacket` and `inspect.getsourcefile()` rejects that object type.
- `bundle.log` contains the complete traceback.
- `reachability_runtime.json` is incomplete stdout from the failed process and
  must not be cited as a JSON result.

The metadata serializer was made tolerant of registered custom-op objects.  A
fresh non-overwriting `20260723b` directory is used for the real campaign.
