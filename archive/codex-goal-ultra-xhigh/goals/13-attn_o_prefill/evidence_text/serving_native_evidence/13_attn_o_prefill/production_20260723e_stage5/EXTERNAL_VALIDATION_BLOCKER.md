# External production-validation blocker

The attention-layer and complete SGLang prefill baselines were not executed on
this host. Their production acceptance contract is preserved, not weakened.

## Missing resources

- The current SGLang GLM-5.2 FP8 end-to-end test is explicitly an eight-GPU
  `tp_size=8` lane and includes a `--dp=8 --enable-dp-attention` variant.
- This host exposes four physical B200 GPUs. A four-rank run is diagnostic only
  under the governing rules and cannot replace the TP8/DP8 gate.
- No usable GLM-5.2 FP8 checkpoint is local. The only GLM-named model path
  inspected, `/mnt/OS-oKqEXySb/models/GLM-5.2-NVFP4`, is an empty 4 KiB
  directory. It is also the wrong quantization family for this FP8 packed-scale
  goal and cannot be substituted.

The raw preflight is in `validation/external_gate_preflight.txt`; the captured
topology is in `topology.txt`. Because there is no model, launching a four-GPU
server would not produce a useful diagnostic and was not attempted.

## Work completed without weakening the gate

- The exact rank-local `Fp8LinearMethod.apply` region was reconstructed with
  production BF16 input, checkpoint-style FP8 weight, packed `int32` UE8M0
  scales, production quantization, stock dispatch, PDL, and BF16 output.
- Reachability, three same-GPU baselines, three existing-dispatcher series,
  three controlled source-experiment series, Nsight Systems, Nsight Compute,
  SASS, ptxas resources, supported configurations, and full-region timing are
  preserved.
- The source experiment already loses the microbenchmark and containing-region
  gates, so it is ineligible for promotion regardless of an unavailable
  end-to-end result.
- The experiment was reverted and stock fallback remains active.

## Required external lane

If a future candidate first clears the local micro and region gates, run the
unchanged SGLang `test/registered/8-gpu-models/test_glm52_fp8.py` contract on
one eight-B200 node with `zai-org/GLM-5.2-FP8`. Collect alternating stock and
candidate attention-layer/prefill measurements for TP8+DP8, preserve the
production local M4096 packed ABI, and validate graph/replay and accuracy
semantics. A TP4 result must remain separately labeled diagnostic evidence.
