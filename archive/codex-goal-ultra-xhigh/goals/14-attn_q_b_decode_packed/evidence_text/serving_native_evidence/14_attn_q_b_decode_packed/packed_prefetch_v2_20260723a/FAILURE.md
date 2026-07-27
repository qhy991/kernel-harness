# Preserved failed v2 bundle

The same-GPU bundle completed all six alternating serving-native pair series, then
stopped before graph/profiler collection.  The first production-layer probe passed
a prequantized tuple to `Fp8LinearMethod`; the stock DeepGEMM path correctly
asserted that its caller-side `input_scale` is `None`.  This is a probe-contract
error, not a kernel correctness result.  The replacement bundle uses the real BF16
caller input so `Fp8LinearMethod` performs production dynamic packed-UE8M0
quantization.
