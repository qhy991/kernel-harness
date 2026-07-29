"""Genuine one-CTA W13 candidate for the four fixed decode points."""

from serving_native.candidates.w13_common import artifact_paths

W13_VARIANT = "bm32_1sm"
CANDIDATE_IDENTITY = "same-base DeepGEMM BM32 one-CTA/one-SM"
DECLARED_FALLBACK = False
ARTIFACT_PATHS = artifact_paths()


def run(inputs, runtime):
    return runtime.candidate_w13(inputs)
