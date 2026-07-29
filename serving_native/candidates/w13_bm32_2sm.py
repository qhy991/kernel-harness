"""Historical BM32 two-CTA anchor rebuilt from the pinned source base."""

from serving_native.candidates.w13_common import artifact_paths

W13_VARIANT = "bm32_2sm"
CANDIDATE_IDENTITY = "same-base DeepGEMM BM32 two-CTA/two-SM"
DECLARED_FALLBACK = False
ARTIFACT_PATHS = artifact_paths()


def run(inputs, runtime):
    return runtime.candidate_w13(inputs)
