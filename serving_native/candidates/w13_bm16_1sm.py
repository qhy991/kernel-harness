"""Exact BM16 one-CTA/one-SM W13 API-v1 provider candidate."""

from serving_native.candidates.w13_common import artifact_paths

W13_VARIANT = "bm16_1sm"
CANDIDATE_IDENTITY = (
    "infini_kernel GLM-5.2 W13 BM16 one-CTA/one-SM stage-11 API-v1"
)
DECLARED_FALLBACK = False
ARTIFACT_PATHS = artifact_paths()


def run(inputs, runtime):
    return runtime.candidate_w13(inputs)
