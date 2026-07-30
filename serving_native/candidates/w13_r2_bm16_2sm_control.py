"""Round-2 control: reproduces the round-1 BM16 two-SM machine code byte-for-byte."""

from serving_native.candidates.w13_common import artifact_paths

W13_VARIANT = "r2_bm16_2sm_control"
CANDIDATE_IDENTITY = (
    "infini_kernel GLM-5.2 W13 r2_bm16_2sm_control API-v1"
)
DECLARED_FALLBACK = False
ARTIFACT_PATHS = artifact_paths()


def run(inputs, runtime):
    return runtime.candidate_w13(inputs)
