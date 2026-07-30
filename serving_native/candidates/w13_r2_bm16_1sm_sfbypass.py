"""Round-2 SF-relay-bypass experiment on BM16 one-SM."""

from serving_native.candidates.w13_common import artifact_paths

W13_VARIANT = "r2_bm16_1sm_sfbypass"
CANDIDATE_IDENTITY = (
    "infini_kernel GLM-5.2 W13 r2_bm16_1sm_sfbypass API-v1"
)
DECLARED_FALLBACK = False
ARTIFACT_PATHS = artifact_paths()


def run(inputs, runtime):
    return runtime.candidate_w13(inputs)
