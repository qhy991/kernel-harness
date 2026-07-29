"""Stock-versus-stock identity control with both runtimes still initialized."""

from serving_native.candidates.w13_common import artifact_paths

W13_VARIANT = "bm16_1sm"
CANDIDATE_IDENTITY = "same-source stock W13 identity control"
IDENTITY_CONTROL = True
DECLARED_FALLBACK = False
ARTIFACT_PATHS = artifact_paths()


def run(inputs, runtime):
    return runtime.reference(inputs)
