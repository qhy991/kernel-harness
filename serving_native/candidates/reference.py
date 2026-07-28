"""Identity candidate: invoke the exact production reference."""

IDENTITY_CONTROL = True
CANDIDATE_IDENTITY = "identity_control:runtime.reference"
ARTIFACT_PATHS = ()


def run(inputs, runtime):
    return runtime.reference(inputs)
