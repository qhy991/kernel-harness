"""Identity candidate: invoke the exact production reference."""


def run(inputs, runtime):
    return runtime.reference(inputs)
