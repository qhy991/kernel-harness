"""Direct PyNCCL control for the caller-owned row-gather ABI.

This is intentionally equivalent to the GroupCoordinator graph-mode branch.
It proves that candidate dispatch itself does not add allocation or copies.
"""

BACKEND = "pynccl_nccl_allgather"
SUPPORTED_FAMILIES = ("allgather",)


def is_applicable(inputs, runtime):
    communicator = getattr(runtime.tp_group, "pynccl_comm", None)
    return (
        runtime.workload.family in SUPPORTED_FAMILIES
        and communicator is not None
        and bool(getattr(communicator, "available", False))
    )


def run(inputs, runtime):
    if not is_applicable(inputs, runtime):
        return runtime.reference(inputs)

    communicator = runtime.tp_group.pynccl_comm
    with communicator.change_state(enable=True):
        communicator.all_gather(inputs["output"], inputs["local"])
    return inputs["output"]
