"""Graph-safe PyNCCL grouped-broadcast AllGather experiment.

PyNCCL implements the ``sizes=`` path as one grouped ``ncclBroadcast`` per
rank.  Equal sizes retain the same rank-ordered, caller-owned output ABI as
``ncclAllGather`` without allocating an adapter buffer.
"""

BACKEND = "pynccl_grouped_nccl_broadcast"
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
    local_rows = inputs["local"].shape[0]
    sizes = [local_rows] * runtime.world_size
    with communicator.change_state(enable=True):
        communicator.all_gather(
            inputs["output"],
            inputs["local"],
            sizes=sizes,
        )
    return inputs["output"]
