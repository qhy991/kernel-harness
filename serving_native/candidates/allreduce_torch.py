"""Compare raw torch/NCCL AllReduce with SGLang's production coordinator."""

import torch


def run(inputs, runtime):
    torch.distributed.all_reduce(inputs["local"], group=runtime.device_group)
    return inputs["local"]
