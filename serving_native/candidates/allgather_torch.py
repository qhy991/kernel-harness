"""Example candidate that compares raw NCCL against SGLang's group wrapper."""

import torch


def run(inputs, runtime):
    output = torch.empty_like(inputs["output"])
    torch.distributed.all_gather_into_tensor(
        output, inputs["local"], group=runtime.device_group
    )
    return output
