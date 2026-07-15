"""GLM-5.2 DSA sparse prefill attention — SGLang FlashMLA baseline.

This is the B200 sparse-prefill kernel used by SGLang's DSA backend:
`flash_mla_sparse_fwd`.  Under TP8 GLM-5.2 has 8 local query heads, while the
Blackwell kernel requires 128 heads.  SGLang pads to 128 before dispatch and
trims the output afterward; the task mirrors that contract while keeping the
padding allocation out of the timed region.

Workloads: query tokens M in {1024, 2048, 4096}, shared BF16 latent KV length
65536, selected top-k 2048, latent head dim 576, output dim 512.
"""

import torch
from sgl_kernel.flash_mla import flash_mla_sparse_fwd

LOCAL_HEADS = 8
PADDED_HEADS = 128
HEAD_DIM = 576
VALUE_DIM = 512
TOPK = 2048


def get_inputs(axes_and_scalars: dict, device: torch.device) -> dict:
    m = axes_and_scalars["M"]
    s_kv = axes_and_scalars["ctx"]

    # SGLang's B200 path pads TP-local heads from 8 to 128 before dispatch.
    q = torch.zeros(
        m, PADDED_HEADS, HEAD_DIM, device=device, dtype=torch.bfloat16
    )
    q[:, :LOCAL_HEADS].normal_()
    kv_cache = torch.randn(s_kv, 1, HEAD_DIM, device=device, dtype=torch.bfloat16)

    # Each row selects TOPK unique positions without constructing M randperms.
    offsets = torch.randint(
        0, s_kv, (m, 1, 1), device=device, dtype=torch.int32
    )
    selected = torch.arange(TOPK, device=device, dtype=torch.int32).view(1, 1, -1)
    indices = (selected + offsets).remainder(s_kv).contiguous()

    return {"q": q, "kv_cache": kv_cache, "indices": indices}


@torch.no_grad()
def run(q, kv_cache, indices):
    out, _, _ = flash_mla_sparse_fwd(
        q=q,
        kv=kv_cache,
        indices=indices,
        sm_scale=HEAD_DIM**-0.5,
        d_v=VALUE_DIM,
    )
    return out[:, :LOCAL_HEADS, :]
