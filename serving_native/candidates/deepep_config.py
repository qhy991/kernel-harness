"""DeepEP config-tuning template.

Edit these fixed values and pass this file to a normal dispatch/combine task.
The runner constructs ``deep_ep.Config`` and keeps the rest of the SGLang ABI
unchanged.  Low-latency DeepEP does not consume this normal-mode Config.
"""


CANDIDATE_API = "reference_with_config_v1"
CANDIDATE_CONFIG = {
    "num_sms": 24,
    "num_max_nvl_chunked_send_tokens": 8,
    "num_max_nvl_chunked_recv_tokens": 512,
    "num_max_rdma_chunked_send_tokens": 16,
    "num_max_rdma_chunked_recv_tokens": 128,
}

CANDIDATE_IDENTITY = "DeepEP production call with explicit candidate Config"
