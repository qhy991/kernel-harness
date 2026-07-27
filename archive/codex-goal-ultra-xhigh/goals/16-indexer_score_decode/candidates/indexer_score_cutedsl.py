"""Use SGLang's SM100 CuTe-DSL paged-MQA score backend."""

CANDIDATE_METADATA = {
    "backend": "cutedsl",
    "mode": "decode_next_n_1",
    "fallback": "candidate is external; stock DeepGEMM remains enabled",
}


def run(inputs, runtime):
    return runtime.run_indexer_score_topk(inputs, backend="cutedsl")
