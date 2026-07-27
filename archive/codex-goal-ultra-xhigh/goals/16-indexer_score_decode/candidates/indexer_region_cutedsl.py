"""Use CuTe-DSL for score/top-k inside the complete indexer regions."""

CANDIDATE_METADATA = {
    "backend": "cutedsl",
    "scope": "complete indexer and selected TRT-LLM DSA containing regions",
    "fallback": "external experiment only; stock DeepGEMM remains active",
}


def run(inputs, runtime):
    return runtime.run_indexer_containing_region(inputs, backend="cutedsl")
