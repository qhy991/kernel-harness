import torch
_compiled = {}
@torch.no_grad()
def run(hidden_states, lm_head_weight):
    h = hidden_states.bfloat16(); w = lm_head_weight.bfloat16()
    M = h.shape[0]
    fn = _compiled.get(M)
    if fn is None:
        def _mm(a, b):
            return torch.matmul(a, b.t())
        fn = torch.compile(_mm, mode="max-autotune-no-cudagraphs", dynamic=False)
        # warm compile
        fn(h, w)
        _compiled[M] = fn
    return fn(h, w)
