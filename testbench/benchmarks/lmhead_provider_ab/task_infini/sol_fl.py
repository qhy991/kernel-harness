import torch, torch.nn.functional as F
@torch.no_grad()
def run(hidden_states, lm_head_weight):
    return F.linear(hidden_states.bfloat16(), lm_head_weight.bfloat16())
