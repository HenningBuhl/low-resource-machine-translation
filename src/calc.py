import torch
from torch.functional import F


def top_k_top_p_filtering(logits, top_k=0, top_p=1.0, filter_value=-float('inf')):
    '''Applies top-K and top-p (nucleus) filtering to the logits.'''

    assert logits.dim() == 1  # batch size 1 for now - could be updated for more but the code would be less clear.
    top_k = min(top_k, logits.size(-1))  # Safety check.
    if top_k > 0:
        # Remove all tokens with a probability less than the last token of the top-k.
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold.
        sorted_indices_to_remove = cumulative_probs > top_p

        # Shift the indices to the right to keep also the first token above the threshold.
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = filter_value
    return logits
