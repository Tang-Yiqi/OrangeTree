import torch
def norm(sequence, mean = None, std = None, eps=1e-8):
    if mean is None or std is None:
        mean = sequence.mean(dim=0, keepdim=True)
        var  = sequence.var(dim=0, unbiased = False, keepdim=True)
        var = torch.clamp(var, min=eps)
        std = torch.sqrt(var)

    NormSequence = (sequence - mean) / std

    return NormSequence, mean, std
