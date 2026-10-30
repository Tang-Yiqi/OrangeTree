import torch
import torch.nn as nn

class CRevIN(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.ChannelNum = configs['dataset']['ChannelNum']

        self.k = nn.Parameter(torch.ones(1, 1, self.ChannelNum))
        self.b = nn.Parameter(torch.zeros(1, 1, self.ChannelNum))

    def norm(self, sequence, mean=None, std=None, eps=1):
        if mean is None or std is None:
            mean = sequence.mean(dim=1, keepdim=True)
            var = sequence.var(dim=1, unbiased=False, keepdim=True)
            var = torch.clamp(var, min=eps)
            std = torch.sqrt(var)
        
        NormSequence = (sequence - mean) / std
        
        return NormSequence, mean, std

    def UnNorm(self, NormSequence, mean, std):
        sequence = NormSequence * std + mean
        return sequence

    def affine(self, NormSequence):
        NormSequence = NormSequence * self.k + self.b
        return NormSequence

    def UnAffine(self, NormSequence):
        result = (NormSequence - self.b) / self.k
        return result

class CLayerNorm(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.ChannelNum = configs['dataset']['ChannelNum']
        
        self.gamma = nn.Parameter(torch.ones(1, 1, self.ChannelNum))
        self.beta = nn.Parameter(torch.zeros(1, 1, self.ChannelNum))

    def norm(self, sequence, eps=1):
        mean = sequence.mean(dim=-1, keepdim=True)
        var = sequence.var(dim=-1, unbiased=False, keepdim=True)
        var = torch.clamp(var, min=eps)
        std = torch.sqrt(var)
        
        NormSequence = (sequence - mean) / std
        
        return NormSequence, mean, std

    def UnNorm(self, NormSequence, mean, std):
        sequence = NormSequence * std + mean
        return sequence

    def affine(self, NormSequence):
        NormSequence = NormSequence * self.gamma + self.beta
        return NormSequence

    def UnAffine(self, NormSequence):
        result = (NormSequence - self.beta) / self.gamma
        return result

    def forward(self, sequence):
        norm_seq, mean, std = self.norm(sequence)
        
        output = self.affine(norm_seq)
        
        return output, mean, std

    def inverse(self, output, mean, std):
        norm_seq = self.UnAffine(output)
        
        sequence = self.UnNorm(norm_seq, mean, std)
        
        return sequence
