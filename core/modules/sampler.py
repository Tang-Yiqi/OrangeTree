import torch
from collections import deque 

def sampler(InputBatch, TimeStepNum, SamplingNum):
    BatchSize, TotalTimeStepNum, ChannelNum = InputBatch.shape
    device = InputBatch.device
    
    SamplingIndices = torch.linspace(0, TimeStepNum - 1, SamplingNum, device=device)
    
    SampledBatch = torch.zeros(BatchSize, SamplingNum, ChannelNum, device=device, dtype=InputBatch.dtype)
    
    for SamplingIndex in range(SamplingNum):
        SamplingPosition = SamplingIndices[SamplingIndex]
        
        FloorIndex = torch.floor(SamplingPosition).long()
        CeilIndex = torch.ceil(SamplingPosition).long()
        Weight = SamplingPosition - FloorIndex.float()

        FloorIndex = torch.clamp(FloorIndex, min=0, max=TimeStepNum - 1)
        CeilIndex = torch.clamp(CeilIndex, min=0, max=TimeStepNum - 1)

        FloorValue = InputBatch[:, FloorIndex, :]
        CeilValue = InputBatch[:, CeilIndex, :]

        InterpolatedValue = (1 - Weight) * FloorValue + Weight * CeilValue
        SampledBatch[:, SamplingIndex, :] = InterpolatedValue
    
    return SampledBatch
    
    






        
