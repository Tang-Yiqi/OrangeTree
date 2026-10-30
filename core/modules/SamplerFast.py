import torch


def sampler(InputBatch, TimeStepNum, SamplingNum):
    if TimeStepNum == SamplingNum:
        return InputBatch

    SamplingPositions = torch.linspace(0, TimeStepNum - 1, SamplingNum, device=InputBatch.device)
    FloorIndices = SamplingPositions.floor().long().clamp_(0, TimeStepNum - 1)
    CeilIndices = SamplingPositions.ceil().long().clamp_(0, TimeStepNum - 1)
    Weights = (SamplingPositions - FloorIndices).to(InputBatch.dtype).view(1, -1, 1)

    FloorValues = InputBatch.index_select(1, FloorIndices)
    CeilValues = InputBatch.index_select(1, CeilIndices)
    return (1 - Weights) * FloorValues + Weights * CeilValues
