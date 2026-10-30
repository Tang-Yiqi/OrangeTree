import torch
import torch.nn.functional as F
import math
from collections import OrderedDict

_RANGE_AFFINE_CACHE = OrderedDict()


def _BuildRangeAffineProjectorFunctional(DownWeight, DownBias, ProjectorWeight, ProjectorBias, InputLen, HiddenSize):
    PreviousLayer = torch.zeros(HiddenSize + 1, HiddenSize, 1, device=DownWeight.device, dtype=DownWeight.dtype)
    PreviousLayer[:HiddenSize, :, 0] = torch.eye(HiddenSize, device=DownWeight.device, dtype=DownWeight.dtype)
    BiasMask = torch.zeros(HiddenSize + 1, 1, 1, device=DownWeight.device, dtype=DownWeight.dtype)
    BiasMask[-1] = 1
    ScoreBiasMask = BiasMask.squeeze(-1)
    ConvolutionKernel = DownWeight.reshape(HiddenSize, 2, HiddenSize).permute(0, 2, 1).contiguous()

    FeatureLayers = [PreviousLayer.permute(0, 2, 1)]
    for _ in range(2, InputLen + 1):
        PreviousLayer = F.conv1d(PreviousLayer, ConvolutionKernel, bias=None, padding=1) + BiasMask * DownBias.view(1, HiddenSize, 1)
        FeatureLayers.append(PreviousLayer.permute(0, 2, 1))

    LayerScores = F.linear(torch.cat(FeatureLayers, dim=1), ProjectorWeight, bias=None).squeeze(-1)
    LayerScores = LayerScores + ScoreBiasMask * ProjectorBias
    return LayerScores[:HiddenSize].transpose(0, 1).contiguous(), LayerScores[HiddenSize]


class CWeight:
    def __init__(self, RangeFrom, RangeTo):
        self.RangeFrom = RangeFrom
        self.RangeTo = RangeTo
        self.feature = None
        self.weight = None

class CPascalWeighter:
    def __init__(self, HiddenSize, WeighterHiddenSize, AllowTF32=False):
        self.HiddenSize = HiddenSize
        self.WeighterHiddenSize = WeighterHiddenSize
        self.AllowTF32 = AllowTF32

        self.HiddenProjector = None
        self.DownLinear = None
        self.WeighterProjector = None

        self.weights = None

        self.device = None
        
        self.InputLen = None
        self.BatchSize = None
        self.ChannelNum = None

        self.ZeroFeature = None

    def GetIndexFromRange(self, RangeFrom, RangeTo):
        RangeLen = RangeTo - RangeFrom + 1
        LayerIndex = self.InputLen - RangeLen + 1
        UpLayersLen = (LayerIndex * LayerIndex - LayerIndex) // 2
        index = UpLayersLen + RangeFrom - 1
        return index
    
    def GetRangeFromIndex(self, index):
        
        LayerIndex = int((1 + math.sqrt(1 + 8 * index)) / 2)
        
        UpLayersLen = (LayerIndex * LayerIndex - LayerIndex) // 2
        if UpLayersLen > index:
            LayerIndex -= 1
            UpLayersLen = (LayerIndex * LayerIndex - LayerIndex) // 2
        
        OffsetInLayer = index - UpLayersLen
        
        RangeFrom = OffsetInLayer + 1
        RangeLen = self.InputLen - LayerIndex + 1
        RangeTo = RangeFrom + RangeLen - 1
        
        return RangeFrom, RangeTo

    def BuildPascal(self, RangeFrom, RangeTo):
        if RangeTo - RangeFrom + 1 == self.InputLen:
            return
        
        UpLeft = CWeight(RangeFrom, RangeTo)
        UpRight = CWeight(RangeFrom, RangeTo)

        if RangeFrom - 1 < 1:
            UpLeft.feature = self.ZeroFeature
        else:
            UpLeft = self.weights[RangeFrom - 1][RangeTo]
            if UpLeft.feature is None:
                self.BuildPascal(RangeFrom - 1, RangeTo)

        if RangeTo + 1 > self.InputLen:
            UpRight.feature = self.ZeroFeature
        else:
            UpRight = self.weights[RangeFrom][RangeTo + 1]
            if UpRight.feature is None:
                self.BuildPascal(RangeFrom, RangeTo + 1)

        ConcatenatedFeature = torch.cat([UpLeft.feature, UpRight.feature], dim=-1)
        BatchSize, ChannelNum, FeatureDim = ConcatenatedFeature.shape
        
        ReshapedFeature = ConcatenatedFeature.view(-1, FeatureDim)
        TransformedFeature = self.DownLinear(ReshapedFeature)
        
        self.weights[RangeFrom][RangeTo].feature = TransformedFeature.view(BatchSize, ChannelNum, self.WeighterHiddenSize)
        
            
    def _BuildRangeAffineProjector(self, dtype):
        HiddenSize = self.WeighterHiddenSize
        ProjectorArguments = (
            self.DownLinear.weight,
            self.DownLinear.bias,
            self.WeighterProjector.weight,
            self.WeighterProjector.bias,
        )
        PreviousAllowTF32 = torch.backends.cudnn.allow_tf32
        torch.backends.cudnn.allow_tf32 = self.AllowTF32
        try:
            return _BuildRangeAffineProjectorFunctional(*ProjectorArguments, self.InputLen, HiddenSize)
        finally:
            torch.backends.cudnn.allow_tf32 = PreviousAllowTF32

    def GetWeights(self, feature, HiddenProjector, DownLinear, WeighterProjector, InputLen):
        if torch.is_grad_enabled() and feature.shape[0] * feature.shape[1] < 1024:
            return self._GetWeightsByFeatureRecurrence(feature, HiddenProjector, DownLinear, WeighterProjector, InputLen)

        self.HiddenProjector = HiddenProjector
        self.DownLinear = DownLinear
        self.WeighterProjector = WeighterProjector
        self.InputLen = InputLen
        self.device = feature.device
        self.BatchSize, self.ChannelNum, self.HiddenSize = feature.shape

        ProjectedFeature = self.HiddenProjector(feature)
        Cacheable = not torch.is_grad_enabled()
        CacheKey = (
            self.DownLinear.weight,
            self.DownLinear.weight._version,
            self.DownLinear.bias._version,
            self.WeighterProjector.weight,
            self.WeighterProjector.weight._version,
            self.WeighterProjector.bias._version,
            self.InputLen,
            self.device.type,
            self.device.index,
            ProjectedFeature.dtype,
        )

        if Cacheable and CacheKey in _RANGE_AFFINE_CACHE:
            ScoreWeights, ScoreBiases = _RANGE_AFFINE_CACHE.pop(CacheKey)
            _RANGE_AFFINE_CACHE[CacheKey] = (ScoreWeights, ScoreBiases)
        else:
            ScoreWeights, ScoreBiases = self._BuildRangeAffineProjector(ProjectedFeature.dtype)
            if Cacheable:
                _RANGE_AFFINE_CACHE[CacheKey] = (ScoreWeights, ScoreBiases)
                while len(_RANGE_AFFINE_CACHE) > 16:
                    _RANGE_AFFINE_CACHE.popitem(last=False)

        return F.linear(ProjectedFeature, ScoreWeights, ScoreBiases)

    def _GetWeightsByFeatureRecurrence(self, feature, HiddenProjector, DownLinear, WeighterProjector, InputLen):
        self.HiddenProjector = HiddenProjector
        self.DownLinear = DownLinear
        self.WeighterProjector = WeighterProjector

        self.InputLen = InputLen
        self.device = feature.device

        self.BatchSize, self.ChannelNum, self.HiddenSize = feature.shape

        PreviousLayer = self.HiddenProjector(feature).unsqueeze(2)
        PendingLayers = [PreviousLayer]
        WeightChunks = []
        ProjectorChunkSize = 16

        if self.InputLen == 1:
            return self.WeighterProjector(PreviousLayer).squeeze(-1)

        ConvolutionKernel = self.DownLinear.weight.reshape(self.WeighterHiddenSize, 2, self.WeighterHiddenSize).permute(0, 2, 1).contiguous()
        BatchChannelSize = self.BatchSize * self.ChannelNum

        PreviousAllowTF32 = torch.backends.cudnn.allow_tf32
        torch.backends.cudnn.allow_tf32 = self.AllowTF32
        try:
            for LayerIndex in range(2, self.InputLen + 1):
                CurrentLayer = F.conv1d(
                    PreviousLayer.permute(0, 1, 3, 2).reshape(BatchChannelSize, self.WeighterHiddenSize, -1),
                    ConvolutionKernel, self.DownLinear.bias, padding=1
                ).reshape(self.BatchSize, self.ChannelNum, self.WeighterHiddenSize, -1).permute(0, 1, 3, 2)
                PendingLayers.append(CurrentLayer)
                PreviousLayer = CurrentLayer

                if len(PendingLayers) == ProjectorChunkSize or LayerIndex == self.InputLen:
                    WeightChunks.append(self.WeighterProjector(torch.cat(PendingLayers, dim=2)).squeeze(-1))
                    PendingLayers.clear()
        finally:
            torch.backends.cudnn.allow_tf32 = PreviousAllowTF32

        return torch.cat(WeightChunks, dim=2)
