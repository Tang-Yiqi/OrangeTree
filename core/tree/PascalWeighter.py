import torch
import math
from collections import deque 
import gc

class CWeight:
    def __init__(self, RangeFrom, RangeTo):
        self.RangeFrom = RangeFrom
        self.RangeTo = RangeTo
        self.feature = None
        self.weight = None

class CPascalWeighter:
    def __init__(self, HiddenSize, WeighterHiddenSize):
        self.HiddenSize = HiddenSize
        self.WeighterHiddenSize = WeighterHiddenSize

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
        
            
    def GetWeights(self, feature, HiddenProjector, DownLinear, WeighterProjector, InputLen):
        self.HiddenProjector = HiddenProjector
        self.DownLinear = DownLinear
        self.WeighterProjector = WeighterProjector

        self.InputLen = InputLen
        self.weights = [[CWeight(i, j) for j in range(self.InputLen + 1)] for i in range(self.InputLen + 1)]

        self.device = feature.device

        self.BatchSize, self.ChannelNum, self.HiddenSize = feature.shape

        self.ZeroFeature = torch.zeros(self.BatchSize, self.ChannelNum, self.WeighterHiddenSize, device=self.device)

        
        ReshapedFeature = feature.view(-1, self.HiddenSize)
        ProjectedFeature = self.HiddenProjector(ReshapedFeature)
        self.weights[1][self.InputLen].feature = ProjectedFeature.view(self.BatchSize, self.ChannelNum, self.WeighterHiddenSize)

        RangeNum = (self.InputLen * (self.InputLen + 1)) // 2
        WeightList = torch.zeros(self.BatchSize, self.ChannelNum, RangeNum, device=self.device)





        


        for LayerIndex in range(1, self.InputLen + 1):
            RangeLength = self.InputLen - LayerIndex + 1
            for RangeFrom in range(1, self.InputLen - RangeLength + 2):
                RangeTo = RangeFrom + RangeLength - 1

                self.BuildPascal(RangeFrom, RangeTo)

                NodeFeature = self.weights[RangeFrom][RangeTo].feature
                ReshapedNodeFeature = NodeFeature.view(-1, self.WeighterHiddenSize)
                WeightValue = self.WeighterProjector(ReshapedNodeFeature)

                index = self.GetIndexFromRange(RangeFrom, RangeTo)
                WeightList[:, :, index] = WeightValue.view(self.BatchSize, self.ChannelNum)

            if LayerIndex > 1:
                PreviousLayerIndex = LayerIndex - 1
                PreviousRangeLength = self.InputLen - PreviousLayerIndex + 1
                for RangeFrom in range(1, self.InputLen - PreviousRangeLength + 2):
                    RangeTo = RangeFrom + PreviousRangeLength - 1
                    self.weights[RangeFrom][RangeTo].feature = None
                
                
        del self.weights

        return WeightList
    




        
