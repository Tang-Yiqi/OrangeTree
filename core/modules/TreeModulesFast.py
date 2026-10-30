from math import e, ceil
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os
import datetime
from core.tree.SegmentTreeEncoderFast import CMultiBranchSegmentTreeLinearUp
from core.tree.PascalWeighter import CPascalWeighter as COriginalPascalWeighter
from core.tree.PascalWeighterFast import CPascalWeighter as CFastPascalWeighter
from core.modules.TreeModules import CTSFModules as COriginalTSFModules
import random
from collections import deque 

_RANGE_INDEX_CACHE = {}
_FLAT_SAMPLE_INDEX_CACHE = {}



class CTSFModules(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.ChannelNum = configs['dataset']['ChannelNum']
        self.PatchLen = configs['model']['PatchLen']
        self.HiddenSize = configs['model']['HiddenSize']


        self.PatchStride = configs['model']['PatchStride']


        self.sigmoid = configs['model']['sigmoid']
        self.avg = configs['model']['avg']

        self.TopK = configs['model']['TopK']
        self.TopKMode = configs['model']['TopKMode']
        self.KRate = configs['model']['KRate']
        self.FixedK = configs['model']['FixedK']

        self.BranchNum = configs['model']['BranchNum']

        self.RangeAdditionalWeighter = configs['model']['RangeAdditionalWeighter']
        self.RangeAdditionalWeighterType = configs['model']['RangeAdditionalWeighterType']

        self.RangePositionEncode = configs['model']['RangePositionEncoding']
        self.RangePositionEncodingFor = configs['model']['RangePositionEncodingFor']

        self.device = configs['device']

        self.WeighterHiddenSize = configs['model']['WeighterHiddenSize']
        self.TF32 = configs['model'].get('TF32', False)
        self.ReuseChannelEmbedding = configs['model'].get('ReuseChannelEmbedding', True)
        self.OptimizeTreeBuild = configs['model'].get('OptimizeTreeBuild', True)
        self.OptimizeRangeScorer = configs['model'].get('OptimizeRangeScorer', True)
        self.OptimizeRangeExtraction = configs['model'].get('OptimizeRangeExtraction', True)

        self.RangeGateType = configs['model']['RangeGateType']

        self.SelectedRangeIndices = None
        self.SelectedRangeVotes = None
        self.SelectedVoteDenominator = None
        

        self.EmbeddingMethod = configs['model']['EmbeddingMethod']

        self.TreeType = configs['model']['TreeType']

        self.residual = configs['model']['residual']
        self.SamplingNum = configs['model']['SamplingNum']



        if self.EmbeddingMethod == 'CNN':
            self.PatchCNN = nn.Sequential(
                nn.Conv1d(in_channels=self.ChannelNum, out_channels=64, kernel_size=3, stride=1, padding=1),
                nn.Conv1d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1),
                nn.AdaptiveAvgPool1d(1), 
                nn.Flatten(),
                nn.Linear(32, self.HiddenSize)
            )
        elif self.EmbeddingMethod == 'ACNN':
            self.PatchCNN = nn.Sequential(
                nn.Conv1d(in_channels=self.ChannelNum, out_channels=64, kernel_size=3, stride=1, padding=1),
                nn.GELU(),
                nn.Dropout(0.5),
                nn.Conv1d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1),
                nn.GELU(),
                nn.Dropout(0.5),
                nn.AdaptiveAvgPool1d(1), 
                nn.Flatten(),
                nn.Linear(32, self.HiddenSize)
            )
        elif self.EmbeddingMethod == 'Linear':
            self.PatchEmbedding = nn.Linear(in_features=self.PatchLen * self.ChannelNum, out_features=self.HiddenSize)
        elif self.EmbeddingMethod == 'MLP':
            self.PatchEmbedding = nn.Sequential(
                nn.Linear(in_features=self.PatchLen * self.ChannelNum, out_features=self.HiddenSize * 2),
                nn.GELU(),
                nn.Dropout(0.5),
                nn.Linear(in_features=self.HiddenSize * 2, out_features=self.HiddenSize),
                nn.GELU(),
                nn.Dropout(0.5),
                nn.Linear(in_features=self.HiddenSize, out_features=self.HiddenSize)
            )


        self.SegmentTree = None

        if self.TreeType == 'LinearUp':
            self.UpFunction = nn.Linear(in_features=self.HiddenSize * self.BranchNum, out_features=self.HiddenSize)
        elif self.TreeType == 'MLPUp':
            self.UpFunction = nn.Sequential(
                nn.Linear(in_features=self.HiddenSize * self.BranchNum, out_features=self.HiddenSize * self.BranchNum * 2),
                nn.GELU(),
                nn.Dropout(0.5),
                nn.Linear(in_features=self.HiddenSize * self.BranchNum * 2, out_features=self.HiddenSize * self.BranchNum),
                nn.GELU(),
                nn.Dropout(0.5),
                nn.Linear(in_features=self.HiddenSize * self.BranchNum, out_features=self.HiddenSize)
            )
        elif self.TreeType == 'CNNUp':
            self.UpFunction = nn.Sequential(
                nn.Unflatten(1, (self.HiddenSize, self.BranchNum)),
                nn.Conv1d(in_channels=self.HiddenSize, out_channels= 64 * self.BranchNum, kernel_size=3, padding=1),
                nn.Conv1d(in_channels=64 * self.BranchNum, out_channels=32 * self.BranchNum, kernel_size=3, padding=1),
                nn.AdaptiveAvgPool1d(1),
                nn.Flatten(),
                nn.Linear(32 * self.BranchNum, self.HiddenSize)
            )
        elif self.TreeType == 'ACNNUp':
            self.UpFunction = nn.Sequential(
                nn.Unflatten(1, (self.HiddenSize, self.BranchNum)),
                nn.Conv1d(in_channels=self.HiddenSize, out_channels= 64 * self.BranchNum, kernel_size=3, padding=1),
                nn.GELU(),
                nn.Dropout(0.5),
                nn.Conv1d(in_channels=64 * self.BranchNum, out_channels=32 * self.BranchNum, kernel_size=3, padding=1),
                nn.GELU(),
                nn.Dropout(0.5),
                nn.AdaptiveAvgPool1d(1),
                nn.Flatten(),
                nn.Linear(32 * self.BranchNum, self.HiddenSize)
            )

        self.ChannelIndependentLinearEmbedding = nn.Linear(in_features=self.SamplingNum, out_features=self.HiddenSize)


        self.PascalWeighter = None
        self.HiddenCompressLinear = nn.Linear(in_features=self.HiddenSize, out_features=self.WeighterHiddenSize)
        self.PascalHiddenProjector = nn.Linear(in_features=self.WeighterHiddenSize, out_features=1)
        self.PascalDownLinear = nn.Linear(in_features=self.WeighterHiddenSize * 2, out_features=self.WeighterHiddenSize)



        if self.RangeAdditionalWeighter == True:
            if self.RangeAdditionalWeighterType == 'Linear':
                self.RangeOutputWeighter = nn.Linear(in_features=self.HiddenSize, out_features=self.HiddenSize)
            elif self.RangeAdditionalWeighterType == 'MLP':
                self.RangeOutputWeighter = nn.Sequential(
                    nn.Linear(in_features=self.HiddenSize, out_features=self.HiddenSize * 2),
                    nn.GELU(),
                    nn.Dropout(0.5),
                    nn.Linear(in_features=self.HiddenSize * 2, out_features=self.HiddenSize),
                    nn.GELU(),
                    nn.Dropout(0.5),
                    nn.Linear(in_features=self.HiddenSize, out_features=self.HiddenSize)
                )
            elif self.RangeAdditionalWeighterType == 'Length':
                self.LongRangeWeighter = nn.Linear(in_features=self.HiddenSize, out_features=self.HiddenSize)
                self.MiddleRangeWeighter = nn.Linear(in_features=self.HiddenSize, out_features=self.HiddenSize)
                self.ShortRangeWeighter = nn.Linear(in_features=self.HiddenSize, out_features=self.HiddenSize)
            elif self.RangeAdditionalWeighterType == 'Position':
                self.EarlyRangeWeighter = nn.Linear(in_features=self.HiddenSize, out_features=self.HiddenSize)
                self.MiddleRangeWeighter = nn.Linear(in_features=self.HiddenSize, out_features=self.HiddenSize)
                self.LateRangeWeighter = nn.Linear(in_features=self.HiddenSize, out_features=self.HiddenSize)
            elif self.RangeAdditionalWeighterType == 'LengthPosition':
                self.LongRangeWeighter = nn.Linear(in_features=self.HiddenSize, out_features=self.HiddenSize)

                self.MiddleEarlyRangeWeighter = nn.Linear(in_features=self.HiddenSize, out_features=self.HiddenSize)
                self.MiddleLateRangeWeighter = nn.Linear(in_features=self.HiddenSize, out_features=self.HiddenSize)

                self.ShortEarlyRangeWeighter = nn.Linear(in_features=self.HiddenSize, out_features=self.HiddenSize)
                self.ShortMiddleRangeWeighter = nn.Linear(in_features=self.HiddenSize, out_features=self.HiddenSize)
                self.ShortLateRangeWeighter = nn.Linear(in_features=self.HiddenSize, out_features=self.HiddenSize)


    def embedding(self, patches):
        BatchSize, PatchNum, PatchLen, ChannelNum = patches.shape
        
        if self.EmbeddingMethod == 'CNN' or self.EmbeddingMethod == 'ACNN':
            ReshapedPatches = patches.reshape(-1, ChannelNum, PatchLen)
            
            PatchFeatures = self.PatchCNN(ReshapedPatches)
        elif self.EmbeddingMethod == 'Linear' or self.EmbeddingMethod == 'MLP':
            ReshapedPatches = patches.reshape(-1, PatchLen * ChannelNum)
            PatchFeatures = self.PatchEmbedding(ReshapedPatches)
            
        PatchFeatures = PatchFeatures.reshape(BatchSize, PatchNum, self.HiddenSize)
        return PatchFeatures
    
    def PositionEncoding(self, PatchFeatures):
        BatchSize, PatchNum, HiddenSize = PatchFeatures.shape
        position = torch.arange(0, PatchNum, device=self.device).unsqueeze(1)
        DivTerm = torch.exp(torch.arange(0, HiddenSize, 2, device=self.device) * -(np.log(100.0) / HiddenSize))

        PositionEncoding = torch.zeros(BatchSize, PatchNum, HiddenSize, device=self.device)
        PositionEncoding[:, :, 0::2] = torch.sin(position * DivTerm)
        PositionEncoding[:, :, 1::2] = torch.cos(position * DivTerm)

        PositionEncodedPatchFeatures = PatchFeatures + PositionEncoding

        return PositionEncodedPatchFeatures

    def TreeInsert(self, PatchFeatures):
        if not self.OptimizeTreeBuild and not self.OptimizeRangeExtraction:
            return COriginalTSFModules.TreeInsert(self, PatchFeatures)

        if self.TreeType == 'LinearUp' or self.TreeType == 'MLPUp' or self.TreeType == 'CNNUp' or self.TreeType == 'ACNNUp':
            self.SegmentTree = CMultiBranchSegmentTreeLinearUp(BranchNum=self.BranchNum, UpLinear=self.UpFunction)

        if self.OptimizeTreeBuild:
            self.SegmentTree.build(PatchFeatures)
        else:
            for PatchIndex in range(PatchFeatures.shape[1]):
                self.SegmentTree.insert(PatchFeatures[:, PatchIndex, :])

    
    def PascalWeight(self, SampledBatch, InputLen, ChannelIndependentFeatures=None):
        if self.OptimizeRangeScorer:
            self.PascalWeighter = CFastPascalWeighter(HiddenSize=self.HiddenSize, WeighterHiddenSize=self.WeighterHiddenSize, AllowTF32=getattr(self, 'TF32', False))
        else:
            self.PascalWeighter = COriginalPascalWeighter(HiddenSize=self.HiddenSize, WeighterHiddenSize=self.WeighterHiddenSize)

        if ChannelIndependentFeatures is None:
            ChannelIndependentFeatures = self.ChannelIndependentLinearEmbedding(SampledBatch.permute(0, 2, 1))

        VoteNum = self.PascalWeighter.GetWeights(ChannelIndependentFeatures, self.HiddenCompressLinear, self.PascalDownLinear, self.PascalHiddenProjector, InputLen)

        if self.OptimizeRangeScorer:
            VoteNum = VoteNum.abs_()
        else:
            VoteNum = torch.abs(VoteNum)
        FinalVotes = torch.mean(VoteNum, dim=1)
        return FinalVotes
    
    def TopKMask(self, FinalVotes):
        if not self.OptimizeRangeExtraction:
            return COriginalTSFModules.TopKMask(self, FinalVotes)

        BatchSize, RangeNum = FinalVotes.shape
        if self.TopKMode == 'fixed':
            k = self.FixedK
        elif self.TopKMode == 'rate':
            k = ceil(RangeNum * self.KRate)
        
        RangeScores = FinalVotes
        
        TopValues, TopIndices = torch.topk(RangeScores, k, dim=1)
        self.SelectedRangeIndices = TopIndices

        if self.RangeGateType == 'votes' and self.sigmoid == False:
            SelectedVotes = TopValues
            if self.avg == True:
                SelectedVotes = SelectedVotes / torch.sum(SelectedVotes, dim=1, keepdim=True)
            SelectedSoftMask = torch.ones_like(SelectedVotes) + (SelectedVotes - SelectedVotes.detach())
            self.SelectedRangeVotes = SelectedVotes
            self.SelectedVoteDenominator = torch.sum(SelectedVotes * SelectedSoftMask, dim=1, keepdim=True)
            return None, None

        self.SelectedRangeVotes = None
        
        mask = torch.zeros(BatchSize, RangeNum, dtype=torch.float32, device=self.device)
        mask.scatter_(1, TopIndices, 1.0)

        FinalVotes = FinalVotes * mask

        if self.avg == True:
            FinalVotes = FinalVotes / torch.sum(FinalVotes, dim=1, keepdim=True)

        if self.sigmoid == True:
            FinalVotes = torch.sigmoid(FinalVotes)

        SoftMask = mask + (FinalVotes - FinalVotes.detach())
        self.SelectedVoteDenominator = None
        return SoftMask, FinalVotes

    def RangePositionEncoding(self, RangeFrom, RangeTo):

        center = (RangeFrom + RangeTo - 1) / 2
        length = RangeTo - RangeFrom + 1

        CenterPos = torch.tensor([[center]], dtype=torch.float32, device=self.device)
        LengthPos = torch.tensor([[length]], dtype=torch.float32, device=self.device)

        CenterDivTerm = torch.exp(torch.arange(0, self.HiddenSize, 2, device=self.device) * -(np.log(200.0) / self.HiddenSize))
        LengthDivTerm = torch.exp(torch.arange(0, self.HiddenSize, 2, device=self.device) * -(np.log(100.0) / self.HiddenSize))

        CenterRangePositionEncoding = torch.zeros(1, self.HiddenSize, device=self.device)
        CenterRangePositionEncoding[:, 0::2] = torch.sin(CenterPos * CenterDivTerm)
        CenterRangePositionEncoding[:, 1::2] = torch.cos(CenterPos * CenterDivTerm)


        LengthRangePositionEncoding = torch.zeros(1, self.HiddenSize, device=self.device)
        LengthRangePositionEncoding[:, 0::2] = torch.sin(LengthPos * LengthDivTerm)
        LengthRangePositionEncoding[:, 1::2] = torch.cos(LengthPos * LengthDivTerm)

        return CenterRangePositionEncoding + LengthRangePositionEncoding

    def RangeAdditionalWeight(self, RangeFeature, RangeFrom, RangeTo, InputLen):
        if self.RangeAdditionalWeighterType == 'Linear' or self.RangeAdditionalWeighterType == 'MLP':
            RangeFeature = self.RangeOutputWeighter(RangeFeature)
        else:
            RangeLength = RangeTo - RangeFrom + 1
            RangeCenter = (RangeFrom + RangeTo) // 2

            OneThirdLen = InputLen // 3
            TwoThirdLen = 2 * OneThirdLen

            OneThirdPos = InputLen // 3
            TwoThirdPos = 2 * OneThirdPos
            if self.RangeAdditionalWeighterType == 'Length':
                if RangeLength > TwoThirdLen:
                    RangeFeature = self.LongRangeWeighter(RangeFeature)
                elif RangeLength > OneThirdLen:
                    RangeFeature = self.MiddleRangeWeighter(RangeFeature)
                else:
                    RangeFeature = self.ShortRangeWeighter(RangeFeature)
            elif self.RangeAdditionalWeighterType == 'Position':
                if RangeCenter > TwoThirdPos:
                    RangeFeature = self.LateRangeWeighter(RangeFeature)
                elif RangeCenter > OneThirdPos:
                    RangeFeature = self.MiddleRangeWeighter(RangeFeature)
                else:
                    RangeFeature = self.EarlyRangeWeighter(RangeFeature)
            elif self.RangeAdditionalWeighterType == 'LengthPosition':
                if RangeLength > TwoThirdLen:
                    RangeFeature = self.LongRangeWeighter(RangeFeature)
                elif RangeLength > OneThirdLen:
                    if RangeCenter > TwoThirdPos:
                        RangeFeature = self.MiddleLateRangeWeighter(RangeFeature)
                    else:
                        RangeFeature = self.MiddleEarlyRangeWeighter(RangeFeature)
                else:
                    if RangeCenter > TwoThirdPos:
                        RangeFeature = self.ShortLateRangeWeighter(RangeFeature)
                    elif RangeCenter > OneThirdPos:
                        RangeFeature = self.ShortMiddleRangeWeighter(RangeFeature)
                    else:
                        RangeFeature = self.ShortEarlyRangeWeighter(RangeFeature)
        return RangeFeature

    def AllRangeFeature(self, mask, FinalVotes, PatchNum):
        if not self.OptimizeRangeExtraction:
            return COriginalTSFModules.AllRangeFeature(self, mask, FinalVotes, PatchNum)

        if not hasattr(self, 'SelectedRangeIndices'):
            self.SelectedRangeIndices = None
        if self.TopK != True or self.SelectedRangeIndices is None:
            raise RuntimeError("AllRangeFeature requires Top-K range indices")

        SelectedRangeIndices = self.SelectedRangeIndices.detach()
        self.SelectedRangeIndices = None
        BatchSize, SelectedRangeNum = SelectedRangeIndices.shape

        FlatRanges = SelectedRangeIndices.reshape(-1)
        SampleCacheKey = (BatchSize, SelectedRangeNum, FlatRanges.device.type, FlatRanges.device.index)
        if SampleCacheKey not in _FLAT_SAMPLE_INDEX_CACHE:
            _FLAT_SAMPLE_INDEX_CACHE[SampleCacheKey] = torch.arange(BatchSize, device=FlatRanges.device).repeat_interleave(SelectedRangeNum)
        FlatSamples = _FLAT_SAMPLE_INDEX_CACHE[SampleCacheKey]
        CacheKey = (PatchNum, FlatRanges.device.type, FlatRanges.device.index)
        if CacheKey not in _RANGE_INDEX_CACHE:
            RangeBegins = []
            RangeEnds = []
            for RangeLength in range(PatchNum, 0, -1):
                for RangeBegin in range(1, PatchNum - RangeLength + 2):
                    RangeBegins.append(RangeBegin)
                    RangeEnds.append(RangeBegin + RangeLength - 1)
            _RANGE_INDEX_CACHE[CacheKey] = (
                torch.tensor(RangeBegins, device=FlatRanges.device, dtype=torch.long),
                torch.tensor(RangeEnds, device=FlatRanges.device, dtype=torch.long),
            )
        RangeBeginLookup, RangeEndLookup = _RANGE_INDEX_CACHE[CacheKey]
        SelectedRangeBegins = RangeBeginLookup.index_select(0, FlatRanges)
        SelectedRangeEnds = RangeEndLookup.index_select(0, FlatRanges)
        RangeFeature = self.SegmentTree.GetRangeFeatures(SelectedRangeBegins, SelectedRangeEnds, FlatSamples)

        Ranges = None
        if self.RangePositionEncode == True or (self.RangeAdditionalWeighter == True and self.RangeAdditionalWeighterType not in ('Linear', 'MLP')):
            Ranges = list(zip(SelectedRangeBegins.tolist(), SelectedRangeEnds.tolist()))

        if self.RangePositionEncode == True and self.RangePositionEncodingFor in ('topk', 'all'):
            PositionFeatures = torch.cat([self.RangePositionEncoding(RangeFrom, RangeTo) for RangeFrom, RangeTo in Ranges], dim=0)
            RangeFeature = RangeFeature + PositionFeatures

        DelayLinearWeighting = self.RangeAdditionalWeighter == True and self.RangeAdditionalWeighterType == 'Linear'
        if self.RangeAdditionalWeighter == True and not DelayLinearWeighting:
            if self.RangeAdditionalWeighterType in ('Linear', 'MLP'):
                RangeFeature = self.RangeOutputWeighter(RangeFeature)
            else:
                RangeFeature = torch.stack(
                    [
                        self.RangeAdditionalWeight(RangeFeature[Index : Index + 1], RangeFrom, RangeTo, PatchNum)[0]
                        for Index, (RangeFrom, RangeTo) in enumerate(Ranges)
                    ],
                    dim=0,
                )

        if self.RangeGateType == 'votes':
            SelectedRangeVotes = getattr(self, 'SelectedRangeVotes', None)
            if SelectedRangeVotes is None:
                SelectedVotes = FinalVotes[FlatSamples, FlatRanges].unsqueeze(-1)
            else:
                SelectedVotes = SelectedRangeVotes.reshape(-1, 1)
                self.SelectedRangeVotes = None
            RangeFeature = RangeFeature * SelectedVotes
            BiasScales = SelectedVotes
        else:
            BiasScales = None

        BatchFeatures = RangeFeature.reshape(BatchSize, SelectedRangeNum, self.HiddenSize).sum(dim=1)

        if DelayLinearWeighting:
            BatchFeatures = F.linear(BatchFeatures, self.RangeOutputWeighter.weight, bias=None)
            if BiasScales is None:
                BatchBiasScales = BatchFeatures.new_full((BatchSize, 1), SelectedRangeNum)
            else:
                BatchBiasScales = BiasScales.to(BatchFeatures.dtype).reshape(BatchSize, SelectedRangeNum, 1).sum(dim=1)
            BatchFeatures = BatchFeatures + BatchBiasScales * self.RangeOutputWeighter.bias

        return BatchFeatures

    
    def gating(self, BatchFeatures, mask, FinalVotes):            
        if not self.OptimizeRangeExtraction:
            return COriginalTSFModules.gating(self, BatchFeatures, mask, FinalVotes)

        if self.RangeGateType == 'votes':
            if self.SelectedVoteDenominator is not None:
                VoteDenominator = self.SelectedVoteDenominator
                self.SelectedVoteDenominator = None
            else:
                MaskedVotes = FinalVotes.unsqueeze(-1)
                MaskedVotes = MaskedVotes * mask.unsqueeze(-1).float()
                VoteDenominator = torch.sum(MaskedVotes, dim=1)

            BatchFeatures = BatchFeatures / (VoteDenominator + 1e-8)
        elif self.RangeGateType == 'bool':
            BatchFeatures = BatchFeatures / torch.sum(mask, dim=1, keepdim=True)

        return BatchFeatures


        
