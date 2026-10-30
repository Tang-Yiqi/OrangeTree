import torch
import torch.nn as nn

from torch import ceil, floor, round

from math import e, log2
from core.modules.TreeModules import CTSFModules
from core.modules.normalization import CRevIN, CLayerNorm
from core.tree.decoder import CSegmentTreeDecoder
from core.tree.DecoderRes import CSegmentTreeDecoderRes
from core.modules.sampler import sampler
import copy

from core.modules.ChannelShrinker import GNNShrinker, CNNShrinker, SimpleCNNShrinker


class OrangeTree(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.ChannelNum = configs['dataset']['ChannelNum']

        self.device = configs['device']

        self.SamplingNum = configs['model']['SamplingNum']

        self.PatchLen = configs['model']['PatchLen']
        self.PatchStride = configs['model']['PatchStride']

        self.RevIN = CRevIN(configs)
        self.LayerNorm = CLayerNorm(configs)

        self.dropout = configs['model']['dropout']

        self.affine = configs['model']['affine']
        self.norm = configs['model']['norm']


        self.HiddenSize = configs['model']['HiddenSize']

        self.LayerNormOn = configs['model']['LayerNorm']

        self.PositionEncoding = configs['model']['PositionEncoding']


        self.ShrinkOrExpand = configs['model']['ShrinkOrExpand']
        self.ShrinkOrExpandChannelNum = configs['model']['ShrinkOrExpandChannelNum']
        self.ShrinkOrExpandMode = configs['model']['ShrinkOrExpandMode']
        self.ShrinkerGNNTemp = configs['model']['ShrinkerGNNTemp']


        self.PredictorType = configs['model']['predictor']
        self.BranchNum = configs['model']['BranchNum']
        self.DecoderBranchNum = configs['model']['DecoderBranchNum']

        self.GlobalResidual = configs['model']['GlobalResidual']
        self.GlobalResidualProvider = configs['model']['GlobalResidualProvider']

        self.residual = configs['model']['residual']

        self.TSFFlatten = nn.Flatten()


        

        if self.ShrinkOrExpand == True:
            if self.ShrinkOrExpandMode == 'GNN':
                self.shrinker = GNNShrinker(self.ChannelNum, self.ShrinkOrExpandChannelNum, self.ShrinkerGNNTemp)
            elif self.ShrinkOrExpandMode == 'CNN':
                self.shrinker = CNNShrinker(self.ChannelNum, self.ShrinkOrExpandChannelNum)
            elif self.ShrinkOrExpandMode == 'SimpleCNN':
                self.shrinker = SimpleCNNShrinker(self.ChannelNum, self.ShrinkOrExpandChannelNum)

            self.OldChannelNum = self.ChannelNum
            self.ChannelNum = self.ShrinkOrExpandChannelNum
            configs['dataset']['ChannelNum'] = self.ShrinkOrExpandChannelNum

        self.TSFModules = CTSFModules(configs)

        if self.GlobalResidual == True:
            if self.GlobalResidualProvider == 'Linear':
                self.ResidualLayer = nn.Linear(in_features=self.SamplingNum * self.ChannelNum, out_features=self.HiddenSize)
            elif self.GlobalResidualProvider == 'MLP':
                self.ResidualLayer = nn.Sequential(
                    nn.Linear(in_features=self.SamplingNum * self.ChannelNum, out_features=self.SamplingNum * self.ChannelNum * 2),
                    nn.GELU(),
                    nn.Dropout(0.5),
                    nn.Linear(in_features=self.SamplingNum * self.ChannelNum * 2, out_features=self.HiddenSize),
                    nn.GELU(),
                    nn.Dropout(0.5),
                    nn.Linear(in_features=self.HiddenSize, out_features=self.HiddenSize)
                )
            elif self.GlobalResidualProvider == 'CNN':
                self.ResidualLayer = nn.Sequential(
                    nn.Conv1d(in_channels=self.ChannelNum, out_channels= 64 * self.SamplingNum, kernel_size=3, padding=1),
                    nn.Conv1d(in_channels=64 * self.SamplingNum, out_channels=32 * self.SamplingNum, kernel_size=3, padding=1),
                    nn.AdaptiveAvgPool1d(1),
                    nn.Flatten(),
                    nn.Linear(in_features=32 * self.SamplingNum, out_features=self.HiddenSize)
                )
        
        if self.PredictorType == 'DecoderMLP' and self.residual == True:
            self.decoder = None
            self.DownLinear = nn.Linear(in_features=self.HiddenSize, out_features=self.DecoderBranchNum * self.HiddenSize)
            self.ResDownLinear = nn.Linear(in_features=self.HiddenSize, out_features=self.DecoderBranchNum * self.HiddenSize)
            self.TSFPredictor = nn.Sequential(
                nn.Linear(in_features=self.HiddenSize, out_features=self.HiddenSize * 2),
                nn.GELU(),
                nn.Dropout(0.5),
                nn.Linear(in_features=self.HiddenSize * 2, out_features=self.HiddenSize),
                nn.GELU(),
                nn.Dropout(0.5),
                nn.Linear(in_features=self.HiddenSize, out_features=1)
            )
        elif self.PredictorType == 'decoder':
            self.decoder = None
            self.DownLinear = nn.Linear(in_features=self.HiddenSize, out_features=self.DecoderBranchNum * self.HiddenSize)
            self.TSFPredictor = nn.Linear(in_features=self.HiddenSize, out_features=self.ChannelNum)
        elif self.PredictorType == 'DecoderMLP':
            self.decoder = None
            self.DownLinear = nn.Linear(in_features=self.HiddenSize, out_features=self.DecoderBranchNum * self.HiddenSize)
            self.TSFPredictor = nn.Sequential(
                nn.Linear(in_features=self.HiddenSize, out_features=self.HiddenSize * 2),
                nn.GELU(),
                nn.Dropout(0.5),
                nn.Linear(in_features=self.HiddenSize * 2, out_features=self.HiddenSize),
                nn.GELU(),
                nn.Dropout(0.5),
                nn.Linear(in_features=self.HiddenSize, out_features=self.ChannelNum)
            )
        


    def SplitPatches(self, x, PatchLen, stride):
        BatchSize, SeqLen, ChannelNum = x.shape
        
        if SeqLen < PatchLen:
            return torch.empty(BatchSize, 0, PatchLen, ChannelNum, device=x.device)

        PatchNum = (SeqLen - PatchLen) // stride + 1

        patches = []
        for i in range(PatchNum):
            start = i * stride
            end = start + PatchLen
            if end <= SeqLen:
                patch = x[:, start:end, :]
                patches.append(patch.unsqueeze(1))
        
        if patches:
            patches = torch.cat(patches, dim=1)
        else:
            patches = torch.empty(BatchSize, 0, PatchLen, ChannelNum, device=x.device)

        return patches
    

    
    def predictor(self, FlattenedFeatures, BatchSize, SampledBatch, TargetLen):
        if self.PredictorType == 'DecoderMLP' and self.residual == True:
            self.decoder = CSegmentTreeDecoderRes(self.DecoderBranchNum, TargetLen, self.DownLinear, self.TSFPredictor, self.TSFModules.ChannelIndependentLinearEmbedding, self.ResDownLinear)
            out = self.decoder.decode(FlattenedFeatures, SampledBatch)
        elif self.PredictorType == 'decoder' or self.PredictorType == 'DecoderMLP':
            self.decoder = CSegmentTreeDecoder(self.DecoderBranchNum, TargetLen, self.DownLinear, self.TSFPredictor)
            out = self.decoder.decode(FlattenedFeatures)

        out = out.reshape(BatchSize, TargetLen, self.ChannelNum)
        return out
    
    def forward(self, InputBatch, TargetLen):
        
        if self.norm == True:
            InputBatch, mean, std = self.RevIN.norm(InputBatch)
        if self.affine == True:
            InputBatch = self.RevIN.affine(InputBatch)
        if self.LayerNormOn == True:
            InputBatch, LayerNormMean, LayerNormStd = self.LayerNorm.forward(InputBatch)

        if self.ShrinkOrExpand != False:
            InputBatch = self.shrinker.shrink(InputBatch)

        SampledBatch = sampler(InputBatch, InputBatch.shape[1], self.SamplingNum)
        
        patches = self.SplitPatches(InputBatch, self.PatchLen, self.PatchStride)
        BatchSize, PatchNum, PatchLen, ChannelNum = patches.shape

        PatchFeatures = self.TSFModules.embedding(patches)


        if self.PositionEncoding == True:
            PatchFeatures = self.TSFModules.PositionEncoding(PatchFeatures)


        self.TSFModules.TreeInsert(PatchFeatures)

        FinalVotes = self.TSFModules.PascalWeight(SampledBatch, PatchNum)
        mask, FinalVotes = self.TSFModules.TopKMask(FinalVotes)

        BatchFeatures = self.TSFModules.AllRangeFeature(mask, FinalVotes, PatchNum)
        GatedFeatures = self.TSFModules.gating(BatchFeatures, mask, FinalVotes)
        GatedFeatures = torch.dropout(GatedFeatures, p=self.dropout, train=self.training)

        if self.GlobalResidual == True:
            if self.GlobalResidualProvider == 'MLP' or self.GlobalResidualProvider == 'Linear':
                FlattenedInput = self.TSFFlatten(SampledBatch)
                ResidualFeature = self.ResidualLayer(FlattenedInput)
            elif self.GlobalResidualProvider == 'CNN':
                TransInput = SampledBatch.permute(0, 2, 1)
                ResidualFeature = self.ResidualLayer(TransInput)
            GatedFeatures = GatedFeatures + ResidualFeature

        out = self.predictor(GatedFeatures, BatchSize, SampledBatch, TargetLen)

        if self.ShrinkOrExpand == True:
            out = self.shrinker.expand(out)

            
        if self.LayerNormOn == True:
            out = self.LayerNorm.inverse(out, LayerNormMean, LayerNormStd)

        if self.affine == True:
            out = self.RevIN.UnAffine(out)
        if self.norm == True:
            out = self.RevIN.UnNorm(out, mean, std)

        return out
