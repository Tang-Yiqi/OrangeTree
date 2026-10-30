
from collections import OrderedDict

import torch
import torch.nn as nn

from core.modules.TreeModulesFast import CTSFModules
from core.modules.normalizationFast import CRevIN, CLayerNorm
from core.modules.SamplerFast import sampler
from core.modules.ChannelShrinker import GNNShrinker, CNNShrinker, SimpleCNNShrinker


class OrangeTree(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.ConfigureTF32(configs)
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

    def ConfigureTF32(self, configs):
        self.TF32 = configs['model'].get('TF32', False)
        torch.set_float32_matmul_precision('high' if self.TF32 else 'highest')
        torch.backends.cuda.matmul.allow_tf32 = self.TF32
        torch.backends.cudnn.allow_tf32 = self.TF32
        torch.backends.cudnn.benchmark = False


    def FastSplitPatches(self, x, PatchLen, stride):
        BatchSize, SeqLen, ChannelNum = x.shape
        if SeqLen < PatchLen:
            return x.new_empty(BatchSize, 0, PatchLen, ChannelNum)
        return x.unfold(1, PatchLen, stride).permute(0, 1, 3, 2)

    def PredictNormalized(self, FlattenedFeatures, BatchSize, TargetLen, ChannelIndependentFeatures=None):
        TargetLen = int(TargetLen)
        if self.PredictorType == 'DecoderMLP' and self.residual == True:
            MainLayer = FlattenedFeatures.unsqueeze(1)
            ResidualLayer = ChannelIndependentFeatures.unsqueeze(2)
            MainLayer, ResidualLayer = self.ExpandDecoderLayers(MainLayer, ResidualLayer, TargetLen)
            out = self.PredictDecoderLayer(MainLayer, ResidualLayer, TargetLen)
        elif self.PredictorType in ('decoder', 'DecoderMLP'):
            MainLayer, _ = self.ExpandDecoderLayers(FlattenedFeatures.unsqueeze(1), None, TargetLen)
            out = self.PredictDecoderLayer(MainLayer, None, TargetLen)
        else:
            raise ValueError(f'Unsupported predictor type: {self.PredictorType}')
        return out.reshape(BatchSize, TargetLen, self.ChannelNum)

    def DecoderDepth(self, TargetLen):
        TargetLen = int(TargetLen)
        NodeNum = 1
        depth = 0
        while NodeNum < TargetLen:
            NodeNum *= self.DecoderBranchNum
            depth += 1
        return depth

    def ExpandDecoderLayers(self, MainLayer, ResidualLayer, TargetLen, CurrentDepth=0):
        TargetLen = int(TargetLen)
        TargetDepth = self.DecoderDepth(TargetLen)
        while CurrentDepth < TargetDepth:
            IsFinalExpansion = CurrentDepth + 1 == TargetDepth
            MainParentNum = MainLayer.shape[1]
            if IsFinalExpansion:
                MainParentNum = min(MainParentNum, (TargetLen + self.DecoderBranchNum - 1) // self.DecoderBranchNum)
            MainLayer = self.DownLinear(MainLayer[:, :MainParentNum, :]).reshape(MainLayer.shape[0], MainParentNum * self.DecoderBranchNum, self.HiddenSize)

            if ResidualLayer is not None:
                ResidualParentNum = ResidualLayer.shape[2]
                if IsFinalExpansion:
                    ResidualParentNum = min(ResidualParentNum, (TargetLen + self.DecoderBranchNum - 1) // self.DecoderBranchNum)
                ResidualLayer = self.ResDownLinear(ResidualLayer[:, :, :ResidualParentNum, :]).reshape(ResidualLayer.shape[0], ResidualLayer.shape[1], ResidualParentNum * self.DecoderBranchNum, self.HiddenSize)
            CurrentDepth += 1
        return MainLayer, ResidualLayer

    def PredictDecoderLayer(self, MainLayer, ResidualLayer, TargetLen):
        TargetLen = int(TargetLen)
        MainFeatures = MainLayer[:, :TargetLen, :]
        if ResidualLayer is None:
            return self.TSFPredictor(MainFeatures)

        ResidualFeatures = ResidualLayer[:, :, :TargetLen, :].permute(0, 2, 1, 3)
        MixedFeatures = MainFeatures.unsqueeze(2) + ResidualFeatures
        return self.TSFPredictor(MixedFeatures).squeeze(-1)

    def EmbedInput(self, InputBatch):
        if self.PatchLen == 1 and self.PatchStride == 1 and self.TSFModules.EmbeddingMethod == 'Linear':
            BatchSize, PatchNum, _ = InputBatch.shape
            PatchFeatures = self.TSFModules.PatchEmbedding(InputBatch)
        else:
            patches = self.FastSplitPatches(InputBatch, self.PatchLen, self.PatchStride)
            BatchSize, PatchNum, _, _ = patches.shape
            PatchFeatures = self.TSFModules.embedding(patches)
        return BatchSize, PatchNum, PatchFeatures


    def BuildEncodedState(self, GatedFeatures, SampledBatch, ChannelIndependentFeatures, BatchSize, mean, std, LayerNormMean, LayerNormStd):
        return {
            'Features': GatedFeatures,
            'SampledBatch': SampledBatch,
            'ChannelIndependentFeatures': ChannelIndependentFeatures,
            'BatchSize': BatchSize,
            'Mean': mean,
            'Std': std,
            'LayerNormMean': LayerNormMean,
            'LayerNormStd': LayerNormStd,
        }


    def encode(self, InputBatch):
        mean = std = None
        LayerNormMean = LayerNormStd = None

        if self.norm == True:
            InputBatch, mean, std = self.RevIN.norm(InputBatch)
        if self.affine == True:
            InputBatch = self.RevIN.affine(InputBatch)
        if self.LayerNormOn == True:
            InputBatch, LayerNormMean, LayerNormStd = self.LayerNorm(InputBatch)
        if self.ShrinkOrExpand != False:
            InputBatch = self.shrinker.shrink(InputBatch)

        SampledBatch = sampler(InputBatch, InputBatch.shape[1], self.SamplingNum)
        ChannelIndependentFeatures = self.TSFModules.ChannelIndependentLinearEmbedding(SampledBatch.permute(0, 2, 1))

        BatchSize, PatchNum, PatchFeatures = self.EmbedInput(InputBatch)

        if self.PositionEncoding == True:
            PatchFeatures = self.TSFModules.PositionEncoding(PatchFeatures)
        self.TSFModules.TreeInsert(PatchFeatures)

        FinalVotes = self.TSFModules.PascalWeight(SampledBatch, PatchNum, ChannelIndependentFeatures)
        mask, FinalVotes = self.TSFModules.TopKMask(FinalVotes)

        BatchFeatures = self.TSFModules.AllRangeFeature(mask, FinalVotes, PatchNum)
        GatedFeatures = self.TSFModules.gating(BatchFeatures, mask, FinalVotes)
        GatedFeatures = torch.dropout(GatedFeatures, p=self.dropout, train=self.training)

        if self.GlobalResidual == True:
            if self.GlobalResidualProvider in ('MLP', 'Linear'):
                ResidualFeature = self.ResidualLayer(self.TSFFlatten(SampledBatch))
            else:
                ResidualFeature = self.ResidualLayer(SampledBatch.permute(0, 2, 1))
            GatedFeatures = GatedFeatures + ResidualFeature

        return self.BuildEncodedState(GatedFeatures, SampledBatch, ChannelIndependentFeatures, BatchSize, mean, std, LayerNormMean, LayerNormStd)

    def RestoreOutput(self, output, EncodedState):
        if self.ShrinkOrExpand == True:
            output = self.shrinker.expand(output)
        if self.LayerNormOn == True:
            output = self.LayerNorm.inverse(output, EncodedState['LayerNormMean'], EncodedState['LayerNormStd'])
        if self.affine == True:
            output = self.RevIN.UnAffine(output)
        if self.norm == True:
            output = self.RevIN.UnNorm(output, EncodedState['Mean'], EncodedState['Std'])
        return output

    def decode(self, EncodedState, TargetLen):
        output = self.PredictNormalized(EncodedState['Features'], EncodedState['BatchSize'], TargetLen, EncodedState['ChannelIndependentFeatures'])
        return self.RestoreOutput(output, EncodedState)

    def BeginDecodeMulti(self, EncodedState, MaxTargetLen):
        if self.training:
            raise RuntimeError('BeginDecodeMulti requires model.eval().')
        ResidualLayer = None
        if self.PredictorType == 'DecoderMLP' and self.residual == True:
            ResidualLayer = EncodedState['ChannelIndependentFeatures'].unsqueeze(2)
        return {
            'EncodedState': EncodedState,
            'MainLayer': EncodedState['Features'].unsqueeze(1),
            'ResidualLayer': ResidualLayer,
            'CurrentDepth': 0,
            'LastTargetLen': 0,
            'MaxTargetLen': int(MaxTargetLen),
            'MaxDepth': self.DecoderDepth(MaxTargetLen),
        }

    def DecodeNext(self, DecodeState, TargetLen):
        TargetLen = int(TargetLen)
        if TargetLen < DecodeState['LastTargetLen']:
            raise ValueError(f"DecodeNext requires nondecreasing target lengths; got {TargetLen} after {DecodeState['LastTargetLen']}.")
        if TargetLen > DecodeState['MaxTargetLen']:
            raise ValueError(f"TargetLen={TargetLen} exceeds the planned maximum {DecodeState['MaxTargetLen']}.")

        TargetDepth = self.DecoderDepth(TargetLen)
        while DecodeState['CurrentDepth'] < TargetDepth:
            NextDepth = DecodeState['CurrentDepth'] + 1
            MainLayer = DecodeState['MainLayer']
            ResidualLayer = DecodeState['ResidualLayer']

            MainParentNum = MainLayer.shape[1]
            if NextDepth == DecodeState['MaxDepth']:
                MainParentNum = min(MainParentNum, (DecodeState['MaxTargetLen'] + self.DecoderBranchNum - 1) // self.DecoderBranchNum)
            DecodeState['MainLayer'] = self.DownLinear(MainLayer[:, :MainParentNum, :]).reshape(MainLayer.shape[0], MainParentNum * self.DecoderBranchNum, self.HiddenSize)

            if ResidualLayer is not None:
                ResidualParentNum = ResidualLayer.shape[2]
                if NextDepth == DecodeState['MaxDepth']:
                    ResidualParentNum = min(ResidualParentNum, (DecodeState['MaxTargetLen'] + self.DecoderBranchNum - 1) // self.DecoderBranchNum)
                DecodeState['ResidualLayer'] = self.ResDownLinear(ResidualLayer[:, :, :ResidualParentNum, :]).reshape(ResidualLayer.shape[0], ResidualLayer.shape[1], ResidualParentNum * self.DecoderBranchNum, self.HiddenSize)
            DecodeState['CurrentDepth'] = NextDepth

        output = self.PredictDecoderLayer(DecodeState['MainLayer'], DecodeState['ResidualLayer'], TargetLen).reshape(DecodeState['EncodedState']['BatchSize'], TargetLen, self.ChannelNum)
        DecodeState['LastTargetLen'] = TargetLen
        return self.RestoreOutput(output, DecodeState['EncodedState'])

    def DecodeMulti(self, EncodedState, TargetLens):
        UniqueTargetLens = list(OrderedDict.fromkeys(int(TargetLen) for TargetLen in TargetLens))
        if self.training:
            return {TargetLen: self.decode(EncodedState, TargetLen) for TargetLen in UniqueTargetLens}

        SortedTargetLens = sorted(UniqueTargetLens)
        DecodeState = self.BeginDecodeMulti(EncodedState, SortedTargetLens[-1])
        outputs = {TargetLen: self.DecodeNext(DecodeState, TargetLen) for TargetLen in SortedTargetLens}
        return {TargetLen: outputs[TargetLen] for TargetLen in UniqueTargetLens}

    def PredictMulti(self, InputBatch, TargetLens):
        return self.DecodeMulti(self.encode(InputBatch), TargetLens)

    def forward(self, InputBatch, TargetLen):
        return self.decode(self.encode(InputBatch), TargetLen)
