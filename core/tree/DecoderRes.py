import torch
from collections import deque 
class CNode:
    def __init__(self):
        self.LayerIndex = None
        self.feature = None



class CSegmentTreeDecoderRes:
    def __init__(self, BranchNum, OutLength, DownLinear, predictor, ResProjctor, ResDownLinear):
        self.BranchNum = BranchNum
        self.OutLength = OutLength
        self.DownLinear = DownLinear
        self.predictor = predictor
        self.ResProjctor = ResProjctor
        self.ResDownLinear = ResDownLinear

    def GetFinalLayer(self, feature):
        FeatureQueue = deque()
        root = CNode()
        root.feature = feature
        root.LayerIndex = 1
        FeatureQueue.append(root)

        NowLayer = 1
        while len(FeatureQueue) < self.OutLength:
            NowLayer += 1
            while True:
                NowNode = FeatureQueue.popleft()
                if NowNode.LayerIndex != NowLayer - 1:
                    FeatureQueue.appendleft(NowNode)
                    break
                features = self.DownLinear(NowNode.feature)
                features = features.view(features.shape[0], self.BranchNum, -1)
                for i in range(self.BranchNum):
                    NewNode = CNode()
                    NewNode.feature = features[:, i, :]
                    NewNode.LayerIndex = NowLayer
                    FeatureQueue.append(NewNode)
        
        OutputFeatures = []
        for i in range(self.OutLength):
            NowNode = FeatureQueue.popleft()
            OutputFeatures.append(NowNode.feature)
        
        return torch.stack(OutputFeatures, dim=1)

    def GetResFeatures(self, OriginalInput):
        OriginalInput = OriginalInput.permute(0, 2, 1)
        ResHiddenRoot = self.ResProjctor(OriginalInput)
        FeatureQueue = deque()
        root = CNode()
        root.feature = ResHiddenRoot
        root.LayerIndex = 1
        FeatureQueue.append(root)

        NowLayer = 1
        while len(FeatureQueue) < self.OutLength:
            NowLayer += 1
            while True:
                NowNode = FeatureQueue.popleft()
                if NowNode.LayerIndex != NowLayer - 1:
                    FeatureQueue.appendleft(NowNode)
                    break
                features = self.ResDownLinear(NowNode.feature)
                features = features.view(features.shape[0], features.shape[1], self.BranchNum, -1)
                for i in range(self.BranchNum):
                    NewNode = CNode()
                    NewNode.feature = features[:, :, i, :]
                    NewNode.LayerIndex = NowLayer
                    FeatureQueue.append(NewNode)

        ResFeatures = []
        for i in range(self.OutLength):
            NowNode = FeatureQueue.popleft()
            ResFeatures.append(NowNode.feature)

        return torch.stack(ResFeatures, dim=1)

    def decode(self, feature, OriginalInput):
        OutputFeatures = self.GetFinalLayer(feature)
        ResFeatures = self.GetResFeatures(OriginalInput)
        MixedFeatures = OutputFeatures.unsqueeze(2) + ResFeatures
        output = self.predictor(MixedFeatures)
        
        return output.squeeze(-1)

        
