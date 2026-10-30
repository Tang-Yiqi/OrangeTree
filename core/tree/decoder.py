import torch
from collections import deque 
class CNode:
    def __init__(self):
        self.LayerIndex = None
        self.feature = None



class CSegmentTreeDecoder:
    def __init__(self, BranchNum, OutLength, DownLinear, predictor):
        self.BranchNum = BranchNum
        self.OutLength = OutLength
        self.DownLinear = DownLinear
        self.predictor = predictor

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


    def decode(self, feature):
        OutputFeatures = self.GetFinalLayer(feature)
        output = self.predictor(OutputFeatures)
        return output


        
