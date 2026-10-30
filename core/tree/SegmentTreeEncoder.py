from re import S
import torch
class CNode:
    def __init__(self, BranchNum):
        self.feature = None
        self.sons = [None] * BranchNum


class CMultiBranchSegmentTreeLinearUp:
    def __init__(self, BranchNum, UpLinear):
        self.BranchNum = BranchNum
        self.root = None
        self.length = 0
        self.LeafNum = 0

        self.UpLinear = UpLinear


    def InsertLeaf(self, feature, root, begin, end):
        if root == None:
            root = CNode(self.BranchNum)
        if begin == end:
            root.feature = feature
        else:
            NodeIndex = self.length
            NodeNumOfEachBranch = (end - begin + 1) // self.BranchNum
            SonIndex = (NodeIndex - begin) // NodeNumOfEachBranch
            SonBegin = begin + SonIndex * NodeNumOfEachBranch
            SonEnd = SonBegin + NodeNumOfEachBranch - 1

            root.sons[SonIndex] = self.InsertLeaf(feature, root.sons[SonIndex], SonBegin, SonEnd)
        
        SonFull = True
        for son in root.sons:
            if son == None or son.feature == None:
                SonFull = False
                break
        if SonFull == True:
            root.feature = self.UpLinear(torch.cat([son.feature for son in root.sons], dim=-1))

        return root
            
    def insert(self, feature):
        self.length += 1
        if self.length > self.LeafNum:
            if self.LeafNum == 0:
                self.LeafNum = 1
            else:
                self.LeafNum *= self.BranchNum
                NewRoot = CNode(self.BranchNum)
                NewRoot.sons[0] = self.root
                self.root = NewRoot
        
        self.root = self.InsertLeaf(feature, self.root, 1, self.LeafNum)

        return self.length

    def CalculateRangeFeature(self, RangeBegin, RangeEnd, root, begin, end):
        if RangeBegin == begin and RangeEnd == end:
            return root.feature
        else:
            NodeNumOfEachBranch = (end - begin + 1) // self.BranchNum
            SonFeature = [None] * self.BranchNum
            ZerosTemplate = None
            for SonIndex in range(self.BranchNum):
                SonBegin = begin + SonIndex * NodeNumOfEachBranch
                SonEnd = SonBegin + NodeNumOfEachBranch - 1
                if RangeBegin <= SonEnd and RangeEnd >= SonBegin:
                    SonFeature[SonIndex] = self.CalculateRangeFeature(max(RangeBegin, SonBegin), min(RangeEnd, SonEnd), root.sons[SonIndex], SonBegin, SonEnd)
                    if ZerosTemplate == None:
                        ZerosTemplate = torch.zeros_like(SonFeature[SonIndex])
            for SonIndex in range(self.BranchNum):
                if SonFeature[SonIndex] == None:
                    SonFeature[SonIndex] = torch.zeros_like(ZerosTemplate)

            return self.UpLinear(torch.cat(SonFeature, dim=-1))


    def CalculateSingleSampleRangeFeature(self, RangeBegin, RangeEnd, root, begin, end, SampleIndex):
        if RangeBegin == begin and RangeEnd == end:
            return root.feature[SampleIndex, :]
        else:
            NodeNumOfEachBranch = (end - begin + 1) // self.BranchNum
            SonFeature = [None] * self.BranchNum
            ZerosTemplate = None
            for SonIndex in range(self.BranchNum):
                SonBegin = begin + SonIndex * NodeNumOfEachBranch
                SonEnd = SonBegin + NodeNumOfEachBranch - 1
                if RangeBegin <= SonEnd and RangeEnd >= SonBegin:
                    SonFeature[SonIndex] = self.CalculateSingleSampleRangeFeature(max(RangeBegin, SonBegin), min(RangeEnd, SonEnd), root.sons[SonIndex], SonBegin, SonEnd, SampleIndex)
                    if ZerosTemplate == None:
                        ZerosTemplate = torch.zeros_like(SonFeature[SonIndex])
            for SonIndex in range(self.BranchNum):
                if SonFeature[SonIndex] == None:
                    SonFeature[SonIndex] = torch.zeros_like(ZerosTemplate)
                    
            return self.UpLinear(torch.cat(SonFeature, dim=-1))


    def GetRangeFeature(self, RangeBegin, RangeEnd, SampleIndex = None):
        if SampleIndex == None:
            return self.CalculateRangeFeature(RangeBegin, RangeEnd, self.root, 1, self.LeafNum)
        else:
            return self.CalculateSingleSampleRangeFeature(RangeBegin, RangeEnd, self.root, 1, self.LeafNum, SampleIndex)
