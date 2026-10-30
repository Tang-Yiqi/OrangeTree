from re import S
import torch
import torch.nn.functional as F
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
        self.TensorLayers = None

    def build(self, LeafFeatures):
        self.length = LeafFeatures.shape[1]
        self.LeafNum = 1
        while self.LeafNum < self.length:
            self.LeafNum *= self.BranchNum

        PaddingLength = self.LeafNum - self.length
        if PaddingLength > 0:
            Padding = LeafFeatures.new_zeros(
                LeafFeatures.shape[0], PaddingLength, LeafFeatures.shape[2]
            )
            CurrentLayer = torch.cat((LeafFeatures, Padding), dim=1)
        else:
            CurrentLayer = LeafFeatures

        self.TensorLayers = [CurrentLayer]
        while CurrentLayer.shape[1] > 1:
            BatchSize, NodeNum, HiddenSize = CurrentLayer.shape
            CurrentLayer = self.UpLinear(
                CurrentLayer.reshape(
                    BatchSize,
                    NodeNum // self.BranchNum,
                    self.BranchNum * HiddenSize,
                )
            )
            self.TensorLayers.append(CurrentLayer)

        self.root = None
        return self.length

    def _GetTensorNodeFeature(self, begin, end, SampleIndex=None):
        Span = end - begin + 1
        LayerIndex = 0
        while Span > 1:
            Span //= self.BranchNum
            LayerIndex += 1

        LayerSpan = self.BranchNum ** LayerIndex
        NodeIndex = (begin - 1) // LayerSpan
        Feature = self.TensorLayers[LayerIndex][:, NodeIndex, :]
        if SampleIndex is not None:
            Feature = Feature[SampleIndex, :]
        return Feature

    def _CalculateTensorRangeFeature(
        self, RangeBegin, RangeEnd, begin, end, SampleIndex=None
    ):
        if RangeBegin == begin and RangeEnd == end:
            return self._GetTensorNodeFeature(begin, end, SampleIndex)

        NodeNumOfEachBranch = (end - begin + 1) // self.BranchNum
        SonFeatures = [None] * self.BranchNum
        ZerosTemplate = None
        for SonIndex in range(self.BranchNum):
            SonBegin = begin + SonIndex * NodeNumOfEachBranch
            SonEnd = SonBegin + NodeNumOfEachBranch - 1
            if RangeBegin <= SonEnd and RangeEnd >= SonBegin:
                SonFeatures[SonIndex] = self._CalculateTensorRangeFeature(
                    max(RangeBegin, SonBegin),
                    min(RangeEnd, SonEnd),
                    SonBegin,
                    SonEnd,
                    SampleIndex,
                )
                if ZerosTemplate is None:
                    ZerosTemplate = torch.zeros_like(SonFeatures[SonIndex])

        for SonIndex in range(self.BranchNum):
            if SonFeatures[SonIndex] is None:
                SonFeatures[SonIndex] = ZerosTemplate

        return self.UpLinear(torch.cat(SonFeatures, dim=-1))

    def GetRangeFeatures(self, RangeBegins, RangeEnds, SampleIndices):
        if self.TensorLayers is None:
            return self._GetRangeFeaturesDense(
                RangeBegins, RangeEnds, SampleIndices
            )
        if not RangeBegins.is_cuda:
            return self._GetRangeFeaturesDense(
                RangeBegins, RangeEnds, SampleIndices
            )

        return self._GetRangeFeaturesDenseCompiled(
            RangeBegins, RangeEnds, SampleIndices
        )

    @torch._dynamo.config.patch(capture_dynamic_output_shape_ops=True)
    @torch.compile(fullgraph=True, mode='default')
    def _GetRangeFeaturesSparseCompiled(
        self, RangeBegins, RangeEnds, SampleIndices
    ):
        return self._GetRangeFeaturesSparse(
            RangeBegins, RangeEnds, SampleIndices
        )

    def _GetRangeFeaturesSparse(
        self, RangeBegins, RangeEnds, SampleIndices
    ):
        if self.TensorLayers is None:
            return self._GetRangeFeaturesDense(RangeBegins, RangeEnds, SampleIndices)

        LeftIndices = RangeBegins - 1
        RightIndices = RangeEnds - 1
        QueryNum = LeftIndices.numel()
        HiddenSize = self.TensorLayers[0].shape[-1]
        TreeDepth = len(self.TensorLayers) - 1
        WeightBlocks = self.UpLinear.weight.reshape(
            HiddenSize, self.BranchNum, HiddenSize
        )
        BranchPositions = torch.arange(self.BranchNum, device=LeftIndices.device)

        def ProjectByPositions(Features, Positions, ActiveMask=None):
            Output = Features.new_zeros(QueryNum, HiddenSize)
            if ActiveMask is None:
                ActiveMask = torch.ones_like(Positions, dtype=torch.bool)
            for Position in range(self.BranchNum):
                Indices = torch.where(
                    ActiveMask & Positions.eq(Position)
                )[0]
                Projected = F.linear(
                    Features.index_select(0, Indices),
                    WeightBlocks[:, Position, :],
                )
                Output = Output.index_copy(0, Indices, Projected)
            return Output

        def ProjectCompleteChildren(ParentIndices, LayerIndex, ActiveMask):
            Output = self.TensorLayers[0].new_zeros(QueryNum, HiddenSize)
            for Position in range(self.BranchNum):
                Indices = torch.where(ActiveMask[:, Position])[0]
                ChildFeatures = self.TensorLayers[LayerIndex - 1][
                    SampleIndices.index_select(0, Indices),
                    ParentIndices.index_select(0, Indices) * self.BranchNum + Position,
                    :,
                ]
                Projected = F.linear(ChildFeatures, WeightBlocks[:, Position, :])
                Output = Output.index_add(0, Indices, Projected)
            return Output

        LeftSuffix = self.TensorLayers[0][
            SampleIndices, LeftIndices, :
        ]
        RightPrefix = self.TensorLayers[0][
            SampleIndices, RightIndices, :
        ]
        LeftSuffixLayers = [LeftSuffix]
        RightPrefixLayers = [RightPrefix]

        ChildSpan = 1
        for LayerIndex in range(1, TreeDepth + 1):
            Span = ChildSpan * self.BranchNum

            LeftParents = torch.div(
                LeftIndices, Span, rounding_mode='floor'
            )
            LeftPositions = torch.div(
                LeftIndices, ChildSpan, rounding_mode='floor'
            ).remainder(self.BranchNum)
            LeftFeatures = ProjectByPositions(LeftSuffix, LeftPositions)
            LeftFeatures = LeftFeatures + ProjectCompleteChildren(
                LeftParents, LayerIndex,
                BranchPositions.unsqueeze(0).gt(LeftPositions.unsqueeze(-1)),
            )
            LeftSuffix = LeftFeatures
            if self.UpLinear.bias is not None:
                LeftSuffix = LeftSuffix + self.UpLinear.bias

            FullLeftParents = self.TensorLayers[LayerIndex][
                SampleIndices, LeftParents, :
            ]
            LeftSuffix = torch.where(
                LeftIndices.remainder(Span).eq(0).unsqueeze(-1),
                FullLeftParents,
                LeftSuffix,
            )
            LeftSuffixLayers.append(LeftSuffix)

            RightParents = torch.div(
                RightIndices, Span, rounding_mode='floor'
            )
            RightPositions = torch.div(
                RightIndices, ChildSpan, rounding_mode='floor'
            ).remainder(self.BranchNum)
            RightFeatures = ProjectByPositions(RightPrefix, RightPositions)
            RightFeatures = RightFeatures + ProjectCompleteChildren(
                RightParents, LayerIndex,
                BranchPositions.unsqueeze(0).lt(RightPositions.unsqueeze(-1)),
            )
            RightPrefix = RightFeatures
            if self.UpLinear.bias is not None:
                RightPrefix = RightPrefix + self.UpLinear.bias
            FullRightParents = self.TensorLayers[LayerIndex][
                SampleIndices, RightParents, :
            ]
            RightPrefix = torch.where(
                RightIndices.remainder(Span).eq(Span - 1).unsqueeze(-1),
                FullRightParents,
                RightPrefix,
            )
            RightPrefixLayers.append(RightPrefix)
            ChildSpan = Span

        LcaLayers = torch.full_like(LeftIndices, TreeDepth)
        FoundLca = torch.zeros_like(LeftIndices, dtype=torch.bool)
        Span = 1
        for LayerIndex in range(TreeDepth + 1):
            SameNode = torch.div(
                LeftIndices, Span, rounding_mode='floor'
            ).eq(torch.div(RightIndices, Span, rounding_mode='floor'))
            SelectLayer = SameNode & ~FoundLca
            LcaLayers = torch.where(
                SelectLayer,
                torch.full_like(LcaLayers, LayerIndex),
                LcaLayers,
            )
            FoundLca = FoundLca | SameNode
            Span *= self.BranchNum

        CurrentFeatures = self.TensorLayers[0].new_zeros(
            QueryNum, HiddenSize
        )
        CurrentFeatures = torch.where(
            LcaLayers.eq(0).unsqueeze(-1),
            LeftSuffixLayers[0],
            CurrentFeatures,
        )

        ChildSpan = 1
        for LayerIndex in range(1, TreeDepth + 1):
            Span = ChildSpan * self.BranchNum
            LeftPositions = torch.div(
                LeftIndices, ChildSpan, rounding_mode='floor'
            ).remainder(self.BranchNum)
            RightPositions = torch.div(
                RightIndices, ChildSpan, rounding_mode='floor'
            ).remainder(self.BranchNum)
            IsBaseLayer = LcaLayers.eq(LayerIndex)
            IsWrapLayer = LcaLayers.lt(LayerIndex)
            IsReady = IsBaseLayer | IsWrapLayer

            PrimaryFeatures = torch.where(
                IsBaseLayer.unsqueeze(-1),
                LeftSuffixLayers[LayerIndex - 1],
                CurrentFeatures,
            )
            LayerFeatures = ProjectByPositions(
                PrimaryFeatures, LeftPositions, IsReady
            )
            LayerFeatures = LayerFeatures + ProjectByPositions(
                RightPrefixLayers[LayerIndex - 1],
                RightPositions,
                IsBaseLayer,
            )
            ParentIndices = torch.div(
                LeftIndices, Span, rounding_mode='floor'
            )
            MiddleMask = (
                BranchPositions.unsqueeze(0).gt(LeftPositions.unsqueeze(-1))
                & BranchPositions.unsqueeze(0).lt(RightPositions.unsqueeze(-1))
                & IsBaseLayer.unsqueeze(-1)
            )
            LayerFeatures = LayerFeatures + ProjectCompleteChildren(
                ParentIndices, LayerIndex, MiddleMask
            )
            if self.UpLinear.bias is not None:
                LayerFeatures = LayerFeatures + self.UpLinear.bias
            ExactNode = (
                LeftIndices.remainder(Span).eq(0)
                & RightIndices.remainder(Span).eq(Span - 1)
                & IsBaseLayer
            )
            FullParents = self.TensorLayers[LayerIndex][
                SampleIndices, ParentIndices, :
            ]
            LayerFeatures = torch.where(
                ExactNode.unsqueeze(-1), FullParents, LayerFeatures
            )
            CurrentFeatures = torch.where(
                IsReady.unsqueeze(-1), LayerFeatures, CurrentFeatures
            )
            ChildSpan = Span

        return CurrentFeatures

    @torch.compile(fullgraph=True, mode='default')
    def _GetRangeFeaturesDenseCompiled(
        self, RangeBegins, RangeEnds, SampleIndices
    ):
        return self._GetRangeFeaturesDense(
            RangeBegins, RangeEnds, SampleIndices
        )

    def _GetRangeFeaturesDense(self, RangeBegins, RangeEnds, SampleIndices):
        if self.TensorLayers is None:
            Ranges = zip(RangeBegins.tolist(), RangeEnds.tolist())
            return torch.stack(
                [
                    self.GetRangeFeature(Begin, End, SampleIndices[Index : Index + 1])[0]
                    for Index, (Begin, End) in enumerate(Ranges)
                ],
                dim=0,
            )

        LeftIndices = RangeBegins - 1
        RightIndices = RangeEnds - 1
        QueryNum = LeftIndices.numel()
        HiddenSize = self.TensorLayers[0].shape[-1]
        TreeDepth = len(self.TensorLayers) - 1
        FeatureDType = self.TensorLayers[0].dtype

        LeftSuffix = self.TensorLayers[0][SampleIndices, LeftIndices, :]
        RightPrefix = self.TensorLayers[0][SampleIndices, RightIndices, :]
        LeftSuffixLayers = [LeftSuffix]
        RightPrefixLayers = [RightPrefix]

        BranchPositions = torch.arange(self.BranchNum, device=LeftIndices.device)

        ChildSpan = 1
        for LayerIndex in range(1, TreeDepth + 1):
            Span = ChildSpan * self.BranchNum

            LeftParents = torch.div(LeftIndices, Span, rounding_mode='floor')
            LeftPositions = torch.div(
                LeftIndices, ChildSpan, rounding_mode='floor'
            ).remainder(self.BranchNum)
            LeftSlots = F.one_hot(
                LeftPositions, num_classes=self.BranchNum
            ).to(FeatureDType).unsqueeze(-1)
            LeftChildren = self.TensorLayers[LayerIndex - 1][
                SampleIndices.unsqueeze(-1),
                LeftParents.unsqueeze(-1) * self.BranchNum + BranchPositions.unsqueeze(0),
                :,
            ]
            LeftInputs = LeftSlots * LeftSuffix.unsqueeze(1)
            LeftInputs = LeftInputs + torch.where(
                BranchPositions.unsqueeze(0).gt(LeftPositions.unsqueeze(-1)).unsqueeze(-1),
                LeftChildren, torch.zeros_like(LeftChildren),
            )
            LeftSuffix = self.UpLinear(
                LeftInputs.reshape(QueryNum, self.BranchNum * HiddenSize)
            )
            FullLeftParents = self.TensorLayers[LayerIndex][
                SampleIndices, LeftParents, :
            ]
            LeftSuffix = torch.where(
                LeftIndices.remainder(Span).eq(0).unsqueeze(-1),
                FullLeftParents,
                LeftSuffix,
            )
            LeftSuffixLayers.append(LeftSuffix)

            RightParents = torch.div(RightIndices, Span, rounding_mode='floor')
            RightPositions = torch.div(
                RightIndices, ChildSpan, rounding_mode='floor'
            ).remainder(self.BranchNum)
            RightSlots = F.one_hot(
                RightPositions, num_classes=self.BranchNum
            ).to(FeatureDType).unsqueeze(-1)
            RightChildren = self.TensorLayers[LayerIndex - 1][
                SampleIndices.unsqueeze(-1),
                RightParents.unsqueeze(-1) * self.BranchNum + BranchPositions.unsqueeze(0),
                :,
            ]
            RightInputs = RightSlots * RightPrefix.unsqueeze(1)
            RightInputs = RightInputs + torch.where(
                BranchPositions.unsqueeze(0).lt(RightPositions.unsqueeze(-1)).unsqueeze(-1),
                RightChildren, torch.zeros_like(RightChildren),
            )
            RightPrefix = self.UpLinear(
                RightInputs.reshape(QueryNum, self.BranchNum * HiddenSize)
            )
            FullRightParents = self.TensorLayers[LayerIndex][
                SampleIndices, RightParents, :
            ]
            RightPrefix = torch.where(
                RightIndices.remainder(Span).eq(Span - 1).unsqueeze(-1),
                FullRightParents,
                RightPrefix,
            )
            RightPrefixLayers.append(RightPrefix)
            ChildSpan = Span

        LcaLayers = torch.full_like(LeftIndices, TreeDepth)
        FoundLca = torch.zeros_like(LeftIndices, dtype=torch.bool)
        Span = 1
        for LayerIndex in range(TreeDepth + 1):
            SameNode = torch.div(
                LeftIndices, Span, rounding_mode='floor'
            ).eq(torch.div(RightIndices, Span, rounding_mode='floor'))
            SelectLayer = SameNode & ~FoundLca
            LcaLayers = torch.where(
                SelectLayer,
                torch.full_like(LcaLayers, LayerIndex),
                LcaLayers,
            )
            FoundLca = FoundLca | SameNode
            Span *= self.BranchNum

        CurrentFeatures = self.TensorLayers[0].new_zeros(QueryNum, HiddenSize)
        CurrentFeatures = torch.where(
            LcaLayers.eq(0).unsqueeze(-1), LeftSuffixLayers[0], CurrentFeatures
        )

        ChildSpan = 1
        for LayerIndex in range(1, TreeDepth + 1):
            Span = ChildSpan * self.BranchNum
            LeftPositions = torch.div(
                LeftIndices, ChildSpan, rounding_mode='floor'
            ).remainder(self.BranchNum)
            RightPositions = torch.div(
                RightIndices, ChildSpan, rounding_mode='floor'
            ).remainder(self.BranchNum)
            LeftSlots = F.one_hot(
                LeftPositions, num_classes=self.BranchNum
            ).to(FeatureDType).unsqueeze(-1)
            RightSlots = F.one_hot(
                RightPositions, num_classes=self.BranchNum
            ).to(FeatureDType).unsqueeze(-1)

            BaseInputs = (
                LeftSlots * LeftSuffixLayers[LayerIndex - 1].unsqueeze(1)
                + RightSlots * RightPrefixLayers[LayerIndex - 1].unsqueeze(1)
            )
            ParentIndices = torch.div(LeftIndices, Span, rounding_mode='floor')
            FullChildren = self.TensorLayers[LayerIndex - 1][
                SampleIndices.unsqueeze(-1),
                ParentIndices.unsqueeze(-1) * self.BranchNum + BranchPositions.unsqueeze(0),
                :,
            ]
            MiddleMask = (
                BranchPositions.unsqueeze(0).gt(LeftPositions.unsqueeze(-1))
                & BranchPositions.unsqueeze(0).lt(RightPositions.unsqueeze(-1))
            ).unsqueeze(-1)
            BaseInputs = BaseInputs + torch.where(
                MiddleMask, FullChildren, torch.zeros_like(FullChildren)
            )
            WrapInputs = LeftSlots * CurrentFeatures.unsqueeze(1)
            IsBaseLayer = LcaLayers.eq(LayerIndex).view(QueryNum, 1, 1)
            IsWrapLayer = LcaLayers.lt(LayerIndex).view(QueryNum, 1, 1)
            LayerInputs = torch.where(
                IsBaseLayer,
                BaseInputs,
                torch.where(IsWrapLayer, WrapInputs, torch.zeros_like(BaseInputs)),
            )
            LayerFeatures = self.UpLinear(
                LayerInputs.reshape(QueryNum, self.BranchNum * HiddenSize)
            )

            ExactNode = (
                LeftIndices.remainder(Span).eq(0)
                & RightIndices.remainder(Span).eq(Span - 1)
                & LcaLayers.eq(LayerIndex)
            )
            FullParents = self.TensorLayers[LayerIndex][
                SampleIndices, ParentIndices, :
            ]
            LayerFeatures = torch.where(
                ExactNode.unsqueeze(-1), FullParents, LayerFeatures
            )
            CurrentFeatures = torch.where(
                LcaLayers.le(LayerIndex).unsqueeze(-1),
                LayerFeatures,
                CurrentFeatures,
            )
            ChildSpan = Span

        return CurrentFeatures


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
        if self.TensorLayers is not None:
            return self._CalculateTensorRangeFeature(
                RangeBegin,
                RangeEnd,
                1,
                self.LeafNum,
                SampleIndex,
            )
        if SampleIndex == None:
            return self.CalculateRangeFeature(RangeBegin, RangeEnd, self.root, 1, self.LeafNum)
        else:
            return self.CalculateSingleSampleRangeFeature(RangeBegin, RangeEnd, self.root, 1, self.LeafNum, SampleIndex)
