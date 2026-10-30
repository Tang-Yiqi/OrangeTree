import torch
import torch.nn as nn
import torch.nn.functional as F

class GNNShrinker(nn.Module):
    def __init__(self, OriginalChannelNum, NewChannelNum, temp=0.1):
        super(GNNShrinker, self).__init__()
        self.OldChannelNum = OriginalChannelNum
        self.NewChannelNum = NewChannelNum
        self.temp = temp
        
        self.AssignmentWeights = nn.Parameter(
            torch.Tensor(self.OldChannelNum, self.NewChannelNum)
        )
        nn.init.xavier_uniform_(self.AssignmentWeights)
        
        
    def GetAssignmentMatrix(self):
        return F.softmax(self.AssignmentWeights / self.temp, dim=-1)

    def shrink(self, x):
        S = self.GetAssignmentMatrix()
        
        PooledX = torch.matmul(x, S)
        return PooledX

    def expand(self, XPooled):
        S = self.GetAssignmentMatrix()
        
        UnpooledX = torch.matmul(XPooled, S.t()) 
        return UnpooledX
    
class CNNShrinker(nn.Module):
    def __init__(self, OriginalChannelNum, NewChannelNum):
        super(CNNShrinker, self).__init__()
        self.OldChannelNum = OriginalChannelNum
        self.NewChannelNum = NewChannelNum
        
        self.ShrinkCNN = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=4, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(in_channels=4, out_channels=1, kernel_size=3, padding=1),
            nn.AdaptiveAvgPool1d(self.NewChannelNum)
        )
        
        self.ExpandCNN = nn.Sequential(
            nn.Upsample(size=self.OldChannelNum, mode='linear', align_corners=False),
            nn.Conv1d(in_channels=1, out_channels=4, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(in_channels=4, out_channels=1, kernel_size=3, padding=1)
        )

    def shrink(self, x):
        BatchSize, SeqLen, _ = x.shape
        x = x.reshape(BatchSize * SeqLen, 1, self.OldChannelNum)
        PooledX = self.ShrinkCNN(x)
        PooledX = PooledX.reshape(BatchSize, SeqLen, self.NewChannelNum)
        return PooledX

    def expand(self, XPooled):
        BatchSize, PredLen, _ = XPooled.shape
        XPooled = XPooled.reshape(BatchSize * PredLen, 1, self.NewChannelNum)
        UnpooledX = self.ExpandCNN(XPooled)
        UnpooledX = UnpooledX.reshape(BatchSize, PredLen, self.OldChannelNum)
        return UnpooledX
    
class SimpleCNNShrinker(nn.Module):
    def __init__(self, OriginalChannelNum, NewChannelNum):
        super(SimpleCNNShrinker, self).__init__()
        self.OldChannelNum = OriginalChannelNum
        self.NewChannelNum = NewChannelNum
        
        KernelSize = 5 
        self.ShrinkCNN = nn.Sequential(
            nn.Conv1d(OriginalChannelNum, NewChannelNum, kernel_size=KernelSize, padding=KernelSize//2),
            nn.GELU()
        )
        
        self.ExpandCNN = nn.Conv1d(NewChannelNum, OriginalChannelNum, kernel_size=1)

    def shrink(self, x):
        x = x.permute(0, 2, 1)
        ShrinkedX = self.ShrinkCNN(x)
        ShrinkedX = ShrinkedX.permute(0, 2, 1)
        return ShrinkedX

    def expand(self, ShrinkedX):
        ShrinkedX = ShrinkedX.permute(0, 2, 1)
        ExpandedX = self.ExpandCNN(ShrinkedX)
        ExpandedX = ExpandedX.permute(0, 2, 1)
        return ExpandedX
