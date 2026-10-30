import torch
import pandas as pd
from modules.GlobalNorm import norm
from torch.utils.data import Dataset

def SplitTS(TimeSeries, mode, InputLen, TargetLen):
    AvaliableStepNum = len(TimeSeries)
    if mode == 0:
        TrainStepNum = int(AvaliableStepNum * 0.7)
        LeftStepNum = AvaliableStepNum - TrainStepNum
        ValStepNum = round(LeftStepNum / 3) + InputLen + 1
        TestStepNum = round(LeftStepNum / 3) * 2 + InputLen
    elif mode == 1:
        TrainStepNum = int(AvaliableStepNum * 0.6)
        LeftStepNum = AvaliableStepNum - TrainStepNum + 2 * InputLen
        ValStepNum = int(LeftStepNum / 2)
        TestStepNum = ValStepNum

    TrainBegin = 0
    TrainEnd = TrainStepNum

    ValBegin = TrainEnd - InputLen
    ValEnd = ValBegin + ValStepNum

    TestBegin = ValEnd - InputLen
    TestEnd = TestBegin + TestStepNum

    TrainSet = TimeSeries[TrainBegin:TrainEnd]
    ValSet = TimeSeries[ValBegin:ValEnd]
    TestSet = TimeSeries[TestBegin:TestEnd]

    return TrainSet, ValSet, TestSet



def LoadData(configs, InputLen, TargetLen):
    DataSetName = configs['dataset']['DatasetName']
    
    TimeSeries = pd.read_csv('datasets/'+ DataSetName +'.csv')
    channels = TimeSeries.columns.tolist()

    channels.remove('date')

    TimeSeries = torch.tensor(TimeSeries[channels].values, dtype=torch.float32, device=configs['device'])
    ChannelNum = len(channels)

    if 'ETT' in DataSetName:
        TrainTS, ValidateTS, TestTS = SplitTS(TimeSeries, mode=1, InputLen=InputLen, TargetLen=TargetLen)
    else:
        TrainTS, ValidateTS, TestTS = SplitTS(TimeSeries, mode=0, InputLen=InputLen, TargetLen=TargetLen)

    TrainTS, mean, std = norm(TrainTS)
    ValidateTS, _, _ = norm(ValidateTS, mean, std)
    TestTS, _, _ = norm(TestTS, mean, std)

    return TrainTS, ValidateTS, TestTS

class LoadTSData(Dataset):
    def __init__(self, config, TSData, InputLen, TargetLen):
        self.stride = config['dataset']['stride']

        self.InputLen = InputLen
        self.TargetLen = TargetLen

        self.SampleLen = self.InputLen + self.TargetLen
        self.length = (len(TSData) - self.SampleLen) // self.stride + 1
        self.TSData = TSData

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        StartIndex = index * self.stride
        EndIndex = StartIndex + self.SampleLen
        
        sample = self.TSData[StartIndex:EndIndex]
        
        x = sample[:self.InputLen]
        y = sample[self.InputLen:]
        
        return x, y
