import torch
import torch.nn.functional as F
from torch.optim import Adam, SGD
import datetime
import numpy as np
from time import sleep
import json
from os import mkdir
from torch.utils.data import Dataset , DataLoader

from modules.LoadData import LoadData, LoadTSData

import random

def ArctanLoss(output, target, reduction='mean'):
    diff = output - target
    loss = torch.atan(diff)
    loss = torch.abs(loss)
    if reduction == 'none':
        return loss
    elif reduction == 'mean':
        return torch.mean(loss)
    elif reduction == 'sum':
        return torch.sum(loss)


def PowerLoss(output, target, power=2, reduction='mean'):
    diff = torch.abs(output - target)
    loss = torch.pow(diff, power)
    if reduction == 'none':
        return loss
    elif reduction == 'mean':
        return torch.mean(loss)
    elif reduction == 'sum':
        return torch.sum(loss)


def TrainModel(model, configs):
    InputLen = configs['dataset']['InputLen']
    TargetLen = configs['dataset']['TargetLen']

    EpochNum = configs['model']['EpochNum']
    LearningRate = configs['model']['LearningRate']
    BatchSize = configs['model']['BatchSize']

    device = configs['device']
    ModelName = configs['model']['name']
    StartTime = configs['StartTime']

    ReduceLRThreshold = configs['model']['ReduceLRThreshold']
    LearningRateReducer = configs['model']['LearningRateReducer']
    RollBack = configs['model']['RollBack']
    MaxContinuousRollBack = configs['model']['MaxContinuousRollBack']

    LossFunction = configs['LossFunction']
    MAERate = configs['MAERate']
    LossPower = configs['LossPower']

    patience = configs['model']['patience']

    ValAndTestBatchSize = configs['model']['ValAndTestBatchSize']


    SavePath = 'models/' + ModelName + StartTime + '.pth'
    BestTrainMSELoss = float('inf')
    BestValidateMSELoss = float('inf')
    BestTestMSELoss = float('inf')

    BestTrainMAELoss = float('inf')
    BestValidateMAELoss = float('inf')
    BestTestMAELoss = float('inf')

    BestTrainLoss = float('inf')
    BestValidateLoss = float('inf')
    BestTestLoss = float('inf')


    NoImprovementCount = 0
    BestEpoch = 0

    NowLearningRate = LearningRate

    RollBackCount = 0


    InputLenNum = len(InputLen)
    TargetLenNum = len(TargetLen)

    PairNum = InputLenNum

    TrainDatasets = [0 for _ in range(PairNum)]
    ValDatasets = [0 for _ in range(PairNum)]
    TestDatasets = [0 for _ in range(PairNum)]


    for i in range(PairNum):
        TrainTS, ValidateTS, TestTS = LoadData(configs, InputLen[i], TargetLen[i])


        
        TrainDatasets[i] = LoadTSData(configs, TrainTS, InputLen[i], TargetLen[i])
        ValDatasets[i] = LoadTSData(configs, ValidateTS, InputLen[i], TargetLen[i])
        TestDatasets[i] = LoadTSData(configs, TestTS, InputLen[i], TargetLen[i])

        print('InputLen: ', InputLen[i], ',TargetLen: ', TargetLen[i])
        print('Train Sample Number:', TrainDatasets[i].__len__())
        print('Val Sample Number:', ValDatasets[i].__len__())
        print('Test Sample Number:', TestDatasets[i].__len__())





    optimizer = Adam(model.parameters(), lr=NowLearningRate)

    print('Train with', LossFunction, 'Loss', '\n\n')

    epoch = 0

    NowPatience = patience
    while epoch < EpochNum:
        model.train()
        print(f"Epoch [{epoch + 1}/{EpochNum}]", flush=True)

        for param in optimizer.param_groups:
            param['lr'] = NowLearningRate

        TrainAvgMSELoss = np.ones(PairNum) * float('inf')
        TrainAvgMAELoss = np.ones(PairNum) * float('inf')
        TrainAvgLoss = np.ones(PairNum) * float('inf')

        ValidateAvgMSELoss = np.ones(PairNum) * float('inf')
        ValidateAvgMAELoss = np.ones(PairNum) * float('inf')
        ValidateAvgLoss = np.ones(PairNum) * float('inf')

        TestAvgMAELoss = np.ones(PairNum) * float('inf')
        TestAvgMSELoss = np.ones(PairNum) * float('inf')
        TestAvgLoss = np.ones(PairNum) * float('inf')


        TrainDataLoader = [0 for _ in range(PairNum)]
        UsedCount = [0 for _ in range(PairNum)]
        BatchNumCount = [0 for _ in range(PairNum)]
        for i in range(PairNum):
            TrainDataset = TrainDatasets[i]
            TrainDataLoader[i] = iter(DataLoader(dataset=TrainDataset, batch_size=BatchSize, shuffle=True, drop_last=True))
            BatchNumCount[i] = TrainDataset.__len__() // BatchSize

        TrainTotalMSELoss = [0 for _ in range(PairNum)]
        TrainTotalMAELoss = [0 for _ in range(PairNum)]
        TrainTotalLoss = [0 for _ in range(PairNum)]
        TrainTotalCount = [0 for _ in range(PairNum)]
        while True:
            SampleIndex = random.randint(0, PairNum - 1)

        
            AllUsed = True
            for i in range(PairNum):
                if UsedCount[i] < BatchNumCount[i]:
                    AllUsed = False
                    break
            if AllUsed == True:
                break

            if UsedCount[SampleIndex] >= BatchNumCount[SampleIndex]: 
                continue
            else:
                UsedCount[SampleIndex] += 1
            

            BatchIndex = UsedCount[SampleIndex] - 1
            x, y = next(TrainDataLoader[SampleIndex])

            x = x.to(device)
            y = y.to(device)


            output = model(x, TargetLen[SampleIndex])

            MSELoss = F.mse_loss(output, y)
            MAELoss = F.l1_loss(output, y)
            
    
            if LossFunction == 'ARCTAN':
                loss = ArctanLoss(output, y)
                loss.backward()
            elif LossFunction == 'POWER':
                loss = PowerLoss(output, y, power=LossPower)
                loss.backward()
            elif LossFunction == 'MIX':
                loss = MAERate * MAELoss + (1 - MAERate) * MSELoss
                loss.backward()

            TrainTotalMSELoss[SampleIndex] += MSELoss.item() * x.size(0)
            TrainTotalMAELoss[SampleIndex] += MAELoss.item() * x.size(0)
            TrainTotalLoss[SampleIndex] += loss.item() * x.size(0)
            TrainTotalCount[SampleIndex] += x.size(0)

            

            optimizer.step()
            optimizer.zero_grad()


        for i in range(PairNum):
            TrainAvgMSELoss[i] = TrainTotalMSELoss[i] / max(1, TrainTotalCount[i])
            TrainAvgMAELoss[i] = TrainTotalMAELoss[i] / max(1, TrainTotalCount[i])
            TrainAvgLoss[i] = TrainTotalLoss[i] / max(1, TrainTotalCount[i])
            
            print(f"Train MSE: {TrainAvgMSELoss[i]:.5f}")
            print(f"Train MAE: {TrainAvgMAELoss[i]:.5f}")
            print(f"Train Loss: {TrainAvgLoss[i]:.5f}")


        for i in range(PairNum):
            print('InputLen: ', InputLen[i], ',TargetLen: ', TargetLen[i])

            model.eval()
            with torch.no_grad():
                ValDataset = ValDatasets[i]

                ValidateTotalMSELoss = 0.0
                ValidateTotalMAELoss = 0.0
                ValidateTotalLoss = 0.0

                ValidateTotalCount = 0
                
                ValDataLoader = DataLoader(dataset=ValDataset, batch_size=ValAndTestBatchSize, shuffle=False, drop_last=False)

                for BatchIndex, (x, y) in enumerate(ValDataLoader):
                    x = x.to(device)
                    y = y.to(device)

                    
                    output = model(x, TargetLen[i])

                    MSELoss = F.mse_loss(output, y)
                    MAELoss = F.l1_loss(output, y)

                    if LossFunction == 'ARCTAN':
                        loss = ArctanLoss(output, y)
                    elif LossFunction == 'POWER':
                        loss = PowerLoss(output, y, power=LossPower)
                    elif LossFunction == 'MIX':
                        loss = MAERate * MAELoss + (1 - MAERate) * MSELoss

                    ValidateTotalMSELoss += MSELoss.item() * x.size(0)
                    ValidateTotalMAELoss += MAELoss.item() * x.size(0)
                    ValidateTotalLoss += loss.item() * x.size(0)
                    ValidateTotalCount += x.size(0)

                ValidateAvgMSELoss[i] = ValidateTotalMSELoss / max(1, ValidateTotalCount)
                ValidateAvgMAELoss[i] = ValidateTotalMAELoss / max(1, ValidateTotalCount)
                ValidateAvgLoss[i] = ValidateTotalLoss / max(1, ValidateTotalCount)

            model.eval()
            with torch.no_grad():
                TestDataset = TestDatasets[i]

                TestTotalMSELoss = 0.0
                TestTotalMAELoss = 0.0
                TestTotalLoss = 0.0
                TestTotalCount = 0

                
                TestDataLoader = DataLoader(dataset=TestDataset, batch_size=ValAndTestBatchSize, shuffle=False, drop_last=False)
                for BatchIndex, (x, y) in enumerate(TestDataLoader):
                    x = x.to(device)
                    y = y.to(device)

                    
                    output = model(x, TargetLen[i])

                    MSELoss = F.mse_loss(output, y)
                    MAELoss = F.l1_loss(output, y)


                    if LossFunction == 'ARCTAN':
                        loss = ArctanLoss(output, y)
                    elif LossFunction == 'POWER':
                        loss = PowerLoss(output, y, power=LossPower)
                    elif LossFunction == 'MIX':
                        loss = MAERate * MAELoss + (1 - MAERate) * MSELoss

                    TestTotalMSELoss += MSELoss.item() * x.size(0)
                    TestTotalMAELoss += MAELoss.item() * x.size(0)
                    TestTotalLoss += loss.item() * x.size(0)
                    TestTotalCount += x.size(0)

                TestAvgMSELoss[i] = TestTotalMSELoss / max(1, TestTotalCount)
                TestAvgMAELoss[i] = TestTotalMAELoss / max(1, TestTotalCount)
                TestAvgLoss[i] = TestTotalLoss / max(1, TestTotalCount)
            print(f"Validate MSE: {ValidateAvgMSELoss[i]:.5f} \t Test MSE: {TestAvgMSELoss[i]:.5f}")
            print(f"Validate MAE: {ValidateAvgMAELoss[i]:.5f} \t Test MAE: {TestAvgMAELoss[i]:.5f}")
            print(f"Validate Loss: {ValidateAvgLoss[i]:.5f} \t Test Loss: {TestAvgLoss[i]:.5f}")




        TrainMSELossAllLen = np.mean(TrainAvgMSELoss)
        ValidateMSELossAllLen = np.mean(ValidateAvgMSELoss)
        TestMSELossAllLen = np.mean(TestAvgMSELoss)

        TrainMAELossAllLen = np.mean(TrainAvgMAELoss)
        ValidateMAELossAllLen = np.mean(ValidateAvgMAELoss)
        TestMAELossAllLen = np.mean(TestAvgMAELoss)

        TrainLossAllLen = np.mean(TrainAvgLoss)
        ValidateLossAllLen = np.mean(ValidateAvgLoss)
        TestLossAllLen = np.mean(TestAvgLoss)

        if PairNum > 1:
            print(f"All Length AVG Train MSE: {TrainMSELossAllLen:.5f} \t All Length AVG Validate MSE: {ValidateMSELossAllLen:.5f} \t All Length AVG Test MSE: {TestMSELossAllLen:.5f}")
            print(f"All Length AVG Train MAE: {TrainMAELossAllLen:.5f} \t All Length AVG Validate MAE: {ValidateMAELossAllLen:.5f} \t All Length AVG Test MAE: {TestMAELossAllLen:.5f}")
            print(f"All Length AVG Train Loss: {TrainLossAllLen:.5f} \t All Length AVG Validate Loss: {ValidateLossAllLen:.5f} \t All Length AVG Test Loss: {TestLossAllLen:.5f}")


        if (ValidateLossAllLen < BestValidateLoss):
            NowPatience = patience
            NoImprovementCount = 0
            RollBackCount = 0

            BestValidateMSELoss = ValidateMSELossAllLen
            BestTrainMSELoss = TrainMSELossAllLen
            BestTestMSELoss = TestMSELossAllLen

            BestTrainMAELoss = TrainMAELossAllLen
            BestValidateMAELoss = ValidateMAELossAllLen
            BestTestMAELoss = TestMAELossAllLen

            BestTrainLoss = TrainLossAllLen
            BestValidateLoss = ValidateLossAllLen
            BestTestLoss = TestLossAllLen

            BestEpoch = epoch
            torch.save(model, SavePath)
            print(f"Model saved: {SavePath} (Validate loss improved)")
        else:
            NoImprovementCount += 1

        if NoImprovementCount >= ReduceLRThreshold:
            NowPatience -= 1

            NowLearningRate *= LearningRateReducer
            print(f"Reducing learning rate to {NowLearningRate:.6f} due to no improvement in validation loss.")
            NoImprovementCount = 0
            if RollBack:
                RollBackCount += 1
                if RollBackCount > MaxContinuousRollBack:
                    print(f"Reached maximum continuous rollbacks ({MaxContinuousRollBack}), stopping training.")
                    break
                epoch = BestEpoch
                
                model = torch.load(SavePath)
                print(f"No improvement in validation loss for {NoImprovementCount} epochs.")
                print(f"Jumping back to epoch {BestEpoch + 1}")

        epoch += 1
        if NowPatience <= 0:
            print("Early stopping triggered due to patience exhaustion.")
            break
        print('\n')
    
    print(f"\nBest Test MSE: {BestTestMSELoss:.10f}")
    print(f"\nBest Test MAE: {BestTestMAELoss:.10f}")
    return model
