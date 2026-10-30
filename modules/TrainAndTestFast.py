
from collections import OrderedDict
import random

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader, Subset

from modules.TrainAndTest import ArctanLoss, PowerLoss
from modules.LoadData import LoadData, LoadTSData


def ConfigureFloat32Execution(model, configs, Training=False):
    device = configs['device']
    EnableTF32 = (
        device.type == 'cuda'
        and configs['model'].get('TF32', False)
    )
    torch.set_float32_matmul_precision(
        'high' if EnableTF32 else 'highest'
    )
    torch.backends.cuda.matmul.allow_tf32 = EnableTF32
    torch.backends.cudnn.allow_tf32 = EnableTF32
    torch.backends.cudnn.benchmark = bool(
        EnableTF32
        and Training
        and configs['model'].get('CudnnBenchmarkTraining', False)
    )
    TSFModules = getattr(model, 'TSFModules', None)
    if TSFModules is not None:
        TSFModules.TF32 = EnableTF32
    return EnableTF32


def _CalculateLoss(Output, Target, configs):
    MSELoss = F.mse_loss(Output, Target)
    MAELoss = F.l1_loss(Output, Target)
    LossFunction = configs['LossFunction']
    if LossFunction == 'ARCTAN':
        Loss = ArctanLoss(Output, Target)
    elif LossFunction == 'POWER':
        Loss = PowerLoss(
            Output, Target, power=configs['LossPower']
        )
    elif LossFunction == 'MIX':
        MAERate = configs['MAERate']
        Loss = MAERate * MAELoss + (1 - MAERate) * MSELoss
    else:
        Loss = MSELoss
    return MSELoss, MAELoss, Loss


def _ValidateLengthPairs(InputLens, TargetLens):
    if len(InputLens) != len(TargetLens):
        raise ValueError(
            'MIOL requires parallel InputLen and TargetLen lists; '
            f'got {len(InputLens)} and {len(TargetLens)} entries.'
        )


def _GroupPairIndicesByInputLen(InputLens):
    Groups = OrderedDict()
    for PairIndex, InputLen in enumerate(InputLens):
        Groups.setdefault(int(InputLen), []).append(PairIndex)
    return Groups


def _NewMetricTotals(PairNum, device):
    return (
        [torch.zeros((), device=device) for _ in range(PairNum)],
        [torch.zeros((), device=device) for _ in range(PairNum)],
        [torch.zeros((), device=device) for _ in range(PairNum)],
        [0 for _ in range(PairNum)],
    )


def _AccumulateMetrics(PairIndex, Output, Target, configs, MSETotals, MAETotals, LossTotals, Counts):
    MSELoss, MAELoss, Loss = _CalculateLoss(Output, Target, configs)
    BatchSize = Target.shape[0]
    MSETotals[PairIndex] += MSELoss * BatchSize
    MAETotals[PairIndex] += MAELoss * BatchSize
    LossTotals[PairIndex] += Loss * BatchSize
    Counts[PairIndex] += BatchSize


def _EvaluateDatasetsWithSharedEncoder(model, Datasets, configs):
    InputLens = configs['dataset']['InputLen']
    TargetLens = configs['dataset']['TargetLen']
    BatchSize = configs['model']['ValAndTestBatchSize']
    device = configs['device']
    PairNum = len(InputLens)
    MSETotals, MAETotals, LossTotals, Counts = _NewMetricTotals(PairNum, device)

    model.eval()
    with torch.no_grad():
        for InputLen, PairIndices in _GroupPairIndicesByInputLen(InputLens).items():
            CommonCount = min(len(Datasets[i]) for i in PairIndices)
            MaxPairIndex = max(
                PairIndices, key=lambda i: int(TargetLens[i])
            )
            SharedDataset = Subset(
                Datasets[MaxPairIndex], range(CommonCount)
            )
            SharedLoader = DataLoader(
                SharedDataset,
                batch_size=BatchSize,
                shuffle=False,
                drop_last=False,
            )
            GroupTargetLens = [
                int(TargetLens[i]) for i in PairIndices
            ]

            for x, MaxTarget in SharedLoader:
                x = x.to(device)
                MaxTarget = MaxTarget.to(device)
                Outputs = model.PredictMulti(x, GroupTargetLens)
                for PairIndex in PairIndices:
                    TargetLen = int(TargetLens[PairIndex])
                    Target = MaxTarget[:, :TargetLen, :]
                    _AccumulateMetrics(PairIndex, Outputs[TargetLen], Target, configs, MSETotals, MAETotals, LossTotals, Counts)

            for PairIndex in PairIndices:
                if len(Datasets[PairIndex]) <= CommonCount:
                    continue
                TailDataset = Subset(
                    Datasets[PairIndex],
                    range(CommonCount, len(Datasets[PairIndex])),
                )
                TailLoader = DataLoader(
                    TailDataset,
                    batch_size=BatchSize,
                    shuffle=False,
                    drop_last=False,
                )
                TargetLen = int(TargetLens[PairIndex])
                for x, Target in TailLoader:
                    x = x.to(device)
                    Target = Target.to(device)
                    Output = model(x, TargetLen)
                    _AccumulateMetrics(PairIndex, Output, Target, configs, MSETotals, MAETotals, LossTotals, Counts)

    AvgMSE = np.empty(PairNum)
    AvgMAE = np.empty(PairNum)
    AvgLoss = np.empty(PairNum)
    for PairIndex in range(PairNum):
        Denominator = max(1, Counts[PairIndex])
        AvgMSE[PairIndex] = (
            MSETotals[PairIndex] / Denominator
        ).item()
        AvgMAE[PairIndex] = (
            MAETotals[PairIndex] / Denominator
        ).item()
        AvgLoss[PairIndex] = (
            LossTotals[PairIndex] / Denominator
        ).item()
    return AvgMSE, AvgMAE, AvgLoss


def _CreateTrainingOptimizer(model, configs, LearningRate, PrintExecutionInfo=False):
    EnableTF32 = ConfigureFloat32Execution(model, configs, Training=True)
    UseFusedAdam = (
        configs['device'].type == 'cuda'
        and configs['model'].get('FusedAdam', False)
    )
    optimizer = Adam(
        model.parameters(),
        lr=LearningRate,
        fused=UseFusedAdam,
    )
    if PrintExecutionInfo:
        print(
            'Float32 execution:',
            'TF32 Tensor Cores enabled' if EnableTF32 else 'strict FP32',
            '| Adam:',
            'fused' if UseFusedAdam else 'default',
            '| MIOL inference: shared Encoder',
        )
    return optimizer


def _AccumulateTrainingMetrics(SampleIndex, MSELoss, MAELoss, loss, BatchSize, TrainTotalMSELoss, TrainTotalMAELoss, TrainTotalLoss, TrainTotalCount):
    TrainTotalMSELoss[SampleIndex] += MSELoss.detach() * BatchSize
    TrainTotalMAELoss[SampleIndex] += MAELoss.detach() * BatchSize
    TrainTotalLoss[SampleIndex] += loss.detach() * BatchSize
    TrainTotalCount[SampleIndex] += BatchSize


def _GetMetricAverage(Total, Count):
    return (Total / max(1, Count)).item()


def _ReloadModelAndOptimizer(SavePath, configs, LearningRate):
    model = torch.load(SavePath, map_location=configs['device'])
    optimizer = _CreateTrainingOptimizer(model, configs, LearningRate)
    return model, optimizer


def _PrintBestMetrics(BestTrainMSELoss, BestTrainMAELoss, BestValidateMSELoss, BestValidateMAELoss, BestTestMSELoss, BestTestMAELoss, BestTestLoss):
    print(f'\nBest Train MSE/MAE: {BestTrainMSELoss:.10f}/{BestTrainMAELoss:.10f}')
    print(f'Best Validate MSE/MAE: {BestValidateMSELoss:.10f}/{BestValidateMAELoss:.10f}')
    print(f'Best Test MSE/MAE: {BestTestMSELoss:.10f}/{BestTestMAELoss:.10f}')
    print(f'Best Test Loss: {BestTestLoss:.10f}')


def TrainModel(model, configs):
    InputLen = configs['dataset']['InputLen']
    TargetLen = configs['dataset']['TargetLen']
    _ValidateLengthPairs(InputLen, TargetLen)

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


    optimizer = _CreateTrainingOptimizer(model, configs, NowLearningRate, PrintExecutionInfo=True)

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
        TotalBatchNum = 0
        for i in range(PairNum):
            TrainDataset = TrainDatasets[i]
            TrainDataLoader[i] = iter(DataLoader(dataset=TrainDataset, batch_size=BatchSize, shuffle=True, drop_last=True))
            BatchNumCount[i] = TrainDataset.__len__() // BatchSize
            TotalBatchNum += BatchNumCount[i]

        GlobalBatchIndex = 0

        TrainTotalMSELoss, TrainTotalMAELoss, TrainTotalLoss, TrainTotalCount = _NewMetricTotals(PairNum, device)
        while GlobalBatchIndex < TotalBatchNum:
            SampleIndex = random.randint(0, PairNum - 1)

            if UsedCount[SampleIndex] >= BatchNumCount[SampleIndex]: 
                continue
            else:
                UsedCount[SampleIndex] += 1
            

            BatchIndex = UsedCount[SampleIndex] - 1
            x, y = next(TrainDataLoader[SampleIndex])

            x = x.to(device)
            y = y.to(device)


            output = model(x, int(TargetLen[SampleIndex]))

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
            else:
                loss = MSELoss
                loss.backward()

            _AccumulateTrainingMetrics(SampleIndex, MSELoss, MAELoss, loss, x.size(0), TrainTotalMSELoss, TrainTotalMAELoss, TrainTotalLoss, TrainTotalCount)

            

            optimizer.step()
            optimizer.zero_grad(set_to_none=True)


            GlobalBatchIndex += 1

        for i in range(PairNum):
            TrainAvgMSELoss[i] = _GetMetricAverage(TrainTotalMSELoss[i], TrainTotalCount[i])
            TrainAvgMAELoss[i] = _GetMetricAverage(TrainTotalMAELoss[i], TrainTotalCount[i])
            TrainAvgLoss[i] = _GetMetricAverage(TrainTotalLoss[i], TrainTotalCount[i])
            
            print(f"Train MSE: {TrainAvgMSELoss[i]:.5f}")
            print(f"Train MAE: {TrainAvgMAELoss[i]:.5f}")
            print(f"Train Loss: {TrainAvgLoss[i]:.5f}")


        ValidateAvgMSELoss, ValidateAvgMAELoss, ValidateAvgLoss = _EvaluateDatasetsWithSharedEncoder(model, ValDatasets, configs)
        TestAvgMSELoss, TestAvgMAELoss, TestAvgLoss = _EvaluateDatasetsWithSharedEncoder(model, TestDatasets, configs)

        for i in range(PairNum):
            print('InputLen: ', InputLen[i], ',TargetLen: ', TargetLen[i])

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
                
                model, optimizer = _ReloadModelAndOptimizer(SavePath, configs, NowLearningRate)
                print(f"No improvement in validation loss for {NoImprovementCount} epochs.")
                print(f"Jumping back to epoch {BestEpoch + 1}")

        epoch += 1
        if NowPatience <= 0:
            print("Early stopping triggered due to patience exhaustion.")
            break
        print('\n')
    
    _PrintBestMetrics(BestTrainMSELoss, BestTrainMAELoss, BestValidateMSELoss, BestValidateMAELoss, BestTestMSELoss, BestTestMAELoss, BestTestLoss)
    return model
