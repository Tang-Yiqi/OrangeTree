for dataset in "etth1" "etth2" "ettm1" "ettm2" "weather" "solar"
do  
    for BranchNum in 2 3 4 5
    do
        for KRate in 0.01 0.001
        do
            if [ "$dataset" = "etth1" ] || [ "$dataset" = "ettm1" ] || [ "$dataset" = "ettm2" ] || [ "$dataset" = "etth2" ]; then
                    BatchSize=256
                    LearningRate=0.0008
                    EpochNum=50
            elif [ "$dataset" = "weather" ]; then
                EpochNum=50
                BatchSize=96
                LearningRate=0.0006
            elif [ "$dataset" = "solar" ]; then
                EpochNum=10
                BatchSize=16
                LearningRate=0.00005
                if [ "$KRate" = "0.1" ] || [ "$KRate" = "0.01" ]; then
                    continue
                fi
            fi

            echo "Starting model training:"
            python main.py --multirun \
            model=OrangeTreeFast \
            dataset=$dataset \
            LossFunction="MIX" \
            device=cuda:0 \
            dataset.InputLen="[48,48,96,96,96,96,144,144]" \
            dataset.TargetLen="[96,192,96,192,336,720,336,720]" \
            model.predictor="DecoderMLP" \
            model.BatchSize=$BatchSize \
            model.ValAndTestBatchSize=$BatchSize \
            model.residual=True \
            model.RangePositionEncoding=False \
            model.RangeGateType="votes" \
            model.RangeAdditionalWeighter=True \
            model.RangeAdditionalWeighterType="Linear" \
            model.WeighterHiddenSize=32 \
            model.LearningRate=$LearningRate \
            model.SamplingNum=96 \
            model.avg=False \
            model.BranchNum=$BranchNum \
            model.DecoderBranchNum=$BranchNum \
            model.KRate=$KRate \
            model.EpochNum=$EpochNum \
            model.patience=3 \
            > results/MIOL/$dataset,B$BranchNum,K$KRate.out 2>&1
            echo "Model training completed"
            echo "----------------------------------------"

        done
    done
done



for dataset in "etth1" "etth2" "ettm1" "ettm2" "weather" "solar"
do  
    for BranchNum in 2 3 4 5
    do
        for FixedK in 4 16 64
        do
            if [ "$dataset" = "etth1" ] || [ "$dataset" = "ettm1" ] || [ "$dataset" = "ettm2" ] || [ "$dataset" = "etth2" ]; then
                BatchSize=256
                LearningRate=0.0008
                EpochNum=50
            elif [ "$dataset" = "weather" ]; then
                EpochNum=50
                BatchSize=96
                LearningRate=0.0006
            elif [ "$dataset" = "solar" ]; then
                EpochNum=10
                BatchSize=16
                LearningRate=0.00005
                if [ "$FixedK" = "64" ] || [ "$FixedK" = "16" ]; then
                    continue
                fi
            fi

            echo "Starting model training:"
            python main.py --multirun \
            model=OrangeTreeFast \
            dataset=$dataset \
            LossFunction="MIX" \
            device=cuda:0 \
            dataset.InputLen="[48,48,96,96,96,96,144,144]" \
            dataset.TargetLen="[96,192,96,192,336,720,336,720]" \
            model.predictor="DecoderMLP" \
            model.BatchSize=$BatchSize \
            model.ValAndTestBatchSize=$BatchSize \
            model.residual=True \
            model.RangePositionEncoding=False \
            model.RangeGateType="votes" \
            model.RangeAdditionalWeighter=True \
            model.RangeAdditionalWeighterType="Linear" \
            model.WeighterHiddenSize=32 \
            model.LearningRate=$LearningRate \
            model.SamplingNum=96 \
            model.avg=False \
            model.BranchNum=$BranchNum \
            model.DecoderBranchNum=$BranchNum \
            model.TopKMode="fixed" \
            model.FixedK=$FixedK \
            model.EpochNum=$EpochNum \
            model.patience=3 \
            > results/MIOL/$dataset,B$BranchNum,FixedK$FixedK.out 2>&1
            echo "Model training completed"
            echo "----------------------------------------"

        done
    done
done
