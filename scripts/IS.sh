for dataset in "etth1" "etth2" "ettm1" "ettm2" "weather" "solar"
do  
    for TargetLen in "[96]" "[192]" "[336]" "[720]"
    do
        for BranchNum in 2 3 4 5
        do
            for KRate in 0.1 0.01 0.001
            do
                if [ "$dataset" = "etth1" ] || [ "$dataset" = "ettm1" ] || [ "$dataset" = "ettm2" ] || [ "$dataset" = "etth2" ]; then
                        BatchSize=256
                        LearningRate=0.0008
                        EpochNum=50
                elif [ "$dataset" = "weather" ]; then
                    EpochNum=50
                    if [ "$TargetLen" = "[96]" ] || [ "$TargetLen" = "[192]" ] || [ "$TargetLen" = "[336]" ]; then
                        BatchSize=128
                        LearningRate=0.0008
                    elif [ "$TargetLen" = "[720]" ]; then
                        BatchSize=96
                        LearningRate=0.0006
                    fi
                elif [ "$dataset" = "solar" ]; then
                    EpochNum=10
                    if [ "$TargetLen" = "[96]" ]; then
                        BatchSize=64
                        LearningRate=0.0002
                    elif [ "$TargetLen" = "[192]" ]; then
                        BatchSize=32
                        LearningRate=0.0001
                    else
                        BatchSize=16
                        LearningRate=0.00005
                    fi
                fi

                echo "Starting model training:"
                python main.py --multirun \
                model=OrangeTreeFast \
                dataset=$dataset \
                LossFunction="MIX" \
                device=cuda:0 \
                dataset.InputLen="[96]" \
                dataset.TargetLen=$TargetLen \
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
                > results/IS/$dataset,$TargetLen,B$BranchNum,K$KRate.out 2>&1
                echo "Model training completed"
                echo "----------------------------------------"

            done
        done
    done
done
