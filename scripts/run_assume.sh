DEVICE=0


seeds=(
    # 691 76134 96165 135 3407
    # 17665 38008 18374 87 6576 
    764 8612 6075 13761 37
    1387 1896 12478 153 6533
    # 479 22617 2617 62 86872
    # 345 41981 92328 197 1233

    # 2987 239 87239 82 1546
    # 65 42 314 6465 315
    # 68975 4357 3575 38638 2574 
    # 3971 395 74 15274 195 
    )

# t=0
# task_num=50
# for seed in "${seeds[@]}"
# do
#     # for tprime in {1..49}
#     for tprime in {49..49}
#     do  
#         for k in {0..8}
#         do  
#             CUDA_VISIBLE_DEVICES=$DEVICE python assume.py\
#                 --backbone ResNet\
#                 --learning_rate 1e-2\
#                 --weight_decay 0\
#                 --epochs 100\
#                 --seed $seed\
#                 --model_path_t "/data/yjkim/cl/outputs/FlatImageNet100Resized32FixedTask${task_num}DynamicKernelAvgPoolChannel8Stage4_FlatMiniResNet/Epoch_500/Seed_${seed}/_Reproduce_FineTuning_ImageNet100Resized32FixedTask${task_num}_FlatMiniResNet_Task_${t}_${seed}.pt"\
#                 --model_path_tprime "/data/yjkim/cl/outputs/FlatImageNet100Resized32FixedTask${task_num}DynamicKernelAvgPoolChannel8Stage4_FlatMiniResNet/Epoch_500/Seed_${seed}/_Reproduce_FineTuning_ImageNet100Resized32FixedTask${task_num}_FlatMiniResNet_Task_${tprime}_${seed}.pt"\
#                 --t $t\
#                 --tprime $tprime\
#                 --target_layer "block${k}"\
#                 --target_conv conv2
#         done
#     done
# done

# task_num=50
# for seed in "${seeds[@]}"
# do
#     # for tprime in {1..49}
#     for t in {0..48}
#     do  
#         for k in {0..8}
#         do  
#             tprime=${t+1}
#             CUDA_VISIBLE_DEVICES=$DEVICE python assume.py\
#                 --backbone ResNet\
#                 --learning_rate 1e-2\
#                 --weight_decay 0\
#                 --epochs 100\
#                 --seed $seed\
#                 --model_path_t "/data/yjkim/cl/outputs/FlatImageNet100Resized32FixedTask${task_num}DynamicKernelAvgPoolChannel8Stage4_FlatMiniResNet/Epoch_500/Seed_${seed}/_Reproduce_FineTuning_ImageNet100Resized32FixedTask${task_num}_FlatMiniResNet_Task_${t}_${seed}.pt"\
#                 --model_path_tprime "/data/yjkim/cl/outputs/FlatImageNet100Resized32FixedTask${task_num}DynamicKernelAvgPoolChannel8Stage4_FlatMiniResNet/Epoch_500/Seed_${seed}/_Reproduce_FineTuning_ImageNet100Resized32FixedTask${task_num}_FlatMiniResNet_Task_${tprime}_${seed}.pt"\
#                 --t $t\
#                 --tprime ${tprime}\
#                 --target_layer "block${k}"\
#                 --target_conv conv2
#         done
#     done
# done

# task_num=20
# for seed in "${seeds[@]}"
# do
#     for t in {0..18}
#     # for tprime in {19..19}
#     do  
#         for k in {0..8}
#         # for k in {1..1}
#         do  
#             # tprime=${t+1}
#             tprime=$((t + 1))
#             backbone='ReLUNetLayer9Width256'
#             CUDA_VISIBLE_DEVICES=$DEVICE python assume.py\
#                 --backbone $backbone\
#                 --learning_rate 1e-2\
#                 --weight_decay 0\
#                 --epochs 100\
#                 --seed $seed\
#                 --model_path_t "/data/yjkim/cl/outputs/FlatSplitCIFAR100Task${task_num}_${backbone}/Epoch_100/Seed_${seed}/_Reproduce_FineTuning_SplitCIFAR100_ReLUNetCIFAR100_${t}_${seed}.pt"\
#                 --model_path_tprime "/data/yjkim/cl/outputs/FlatSplitCIFAR100Task${task_num}_${backbone}/Epoch_100/Seed_${seed}/_Reproduce_FineTuning_SplitCIFAR100_ReLUNetCIFAR100_${tprime}_${seed}.pt"\
#                 --t $t\
#                 --tprime $tprime\
#                 --target_layer "block${k}"
#         done
#     done
# done

# For FlatViT
t=0
task_num=50
for seed in "${seeds[@]}"
do
    for tprime in {49..49}
    do  
        for k in {0..8}
        do  
            backbone='FlatViT'
            CUDA_VISIBLE_DEVICES=$DEVICE python assume.py\
                --backbone $backbone\
                --learning_rate 1e-2\
                --weight_decay 0\
                --epochs 100\
                --seed $seed\
                --model_path_t "/data/yjkim/cl/outputs/FlatSplitCIFAR100FixedTask${task_num}_${backbone}/Epoch_100/Seed_${seed}/_Reproduce_FineTuning_SplitCIFAR100FixedTask50_FlatVisionTransformer_Task_${t}_${seed}.pt"\
                --model_path_tprime "/data/yjkim/cl/outputs/FlatSplitCIFAR100FixedTask${task_num}_${backbone}/Epoch_100/Seed_${seed}/_Reproduce_FineTuning_SplitCIFAR100FixedTask50_FlatVisionTransformer_Task_${tprime}_${seed}.pt"\
                --t $t\
                --tprime $tprime\
                --target_layer $k
        done
    done
done