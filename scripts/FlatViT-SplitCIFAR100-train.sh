export WANDB_KEY=5d69e23a8dc89acb4cbced8940a8254c470d7bad
export WANDB_ENTITY='yjkim-stat'
export WANDB_PROJECT='Continual-Learning'

export DATA_ROOT=/data/yjkim/cl/torchvision

export SRC_ROOT='/home/yjkim/cl-git'
DEVICE=0

export EXP_TYPE='train'

SEEDS=(
    # 691 76134 96165 135 3407 
    # 17665 38008 18374 87 6576 
    # 764 8612 6075 13761 37
    # 1387 1896 12478 153 6533
    # 479 22617 2617 62 86872
    345 41981 92328 197 1233
    
    # 2987 239 87239 82 1546
    # 65 42 314 6465 315
    # 68975 4357 3575 38638 2574 
    # 3971 395 74 15274 195 
    )


backbone=FlatViT
nb_epochs=100

scenario=SplitCIFAR100FixedTask50

for seed in "${SEEDS[@]}"
do
    OUTPUT_DIR="/data/yjkim/cl/outputs/FlatSplitCIFAR100FixedTask50_${backbone}/Epoch_${nb_epochs}/Seed_${seed}"
    export RUN_NAME_PREFIX="Reproduce"    
    CUDA_VISIBLE_DEVICES=$DEVICE  python main_train.py\
        --data_root $DATA_ROOT \
        --saving_dir $OUTPUT_DIR\
        --scenario "SplitCIFAR100FixedTask50" \
        --backbone $backbone\
        --strategy "FineTuning"\
        --nb_epochs $nb_epochs\
        --seed_value $seed\
        --num_workers 1
  
done