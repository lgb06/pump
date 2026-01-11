#!/bin/bash

# 检查第一个参数是否存在
if [ -n "$1" ]; then
  # 如果存在，则设置 save_path 为 ./checkpoints/后面跟着传的参数
  SAVE_PARAM="$1"
  SAVE_PATH="./checkpoints_sft_2cls/${SAVE_PARAM}/"
else
  # 如果不存在，则设置 save_path 为 ./checkpoints
  SAVE_PATH="./checkpoints_sft_2cls/"
fi

export CUDA_VISIBLE_DEVICES=1

model_name=RmGPT
exp_name=based_on_pretrained
wandb_mode=disabled
ptune_name=sft_2cls
d_model=512
device='cuda:0'

# 
timestamp=$(date +"%Y%m%d_%H%M%S")
full_exp_name="${exp_name}_${timestamp}"


# PRETRAINED_WEIGHT_NAME="rm+nlnVib"
# PRETRAINED_WEIGHT_NAME="rm+nlnElec"
PRETRAINED_WEIGHT_NAME="rm+nlnVibElec"

PRETRAINED_WEIGHT_PATH="/inspire/hdd/project/continuinglearinginlm/lijiapeng-CZXS25110021/rmgpt_pump/rmgpt_github/checkpoints_pretrain/${PRETRAINED_WEIGHT_NAME}/Base_RmGPT2_pretrain_x512_RmGPT_hd512_el4_en8_at16_it0/pretrain_checkpoint.pth"

#  test on nln-emp
python run.py \
    --is_training 0 \
    --model_id $full_exp_name \
    --model $model_name \
    --lradj head_tuning \
    --patch_len 256 \
    --stride 256 \
    --e_layers 4 \
    --d_model $d_model \
    --des 'Exp' \
    --itr 1 \
    --weight_decay 0 \
    --train_epochs 20 \
    --debug $wandb_mode \
    --project_name $ptune_name \
    --task_data_config_path  data_provider/data_config/pump/NLNEMP.yaml \
    --batch_size 256 \
    --checkpoints $SAVE_PATH \
    --pretrained_weight $PRETRAINED_WEIGHT_PATH \
    --visualize True \
    --visualize_output_dir "./visualization_results/${PRETRAINED_WEIGHT_NAME}_pretrained/${full_exp_name}_nlnemp_visualization/"
