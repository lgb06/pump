#!/bin/bash

# 检查第一个参数是否存在
if [ -n "$1" ]; then
  # 如果存在，则设置 save_path 为 ./checkpoints/后面跟着传的参数
  SAVE_PARAM="$1"
  SAVE_PATH="./checkpoints/${SAVE_PARAM}/"
else
  # 如果不存在，则设置 save_path 为 ./checkpoints
  SAVE_PATH="./checkpoints/"
fi


export CUDA_VISIBLE_DEVICES=5

model_name=RmGPT
exp_name=pump7
wandb_mode=disabled
ptune_name=supervised_learning
d_model=512
device='cuda:0'

data_type=Electric

# 
timestamp=$(date +"%Y%m%d_%H%M%S")
full_exp_name="${exp_name}_${data_type}_${timestamp}"

# Supervised learning
python run.py \
    --is_training 1 \
    --model_id $full_exp_name \
    --model $model_name \
    --lradj supervised \
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
    --learning_rate 1e-7\
    --batch_size 256 \
    --checkpoints $SAVE_PATH \
    --cls_mode 12 \
    --data $data_type     
  # --pretrained_weight 'checkpoints/ALL_task_test_RmGPT_All_ftM_dm512_el4_test_0/pretrain_checkpoint.pth'\
  # --prompt_tune_epoch 20\
  # --prompt_num 10 \


# # Supervised learning
# python run.py \
#   --is_training 1 \
#   --model_id $exp_name \
#   --model $model_name \
#   --lradj supervised \
#   --prompt_num 10 \
#   --patch_len 256 \
#   --stride 256 \
#   --e_layers 4 \
#   --d_model $d_model \
#   --des 'Exp' \
#   --itr 1 \
#   --weight_decay 0 \
#   --prompt_tune_epoch 20\
#   --train_epochs 0 \
#   --debug $wandb_mode \
#   --project_name $ptune_name \
#   --task_data_config_path  data_provider/data_config/main_result/multi_task.yaml \
#   --pretrained_weight 'checkpoints/ALL_task_test_RmGPT_All_ftM_dm512_el4_test_0/pretrain_checkpoint.pth'\
#   --learning_rate 1e-7\
#   --batch_size 256 \
