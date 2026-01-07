#!/bin/bash

# 检查第一个参数是否存在
if [ -n "$1" ]; then
  # 如果存在，则设置 save_path 为 ./checkpoints/后面跟着传的参数
  SAVE_PARAM="$1"
  SAVE_PATH="./checkpoints_KNN/${SAVE_PARAM}/"
else
  # 如果不存在，则设置 save_path 为 ./checkpoints
  SAVE_PATH="./checkpoints_KNN/"
fi

export CUDA_VISIBLE_DEVICES=1

model_name=RmGPT_KNN
exp_name=multi_rm_KNN
wandb_mode=disabled
ptune_name=KNN_2cls
d_model=512
device='cuda:0'

# 
timestamp=$(date +"%Y%m%d_%H%M%S")
full_exp_name="${exp_name}_${timestamp}"


python run_KNN.py \
    --model_id $full_exp_name \
    --model $model_name \
    --patch_len 256 \
    --stride 256 \
    --e_layers 4 \
    --d_model $d_model \
    --project_name $ptune_name \
    --batch_size 256 \
    --checkpoints $SAVE_PATH \
    --knn_k 10 \
    --pretrained_weight '/inspire/hdd/project/continuinglearinginlm/lijiapeng-CZXS25110021/rmgpt_pump/rmgpt_github/checkpoints_pretrain/rm+nlnVib/Base_RmGPT2_pretrain_x512_RmGPT_hd512_el4_en8_at16_it0/pretrain_checkpoint.pth'\