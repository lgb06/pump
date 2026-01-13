#!/bin/bash

# 检查第一个参数是否存在
if [ -n "$1" ]; then
  # 如果存在
  parent_dir="$1"
else
  # 如果第一个参数不存在，打印错误信息并退出脚本
  echo "错误：未提供第一个参数。" >&2 # 将错误信息输出到标准错误
  echo "请提供一个父目录作为第一个参数。" >&2
  exit 1 # 以非零状态码退出，表示脚本执行失败
fi

export CUDA_VISIBLE_DEVICES=2

model_name=RmGPT
exp_name=multi_rm
wandb_mode=disabled
ptune_name=sft_2cls
d_model=512
device='cuda:0'

# 
timestamp=$(date +"%Y%m%d_%H%M%S")
full_exp_name="${exp_name}_${timestamp}"



PRETRAINED_WEIGHT_PARENTDIR="${parent_dir}/NLN-EMP-test/"
# PRETRAINED_WEIGHT_PATH="${parent_dir}/checkpoint.pth"
PRETRAINED_WEIGHT_PATH="${parent_dir}/pretrain_checkpoint.pth"
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
    --checkpoints $PRETRAINED_WEIGHT_PARENTDIR \
    --pretrained_weight $PRETRAINED_WEIGHT_PATH \
    