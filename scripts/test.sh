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
PRETRAINED_WEIGHT_PATH="${parent_dir}/checkpoint.pth"
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
    
  # --prompt_tune_epoch 20\
  # --prompt_num 10 \
### --checkpoints $SAVE_PATH \ # output file地址



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
#   --learning_rate 1e-6\
#   --batch_size 256 \
