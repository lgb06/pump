#!/bin/bash

# 环境设置
source ~/anaconda3/etc/profile.d/conda.sh
conda activate time_series

# 模型参数
model_name="RmGPT"
exp_name="RmGPT2_Scale_exp"
wandb_mode="online"

# GPU 分配
# GPU 分配
# gpu_ids=(0 1 2 3 4 5 6 7)

# 每个任务的 d_model 和 e_layers
# depths=(64 64 128 256 256 512 1024 2048)  # d_model
# layers=(1 4 3 2 4 3 2 4)                  # e_layers
gpu_ids=(4 5 6 7)
depths=(768 768 1024 1536)
layers=(8 12 24 24)
heads=(12 16 16 16)  
# 创建日志文件夹
mkdir -p logs

# 批量启动任务
for i in "${!depths[@]}"; do
  echo "Starting task $i on GPU ${gpu_ids[i]}"
  tmux new-session -d -s "pretrain_task_norm_$i" \
    " source ~/anaconda3/etc/profile.d/conda.sh;\
     conda activate time_series;\
     python run_pretrain.py \
        --n_heads ${heads[i]} \
        --is_training 1 \
        --model_id $exp_name \
        --model $model_name \
        --prompt_num 10 \
        --patch_len 256 \
        --stride 256 \
        --e_layers ${layers[i]} \
        --d_model ${depths[i]} \
        --des 'Exp' \
        --acc_it 8 \
        --batch_size 2048 \
        --learning_rate 5e-8 \
        --min_lr 1e-9 \
        --weight_decay 5e-10 \
        --min_mask_ratio 0.7 \
        --wandb_debug $wandb_mode \
        --task_data_config_path data_provider/data_config/main_result/multi_task_pretrain.yaml \
        --memory_check True \
        --device cuda:${gpu_ids[i]} > logs/pretrain_task_$i.log 2>&1 && exec bash"
done

# 显示所有 tmux 会话
tmux list-sessions
