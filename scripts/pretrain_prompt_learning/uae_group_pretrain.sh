#!/usr/bin/env bash
# Pretrain RmGPT only on UEA ECG200 (2-class) with max_seq_len slicing

model_name=RmGPT
exp_name=RmGPT2_pretrain_x512
wandb_mode=disabled
ptune_name=prompt_tuning
d_model=512

# 
timestamp=$(date +"%Y%m%d_%H%M%S")
full_exp_name="${exp_name}_${timestamp}"

no=3

export CUDA_LAUNCH_BLOCKING=1
export CUDA_VISIBLE_DEVICES="${no}"

export PYTORCH_SDP_DISABLE_FLASH=1
export PYTORCH_SDP_DISABLE_MEM_EFFICIENT=1


python run_pretrain.py \
  --is_training 1 \
  --task_name UEA_only_pretrain \
  --model_id $exp_name \
  --model RmGPT \
  --pretrain_data_config_path "data_provider/data_config/main_result/uae_3_pretrain${no}.yaml" \
  --prompt_num 10 \
  --patch_len 256 \
  --stride 256 \
  --e_layers 4 \
  --d_model $d_model \
  --des 'Exp' \
  --acc_it 1 \
  --batch_size 512 \
  --learning_rate 5e-7 \
  --min_lr 1e-4 \
  --weight_decay 5e-10 \
  --pretrain_epochs 30 \
  --warmup_epochs 0 \
  --wandb_debug $wandb_mode \
  --device 'cuda:0' \
  --checkpoints "./checkpoints_pretrain/uae/group${no}_${full_exp_name}" \
  --expert_num 8