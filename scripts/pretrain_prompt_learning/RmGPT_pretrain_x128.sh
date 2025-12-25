model_name=RmGPT
exp_name=RmGPT2_pretrain_x512
wandb_mode=disabled
ptune_name=prompt_tuning
d_model=512

random_port=$((RANDOM % 9000 + 1000))

export WANDB_API_KEY="7c05777e3b354a6cc5956d04e1b3a1cda2a16be0"
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export OMP_NUM_THREADS=1

# Pretrain
# /home/ps/anaconda3/envs/time_series/bin/torchrun --nnodes 1 --nproc-per-node 8 --master_port $random_port run_pretrain.py \
#   --is_training 1 \
#   --model_id $exp_name \
#   --model $model_name \
#   --prompt_num 10 \
#   --patch_len 256 \
#   --stride 256 \
#   --e_layers 4 \
#   --d_model $d_model \
#   --des 'Exp' \
#   --acc_it 1 \
#   --batch_size 200 \
#   --learning_rate 5e-9 \
#   --min_lr 1e-4 \
#   --weight_decay 5e-10\
#   --train_epochs 10 \
#   --warmup_epochs 0 \
#   --min_keep_ratio 0.5 \
#   --right_prob 0.5 \
#   --min_mask_ratio 0.3 \
#   --max_mask_ratio 0.5 \
#   --wandb_debug $wandb_mode \
#   --task_data_config_path data_provider/data_config/main_result/multi_task_pretrain.yaml\
#   --min_keep_ratio 0.1 \
#   --ddp True\
#   # --device 'cuda:0'



# # 单机卡启动方法
python run_pretrain.py \
  --is_training 1 \
  --model_id $exp_name \
  --model $model_name \
  --prompt_num 10 \
  --patch_len 256 \
  --stride 256 \
  --e_layers 4 \
  --d_model $d_model \
  --des 'Exp' \
  --acc_it 1 \
  --batch_size 512\
  --learning_rate 5e-7 \
  --min_lr 1e-4 \
  --weight_decay 5e-10\
  --pretrain_epochs 10 \
  --warmup_epochs 0 \
  --wandb_debug $wandb_mode \
  --pretrain_data_config_path data_provider/data_config/main_result/multi_task_pretrain.yaml\
  --device 'cuda:5' \
  --checkpoints './checkpoints_temp/' \

  # --min_mask_ratio 0.6 \
  # --max_mask_ratio 0.8 \
  # --min_keep_ratio 0.5 \
  # --right_prob 0.5 \


# # # 单机卡启动方法
# python run_pretrain.py \
#   --is_training 1 \
#   --model_id $exp_name \
#   --model $model_name \
#   --prompt_num 10 \
#   --patch_len 256 \
#   --stride 256 \
#   --e_layers 4 \
#   --d_model $d_model \
#   --des 'Exp' \
#   --acc_it 1 \
#   --batch_size 512\
#   --learning_rate 5e-7 \
#   --min_lr 1e-4 \
#   --weight_decay 5e-10\
#   --train_epochs 10 \
#   --warmup_epochs 0 \
#   --min_keep_ratio 0.5 \
#   --right_prob 0.5 \
#   --min_mask_ratio 0.6 \
#   --max_mask_ratio 0.8 \
#   --wandb_debug $wandb_mode \
#   --task_data_config_path data_provider/data_config/main_result/multi_task_pretrain.yaml\
#   --min_keep_ratio 0.1 \
#   --device 'cuda:5'

# # Prompt tuning
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
#   --prompt_tune_epoch 10\
#   --train_epochs 10 \
#   --debug $wandb_mode \
#   --project_name $ptune_name \
#   --task_data_config_path  data_provider/multi_task.yaml \
#   --pretrained_weight 'checkpoints/ALL_task_test_RmGPT_All_ftM_dm512_el4_test_0/pretrain_checkpoint.pth'
#   --batch_size 256 \

#   # --device cuda:0
#   # --learning_rate 0.0001 \
#   # --acc_it 32 \
#   # --clip_grad 100 \
