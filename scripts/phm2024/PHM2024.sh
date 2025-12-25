model_name=RmGPT
exp_name=test
wandb_mode=online
ptune_name=PHM2024_stage2
d_model=512

# Supervised learning
python run.py \
  --is_training 1 \
  --model_id $exp_name \
  --model $model_name \
  --lradj supervised \
  --prompt_num 10 \
  --patch_len 256 \
  --stride 256 \
  --e_layers 4 \
  --d_model $d_model \
  --des 'Exp' \
  --itr 1 \
  --weight_decay 0 \
  --prompt_tune_epoch 0\
  --train_epochs 4 \
  --debug $wandb_mode \
  --project_name $ptune_name \
  --input_len 2048\
  --task_data_config_path  data_provider/data_config/PHM_competition/PHMchallenge_train.yaml \
  --learning_rate 3e-5\
  --batch_size 32 \
  # --pretrained_weight 'checkpoints/ALL_task_RmGPT_pretrain_x512_RmGPT_All_ftM_dm512_el4_Exp_0/ptune_checkpoint.pth'\
  # --pretrained_weight 'None'\

