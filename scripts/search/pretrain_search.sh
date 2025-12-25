min_mask_ratio=$1
max_mask_ratio=$(echo "$min_mask_ratio + 0.2" | bc)

python run_pretrain.py \
    --min_mask_ratio $min_mask_ratio \
    --max_mask_ratio $max_mask_ratio \

python run_fewshot.py \
    --pretrained_weight "checkpoints/Pretrain_test_RmGPT_All_ftM_dm512_el4_test_0_${min_mask_ratio}/pretrain_checkpoint.pth"
