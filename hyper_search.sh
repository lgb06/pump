#!/bin/bash

# 定义 wandb agent 命令
COMMAND="wandb agent aipt-alchemy-team/RmGPT2_pretrain/x9gr6pmr"

# 定义每个 tmux 窗口对应的 GPU ID（从 1 到 7）
GPU_IDS=(2 3 4 5 6 7)

# 启动 tmux 会话
tmux new-session -d -s hyperparam_search-0

# 循环创建窗口并运行命令
for i in {0..5}
do
    # 创建新的 tmux 窗口，窗口名称为 Hyperparam_Search_(i+1)
    tmux new-window -t hyperparam_search-0: -n "Hyperparam_Search_$((i+1))"
    
    # 在窗口中先激活 conda 环境，然后指定对应的 GPU 并运行 wandb agent
    tmux send-keys -t "Hyperparam_Search_$((i+1))" "source ~/.bashrc && conda activate time_series && CUDA_VISIBLE_DEVICES=${GPU_IDS[$i]} $COMMAND" Enter
done

# 附加到 tmux 会话查看运行状态
tmux attach -t hyperparam_search-0
