import os
import sys

if __name__ == "__main__":
    # 从命令行获取传递的超参数
    min_mask_ratio = sys.argv[1].split('=')[1]

    # 构建 bash 命令
    command = f"bash scripts/search/pretrain_search.sh {min_mask_ratio}"
    print(f"Executing command: {command}")
    os.system(command)