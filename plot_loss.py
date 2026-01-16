import json
import matplotlib.pyplot as plt
import os
import sys

def plot_loss_from_json(json_filepath):
    """
    Reads a JSON file, checks if a corresponding plot already exists,
    and if not, plots the 'train_loss_NLN-EMP' curve and saves it as a PNG.

    Args:
        json_filepath (str): The path to the JSON file.
    """
    # 1. 构建输出图像文件的路径
    #  - 移除 .json 后缀，添加 .png
    #  - 图像文件将保存在与 JSON 文件相同的目录下
    base, _ = os.path.splitext(json_filepath)
    png_filepath = base + ".png"

    # 2. 检查图像文件是否已存在
    if os.path.exists(png_filepath):
        print(f"Plot already exists for {json_filepath}. Skipping.")
        return

    print(f"Plotting loss for: {json_filepath}")

    # 3. 读取 JSON 文件
    try:
        with open(json_filepath, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: JSON file not found at {json_filepath}")
        return
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {json_filepath}")
        return
    except Exception as e:
        print(f"An unexpected error occurred while reading {json_filepath}: {e}")
        return

    # 4. 提取 loss 值
    # 您可以根据需要修改这个键，例如 'train_loss'
    # loss_key = "train_loss_PHM_SLIET"
    loss_key = "train_loss_NLN-EMP"
    if loss_key not in data or not isinstance(data[loss_key], list):
        print(f"Error: '{loss_key}' not found or is not a list in {json_filepath}")
        return

    loss_values = data[loss_key]

    if not loss_values:
        print(f"Warning: '{loss_key}' is empty in {json_filepath}. No plot will be generated.")
        return

    # 5. 绘制 loss 曲线
    plt.figure(figsize=(12, 6)) # 设置图的大小
    plt.plot(loss_values, label='Training Loss (NLN-EMP)')
    plt.xlabel("Epoch/Step")
    plt.ylabel("Loss Value")
    plt.title(f"Loss Curve for {os.path.basename(json_filepath)}")
    plt.grid(True)
    plt.legend()
    plt.tight_layout() # 调整布局以防止标签重叠

    # 6. 保存图像
    try:
        plt.savefig(png_filepath)
        print(f"Successfully saved plot to: {png_filepath}")
    except Exception as e:
        print(f"Error saving plot to {png_filepath}: {e}")

    plt.close() # 关闭当前图形，释放内存

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python plot_loss.py <path_to_json_file>")
        sys.exit(1)

    json_file_path = sys.argv[1]
    plot_loss_from_json(json_file_path)

    