import json
import matplotlib.pyplot as plt
import os
import sys


def _load_json(json_filepath):
    try:
        with open(json_filepath, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Error: JSON file not found at {json_filepath}")
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {json_filepath}")
    except Exception as e:
        print(f"An unexpected error occurred while reading {json_filepath}: {e}")
    return None


def _default_png_path(json_filepath, loss_key, output_path=None):
    if output_path is not None:
        return output_path
    base, _ = os.path.splitext(json_filepath)
    if loss_key == "train_loss_NLN-EMP":
        return base + ".png"
    safe_key = loss_key.replace(os.sep, "_").replace(" ", "_")
    return f"{base}_{safe_key}.png"


def _plot_loss_values(loss_values, json_filepath, loss_key, png_filepath):
    if os.path.exists(png_filepath):
        print(f"Plot already exists for {loss_key} in {json_filepath}. Skipping.")
        return

    if not loss_values:
        print(f"Warning: '{loss_key}' is empty in {json_filepath}. No plot will be generated.")
        return

    plt.figure(figsize=(12, 6))  # 设置图的大小
    plt.plot(loss_values, label=f"Training Loss ({loss_key})")
    plt.xlabel("Epoch/Step")
    plt.ylabel("Loss Value")
    plt.title(f"Loss Curve for {os.path.basename(json_filepath)}")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()  # 调整布局以防止标签重�?

    try:
        plt.savefig(png_filepath)
        print(f"Successfully saved plot to: {png_filepath}")
    except Exception as e:
        print(f"Error saving plot to {png_filepath}: {e}")
    finally:
        plt.close()  # 关闭当前图形，释放内�?


def plot_loss_from_json(json_filepath, loss_key="train_loss_NLN-EMP", output_path=None):
    """
    Reads a JSON file and plots the specified loss curve. If the plot already
    exists, it will be skipped.

    Args:
        json_filepath (str): The path to the JSON file.
        loss_key (str): The key in the JSON containing loss values.
        output_path (str, optional): Optional explicit output path for the PNG.
    """
    data = _load_json(json_filepath)
    if data is None:
        return

    if loss_key not in data or not isinstance(data[loss_key], list):
        print(f"Error: '{loss_key}' not found or is not a list in {json_filepath}")
        return

    png_filepath = _default_png_path(json_filepath, loss_key, output_path)
    _plot_loss_values(data[loss_key], json_filepath, loss_key, png_filepath)


def plot_all_train_losses(json_filepath):
    """
    Plot all loss curves whose keys start with 'train_loss' from a JSON file.
    Each curve is saved as a separate PNG named after the loss key.
    """
    data = _load_json(json_filepath)
    if data is None:
        return

    loss_keys = [k for k in data.keys() if k.startswith("train_loss")]
    if not loss_keys:
        print(f"No train_loss entries found in {json_filepath}")
        return

    for loss_key in loss_keys:
        loss_values = data.get(loss_key, [])
        png_filepath = _default_png_path(json_filepath, loss_key)
        _plot_loss_values(loss_values, json_filepath, loss_key, png_filepath)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python plot_loss.py <path_to_json_file>")
        sys.exit(1)

    json_file_path = sys.argv[1]
    plot_loss_from_json(json_file_path)
