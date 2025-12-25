# ROT Fault Diagnosis Visualization Analysis

This document describes how to use the comprehensive visualization analysis tools for the ROT fault diagnosis task.

## Overview

The visualization analysis includes:
1. **Training Curves**: Loss and accuracy over training epochs
2. **Confusion Matrix**: Classification performance visualization
3. **ROC Curves**: Multi-class ROC curves with AUC scores
4. **Attention Visualization**: Attention weights across different layers

## Usage

### Method 1: Through run.py (Recommended)

After training or testing, add the `--visualize` flag to automatically generate visualizations:

```bash
# For testing with visualization
python run.py \
    --is_training 0 \
    --pretrained_weight checkpoints/your_model/checkpoint.pth \
    --visualize \
    --visualize_output_dir visualization_results \
    --extract_attention

# For training with visualization (after training completes)
python run.py \
    --is_training 1 \
    --task_name sft_wo_cwru \
    --task_data_config_path data_provider/data_config/baseline/ROT.yaml \
    --visualize \
    --visualize_output_dir visualization_results
```

### Method 2: Standalone Visualization Script

You can also run the visualization analysis independently:

```bash
python visualize_analysis.py \
    --checkpoint_path checkpoints/your_model/checkpoint.pth \
    --output_dir visualization_results \
    --extract_attention \
    --task_data_config_path data_provider/data_config/baseline/ROT.yaml \
    --d_model 256 \
    --n_heads 16 \
    --e_layers 5 \
    --expert_num 8 \
    --activated_expert 4
```

## Output Files

The visualization analysis will generate the following files in the output directory:

1. **training_curves.png**: Training loss and validation accuracy curves
2. **confusion_matrix.png**: Confusion matrix (both raw counts and normalized)
3. **roc_curves.png**: ROC curves for all classes with AUC scores
4. **attention_visualization.png**: Attention weights visualization across layers
5. **visualization_report.md**: Comprehensive markdown report with all results

## Requirements

The visualization tools require the following packages:
- matplotlib
- seaborn
- scikit-learn
- numpy
- torch

All chart labels and text are in English.

## Notes

- Training history is automatically saved during training in `training_history.json` in the checkpoint directory
- If training history is not available, training curves will be skipped
- Attention visualization requires the `--extract_attention` flag (enabled by default)
- The visualization script automatically handles model loading and data preparation

## Example Output Structure

```
visualization_results/
├── training_curves.png
├── confusion_matrix.png
├── roc_curves.png
├── attention_visualization.png
└── visualization_report.md
```

