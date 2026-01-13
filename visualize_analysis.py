"""
Comprehensive visualization analysis for ROT fault diagnosis task
Includes: training curves, confusion matrix, ROC curves, and attention visualization
"""
import argparse
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc, roc_auc_score
from sklearn.preprocessing import label_binarize
from sklearn.manifold import TSNE
from scipy.interpolate import interp1d
from itertools import cycle
import os
import json
import sys
from pathlib import Path
from tqdm import tqdm
import yaml
from collections import defaultdict

# Add project root to path
sys.path.append('/dataWYL/WYL/PHM-Large-Model/')

from exp.exp_sup import Exp_All_Task as Exp_All_Task_SUP
from data_provider.data_factory import data_provider
from utils.ddp import is_main_process

# Set style
try:
    plt.style.use('seaborn-v0_8-darkgrid')
except:
    try:
        plt.style.use('seaborn-darkgrid')
    except:
        plt.style.use('default')
sns.set_palette("husl")

class AttentionHook:
    """Hook to extract attention weights from model"""
    def __init__(self):
        self.attention_weights = []
        self.hooks = []
    
    def register_hooks(self, model):
        """Register hooks to attention layers"""
        def seq_attn_hook(module, input, output):
            # Extract attention weights from SeqAttention module
            try:
                x = input[0] if isinstance(input, tuple) else input
                if x is None:
                    return
                B, N, C = x.shape
                # Compute QKV
                qkv = module.qkv(x).reshape(B, N, 3, module.num_heads, module.head_dim)
                q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)
                # Normalize
                q = module.q_norm(q)
                k = module.k_norm(k)
                # Compute attention weights
                attn_weights = torch.matmul(q, k.transpose(-2, -1)) * module.scale
                attn_weights = F.softmax(attn_weights, dim=-1)
                # Average across heads
                attn_weights = attn_weights.mean(dim=1)  # [B, N, N]
                self.attention_weights.append(attn_weights.detach().cpu().numpy())
            except Exception as e:
                pass
        
        # Register hooks for SeqAttention modules
        model_to_hook = model.module if hasattr(model, 'module') else model
        for name, module in model_to_hook.named_modules():
            # Hook into SeqAttention modules
            if 'attn_seq' in name or (hasattr(module, 'qkv') and hasattr(module, 'num_heads')):
                hook = module.register_forward_hook(seq_attn_hook)
                self.hooks.append(hook)
    
    def remove_hooks(self):
        """Remove all registered hooks"""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []


class VisualizationAnalyzer:
    """Comprehensive visualization analyzer for ROT fault diagnosis"""
    
    def __init__(self, args, checkpoint_path=None):
        self.args = args
        self.checkpoint_path = checkpoint_path
        self.device = args.device if isinstance(args.device, str) else f'cuda:{args.device}'
        # Get number of classes from config or use default
        if hasattr(args, 'task_data_config_path'):
            try:
                import yaml
                with open(args.task_data_config_path, 'r') as f:
                    config = yaml.load(f, Loader=yaml.FullLoader)
                    task_config = config.get('task_dataset', {}).get('PHM_ROT', {})
                    num_classes = task_config.get('num_class', 4)
                    self.num_classes = num_classes
            except:
                self.num_classes = 4
        else:
            self.num_classes = 4
        
        # ROT dataset has 4 classes: Normal, Runout, Bearing_NG, Gear_NG
        # self.class_names = ['Normal', 'Runout', 'Bearing_NG', 'Gear_NG']
        self.class_names = ['Normal', 'Abnormal']
        
        # Load experiment
        self.exp = Exp_All_Task_SUP(args)
        if checkpoint_path and os.path.exists(checkpoint_path):
            print(f'Loading checkpoint: {checkpoint_path}')
            checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
            if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
                self.exp.model.load_state_dict(checkpoint['state_dict'], strict=False)
            else:
                self.exp.model.load_state_dict(checkpoint, strict=False)
        
        self.exp.model.eval()
        self.exp.model.to(self.device)
        
        # Get test data
        _, self.test_loader_list = self.exp._get_data(flag='test')
        self.task_id = 0  # ROT task
        
        # Storage for results
        self.train_history = defaultdict(list)
        self.predictions = []
        self.true_labels = []
        self.pred_probs = []
        self.attention_weights = []
        self.features = []  # For t-SNE visualization
        self.sample_data = []  # Store sample data for attention comparison
    
    def load_training_history(self, history_file=None):
        """Load training history from file or wandb"""
        if history_file and os.path.exists(history_file):
            with open(history_file, 'r') as f:
                self.train_history = json.load(f)
            print(f'Loaded training history from {history_file}')
        else:
            # Try to load from checkpoints directory
            if self.checkpoint_path:
                checkpoint_dir = os.path.dirname(self.checkpoint_path)
                history_file = os.path.join(checkpoint_dir, 'training_history.json')
                if os.path.exists(history_file):
                    with open(history_file, 'r') as f:
                        self.train_history = json.load(f)
                    print(f'Loaded training history from {history_file}')
                else:
                    print('Warning: No training history found. Training curves will be empty.')
            else:
                print('Warning: No checkpoint path provided. Training curves will be empty.')
    
    def collect_test_results(self, extract_attention=False):
        """Collect predictions, true labels, features, and attention weights from test set"""
        print('Collecting test results...')
        self.predictions = []
        self.true_labels = []
        self.pred_probs = []
        self.features = []
        self.sample_data = []
        
        # Setup attention hook if needed
        attention_hook = None
        if extract_attention:
            attention_hook = AttentionHook()
            model_to_hook = self.exp.model.module if hasattr(self.exp.model, 'module') else self.exp.model
            attention_hook.register_hooks(model_to_hook)
        
        # Store samples for each class (for attention comparison)
        samples_per_class = {i: [] for i in range(self.num_classes)}
        
        with torch.no_grad():
            for i, (batch_x, label, condition) in enumerate(tqdm(self.test_loader_list[0], desc='Testing')):
                batch_x = batch_x.float().to(self.device)
                label = label.to(self.device)
                
                # Get classification outputs
                outputs = self.exp.model(
                    batch_x, condition, None, None, 
                    task_id=self.task_id, 
                    task_name='classification'
                )
                
                # Get features for t-SNE
                features = self.exp.model(
                    batch_x, condition, None, None,
                    task_id=self.task_id,
                    task_name='get_feature'
                )
                
                # Get probabilities
                probs = F.softmax(outputs, dim=1)
                preds = torch.argmax(outputs, dim=1)
                true = torch.argmax(label, dim=1)
                
                self.predictions.extend(preds.cpu().numpy())
                self.true_labels.extend(true.cpu().numpy())
                self.pred_probs.extend(probs.cpu().numpy())
                self.features.extend(features.cpu().numpy())
                
                # Store sample data for attention visualization (one sample per class)
                for j in range(batch_x.shape[0]):
                    true_label = true[j].item()
                    if len(samples_per_class[true_label]) < 2:  # Store 2 samples per class
                        samples_per_class[true_label].append({
                            'data': batch_x[j:j+1].cpu(),
                            'label': true_label,
                            'pred': preds[j].item(),
                            'prob': probs[j:j+1].cpu().numpy()
                        })
        
        self.predictions = np.array(self.predictions)
        self.true_labels = np.array(self.true_labels)
        self.pred_probs = np.array(self.pred_probs)
        self.features = np.array(self.features)
        
        # Store sample data for visualization
        for class_id in range(self.num_classes):
            if samples_per_class[class_id]:
                self.sample_data.extend(samples_per_class[class_id][:2])
        
        # Extract attention weights if hook was used
        if extract_attention and attention_hook:
            if attention_hook.attention_weights:
                self.attention_weights = attention_hook.attention_weights
            attention_hook.remove_hooks()
        
        print(f'Collected {len(self.predictions)} test samples')
        print(f'Extracted features shape: {self.features.shape}')
    
    def plot_tsne_visualization(self, save_path, n_samples=1000, perplexity=30):
        """Plot t-SNE visualization of features for different fault types"""
        print('Computing t-SNE visualization...')
        
        # Sample data if too large
        if len(self.features) > n_samples:
            indices = np.random.choice(len(self.features), n_samples, replace=False)
            features_subset = self.features[indices]
            labels_subset = self.true_labels[indices]
        else:
            features_subset = self.features
            labels_subset = self.true_labels
        
        # Compute t-SNE
        tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity, n_iter=1000)
        features_2d = tsne.fit_transform(features_subset)
        
        # Plot
        plt.figure(figsize=(12, 10))
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']  # Blue, Orange, Green, Red
        for i, class_name in enumerate(self.class_names):
            if i < len(labels_subset):  # Safety check
                mask = labels_subset == i
                if np.any(mask):  # Only plot if there are samples for this class
                    plt.scatter(features_2d[mask, 0], features_2d[mask, 1], 
                               c=colors[i % len(colors)], label=class_name, alpha=0.6, s=50, 
                               edgecolors='black', linewidth=0.5)
        
        plt.xlabel('t-SNE Dimension 1', fontsize=12)
        plt.ylabel('t-SNE Dimension 2', fontsize=12)
        plt.title('t-SNE Visualization of Feature Representations', fontsize=14, fontweight='bold')
        plt.legend(fontsize=11, loc='best')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f'Saved t-SNE visualization to {save_path}')
    
    def plot_confusion_matrix(self, save_path):
        """Plot confusion matrix"""
        cm = confusion_matrix(self.true_labels, self.predictions)
        
        # Normalize confusion matrix
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Plot raw confusion matrix
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0],
                   xticklabels=self.class_names, yticklabels=self.class_names,
                   cbar_kws={'label': 'Count'})
        axes[0].set_title('Confusion Matrix (Counts)', fontsize=14, fontweight='bold')
        axes[0].set_xlabel('Predicted Label', fontsize=12)
        axes[0].set_ylabel('True Label', fontsize=12)
        
        # Plot normalized confusion matrix
        sns.heatmap(cm_normalized, annot=True, fmt='.2%', cmap='Blues', ax=axes[1],
                   xticklabels=self.class_names, yticklabels=self.class_names,
                   cbar_kws={'label': 'Percentage'})
        axes[1].set_title('Confusion Matrix (Normalized)', fontsize=14, fontweight='bold')
        axes[1].set_xlabel('Predicted Label', fontsize=12)
        axes[1].set_ylabel('True Label', fontsize=12)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f'Saved confusion matrix to {save_path}')
        
        # Calculate and return metrics
        accuracy = np.trace(cm) / np.sum(cm)
        return cm, accuracy
    
    def plot_roc_curves(self, save_path):
        """Plot ROC curves for multi-class classification"""
        # Binarize the output
        classes_list = list(range(self.num_classes))
        y_test_bin = label_binarize(self.true_labels, classes=classes_list)
        n_classes = y_test_bin.shape[1]
        
        # Compute ROC curve and ROC area for each class
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        
        for i in range(n_classes):
            if i < self.pred_probs.shape[1]:  # Safety check for number of classes
                fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], self.pred_probs[:, i])
                roc_auc[i] = auc(fpr[i], tpr[i])
        
        # Compute micro-average ROC curve and ROC area
        fpr["micro"], tpr["micro"], _ = roc_curve(y_test_bin.ravel(), self.pred_probs.ravel())
        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
        
        # Compute macro-average ROC curve and ROC area
        all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
        mean_tpr = np.zeros_like(all_fpr)
        for i in range(n_classes):
            mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
        mean_tpr /= n_classes
        fpr["macro"] = all_fpr
        tpr["macro"] = mean_tpr
        roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
        
        # Plot all ROC curves
        plt.figure(figsize=(10, 8))
        
        colors_list = ['aqua', 'darkorange', 'cornflowerblue', 'crimson', 'mediumseagreen', 'gold']
        colors = cycle(colors_list)
        for i, color in zip(range(n_classes), colors):
            if i < len(self.class_names):  # Safety check
                plt.plot(fpr[i], tpr[i], color=color, lw=2,
                        label=f'ROC curve of {self.class_names[i]} (AUC = {roc_auc[i]:.2f})')
        
        plt.plot(fpr["micro"], tpr["micro"], color='deeppink', linestyle='--', lw=2,
                label=f'Micro-average ROC curve (AUC = {roc_auc["micro"]:.2f})')
        plt.plot(fpr["macro"], tpr["macro"], color='navy', linestyle='--', lw=2,
                label=f'Macro-average ROC curve (AUC = {roc_auc["macro"]:.2f})')
        
        plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Random')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title('Multi-class ROC Curves', fontsize=14, fontweight='bold')
        plt.legend(loc="lower right", fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f'Saved ROC curves to {save_path}')
        
        return roc_auc
    
    def plot_attention_visualization(self, save_path, num_samples_per_class=2):
        """Visualize attention weights for different fault types - comparing different classes"""
        if not self.sample_data:
            print('No sample data available. Skipping attention visualization.')
            return
        
        print('Generating attention visualization for different fault types...')
        
        # Re-extract attention for specific samples
        attention_hook = AttentionHook()
        model_to_hook = self.exp.model.module if hasattr(self.exp.model, 'module') else self.exp.model
        attention_hook.register_hooks(model_to_hook)
        
        sample_attentions = []
        sample_labels = []
        
        with torch.no_grad():
            for sample in self.sample_data[:num_samples_per_class * self.num_classes]:
                batch_x = sample['data'].to(self.device)
                label = sample['label']
                
                # Clear previous attention weights
                attention_hook.attention_weights = []
                
                # Forward pass to get attention
                _ = self.exp.model(
                    batch_x, batch_x, None, None,
                    task_id=self.task_id,
                    task_name='classification'
                )
                
                # Get attention from last layer
                if attention_hook.attention_weights:
                    # Get attention from the last layer
                    last_attn = attention_hook.attention_weights[-1]
                    if len(last_attn.shape) == 4:  # [batch, heads, seq_len, seq_len]
                        last_attn = last_attn.mean(axis=1)  # Average across heads
                    if len(last_attn.shape) == 3:  # [batch, seq_len, seq_len]
                        last_attn = last_attn[0]  # Take first sample
                    
                    sample_attentions.append(last_attn)
                    sample_labels.append(label)
        
        attention_hook.remove_hooks()
        
        if not sample_attentions:
            print('No attention weights extracted. Skipping attention visualization.')
            return
        
        # Organize by class
        class_attentions = {i: [] for i in range(self.num_classes)}
        for attn, label in zip(sample_attentions, sample_labels):
            if len(class_attentions[label]) < num_samples_per_class:
                class_attentions[label].append(attn)
        
        # Create visualization
        fig, axes = plt.subplots(self.num_classes, num_samples_per_class, 
                                figsize=(5*num_samples_per_class, 4*self.num_classes))
        if self.num_classes == 1:
            axes = axes.reshape(1, -1)
        if num_samples_per_class == 1:
            axes = axes.reshape(-1, 1)
        
        for class_idx, class_name in enumerate(self.class_names):
            for sample_idx in range(num_samples_per_class):
                ax = axes[class_idx, sample_idx]
                
                if class_idx < len(class_attentions) and sample_idx < len(class_attentions[class_idx]):
                    attn = class_attentions[class_idx][sample_idx]
                    im = ax.imshow(attn, cmap='hot', aspect='auto', interpolation='nearest')
                    
                    if sample_idx == 0:
                        ax.set_ylabel(f'{class_name}\nQuery Position', fontsize=11, fontweight='bold')
                    if class_idx == self.num_classes - 1:
                        ax.set_xlabel('Key Position', fontsize=10)
                    if class_idx == 0:
                        ax.set_title(f'Sample {sample_idx+1}', fontsize=10, fontweight='bold')
                    
                    plt.colorbar(im, ax=ax, fraction=0.046)
                else:
                    ax.axis('off')
        
        plt.suptitle('Attention Patterns Comparison Across Different Fault Types', 
                    fontsize=14, fontweight='bold', y=0.995)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f'Saved attention visualization to {save_path}')
    
    def generate_markdown_report(self, output_dir, cm, accuracy, roc_auc):
        """Generate comprehensive markdown report"""
        report_path = os.path.join(output_dir, 'visualization_report.md')
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write('# ROT Fault Diagnosis - Comprehensive Analysis Report\n\n')
            f.write('## 1. Overview\n\n')
            f.write('This report presents a comprehensive analysis of the ROT fault diagnosis model, ')
            f.write('including classification results, feature representations, and model interpretability.\n\n')
            
            f.write('## 2. Classification Performance\n\n')
            f.write('### 2.1 Confusion Matrix\n\n')
            f.write('The confusion matrix shows the classification performance for each fault type.\n\n')
            f.write('![Confusion Matrix](confusion_matrix.png)\n\n')
            
            f.write('**Classification Accuracy:** {:.2%}\n\n'.format(accuracy))
            
            f.write('**Per-class Performance:**\n\n')
            f.write('| Class | True Positives | False Positives | False Negatives |\n')
            f.write('|-------|---------------|-----------------|-----------------|\n')
            for i, class_name in enumerate(self.class_names):
                tp = cm[i, i]
                fp = cm[:, i].sum() - tp
                fn = cm[i, :].sum() - tp
                f.write(f'| {class_name} | {tp} | {fp} | {fn} |\n')
            f.write('\n')
            
            f.write('### 2.2 ROC Curves\n\n')
            f.write('The ROC curves demonstrate the model\'s discriminative ability for each class.\n\n')
            f.write('![ROC Curves](roc_curves.png)\n\n')
            
            f.write('**AUC Scores:**\n\n')
            f.write('| Class | AUC Score |\n')
            f.write('|-------|----------|\n')
            for i, class_name in enumerate(self.class_names):
                f.write(f'| {class_name} | {roc_auc[i]:.4f} |\n')
            f.write(f'| Micro-average | {roc_auc["micro"]:.4f} |\n')
            f.write(f'| Macro-average | {roc_auc["macro"]:.4f} |\n')
            f.write('\n')
            
            f.write('## 3. Feature Representation Analysis\n\n')
            f.write('### 3.1 t-SNE Visualization\n\n')
            f.write('The t-SNE visualization shows the distribution of learned features in a 2D space. ')
            f.write('Different fault types should form distinct clusters, demonstrating the model\'s ability to learn discriminative representations.\n\n')
            f.write('![t-SNE Visualization](tsne_visualization.png)\n\n')
            
            f.write('**Analysis:**\n\n')
            f.write('- Well-separated clusters indicate that the model has learned distinct feature representations for each fault type.\n')
            f.write('- Overlapping clusters suggest that some fault types may have similar characteristics.\n')
            f.write('- The spatial distribution reflects the model\'s internal representation of different fault patterns.\n\n')
            
            f.write('## 4. Model Interpretability\n\n')
            f.write('### 4.1 Attention Visualization Across Fault Types\n\n')
            if self.attention_weights or self.sample_data:
                f.write('The attention visualization compares attention patterns across different fault types. ')
                f.write('This shows which parts of the input signal the model focuses on when classifying different fault types.\n\n')
                f.write('![Attention Visualization](attention_visualization.png)\n\n')
                f.write('**Key Observations:**\n\n')
                f.write('- Different fault types may exhibit distinct attention patterns.\n')
                f.write('- The attention maps reveal which temporal regions are most important for each fault type.\n')
                f.write('- Comparing attention patterns helps understand how the model distinguishes between different fault conditions.\n\n')
            else:
                f.write('Attention weights not available.\n\n')
            
            f.write('## 5. Summary\n\n')
            f.write(f'- **Overall Accuracy:** {accuracy:.2%}\n')
            f.write(f'- **Macro-average AUC:** {roc_auc["macro"]:.4f}\n')
            f.write(f'- **Micro-average AUC:** {roc_auc["micro"]:.4f}\n')
            f.write('\n')
            f.write('The model demonstrates good performance in classifying ROT fault types, ')
            f.write('with attention mechanisms effectively focusing on relevant signal features.\n')
        
        print(f'Generated markdown report: {report_path}')
    
    def run_full_analysis(self, output_dir, extract_attention=True):
        """Run complete visualization analysis"""
        os.makedirs(output_dir, exist_ok=True)
        
        print('='*60)
        print('Starting Comprehensive Visualization Analysis')
        print('='*60)
        
        # Collect test results
        self.collect_test_results(extract_attention=extract_attention)
        
        # Generate all visualizations
        print('\nGenerating visualizations...')
        cm, accuracy = self.plot_confusion_matrix(os.path.join(output_dir, 'confusion_matrix.png'))
        # roc_auc = self.plot_roc_curves(os.path.join(output_dir, 'roc_curves.png'))
        
        # Generate t-SNE visualization
        self.plot_tsne_visualization(os.path.join(output_dir, 'tsne_visualization.png'))
        
        # Generate attention visualization
        if extract_attention:
            self.plot_attention_visualization(os.path.join(output_dir, 'attention_visualization.png'))
        
        # Generate markdown report
        # self.generate_markdown_report(output_dir, cm, accuracy, roc_auc)
        
        print('\n' + '='*60)
        print('Visualization Analysis Complete!')
        print(f'Results saved to: {output_dir}')
        print('='*60)


def main():
    parser = argparse.ArgumentParser(description='ROT Fault Diagnosis Visualization Analysis')
    
    # Basic config (same as run.py)
    parser.add_argument('--task_name', type=str, default='sft_wo_cwru')
    parser.add_argument('--model_id', type=str, default='full_lora')
    parser.add_argument('--model', type=str, default='RmGPT')
    parser.add_argument('--data', type=str, default='All')
    parser.add_argument('--task_data_config_path', type=str,
                       default='data_provider/data_config/baseline/ROT.yaml')
    parser.add_argument('--device', type=str, default='cuda:0')
    
    # Model settings
    parser.add_argument('--d_model', type=int, default=256)
    parser.add_argument('--n_heads', type=int, default=16)
    parser.add_argument('--e_layers', type=int, default=5)
    parser.add_argument('--input_len', type=int, default=2048)
    parser.add_argument('--expert_num', type=int, default=8)
    parser.add_argument('--activated_expert', type=int, default=4)
    
    # Tokenizer settings
    parser.add_argument('--patch_len', type=int, default=256)
    parser.add_argument('--stride', type=int, default=256)
    parser.add_argument('--codebook_size', type=int, default=1024)
    parser.add_argument('--tokenizer_path', type=str,
                       default='checkpoints/Pretrain_Token_pretrain_Tokenizer_cs1024_it0/pretrain_checkpoint.pth')
    
    # Other settings
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--label_type', type=str, default='local')
    parser.add_argument('--num_workers', type=int, default=16)
    parser.add_argument('--ddp', type=bool, default=False)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--pretrained_weight', type=str, default=None)
    parser.add_argument('--checkpoints', type=str, default='checkpoints/')
    parser.add_argument('--lora_transform', type=bool, default=False)
    parser.add_argument('--efficiency_tuning', type=bool, default=False)
    parser.add_argument('--large_model', action='store_true', default=True)
    parser.add_argument('--mode_debug', type=bool, default=False)
    
    # Visualization specific
    parser.add_argument('--checkpoint_path', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--output_dir', type=str, default='visualization_results',
                       help='Output directory for visualization results')
    parser.add_argument('--extract_attention', action='store_true', default=True,
                       help='Whether to extract and visualize attention weights')
    parser.add_argument('--history_file', type=str, default=None,
                       help='Path to training history JSON file')
    
    args = parser.parse_args()
    
    # Create analyzer
    analyzer = VisualizationAnalyzer(args, checkpoint_path=args.checkpoint_path)
    
    # Run analysis
    analyzer.run_full_analysis(
        output_dir=args.output_dir,
        extract_attention=args.extract_attention
    )


if __name__ == '__main__':
    main()

