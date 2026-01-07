from datetime import datetime
import importlib
import os
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import yaml
from data_provider.data_factory import data_provider


def read_task_data_config(config_path: str) -> Dict:
    with open(config_path, "r") as config_file:
        config = yaml.load(config_file, Loader=yaml.FullLoader)
    return config.get("task_dataset", {})


def get_task_data_config_list(task_data_config: Dict, default_batch_size: int = None) -> List[Tuple[str, Dict]]:
    task_data_config_list = []
    for task_name, task_config in task_data_config.items():
        task_config["max_batch"] = default_batch_size
        task_data_config_list.append([task_name, task_config])
    return task_data_config_list


class ExpKNN:
    """
    Build a KNN memory bank from normal training samples and evaluate anomaly scores.
    """

    def __init__(self, args):
        self.args = args
        self.device = torch.device(args.device if torch.cuda.is_available() else "cpu")

        # Configs
        self.train_task_data_config = read_task_data_config(args.train_task_data_config_path)
        self.train_task_data_config_list = get_task_data_config_list(
            self.train_task_data_config, default_batch_size=args.batch_size
        )
        self.test_task_data_config_list = []
        for cfg_path in args.test_task_data_config_paths:
            cfg = read_task_data_config(cfg_path)
            self.test_task_data_config_list.extend(
                get_task_data_config_list(cfg, default_batch_size=args.batch_size)
            )

        # Model setup
        self.configs_list = self.train_task_data_config_list + self.test_task_data_config_list
        self.task_name_to_id = {name: idx for idx, (name, _) in enumerate(self.configs_list)}
        self.model = self._build_model()

        # Data
        self.train_loaders = self._get_loaders(self.train_task_data_config_list, flag="train")
        self.test_loaders = self._get_loaders(self.test_task_data_config_list, flag="test")
        self.visualization_dir = Path(getattr(args, "visualization_dir", "logs/KNN_2cls/KNN/visual_anoscore_distribution"))

    def _build_model(self):
        module = importlib.import_module("models." + self.args.model)
        model = module.Model(self.args, self.configs_list).to(self.device)
        if self.args.pretrained_weight is not None and os.path.exists(self.args.pretrained_weight):
            pretrain_weight_path = self.args.pretrained_weight
            print('loading pretrained model:', pretrain_weight_path)
            if 'pretrain_checkpoint.pth' in pretrain_weight_path:
                state_dict = torch.load(
                    pretrain_weight_path, map_location='cpu', weights_only=False)['student']
                ckpt = {}
                for k, v in state_dict.items():
                    if not ('cls_prompts' in k):
                        ckpt[k] = v
            else:
                ckpt = torch.load(pretrain_weight_path, map_location='cpu')
            msg = model.load_state_dict(ckpt, strict=False)
            print(msg)
        model.eval()
        return model

    def _get_loaders(self, config_list: List[Tuple[str, Dict]], flag: str):
        loaders = []
        for task_name, task_config in config_list:
            data_set, data_loader = data_provider(self.args, task_config, flag, ddp=False)
            task_id = self.task_name_to_id[task_name]
            loaders.append(
                {
                    "task_name": task_name,
                    "task_id": task_id,
                    "config": task_config,
                    "loader": data_loader,
                }
            )
            print(f"{flag} dataset {task_name}: {len(data_set)} samples")
        return loaders

    def _prepare_sorted_scores(self, scores: torch.Tensor, labels: torch.Tensor):
        """
        Return scores/labels sorted by score in ascending order.
        """
        scores_cpu = scores.detach().cpu()
        labels_cpu = labels.detach().cpu()
        sorted_scores, sorted_indices = torch.sort(scores_cpu)
        sorted_labels = labels_cpu[sorted_indices]
        return sorted_scores, sorted_labels

    def _save_scores_and_plot(self, sorted_scores: torch.Tensor, sorted_labels: torch.Tensor, dataset_name: str, step: int = 2000):
        """
        Save sorted scores/labels to disk and generate a scatter plot colored by label.
        """
        safe_name = dataset_name.replace("/", "_")
        self.visualization_dir.mkdir(parents=True, exist_ok=True)
         # 生成当前时间字符串（文件名安全）
        ts = datetime.now().strftime("%Y%m%d_%H%M%S") 

        data_path = self.visualization_dir / f"{safe_name}_K{self.args.knn_k}_{ts}_scores_labels.pt"
        torch.save(
            {"scores": sorted_scores, "labels": sorted_labels, "dataset": dataset_name},
            data_path,
        )

        x_axis = range(len(sorted_scores))
        label_values = sorted_labels.numpy()
        score_values = sorted_scores.numpy()
        unique_labels = sorted({int(l) for l in label_values.tolist()})
    
        # ---- subsample: take 1 sample every `step` (the first in each block) ----
        n = score_values.numel()
        idx = torch.arange(0, n, step, dtype=torch.long)   # [M]
        scores_s = score_values[idx]
        labels_s = label_values[idx]
    
        # x-axis uses the ORIGINAL indices to keep the x meaning unchanged
        x_axis = idx.numpy()
        score_values = scores_s.numpy()
        label_values = labels_s.numpy()
    
        plt.figure(figsize=(8, 4))
    
        # 1) anomaly score：淡蓝色点
        plt.scatter(
            x_axis,
            score_values,
            c="lightblue",
            s=18,
            alpha=0.75,
            edgecolors="none",
            label="Anomaly score (subsampled)",
        )
    
        # 2) label(0/1)：红色点（y=0 或 y=1）
        plt.scatter(
            x_axis,
            label_values,
            c="red",
            s=22,
            alpha=0.85,
            edgecolors="none",
            label="Label (0/1, subsampled)",
        )
    
        plt.xlabel(f"Sample index (sorted by anomaly score, every {step} samples)")
        plt.ylabel("Anomaly score")
        plt.title(f"{dataset_name} anomaly scores (blue) + labels (red)")
        plt.legend(frameon=False, loc="best")
        plt.tight_layout()
    
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        plot_path = self.visualization_dir / f"{safe_name}_K{self.args.knn_k}_{ts}_scores_plot.png"
        plt.savefig(plot_path, dpi=300)
        plt.close()

        return data_path, plot_path

    def build_memory_bank(self):
        """
        Populate memory bank using normal samples (label id == 0) from training datasets.
        """
        print("======= Building KNN memory bank from training data... ======= ")
        self.model.reset_memory_bank()
        self.model.eval()
        with torch.no_grad():
            for entry in self.train_loaders:
                for batch in entry["loader"]:
                    batch_x, labels = batch[:2]
                    batch_x = batch_x.float().to(self.device)
                    labels = labels.to(self.device)
                    _ = self.model(
                        batch_x,
                        batch_x,
                        None,
                        None,
                        task_id=entry["task_id"],
                        task_name=entry["config"]["task_name"],
                        labels=labels,
                        update_memory=True,
                        return_anomaly_score=False,
                    )
        print(f"Memory bank size: {self.model.memory_bank.shape[0]}")

    def _evaluate_single_loader(self, entry: Dict):
        all_scores = []
        all_labels = []
        with torch.no_grad():
            for batch in entry["loader"]:
                batch_x, labels = batch[:2]
                batch_x = batch_x.float().to(self.device)
                labels = labels.to(self.device)
                _, anomaly_scores = self.model(
                    batch_x,
                    batch_x,
                    None,
                    None,
                    task_id=entry["task_id"],
                    task_name=entry["config"]["task_name"],
                    labels=labels,
                    update_memory=False,
                    return_anomaly_score=True,
                )
                all_scores.append(anomaly_scores.detach().cpu())
                all_labels.append(labels.detach().cpu())

        if not all_scores:
            return None

        scores = torch.cat(all_scores)
        label_ids = torch.cat(all_labels)
        if label_ids.dim() > 1:
            label_ids = label_ids.argmax(dim=1)

        preds = (scores > self.args.knn_threshold).long()
        acc = (preds.cpu() == label_ids.cpu()).float().mean().item()

        auc = None
        ap = None
        try:
            from sklearn.metrics import average_precision_score, roc_auc_score

            unique_labels = torch.unique(label_ids)
            if unique_labels.numel() > 1:
                auc = roc_auc_score(label_ids.numpy(), scores.numpy())
                ap = average_precision_score(label_ids.numpy(), scores.numpy())
        except Exception as exc:
            print(f"Metric computation skipped for {entry['task_name']}: {exc}")

        score_stats = {
            "mean": scores.mean().item(),
            "std": scores.std().item(),
            "normal_mean": scores[label_ids == 0].mean().item() if (label_ids == 0).any() else None,
            "abnormal_mean": scores[label_ids == 1].mean().item() if (label_ids == 1).any() else None,
        }

        return {
            "task": entry["config"]["dataset_name"],
            "acc": acc,
            "auc": auc,
            "ap": ap,
            "threshold": self.args.knn_threshold,
            "score_stats": score_stats,
            "count": len(scores),
            "scores": scores,
            "labels": label_ids,
        }

    def evaluate(self):
        print("======= Evaluating on test datasets... ======= ")
        results = []
        global_scores = []
        global_labels = []
        for entry in self.test_loaders:
            result = self._evaluate_single_loader(entry)
            if result is not None:
                results.append(result)
                global_scores.append(result["scores"])
                global_labels.append(result["labels"])
                print(
                    f"[{result['task']}] samples={result['count']} "
                    f"acc={result['acc']:.4f} "
                    f"auc={result['auc'] if result['auc'] is not None else 'N/A'} "
                    f"ap={result['ap'] if result['ap'] is not None else 'N/A'} "
                    f"score_mean={result['score_stats']['mean']:.4f}"
                )
        if global_scores:
            merged_scores = torch.cat(global_scores)
            merged_labels = torch.cat(global_labels)
            sorted_scores, sorted_labels = self._prepare_sorted_scores(merged_scores, merged_labels)
            data_path, plot_path = self._save_scores_and_plot(
                sorted_scores, sorted_labels, dataset_name="all_datasets"
            )
            print(
                f"[ALL] samples={len(sorted_scores)} "
                f"Saved sorted scores/labels to {data_path} | plot: {plot_path}"
            )
        return results

    def run(self):
        # Build memory bank from training data
        self.build_memory_bank()
        # Evaluate on test datasets in the same run
        return self.evaluate()
