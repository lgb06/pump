import importlib
import os
from typing import Dict, List, Tuple

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

    def build_memory_bank(self):
        """
        Populate memory bank using normal samples (label id == 0) from training datasets.
        """
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
            "task": entry["task_name"],
            "acc": acc,
            "auc": auc,
            "ap": ap,
            "threshold": self.args.knn_threshold,
            "score_stats": score_stats,
            "count": len(scores),
        }

    def evaluate(self):
        results = []
        for entry in self.test_loaders:
            result = self._evaluate_single_loader(entry)
            if result is not None:
                results.append(result)
                print(
                    f"[{result['task']}] samples={result['count']} "
                    f"acc={result['acc']:.4f} "
                    f"auc={result['auc'] if result['auc'] is not None else 'N/A'} "
                    f"ap={result['ap'] if result['ap'] is not None else 'N/A'} "
                    f"score_mean={result['score_stats']['mean']:.4f}"
                )
        return results

    def run(self):
        # Build memory bank from training data
        self.build_memory_bank()
        # Evaluate on test datasets in the same run
        return self.evaluate()
