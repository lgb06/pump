import argparse
import random
from pprint import pformat, pprint

import numpy as np
import torch

from exp.exp_KNN import ExpKNN


def parse_test_config_paths(raw: str):
    return [p.strip() for p in raw.split(",") if p.strip()]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RmGPT KNN anomaly detection")

    # Basic config
    parser.add_argument("--task_name", type=str, default="knn_anomaly", help="task name")
    parser.add_argument("--model_id", type=str, default="knn", help="model id")
    parser.add_argument("--model", type=str, default="RmGPT_KNN", help="model name")

    # Data
    parser.add_argument(
        "--train_task_data_config_path",
        type=str,
        default="data_provider/data_config/pump/multi_task.yaml",
        help="path to training data config (used for memory bank build)",
    )
    parser.add_argument(
        "--test_task_data_config_paths",
        type=str,
        default="data_provider/data_config/pump/NLNEMP.yaml",
        # default="data_provider/data_config/pump/NLNEMP.yaml,data_provider/data_config/pump/NLNEMP_Elec.yaml",
        help="comma separated paths to test data configs",
    )
    parser.add_argument("--data", type=str, default="All", help="dataset type hint")
    parser.add_argument("--freq", type=str, default="h", help="frequency for time features encoding")
    parser.add_argument("--batch_size", type=int, default=16, help="batch size")
    parser.add_argument("--num_workers", type=int, default=8, help="dataloader workers")

    # Device
    parser.add_argument("--device", type=str, default="cuda:0", help="device id")

    # Model
    parser.add_argument("--d_model", type=int, default=256, help="model dimension")
    parser.add_argument("--n_heads", type=int, default=16, help="number of heads")
    parser.add_argument("--e_layers", type=int, default=5, help="number of encoder layers")
    parser.add_argument("--patch_len", type=int, default=256, help="patch length")
    parser.add_argument("--stride", type=int, default=256, help="stride length")
    parser.add_argument("--input_len", type=int, default=2048, help="input sequence length")
    parser.add_argument("--dropout", type=float, default=0.1, help="dropout")
    parser.add_argument("--expert_num", type=int, default=8, help="number of shared experts")
    parser.add_argument("--activated_expert", type=int, default=4, help="activated experts")
    parser.add_argument("--mode_debug", action="store_true", default=False, help="enable debug mode")

    # KNN settings
    parser.add_argument("--knn_threshold", type=float, default=0.5, help="threshold on anomaly score")
    parser.add_argument("--knn_k", type=int, default=5, help="K value for KNN ")

    # Checkpoints
    parser.add_argument("--pretrained_weight", type=str, default=None, help="path to pretrained weights")
    parser.add_argument("--checkpoints", type=str, default="checkpoints", help="checkpoint directory (unused)")

    # NLN-EMP specific
    parser.add_argument("--cls_mode", type=int, default=-1, help="class grouping mode for NLN-EMP")
    parser.add_argument("--project_name", type=str, default="knn", help="project name for loaders")

    # Reproducibility
    parser.add_argument("--fix_seed", type=int, default=2024, help="random seed")

    args = parser.parse_args()

    args.test_task_data_config_paths = parse_test_config_paths(args.test_task_data_config_paths)

    # Derived/forced settings for KNN
    args.classification_method = "knn"
    args.knn_distance = "cosine"
    args.task_data_config_path = args.train_task_data_config_path

    # num_classes mapping (align with existing scripts)
    cls_to_num_class_mapping = {
        -1: 2,
        0: 21,
        1: 21,
        3: 12,
        11: 6,
        12: 7,
        13: 8,
    }
    args.num_classes = cls_to_num_class_mapping.get(args.cls_mode, 2)

    # num_channels hint (used by some modules for NLN-EMP)
    if not hasattr(args, "num_channels"):
        args.num_channels = 5
        if args.data.lower() == "electric":
            args.num_channels = 6

    # Seed
    if args.fix_seed is not None:
        random.seed(args.fix_seed)
        torch.manual_seed(args.fix_seed)
        np.random.seed(args.fix_seed)

    pprint("Args in experiment:")
    pprint(pformat(vars(args), indent=4))

    exp = ExpKNN(args)
    results = exp.run()

    print("======= KNN evaluation completed =======")
    for res in results:
        print(
            f"{res['task']}: count={res['count']} acc={res['acc']:.4f} "
            f"auc={res['auc'] if res['auc'] is not None else 'N/A'} "
            f"ap={res['ap'] if res['ap'] is not None else 'N/A'} "
            f"threshold={res['threshold']} "
            f"score_mean={res['score_stats']['mean']:.4f}"
        )
