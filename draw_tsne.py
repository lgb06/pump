import argparse
import os
from pathlib import Path
from typing import List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from sklearn.manifold import TSNE  # noqa: E402


def _sample_points(data: np.ndarray, labels: np.ndarray, max_points: int, highlight_mask=None):
    if data.shape[0] <= max_points:
        return data, labels, highlight_mask
    rng = np.random.default_rng(42)
    if highlight_mask is None:
        indices = rng.choice(data.shape[0], size=max_points, replace=False)
        return data[indices], labels[indices], None

    highlight_mask = highlight_mask.astype(bool)
    highlight_idx = np.flatnonzero(highlight_mask)
    other_idx = np.flatnonzero(~highlight_mask)

    if len(highlight_idx) >= max_points:
        sel_high = rng.choice(highlight_idx, size=max_points, replace=False)
        sel_mask = np.ones(max_points, dtype=bool)
        return data[sel_high], labels[sel_high], sel_mask

    remaining = max_points - len(highlight_idx)
    if remaining > len(other_idx):
        remaining = len(other_idx)
    sel_other = rng.choice(other_idx, size=remaining, replace=False)
    indices = np.concatenate([highlight_idx, sel_other])
    sel_mask = np.zeros(indices.shape[0], dtype=bool)
    sel_mask[: len(highlight_idx)] = True
    return data[indices], labels[indices], sel_mask


def _compute_channel_stats(tokens_list: List[torch.Tensor], eps: float = 1e-6):
    """Compute per-channel mean/std over B and D for tokens shaped [B, V, D]."""
    if not tokens_list:
        return None, None
    sum_per_ch = None
    sumsq_per_ch = None
    count = 0
    for t in tokens_list:
        if t is None or t.dim() != 3:
            continue
        t = t.float()
        if sum_per_ch is None:
            sum_per_ch = torch.zeros(t.shape[1], dtype=torch.float64)
            sumsq_per_ch = torch.zeros(t.shape[1], dtype=torch.float64)
        sum_per_ch += t.sum(dim=(0, 2)).double()
        sumsq_per_ch += (t ** 2).sum(dim=(0, 2)).double()
        count += t.shape[0] * t.shape[2]
    if count == 0:
        return None, None
    mean = sum_per_ch / count
    var = sumsq_per_ch / count - mean ** 2
    std = torch.sqrt(torch.clamp(var, min=eps))
    return mean.float(), std.float()


def _flatten_tokens(
    tokens: torch.Tensor,
    labels: torch.Tensor = None,
    mode: str = "concat_norm",
    channel_mean: torch.Tensor = None,
    channel_std: torch.Tensor = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert tokens into [N, F] for t-SNE.
    - concat_norm: per-channel normalize [B, V, D] then flatten to [B, V*D]
    - mean: average over V -> [B, D]
    """
    if tokens.dim() == 2:
        tokens_np = tokens.cpu().numpy()
    else:
        if mode == "mean":
            tokens_np = tokens.mean(dim=1).cpu().numpy()
        else:
            t = tokens
            if channel_mean is not None and channel_std is not None:
                t = (t - channel_mean.view(1, -1, 1)) / (channel_std.view(1, -1, 1) + 1e-6)
            tokens_np = t.reshape(t.shape[0], -1).cpu().numpy()

    if labels is None:
        labels_np = -1 * np.ones(tokens_np.shape[0], dtype=int)
    else:
        if labels.dim() > 1:
            labels = labels.argmax(dim=-1)
        labels_np = labels.view(-1).cpu().numpy()
    return tokens_np, labels_np


def load_debug_records(pt_path: Path) -> List[dict]:
    records = torch.load(pt_path, map_location="cpu")
    if not isinstance(records, list):
        raise ValueError(f"Expected a list of records in {pt_path}, got {type(records)}")
    return records


def plot_tsne(data: np.ndarray, labels: np.ndarray, title: str, save_path: Path, highlight_mask=None):
    """
    highlight_mask: optional boolean mask same length as data to emphasize points (e.g., category tokens).
    """
    tsne = TSNE(n_components=2, random_state=42, init="pca", learning_rate="auto")
    embedding = tsne.fit_transform(data)

    plt.figure(figsize=(8, 6))
    unique_labels = np.unique(labels)

    # Ensure highlight_mask
    if highlight_mask is None:
        highlight_mask = np.zeros(len(data), dtype=bool)

    for lbl in unique_labels:
        mask = labels == lbl
        if lbl == -1:
            label_name = "unknown"
        elif lbl == 9999:
            label_name = "category_token"
        else:
            label_name = str(lbl)

        # Split highlight/non-highlight for this label
        mask_high = mask & highlight_mask
        mask_norm = mask & (~highlight_mask)

        if mask_norm.any():
            plt.scatter(
                embedding[mask_norm, 0],
                embedding[mask_norm, 1],
                s=18,
                alpha=0.7,
                label=label_name if not mask_high.any() else f"{label_name} (others)",
            )
        if mask_high.any():
            plt.scatter(
                embedding[mask_high, 0],
                embedding[mask_high, 1],
                s=80,
                marker="X",
                edgecolor="k",
                linewidths=0.8,
                alpha=0.95,
                label=f"{label_name} (highlight)",
            )

    plt.title(title)
    plt.legend(loc="best", fontsize=8)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Draw t-SNE from debug_tokens_epoch*.pt")
    parser.add_argument("--pt_path", type=str, required=True, help="Path to debug_tokens_epoch*.pt file")
    parser.add_argument("--output_dir", type=str, default=None, help="Directory to save figures")
    parser.add_argument(
        "--flatten_mode",
        type=str,
        default="concat_norm",
        choices=["concat_norm", "mean"],
        help="How to turn [B, V, D] into t-SNE input.",
    )
    parser.add_argument(
        "--max_points",
        type=int,
        default=2500,
        help="Max number of points to use for t-SNE (randomly sampled if exceeded).",
    )
    args = parser.parse_args()

    pt_path = Path(args.pt_path)
    out_dir = Path(args.output_dir) if args.output_dir else pt_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    records = load_debug_records(pt_path)

    # ----------------------
    # Plot 1: debug_cls_token
    # ----------------------
    cls_token_raw = []
    cls_tokens_list = []
    cls_labels_list = []
    for rec in records:
        cls_token = rec.get("cls_token")
        labels = rec.get("labels")
        if cls_token is None:
            continue
        cls_token_raw.append(cls_token)
    ch_mean, ch_std = _compute_channel_stats(cls_token_raw) if args.flatten_mode == "concat_norm" else (None, None)
    for rec in records:
        cls_token = rec.get("cls_token")
        labels = rec.get("labels")
        if cls_token is None:
            continue
        ct, lb = _flatten_tokens(cls_token, labels, mode=args.flatten_mode, channel_mean=ch_mean, channel_std=ch_std)
        cls_tokens_list.append(ct)
        cls_labels_list.append(lb)

    if cls_tokens_list:
        data_cls = np.concatenate(cls_tokens_list, axis=0)
        labels_cls = np.concatenate(cls_labels_list, axis=0)
        data_cls, labels_cls, _ = _sample_points(data_cls, labels_cls, args.max_points)
        plot_tsne(
            data_cls,
            labels_cls,
            title="t-SNE of debug_cls_token",
            save_path=out_dir / f"{pt_path.stem}_cls_token.png",
        )
    else:
        print("No cls_token found in records; skipping first plot.")

    # -----------------------------------------
    # Plot 2: debug_cls_token_projected + category_token
    # -----------------------------------------
    proj_token_raw = []
    proj_tokens_list = []
    proj_labels_list = []
    category_tokens = None
    for rec in records:
        proj = rec.get("cls_token_projected")
        labels = rec.get("labels")
        if proj is not None:
            proj_token_raw.append(proj)
        cat = rec.get("category_token")
        if cat is not None:
            category_tokens = cat  # keep last available (per需求：最后一个batch)
    ch_mean_proj, ch_std_proj = _compute_channel_stats(proj_token_raw) if args.flatten_mode == "concat_norm" else (None, None)
    for rec in records:
        proj = rec.get("cls_token_projected")
        labels = rec.get("labels")
        if proj is not None:
            ct, lb = _flatten_tokens(
                proj,
                labels,
                mode=args.flatten_mode,
                channel_mean=ch_mean_proj,
                channel_std=ch_std_proj,
            )
            proj_tokens_list.append(ct)
            proj_labels_list.append(lb)

    if proj_tokens_list:
        data_proj = np.concatenate(proj_tokens_list, axis=0)
        labels_proj = np.concatenate(proj_labels_list, axis=0)
    else:
        data_proj = np.empty((0, 2))
        labels_proj = np.empty((0,))

    if len(data_proj) > 0:
        data_proj, labels_proj, _ = _sample_points(data_proj, labels_proj, args.max_points)

    # Append category tokens (from last batch) and mark them
    if category_tokens is not None:
        # category_tokens shape: [M, V, D] or [M, D]; flatten all to points
        cat_tensor = category_tokens
        if cat_tensor.dim() == 2:
            cat_flat = cat_tensor.cpu().numpy()
        else:
            cat_flat, _ = _flatten_tokens(
                cat_tensor,
                labels=None,
                mode=args.flatten_mode,
                channel_mean=ch_mean_proj,
                channel_std=ch_std_proj,
            )
        cat_labels = np.full(cat_flat.shape[0], 9999, dtype=int)  # special label for category_token

        data_proj = np.concatenate([data_proj, cat_flat], axis=0) if data_proj.size else cat_flat
        labels_proj = np.concatenate([labels_proj, cat_labels], axis=0) if labels_proj.size else cat_labels

    if len(data_proj) > 0:
        plot_tsne(
            data_proj,
            labels_proj,
            title="t-SNE of cls_token_projected (all) with category_token highlighted",
            save_path=out_dir / f"{pt_path.stem}_cls_proj_and_category.png",
        )
    else:
        print("No projected tokens found; skipping second plot.")


if __name__ == "__main__":
    main()
