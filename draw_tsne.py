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


def _flatten_tokens(tokens: torch.Tensor, labels: torch.Tensor = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Flatten token tensor [B, V, D] into [B*V, D] and expand labels to match.
    If labels is one-hot or [B, *], squeeze to [B] before expansion.
    """
    tokens_np = tokens.reshape(-1, tokens.shape[-1]).cpu().numpy()
    if labels is None:
        labels_np = -1 * np.ones(tokens_np.shape[0], dtype=int)
    else:
        if labels.dim() > 1:
            labels = labels.argmax(dim=-1)
        labels_np = labels.view(-1, 1).repeat(1, tokens.shape[1]).reshape(-1).cpu().numpy()
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
    parser.add_argument("--output_dir", type=str, default="tsne_outputs", help="Directory to save figures")
    args = parser.parse_args()

    pt_path = Path(args.pt_path)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    records = load_debug_records(pt_path)

    # ----------------------
    # Plot 1: debug_cls_token
    # ----------------------
    cls_tokens_list = []
    cls_labels_list = []
    for rec in records:
        cls_token = rec.get("cls_token")
        labels = rec.get("labels")
        if cls_token is None:
            continue
        ct, lb = _flatten_tokens(cls_token, labels)
        cls_tokens_list.append(ct)
        cls_labels_list.append(lb)

    if cls_tokens_list:
        data_cls = np.concatenate(cls_tokens_list, axis=0)
        labels_cls = np.concatenate(cls_labels_list, axis=0)
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
    proj_tokens_list = []
    proj_labels_list = []
    category_tokens = None
    for rec in records:
        proj = rec.get("cls_token_projected")
        labels = rec.get("labels")
        if proj is not None:
            ct, lb = _flatten_tokens(proj, labels)
            proj_tokens_list.append(ct)
            proj_labels_list.append(lb)
        cat = rec.get("category_token")
        if cat is not None:
            category_tokens = cat  # keep last available (per需求：最后一个batch)

    if proj_tokens_list:
        data_proj = np.concatenate(proj_tokens_list, axis=0)
        labels_proj = np.concatenate(proj_labels_list, axis=0)
    else:
        data_proj = np.empty((0, 2))
        labels_proj = np.empty((0,))

    # Append category tokens (from last batch) and mark them
    highlight_mask = None
    if category_tokens is not None:
        # category_tokens shape: [M, V, D] or [M, D]; flatten all to points
        cat_tensor = category_tokens
        if cat_tensor.dim() == 2:
            cat_flat = cat_tensor.cpu().numpy()
        else:
            cat_flat = cat_tensor.reshape(-1, cat_tensor.shape[-1]).cpu().numpy()
        cat_labels = np.full(cat_flat.shape[0], 9999, dtype=int)  # special label for category_token

        data_proj = np.concatenate([data_proj, cat_flat], axis=0) if data_proj.size else cat_flat
        labels_proj = np.concatenate([labels_proj, cat_labels], axis=0) if labels_proj.size else cat_labels

        highlight_mask = np.zeros(len(labels_proj), dtype=bool)
        highlight_mask[-len(cat_flat) :] = True

    if len(data_proj) > 0:
        plot_tsne(
            data_proj,
            labels_proj,
            title="t-SNE of cls_token_projected (all) with category_token highlighted",
            save_path=out_dir / f"{pt_path.stem}_cls_proj_and_category.png",
            highlight_mask=highlight_mask,
        )
    else:
        print("No projected tokens found; skipping second plot.")


if __name__ == "__main__":
    main()
