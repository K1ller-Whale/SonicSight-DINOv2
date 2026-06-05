"""Evaluation: Source localisation accuracy via cross-modal attention weights.

Maps attention weights from the cross-modal attention module back to
video frame positions and measures IoU with ground-truth bounding boxes.

Usage:
    python evaluation/eval_localisation.py \
        --checkpoint checkpoints/best.ckpt \
        --index_file cache/index.json

Output (JSON):
    {"iou_mean": 0.67, "per_sample": [...]}
"""
import argparse
import json
import os
import sys
from typing import Dict, List, Tuple

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.models.separator import SeparatorModule
from src.data.datamodule import AudioVisualDataModule


def compute_iou(pred_box: tuple, gt_box: tuple) -> float:
    """Compute IoU between two boxes (x1, y1, x2, y2)."""
    x1 = max(pred_box[0], gt_box[0])
    y1 = max(pred_box[1], gt_box[1])
    x2 = min(pred_box[2], gt_box[2])
    y2 = min(pred_box[3], gt_box[3])

    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area_pred = (pred_box[2] - pred_box[0]) * (pred_box[3] - pred_box[1])
    area_gt = (gt_box[2] - gt_box[0]) * (gt_box[3] - gt_box[1])
    union = area_pred + area_gt - inter
    return inter / union if union > 0 else 0.0


class AttentionHook:
    """Context manager to register forward hooks on MultiheadAttention modules."""
    def __init__(self, model: SeparatorModule):
        self.model = model
        self.weights = []  # List of [layer_idx, attn_weights]
        self.hooks = []

    def __enter__(self):
        for layer_idx, block in enumerate(self.model.cross_attn.blocks):
            hook = block.attn.register_forward_hook(
                lambda module, input, output, idx=layer_idx: self._hook_fn(idx, output)
            )
            self.hooks.append(hook)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        for hook in self.hooks:
            hook.remove()
        self.hooks = []

    def _hook_fn(self, layer_idx: int, output):
        # output is (attn_output, attn_weights) when need_weights=True
        if isinstance(output, tuple) and len(output) == 2:
            attn_weights = output[1]  # [B, num_heads, T_q, T_k]
            self.weights.append((layer_idx, attn_weights.detach().cpu()))


def extract_attention_weights(model: SeparatorModule, mixture_stft: torch.Tensor,
                               video_frames: torch.Tensor) -> List[torch.Tensor]:
    """Extract attention weights from cross-attention blocks via forward hooks."""
    with AttentionHook(model) as hook:
        _ = model(mixture_stft, video_frames=video_frames)
    # Return list of attention weights per layer: [B, num_heads, T_q, T_k]
    return [w for _, w in sorted(hook.weights, key=lambda x: x[0])]


def attention_to_bbox(attn_weights: torch.Tensor, video_h: int, video_w: int) -> tuple:
    """Map attention weights to predicted bounding box.

    Args:
        attn_weights: [num_heads, T_q, T_k] - attention from source queries to visual patches
        video_h, video_w: original video frame dimensions
    Returns:
        (x1, y1, x2, y2) normalized coordinates
    """
    # Average across heads: [T_q, T_k]
    attn_map = attn_weights.mean(dim=0)
    # Average across source query positions (first n_sources positions): [T_k]
    # T_q = N_sources + T_a (source queries + bottleneck positions)
    # We only care about source query attention to visual patches
    n_sources = 2  # Default, adjust if needed from model
    source_attn = attn_map[:n_sources, :].mean(dim=0)  # [T_k]

    # T_k = T_a * P (n_bottleneck * 1024 patches)
    # Each bottleneck position has 1024 patches (32x32 grid)
    # Map max attention patch to spatial location
    P = 1024
    T_a = source_attn.shape[0] // P
    # Reshape to [T_a, P] and take max per temporal position
    source_attn = source_attn.view(T_a, P).mean(dim=0)  # Average over time -> [P]

    # Find top-k patches by attention
    patch_idx = source_attn.argmax().item()
    patch_h = patch_idx // 32
    patch_w = patch_idx % 32

    # Normalize to [0, 1]
    x_center = (patch_w + 0.5) / 32
    y_center = (patch_h + 0.5) / 32

    # Box size as percentile-based (e.g., 25% of frame)
    box_size = 0.25
    x1 = max(0.0, x_center - box_size / 2)
    y1 = max(0.0, y_center - box_size / 2)
    x2 = min(1.0, x_center + box_size / 2)
    y2 = min(1.0, y_center + box_size / 2)

    return (x1, y1, x2, y2)


def evaluate_localisation(args) -> Dict:
    """Evaluate attention-based source localisation."""
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")

    model = SeparatorModule.load_from_checkpoint(args.checkpoint)
    model = model.to(device)
    model.eval()

    dm = AudioVisualDataModule(
        index_file=args.index_file,
        n_sources=args.n_sources,
        batch_size=1,
        num_workers=0,
        include_visual=True,
    )
    dm.setup("test")
    dataloader = dm.test_dataloader()

    # Load ground-truth bounding boxes if available
    gt_boxes = {}
    if args.gt_boxes and os.path.exists(args.gt_boxes):
        with open(args.gt_boxes) as f:
            gt_boxes = json.load(f)

    iou_scores = []
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            mixture_stft = batch["mixture_stft"].to(device)
            video = batch.get("video_frames")  # [1, N_frames, 3, H, W]
            if video is None:
                iou_scores.append(0.0)
                continue

            video = video.to(device)
            _, _, _, H, W = video.shape

            # Forward with attention hook to extract weights
            attn_weights_list = extract_attention_weights(model, mixture_stft, video)

            # Use last layer's attention weights for localisation
            if attn_weights_list:
                # attn_weights: [B, num_heads, T_q, T_k], B=1
                attn_last = attn_weights_list[-1][0]  # [num_heads, T_q, T_k]
                pred_box = attention_to_bbox(attn_last, H, W)
            else:
                pred_box = (0.25, 0.25, 0.75, 0.75)

            if gt_boxes and str(batch_idx) in gt_boxes:
                gt_box = gt_boxes[str(batch_idx)]
                iou = compute_iou(pred_box, gt_box)
                iou_scores.append(iou)
            else:
                iou_scores.append(0.0)

    return {
        "iou_mean": sum(iou_scores) / len(iou_scores) if iou_scores else 0.0,
        "num_samples": len(iou_scores),
        "per_sample": iou_scores,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate localisation IoU")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--index_file", type=str, default="cache/index.json")
    parser.add_argument("--n_sources", type=int, default=2)
    parser.add_argument("--gt_boxes", type=str, default=None,
                        help="Path to ground-truth bounding boxes JSON")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--output", type=str, default="localisation_results.json")
    args = parser.parse_args()

    results = evaluate_localisation(args)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {args.output}")
    print(f"  IoU mean: {results['iou_mean']:.3f}")
    print(f"  Samples:  {results['num_samples']}")


if __name__ == "__main__":
    main()
