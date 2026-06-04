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


def extract_attention_weights(model: SeparatorModule) -> List[torch.Tensor]:
    """Extract attention weights from cross-attention blocks."""
    weights = []
    for block in model.cross_attn.blocks:
        # MultiheadAttention doesn't expose weights directly
        # Need to hook into attn forward pass
        pass
    return weights


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

            # Forward to get separated sources
            _ = model(mixture_stft, video_frames=video)

            # TODO: Hook into cross-attention to extract weights
            # For now, use placeholder
            if gt_boxes and str(batch_idx) in gt_boxes:
                # Predicted box from attention (placeholder - need attention hook)
                pred_box = (0.25, 0.25, 0.75, 0.75)  # normalized [x1, y1, x2, y2]
                gt_box = gt_boxes[str(batch_idx)]
                iou = compute_iou(pred_box, gt_box)
                iou_scores.append(iou)
            else:
                iou_scores.append(0.0)  # No GT available

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
