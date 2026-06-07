"""Evaluation: Source localisation accuracy via cross-modal attention weights.

Maps attention weights from the cross-modal attention module back to
video frame positions and measures IoU with ground-truth bounding boxes.
Uses top-50 patches and YOLOv8 for GT bounding boxes.

Usage:
    python evaluation/eval_localisation.py \
        --checkpoint checkpoints/best.ckpt \
        --index_file cache/index.json

Output (JSON):
    {"loc_acc": 0.67, "mean_iou": 0.52, "n_frames_evaluated": 100}
"""
import argparse
import json
import os
import sys
from typing import Dict, List, Tuple

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.models.separator import SeparatorModule
from src.data.datamodule import AudioVisualDataModule


def compute_iou(pred_box: Tuple[float, float, float, float],
                gt_box: Tuple[float, float, float, float]) -> float:
    """Compute IoU between two boxes (x1, y1, x2, y2) normalized [0,1]."""
    x1 = max(pred_box[0], gt_box[0])
    y1 = max(pred_box[1], gt_box[1])
    x2 = min(pred_box[2], gt_box[2])
    y2 = min(pred_box[3], gt_box[3])

    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_pred = (pred_box[2] - pred_box[0]) * (pred_box[3] - pred_box[1])
    area_gt = (gt_box[2] - gt_box[0]) * (gt_box[3] - gt_box[1])
    union = area_pred + area_gt - inter
    return inter / union if union > 0 else 0.0


def bbox_from_patches(patch_indices: List[int], grid_size: int = 32) -> Tuple[float, float, float, float]:
    """Create bounding box from top-k patch indices on a grid_size x grid_size grid."""
    if not patch_indices:
        return (0.25, 0.25, 0.75, 0.75)

    # Convert patch indices to grid coordinates
    rows = [idx // grid_size for idx in patch_indices]
    cols = [idx % grid_size for idx in patch_indices]

    # Bounding box around all patches
    min_row, max_row = min(rows), max(rows)
    min_col, max_col = min(cols), max(cols)

    # Normalize to [0, 1] with padding
    pad = 0.0
    x1 = max(0.0, min_col / grid_size - pad)
    y1 = max(0.0, min_row / grid_size - pad)
    x2 = min(1.0, (max_col + 1) / grid_size + pad)
    y2 = min(1.0, (max_row + 1) / grid_size + pad)

    return (x1, y1, x2, y2)


class AttentionHook:
    """Context manager to register forward hooks on MultiheadAttention modules."""
    def __init__(self, model: SeparatorModule, n_sources: int = 2):
        self.model = model
        self.n_sources = n_sources
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
        if isinstance(output, tuple) and len(output) == 2:
            attn_weights = output[1]  # [B, num_heads, T_q, T_k]
            self.weights.append((layer_idx, attn_weights.detach().cpu()))


def extract_attention_weights(model: SeparatorModule, mixture_stft: torch.Tensor,
                              video_frames: torch.Tensor) -> List[torch.Tensor]:
    """Extract attention weights from cross-attention blocks via forward hooks."""
    with AttentionHook(model) as hook:
        _ = model(mixture_stft, video_frames=video_frames)
    return [w for _, w in sorted(hook.weights, key=lambda x: x[0])]


def attention_to_top50_bbox(attn_weights: torch.Tensor) -> Tuple[float, float, float, float]:
    """
    Map attention weights to predicted bounding box using top-50 patches.

    Args:
        attn_weights: [num_heads, T_q, T_k] - attention from source queries to visual patches
    Returns:
        (x1, y1, x2, y2) normalized coordinates
    """
    # Average across heads: [T_q, T_k]
    attn_map = attn_weights.mean(dim=0)

    # Source queries are first N positions: [N, T_k]
    n_sources = min(attn_map.shape[0], 4)  # Handle N=2,3,4
    source_attn = attn_map[:n_sources, :].mean(dim=0)  # [T_k]

    # T_k = T_a * P where P=1024 (32x32 patches per frame)
    P = 1024
    T_a = source_attn.shape[0] // P

    # Reshape to [T_a, P] and average across temporal positions
    source_attn_temporal = source_attn.view(T_a, P).mean(dim=0)  # [P]

    # Get top-50 patches by attention weight
    top50_indices = source_attn_temporal.topk(min(50, P)).indices.tolist()

    # Create bounding box from top-50 patch indices
    return bbox_from_patches(top50_indices)


def load_yolo_detector(model_name: str = 'yolov8n.pt', device: str = 'cpu'):
    """Load YOLOv8 detector for ground truth bounding boxes."""
    try:
        from ultralytics import YOLO
        detector = YOLO(model_name)
        detector.to(device)
        return detector
    except Exception as e:
        print(f"WARNING: Could not load YOLOv8 ({model_name}): {e}")
        return None


def get_yolo_boxes(detector, frame: np.ndarray, conf_threshold: float = 0.5) -> List[Tuple[float, float, float, float]]:
    """Run YOLOv8 on frame and return list of bounding boxes normalized [0,1]."""
    if detector is None:
        return []

    H, W = frame.shape[:2]
    results = detector(frame, verbose=False)[0]

    boxes = []
    for box in results.boxes:
        if box.conf.item() < conf_threshold:
            continue
        # box.xyxy is [x1, y1, x2, y2] in pixel coordinates
        xyxy = box.xyxy[0].cpu().numpy()
        # Normalize
        x1 = xyxy[0] / W
        y1 = xyxy[1] / H
        x2 = xyxy[2] / W
        y2 = xyxy[3] / H
        boxes.append((float(x1), float(y1), float(x2), float(y2)))

    return boxes


def evaluate_localisation(args) -> Dict:
    """Evaluate attention-based source localisation with top-50 patches and YOLOv8 GT."""
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")

    model = SeparatorModule.load_from_checkpoint(args.checkpoint)
    model = model.to(device)
    model.eval()

    # Load YOLO for GT boxes if requested
    yolo_detector = None
    if args.use_yolo:
        yolo_detector = load_yolo_detector(args.yolo_model, device.type)

    dm = AudioVisualDataModule(
        index_file=args.index_file,
        n_sources=args.n_sources,
        batch_size=1,
        num_workers=0,
        include_visual=True,
    )
    dm.setup("test")
    dataloader = dm.test_dataloader()

    # Load ground-truth bounding boxes if available (JSON format: {clip_id: [boxes]})
    gt_boxes = {}
    if args.gt_boxes and os.path.exists(args.gt_boxes):
        with open(args.gt_boxes) as f:
            gt_boxes = json.load(f)

    iou_scores = []
    frames_with_iou = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            mixture_stft = batch["mixture_stft"].to(device)
            video = batch.get("video_frames")  # [1, N_frames, 3, H, W]
            clip_id = batch.get("clip_id", [str(batch_idx)])[0] if batch.get("clip_id") else str(batch_idx)

            if video is None:
                continue

            video = video.to(device)
            _, _, _, H, W = video.shape

            # Forward with attention hook to extract weights
            attn_weights_list = extract_attention_weights(model, mixture_stft, video)

            # Use last layer's attention weights for localisation
            if attn_weights_list:
                attn_last = attn_weights_list[-1][0]  # [num_heads, T_q, T_k], B=1
                pred_box = attention_to_top50_bbox(attn_last)
            else:
                pred_box = (0.25, 0.25, 0.75, 0.75)

            # Get GT boxes: prefer provided JSON, fallback to YOLO on middle frame
            gt_box_list = []
            if gt_boxes and clip_id in gt_boxes:
                gt_box_list = gt_boxes[clip_id]
            elif yolo_detector is not None:
                # Use middle frame for YOLO detection
                mid_frame_idx = video.shape[1] // 2
                frame_tensor = video[0, mid_frame_idx].cpu().numpy()  # [3, H, W]
                frame_np = (frame_tensor.transpose(1, 2, 0) * 255).astype(np.uint8)
                gt_box_list = get_yolo_boxes(yolo_detector, frame_np, args.yolo_conf)

            if gt_box_list:
                # Compute max IoU against all GT boxes
                best_iou = max(compute_iou(pred_box, gt_box) for gt_box in gt_box_list)
                iou_scores.append(best_iou)
                if best_iou > 0.3:
                    frames_with_iou += 1
            else:
                # No GT available
                iou_scores.append(0.0)

    loc_acc = frames_with_iou / len(iou_scores) if iou_scores else 0.0
    mean_iou = np.mean(iou_scores) if iou_scores else 0.0

    return {
        "loc_acc": float(loc_acc),
        "mean_iou": float(mean_iou),
        "n_frames_evaluated": len(iou_scores),
        "per_sample_iou": iou_scores,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate localisation accuracy (IoU) with top-50 patches and YOLOv8")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--index_file", type=str, default="cache/index.json")
    parser.add_argument("--n_sources", type=int, default=2)
    parser.add_argument("--gt_boxes", type=str, default=None,
                        help="Path to ground-truth bounding boxes JSON")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--use_yolo", action="store_true",
                        help="Use YOLOv8 to generate GT boxes if JSON not provided")
    parser.add_argument("--yolo_model", type=str, default="yolov8n.pt",
                        help="YOLOv8 model name")
    parser.add_argument("--yolo_conf", type=float, default=0.5,
                        help="YOLOv8 confidence threshold")
    parser.add_argument("--output", type=str, default="outputs/eval_localisation.json")
    args = parser.parse_args()

    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
    if not os.path.exists(args.index_file):
        raise FileNotFoundError(f"Index file not found: {args.index_file}")

    # Ensure outputs directory exists
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    results = evaluate_localisation(args)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {args.output}")
    print(f"  Loc Acc (IoU>0.3): {results['loc_acc']:.3f}")
    print(f"  Mean IoU:          {results['mean_iou']:.3f}")
    print(f"  Frames evaluated:  {results['n_frames_evaluated']}")


if __name__ == "__main__":
    main()