#!/usr/bin/env python
"""Pre-compute frozen DINOv2 patch tokens for every clip and cache them to disk.

Why: DINOv2 is frozen during Phases 2 and 3, so its patch tokens never change.
Running it inside the training loop is the dominant cost (~26 h/epoch on a T4).
This script runs DINOv2 exactly once per clip, stores the tokens, and repoints
``visual_paths`` in ``index.json`` to the cached tensors. Training then only runs
the audio U-Net encoder + visual_proj + cross-attention, dropping the epoch time
to single-digit minutes.

By default only ``--num-frames`` video frames are kept (evenly spaced across the
clip), stored in float16. The model's SeparatorModule._frame_alignment_idx()
maps the 19 bottleneck time-columns onto however many frames are cached, so any
frame count >= 1 works with no model change.

    K x 1024 x 768 x 2 bytes per clip:
        K=19 -> ~30 MB/clip   K=6 -> ~9.2 MB/clip   K=4 -> ~6.0 MB/clip

Use ``--all-frames`` to instead cache every decoded frame (~150), or set
``--num-frames`` to fit a disk budget (e.g. 4 for ~54 GB over ~8957 clips).

Usage:
    python scripts/precompute_dino_features.py \
        --index-file ./cache/index.json \
        --output-dir ./cache/visual \
        --device cuda \
        --dinov2-batch-size 16 \
        --num-frames 4
"""

import sys
from pathlib import Path

# Ensure repo root is on sys.path so `src` imports work regardless of CWD/invocation
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import json
import os
import shutil
from typing import List

import torch

from src.data.dataset import MixAndSepareDataset
from src.visual.dinov2 import DINOv2FeatureExtractor

# ImageNet normalization (identical to SeparatorModule._normalize_video_frames)
_IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
_IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

# The model flattens the audio bottleneck [B, 512, 9, 19] into 171 positions and
# aligns each of the 19 time-columns (w) to a single video frame. We cache K
# evenly spaced frames; SeparatorModule._frame_alignment_idx(K) then maps the 19
# columns onto those K frames at train time.
N_BOTTLENECK_COLS = 19
DEFAULT_NUM_FRAMES = 19

VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


def select_frame_indices(n_frames: int, n_keep: int = DEFAULT_NUM_FRAMES) -> List[int]:
    """Evenly spaced frame indices: j -> int(j * n_frames / n_keep), clamped."""
    n_keep = max(1, min(n_keep, n_frames))
    return [min(int(j * n_frames / n_keep), n_frames - 1) for j in range(n_keep)]


def normalize_frames(frames: torch.Tensor) -> torch.Tensor:
    """[T, 3, H, W] uint8/float -> ImageNet-normalized float32."""
    frames = frames.float()
    if frames.amax() > 2.0:
        frames = frames / 255.0
    frames = (frames - _IMAGENET_MEAN.to(frames.device)) / _IMAGENET_STD.to(
        frames.device
    )
    return frames


@torch.no_grad()
def encode_clip(
    mp4_path: str,
    model: DINOv2FeatureExtractor,
    device: torch.device,
    keep_all: bool,
    num_frames: int,
    batch_size: int,
) -> torch.Tensor:
    """Decode an mp4 visual clip and return DINOv2 tokens [V, 1024, 768] (fp16)."""
    frames = MixAndSepareDataset._load_frames_from_mp4(
        mp4_path
    )  # [150, 3, 448, 448] uint8
    if not keep_all:
        idx = select_frame_indices(frames.shape[0], num_frames)
        frames = frames[idx]  # [K, 3, 448, 448]
    frames = normalize_frames(frames).to(device)
    feats = []
    for start in range(0, frames.shape[0], max(1, batch_size)):
        chunk = frames[start : start + max(1, batch_size)]
        feats.append(model(chunk).cpu())
    tokens = torch.cat(feats, dim=0)  # [V, 1024, 768]
    return tokens.to(torch.float16)


def is_video_path(path: str) -> bool:
    return Path(path).suffix.lower() in VIDEO_EXTS


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pre-compute frozen DINOv2 features and cache them to disk."
    )
    parser.add_argument(
        "--index-file",
        default="./cache/index.json",
        help="Path to index.json produced by preprocess_data.py.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory for cached features (default: <cache>/visual).",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run DINOv2 on.",
    )
    parser.add_argument(
        "--dinov2-batch-size",
        type=int,
        default=16,
        help="Frames per DINOv2 forward pass.",
    )
    parser.add_argument(
        "--num-frames",
        type=int,
        default=DEFAULT_NUM_FRAMES,
        help="Frames cached per clip (evenly spaced). The model aligns "
        "dynamically, so lower = smaller cache. 4 ~= 54 GB over "
        "~8957 clips; 19 (default) ~= 260 GB. Ignored with --all-frames.",
    )
    parser.add_argument(
        "--all-frames",
        action="store_true",
        help="Cache every decoded frame (~150) instead of --num-frames.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-encode clips even if a cached .pt already exists.",
    )
    parser.add_argument(
        "--no-backup",
        action="store_true",
        help="Do not write index.json.bak before updating the index.",
    )
    args = parser.parse_args()

    if args.num_frames < 1:
        parser.error("--num-frames must be >= 1")

    index_file = args.index_file
    if not os.path.exists(index_file):
        raise FileNotFoundError(f"index.json not found: {index_file}")

    cache_root = os.path.dirname(os.path.abspath(index_file))
    output_dir = args.output_dir or os.path.join(cache_root, "visual")
    os.makedirs(output_dir, exist_ok=True)

    with open(index_file, "r") as f:
        index = json.load(f)

    device = torch.device(args.device)
    model = DINOv2FeatureExtractor().to(device).eval()

    n_total = len(index)
    n_encoded = 0
    n_skipped = 0

    for i, (clip_id, meta) in enumerate(index.items()):
        visual_paths = meta.get("visual_paths", []) or []
        if not visual_paths:
            print(f"[{i + 1}/{n_total}] {clip_id}: no visual_paths, skipping")
            continue

        src_path = visual_paths[0]
        out_path = os.path.join(output_dir, f"{clip_id}.pt")

        if os.path.exists(out_path) and not args.overwrite:
            # Already cached: just make sure the index points at it.
            meta["visual_paths"] = [out_path for _ in visual_paths]
            n_skipped += 1
            print(f"[{i + 1}/{n_total}] {clip_id}: cached, repointing index")
            continue

        if not src_path or not os.path.exists(src_path):
            print(f"[{i + 1}/{n_total}] {clip_id}: source visual missing ({src_path})")
            continue

        if is_video_path(src_path):
            tokens = encode_clip(
                src_path,
                model,
                device,
                keep_all=args.all_frames,
                num_frames=args.num_frames,
                batch_size=args.dinov2_batch_size,
            )
        else:
            # Already a precomputed .pt tensor; load and (optionally) trim/recast.
            tokens = torch.load(src_path, weights_only=False)
            if not args.all_frames and tokens.shape[0] > args.num_frames:
                idx = select_frame_indices(tokens.shape[0], args.num_frames)
                tokens = tokens[idx]
            tokens = tokens.to(torch.float16)

        torch.save(tokens, out_path)
        meta["visual_paths"] = [out_path for _ in visual_paths]
        n_encoded += 1
        print(
            f"[{i + 1}/{n_total}] {clip_id}: saved {tuple(tokens.shape)} -> {out_path}"
        )

    if not args.no_backup:
        backup = index_file + ".bak"
        if not os.path.exists(backup):
            shutil.copyfile(index_file, backup)
            print(f"Backed up original index to {backup}")

    with open(index_file, "w") as f:
        json.dump(index, f, indent=2)

    print("\nDone.")
    print(f"  Encoded: {n_encoded}")
    print(f"  Skipped (already cached): {n_skipped}")
    print(f"  Index updated: {index_file}")
    print(f"  Features dir: {output_dir}")


if __name__ == "__main__":
    main()
