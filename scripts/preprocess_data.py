#!/usr/bin/env python
"""Preprocessing pipeline for DINOv2 audio-visual source separation.

Scans clip directories, computes cRM targets and DINOv2 features,
saves cache with index.json.

Cache structure:
    <output_dir>/
        crm/
            <clip_id>.pt          # float16 [N, 2, F, T]
        visual/
            <clip_id>.pt          # float16 [V, 1024, 768] (DINOv2 patch tokens)
        index.json                # {clip_id: {"crm_path", "visual_path", "n_sources", "split"}}

Usage:
    python scripts/preprocess_data.py \
        --input_dir data/raw/ \
        --output_dir cache/ \
        --train_ratio 0.8 \
        --val_ratio 0.1 \
        --test_ratio 0.1
"""
import argparse
import json
import os
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import torch
import torch.nn.functional as F

from src.data.preprocessing import (
    STFTModule,
    compute_crm_targets,
    AudioPreprocessor,
)
from src.visual.dinov2 import DINOv2FeatureExtractor

# --- Constants ---
SOURCE_EXTS = {".pt", ".wav", ".flac", ".mp3", ".m4a", ".ogg"}
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".pt"}


def find_clip_dirs(input_dir: str) -> List[str]:
    """Find all subdirectories in input_dir (each is a clip)."""
    input_path = Path(input_dir)
    if not input_path.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")
    clip_dirs = [d for d in input_path.iterdir() if d.is_dir()]
    return sorted([str(d) for d in clip_dirs])


def load_source_audios(clip_dir: str) -> List[torch.Tensor]:
    """Load all source audio files from a clip directory.

    Supports .pt (preprocessed tensor) and standard audio formats.
    For .pt files, expects shape [L] or [1, L].
    For audio files, loads via torchaudio and preprocesses to 16kHz mono.

    Returns:
        list of [L] tensors
    """
    clip_path = Path(clip_dir)
    audio_preproc = AudioPreprocessor()

    # Find source files (exclude video.*)
    source_files = []
    for ext in SOURCE_EXTS:
        candidates = list(clip_path.glob(f"source_*{ext}")) + list(clip_path.glob(f"src_*{ext}"))
        # Also catch any non-video .pt/wav/etc files
        if ext == ".pt":
            for cand in clip_path.glob(f"*{ext}"):
                if not cand.name.startswith("video") and "video" not in cand.name.lower():
                    if cand not in source_files:
                        source_files.append(cand)
        else:
            for cand in clip_path.glob(f"*{ext}"):
                if not cand.name.startswith("video") and cand not in source_files:
                    source_files.append(cand)

    source_files = sorted(set(source_files), key=lambda p: p.name)

    waveforms = []
    for sp in source_files:
        if sp.suffix == ".pt":
            data = torch.load(str(sp), weights_only=False)
            if isinstance(data, dict) and "waveform" in data:
                wave = data["waveform"]
            else:
                wave = data
        else:
            import torchaudio
            wave, sr = torchaudio.load(str(sp))
            wave = audio_preproc(wave, sr)

        # Ensure 1D
        if wave.dim() == 2:
            wave = wave.squeeze(0)
        waveforms.append(wave)

    return waveforms


def load_video(clip_dir: str) -> Optional[torch.Tensor]:
    """Load video frames from a clip directory.

    Supports .pt (preprocessed tensor) and video files.
    For .pt files, expects shape [N, 3, H, W] or dict with 'frames' key.
    For video files, loads via torchvision.
    """
    from src.data.preprocessing import VideoPreprocessor
    clip_path = Path(clip_dir)

    # Find video file
    video_file = None
    for ext in VIDEO_EXTS:
        candidates = list(clip_path.glob(f"video*{ext}"))
        if candidates:
            video_file = candidates[0]
            break

    if video_file is None:
        return None

    if video_file.suffix == ".pt":
        data = torch.load(str(video_file), weights_only=False)
        if isinstance(data, dict) and "frames" in data:
            return data["frames"]
        return data
    else:
        from torchvision import io as video_io
        frames, _, _ = video_io.read_video(str(video_file), pts_unit="sec")
        if len(frames) == 0:
            return None
        # Convert to [N, 3, H, W] float32 in [0,1]
        frames = frames.permute(0, 3, 1, 2).float() / 255.0
        # Preprocess
        video_preproc = VideoPreprocessor()
        frames = video_preproc(frames)
        return frames  # [N, 3, H, W]


def extract_dinov2_features(
    frames: torch.Tensor,
    model: DINOv2FeatureExtractor,
    batch_size: int = 4,
    device: str = "cpu",
) -> torch.Tensor:
    """Extract DINOv2 features from video frames.

    Args:
        frames: [N, 3, H, W] float tensor
        model: DINOv2FeatureExtractor instance
        batch_size: frames per batch
        device: device to run on

    Returns:
        features: [N, 1024, 768] (float32)
    """
    model = model.to(device)
    model.eval()

    n_frames = frames.shape[0]
    features_list = []

    with torch.no_grad():
        for i in range(0, n_frames, batch_size):
            batch = frames[i : i + batch_size].to(device)
            feat = model(batch)  # [B, 1024(1025?), 768]
            features_list.append(feat.cpu())

    all_features = torch.cat(features_list, dim=0)  # [N, 1024, 768]
    return all_features


def get_source_files(clip_dir: str) -> List[Path]:
    """Find all source audio files in a clip directory."""
    clip_path = Path(clip_dir)
    source_files = []
    for ext in SOURCE_EXTS:
        if ext == ".pt":
            for cand in clip_path.glob(f"*{ext}"):
                if not cand.name.startswith("video"):
                    source_files.append(cand)
        else:
            for cand in clip_path.glob(f"*{ext}"):
                if not cand.name.startswith("video"):
                    source_files.append(cand)
    return sorted(set(source_files), key=lambda p: p.name)


def process_single_clip(
    clip_dir: str,
    stft_module: STFTModule,
    dinov2_model: Optional[DINOv2FeatureExtractor],
    device: str = "cpu",
    dinov2_batch_size: int = 4,
) -> Optional[Dict[str, Any]]:
    """Process a single clip directory.

    Returns:
        dict with "crm" (float16 tensor), "visual" (float16 tensor or None),
        "n_sources" (int), "source_paths" (list of str), or None if failed.
    """
    clip_path = Path(clip_dir)
    clip_name = clip_path.name

    # Identify source files
    source_files = get_source_files(clip_dir)
    if len(source_files) == 0:
        print(f"    WARNING: No source audio found in {clip_dir}")
        return None

    # Load source audios
    try:
        source_waves = load_source_audios(clip_dir)
    except Exception as e:
        print(f"    ERROR loading audio for {clip_name}: {e}")
        return None

    if len(source_waves) == 0:
        print(f"    WARNING: No source audio found in {clip_dir}")
        return None

    # Pad all to same length
    max_len = max(w.shape[-1] for w in source_waves)
    padded = []
    for w in source_waves:
        if w.shape[-1] < max_len:
            w = F.pad(w, (0, max_len - w.shape[-1]))
        padded.append(w)

    # Compute STFT for each source
    source_stfts = []
    for w in padded:
        stft = stft_module(w)  # [2, F, T]
        source_stfts.append(stft)
    source_stfts = torch.stack(source_stfts)  # [N, 2, F, T]

    # Mix: sum over sources
    sources_stacked = torch.stack(padded)  # [N, L]
    mixture_wave = sources_stacked.sum(dim=0)  # [L]
    mixture_stft = stft_module(mixture_wave)  # [2, F, T]

    # Compute cRM targets (float16)
    try:
        crm = compute_crm_targets(source_stfts, mixture_stft)  # [N, 2, F, T]
        crm = crm.to(torch.float16)
    except Exception as e:
        print(f"    ERROR computing cRM for {clip_name}: {e}")
        return None

    # Load and process video
    frames = load_video(clip_dir)
    if frames is not None and dinov2_model is not None:
        try:
            visual_features = extract_dinov2_features(
                frames, dinov2_model, batch_size=dinov2_batch_size, device=device
            )
            visual_features = visual_features.to(torch.float16)
        except Exception as e:
            print(f"    ERROR extracting DINOv2 features for {clip_name}: {e}")
            visual_features = None
    else:
        visual_features = None

    return {
        "crm": crm,
        "visual": visual_features,
        "n_sources": len(source_waves),
        "source_paths": [str(p) for p in source_files],
    }


def preprocess_dataset(
    input_dir: str,
    output_dir: str,
    dinov2_model: Optional[DINOv2FeatureExtractor] = None,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
    device: str = "cpu",
    dinov2_batch_size: int = 4,
) -> Dict[str, Any]:
    """Preprocess all clips and save cache.

    Args:
        input_dir: Root directory containing clip subdirectories
        output_dir: Output directory for cache
        dinov2_model: DINOv2 model (None = create new)
        train_ratio: Training split ratio
        val_ratio: Validation split ratio
        test_ratio: Test split ratio
        seed: Random seed
        device: torch device
        dinov2_batch_size: Batch size for DINOv2 feature extraction

    Returns:
        Index dict mapping clip_id to cachedOutput (just metadata)
    """
    # Find clip directories
    clip_dirs = find_clip_dirs(input_dir)
    if not clip_dirs:
        raise ValueError(f"No clip directories found in {input_dir}")

    # Create output dirs
    output_path = Path(output_dir)
    crm_dir = output_path / "crm"
    visual_dir = output_path / "visual"
    crm_dir.mkdir(parents=True, exist_ok=True)
    visual_dir.mkdir(parents=True, exist_ok=True)

    # Create splits
    random.seed(seed)
    shuffled = clip_dirs.copy()
    random.shuffle(shuffled)

    n = len(shuffled)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    split_map = {}
    for i, clip_dir in enumerate(shuffled):
        clip_name = Path(clip_dir).name
        if i < train_end:
            split_map[clip_name] = "train"
        elif i < val_end:
            split_map[clip_name] = "val"
        else:
            split_map[clip_name] = "test"

    # Initialize modules
    stft_module = STFTModule()
    if dinov2_model is None:
        dinov2_model = DINOv2FeatureExtractor()

    # Process each clip
    index = {}
    total = len(clip_dirs)

    for i, clip_dir in enumerate(clip_dirs):
        clip_name = Path(clip_dir).name
        print(f"[{i+1}/{total}] Processing {clip_name}")

        result = process_single_clip(
            clip_dir, stft_module, dinov2_model, device=device, dinov2_batch_size=dinov2_batch_size
        )

        if result is None:
            continue

        # Save cRM
        crm_path = crm_dir / f"{clip_name}.pt"
        torch.save(result["crm"], str(crm_path))

        # Save visual features
        if result["visual"] is not None:
            visual_path = visual_dir / f"{clip_name}.pt"
            torch.save(result["visual"], str(visual_path))
        else:
            visual_path = None

        # Build index entry
        index[clip_name] = {
            "crm_path": str(crm_path),
            "visual_path": str(visual_path) if visual_path is not None else None,
            "n_sources": result["n_sources"],
            "source_paths": result.get("source_paths", []),
            "split": split_map.get(clip_name, "train"),
        }

    # Save index
    index_path = output_path / "index.json"
    with open(index_path, "w") as f:
        json.dump(index, f, indent=2)

    print(f"\nDone! Cache saved to {output_dir}")
    print(f"  Total clips: {len(index)}")
    print(f"  Train: {sum(1 for v in index.values() if v['split'] == 'train')}")
    print(f"  Val:   {sum(1 for v in index.values() if v['split'] == 'val')}")
    print(f"  Test:  {sum(1 for v in index.values() if v['split'] == 'test')}")

    return index


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess dataset for DINOv2 audio-visual separation"
    )
    parser.add_argument("--input_dir", type=str, required=True, help="Input directory containing clip subdirectories")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for cache")
    parser.add_argument("--train_ratio", type=float, default=0.8, help="Training split ratio")
    parser.add_argument("--val_ratio", type=float, default=0.1, help="Validation split ratio")
    parser.add_argument("--test_ratio", type=float, default=0.1, help="Test split ratio")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", type=str, default="cpu", help="Device for DINOv2")
    parser.add_argument("--dinov2_batch_size", type=int, default=4, help="Batch size for DINOv2 feature extraction")

    args = parser.parse_args()

    preprocess_dataset(
        args.input_dir,
        args.output_dir,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
        device=args.device,
        dinov2_batch_size=args.dinov2_batch_size,
    )


if __name__ == "__main__":
    main()
