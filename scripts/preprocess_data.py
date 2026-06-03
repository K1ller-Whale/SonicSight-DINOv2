#!/usr/bin/env python
"""Preprocessing pipeline for DINOv2 audio-visual separation.

Scans dataset directories for audio/video pairs, creates JSON index with
train/val/test splits (80/10/10), caches STFT spectrograms and cRM targets.

SPEC 11.2:
- Audio: 16kHz mono, 6s clips (96000 samples)
- Video: 448x448 frames, 25fps (150 frames)
- STFT: N_FFT=512, hop=160, win=400 → [2, F, T]
- cRM: tanh-compressed ideal ratio mask

Usage:
    C:/Users/H/AppData/Local/Programs/Python/Python312/python.exe scripts/preprocess_data.py \
        --data_dir D:/datasets/MUSIC \
        --output_dir D:/datasets/MUSIC/preprocessed \
        --exts .wav,.mp4
"""
import argparse
import json
import os
import random
import shutil
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

import torch
import torch.nn.functional as F
import torchaudio
from torchvision import io as video_io
from PIL import Image
import numpy as np

from src.data.preprocessing import (
    AudioPreprocessor,
    VideoPreprocessor,
    STFTModule,
    compute_crm_targets,
    TARGET_SR,
    CLIP_DURATION,
    CLIP_LENGTH,
    N_FFT,
    HOP_LENGTH,
    WIN_LENGTH,
    N_STFT_FRAMES,
    N_VIDEO_FRAMES,
    IMAGE_SIZE,
)


# --- Constants ---
AUDIO_EXTENSIONS = {".wav", ".flac", ".mp3", ".ogg", ".m4a"}
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
VALID_EXTENSIONS = AUDIO_EXTENSIONS | VIDEO_EXTENSIONS


def find_audio_video_pairs(
    data_dir: str,
    audio_exts: Optional[set] = None,
    video_exts: Optional[set] = None,
) -> List[Dict[str, str]]:
    """Scan directory for audio/video pairs.

    Matches files by stem name: audio.wav pairs with video.mp4 if same stem.

    Args:
        data_dir: Root directory to scan
        audio_exts: Set of audio extensions to match
        video_exts: Set of video extensions to match

    Returns:
        List of dicts with 'audio_path', 'video_path', 'stem'
    """
    audio_exts = audio_exts or AUDIO_EXTENSIONS
    video_exts = video_exts or VIDEO_EXTENSIONS

    data_path = Path(data_dir)
    audio_files: Dict[str, Path] = {}
    video_files: Dict[str, Path] = {}

    # Collect all files
    for path in data_path.rglob("*"):
        if not path.is_file():
            continue
        ext = path.suffix.lower()
        stem = path.stem

        if ext in audio_exts:
            audio_files[stem] = path
        elif ext in video_exts:
            video_files[stem] = path

    # Find pairs (stems that have both audio and video)
    pairs = []
    for stem in audio_files.keys() & video_files.keys():
        pairs.append({
            "audio_path": str(audio_files[stem]),
            "video_path": str(video_files[stem]),
            "stem": stem,
        })

    return sorted(pairs, key=lambda x: x["stem"])


def create_splits(
    pairs: List[Dict[str, str]],
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
) -> Dict[str, List[Dict[str, str]]]:
    """Create train/val/test splits.

    Args:
        pairs: List of audio/video pairs
        train_ratio: Fraction for training
        val_ratio: Fraction for validation
        test_ratio: Fraction for testing
        seed: Random seed for reproducibility

    Returns:
        Dict with 'train', 'val', 'test' keys
    """
    pairs = pairs.copy()
    random.seed(seed)
    random.shuffle(pairs)

    n = len(pairs)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    return {
        "train": pairs[:train_end],
        "val": pairs[train_end:val_end],
        "test": pairs[val_end:],
    }


def load_audio(audio_path: str, target_sr: int = TARGET_SR) -> Tuple[torch.Tensor, int]:
    """Load audio file and return waveform with sample rate.

    Args:
        audio_path: Path to audio file
        target_sr: Target sample rate for resampling info

    Returns:
        (waveform, original_sr) - waveform is [C, L]
    """
    waveform, sr = torchaudio.load(audio_path)
    return waveform, sr


def load_video_frames(
    video_path: str,
    num_frames: int = N_VIDEO_FRAMES,
    image_size: int = IMAGE_SIZE,
) -> torch.Tensor:
    """Load video frames using torchvision.

    Args:
        video_path: Path to video file
        num_frames: Number of frames to extract
        image_size: Target frame size

    Returns:
        frames: [N, 3, H, W] float tensor in [0, 1]
    """
    # Read all frames
    frames, _, _ = video_io.read_video(video_path, pts_unit="sec")

    if len(frames) == 0:
        raise ValueError(f"Empty video: {video_path}")

    # Sample frames evenly
    if len(frames) >= num_frames:
        indices = torch.linspace(0, len(frames) - 1, num_frames).long()
        sampled = frames[indices]
    else:
        # Pad by repeating
        indices = torch.arange(num_frames) % len(frames)
        sampled = frames[indices]

    # Convert to [N, C, H, W] float in [0, 1]
    frames = sampled.permute(0, 3, 1, 2).float() / 255.0

    # Resize if needed
    if frames.shape[-2:] != (image_size, image_size):
        frames = F.interpolate(
            frames,
            size=(image_size, image_size),
            mode="bicubic",
            align_corners=False,
        )

    return frames


def process_audio_sample(
    audio_path: str,
    audio_preproc: AudioPreprocessor,
    stft_module: STFTModule,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Process audio file: preprocess → STFT.

    Args:
        audio_path: Path to audio file
        audio_preproc: AudioPreprocessor instance
        stft_module: STFTModule instance

    Returns:
        (waveform_6s, stft_spec) where waveform is [L] and stft is [2, F, T]
    """
    # Load audio
    waveform, sr = load_audio(audio_path)

    # Preprocess: resample, crop/pad to 6s
    waveform_6s = audio_preproc(waveform, sr)

    # Compute STFT
    stft_spec = stft_module(waveform_6s)

    return waveform_6s, stft_spec


def compute_and_cache_crm(
    source_stfts: torch.Tensor,
    mixture_stft: torch.Tensor,
    output_path: str,
) -> torch.Tensor:
    """Compute cRM targets and cache to disk.

    Args:
        source_stfts: [N, 2, F, T] - source spectrograms
        mixture_stft: [2, F, T] - mixture spectrogram
        output_path: Path to save .pt file

    Returns:
        crm: [N, 2, F, T] - compressed ratio masks
    """
    crm = compute_crm_targets(source_stfts, mixture_stft)
    torch.save(crm, output_path)
    return crm


def build_index_entry(
    pair: Dict[str, str],
    cache_dir: str,
    idx: int,
) -> Dict[str, Any]:
    """Build index entry for a sample.

    Args:
        pair: Audio/video pair dict
        cache_dir: Directory for cached files
        idx: Sample index

    Returns:
        Dict with paths to cached files + metadata
    """
    stem = pair["stem"]
    audio_cache = os.path.join(cache_dir, f"{idx:06d}_audio.pt")
    video_cache = os.path.join(cache_dir, f"{idx:06d}_video.pt")
    crm_cache = os.path.join(cache_dir, f"{idx:06d}_crm.pt")

    return {
        "audio_path": audio_cache,
        "video_path": video_cache,
        "crm_path": crm_cache,
        "original_audio": pair["audio_path"],
        "original_video": pair["video_path"],
        "stem": stem,
        "idx": idx,
    }


def preprocess_dataset(
    data_dir: str,
    output_dir: str,
    splits: Dict[str, List[Dict[str, str]]],
    num_workers: int = 4,
) -> Dict[str, List[Dict[str, Any]]]:
    """Preprocess all samples and cache results.

    Args:
        data_dir: Original data directory
        output_dir: Output directory for cached files
        splits: Train/val/test splits
        num_workers: Reserved for future parallel processing

    Returns:
        Index dict with processed sample info
    """
    # Create output directories
    cache_dir = os.path.join(output_dir, "cache")
    os.makedirs(cache_dir, exist_ok=True)

    # Preprocessing modules
    audio_preproc = AudioPreprocessor()
    video_preproc = VideoPreprocessor()
    stft_module = STFTModule()

    # Flatten all pairs with split info
    all_pairs = []
    for split_name, pairs in splits.items():
        for pair in pairs:
            all_pairs.append((split_name, pair))

    # Build index
    index: Dict[str, List[Dict[str, Any]]] = {
        "train": [],
        "val": [],
        "test": [],
    }

    total = len(all_pairs)
    for global_idx, (split_name, pair) in enumerate(all_pairs):
        sample_idx = len(index[split_name])
        print(f"[{global_idx + 1}/{total}] Processing {pair['stem']} ({split_name})")

        try:
            # Process audio
            waveform_6s, stft_spec = process_audio_sample(
                pair["audio_path"],
                audio_preproc,
                stft_module,
            )

            # Process video
            video_frames = load_video_frames(pair["video_path"])
            video_normalized = video_preproc(video_frames)

            # Cache waveform and STFT
            audio_cache_path = os.path.join(cache_dir, f"{global_idx:06d}_audio.pt")
            stft_cache_path = os.path.join(cache_dir, f"{global_idx:06d}_stft.pt")
            torch.save(
                {"waveform": waveform_6s, "stft": stft_spec},
                audio_cache_path,
            )

            # Cache video frames
            video_cache_path = os.path.join(cache_dir, f"{global_idx:06d}_video.pt")
            torch.save(video_normalized, video_cache_path)

            # Build entry
            entry = {
                "audio_path": audio_cache_path,
                "video_path": video_cache_path,
                "stft_path": stft_cache_path,
                "original_audio": pair["audio_path"],
                "original_video": pair["video_path"],
                "stem": pair["stem"],
                "split": split_name,
                "idx": global_idx,
                "n_frames": video_frames.shape[0],
                "audio_samples": waveform_6s.shape[-1],
            }
            index[split_name].append(entry)

        except Exception as e:
            print(f"  ERROR processing {pair['stem']}: {e}")
            continue

    # Save index summary
    summary = {
        "train_samples": len(index["train"]),
        "val_samples": len(index["val"]),
        "test_samples": len(index["test"]),
        "total_samples": sum(len(v) for v in index.values()),
        "config": {
            "sample_rate": TARGET_SR,
            "clip_duration": CLIP_DURATION,
            "n_fft": N_FFT,
            "hop_length": HOP_LENGTH,
            "win_length": WIN_LENGTH,
            "video_fps": 25,
            "n_video_frames": N_VIDEO_FRAMES,
            "image_size": IMAGE_SIZE,
        },
    }

    with open(os.path.join(output_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    return index


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess audio-visual dataset for DINOv2 separation"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Root directory containing audio/video files",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory for preprocessed data",
    )
    parser.add_argument(
        "--train_ratio",
        type=float,
        default=0.8,
        help="Training set ratio (default: 0.8)",
    )
    parser.add_argument(
        "--val_ratio",
        type=float,
        default=0.1,
        help="Validation set ratio (default: 0.1)",
    )
    parser.add_argument(
        "--test_ratio",
        type=float,
        default=0.1,
        help="Test set ratio (default: 0.1)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for splitting (default: 42)",
    )
    parser.add_argument(
        "--audio_exts",
        type=str,
        default=".wav,.flac,.mp3,.ogg,.m4a",
        help="Comma-separated audio extensions",
    )
    parser.add_argument(
        "--video_exts",
        type=str,
        default=".mp4,.avi,.mov,.mkv,.webm",
        help="Comma-separated video extensions",
    )
    parser.add_argument(
        "--no_video",
        action="store_true",
        help="Process audio only, skip video frames",
    )

    args = parser.parse_args()

    # Validate ratios
    total_ratio = args.train_ratio + args.val_ratio + args.test_ratio
    if abs(total_ratio - 1.0) > 1e-6:
        raise ValueError(f"Split ratios must sum to 1.0, got {total_ratio}")

    # Parse extensions
    audio_exts = {ext.strip() for ext in args.audio_exts.split(",")}
    video_exts = {ext.strip() for ext in args.video_exts.split(",")}

    print(f"Scanning {args.data_dir} for audio/video pairs...")
    print(f"  Audio extensions: {audio_exts}")
    print(f"  Video extensions: {video_exts}")

    # Find pairs
    pairs = find_audio_video_pairs(args.data_dir, audio_exts, video_exts)

    if len(pairs) == 0:
        print("ERROR: No audio/video pairs found!")
        print("Ensure files are named with matching stems (e.g., 'song1.wav' and 'song1.mp4')")
        return

    print(f"Found {len(pairs)} audio/video pairs")

    # Create splits
    splits = create_splits(
        pairs,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )

    print(f"\nDataset splits:")
    print(f"  Train: {len(splits['train'])} samples")
    print(f"  Val:   {len(splits['val'])} samples")
    print(f"  Test:  {len(splits['test'])} samples")

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Save raw splits (before preprocessing)
    with open(os.path.join(args.output_dir, "raw_splits.json"), "w") as f:
        json.dump(splits, f, indent=2)

    # Preprocess and cache
    print(f"\nCaching preprocessed data to {args.output_dir}/cache...")
    index = preprocess_dataset(
        args.data_dir,
        args.output_dir,
        splits,
    )

    # Save final index
    with open(os.path.join(args.output_dir, "index.json"), "w") as f:
        json.dump(index, f, indent=2)

    print(f"\nDone! Preprocessed data saved to {args.output_dir}")
    print(f"Index file: {os.path.join(args.output_dir, 'index.json')}")


if __name__ == "__main__":
    main()
