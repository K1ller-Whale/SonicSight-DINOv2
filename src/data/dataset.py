"""Audio-Visual dataset with on-the-fly mix-and-separate.

SPEC 11.2, 7.1: Synthetic mixtures from clean (source, video) pairs.
"""
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import numpy as np
import os
import json
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any
import random

from src.data.preprocessing import (
    AudioPreprocessor, VideoPreprocessor, STFTModule,
    get_temporal_alignment_table, CLIP_LENGTH, N_STFT_FRAMES,
)


class AudioVisualSource:
    """Represents a single source: audio waveform + video frames + metadata."""

    def __init__(self, audio_path: str, video_path: Optional[str] = None,
                 category: str = "", label: str = ""):
        self.audio_path = audio_path
        self.video_path = video_path
        self.category = category
        self.label = label


class AudioVisualDataset(Dataset):
    """
    Dataset for training the DINOv2-Guided Separator.
    Samples N clean sources, loads their preprocessed audio/video, mixes them.

    Args:
        index_file: Path to JSON index (from preprocess pipeline)
        n_sources: Number of sources to mix per sample
        audio_preproc: AudioPreprocessor instance
        video_preproc: VideoPreprocessor instance
        split: train / val / test
        include_visual: include video frames? (False for Phase 1)
    """

    def __init__(self, index_file: str, n_sources: int = 2,
                 audio_preproc: Optional[AudioPreprocessor] = None,
                 video_preproc: Optional[VideoPreprocessor] = None,
                 split: str = "train", include_visual: bool = True):
        self.index_file = index_file
        self.n_sources = n_sources
        self.audio_preproc = audio_preproc or AudioPreprocessor()
        self.video_preproc = video_preproc or VideoPreprocessor()
        self.stft = STFTModule()
        self.split = split
        self.include_visual = include_visual
        self.alignment = get_temporal_alignment_table()

        # Load index
        if not os.path.exists(index_file):
            raise FileNotFoundError(f"Index file not found: {index_file}")
        with open(index_file, "r") as f:
            self.index = json.load(f)
        if split not in self.index:
            raise ValueError(f"Split '{split}' not in index. Keys: {list(self.index.keys())}")
        self.samples = self.index[split]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Sample N sources, mix, and return."""
        # Sample N unique sources
        if len(self.samples) < self.n_sources:
            raise ValueError(f"Not enough samples ({len(self.samples)}) for {self.n_sources} sources")

        source_indices = random.sample(range(len(self.samples)), self.n_sources)
        sources = [self.samples[i] for i in source_indices]

        # Load and process each source
        audio_waveforms = []
        video_batches = []
        for source in sources:
            wave, frames = self._load_source(source)
            audio_waveforms.append(wave)
            if self.include_visual and frames is not None:
                video_batches.append(frames)

        # Mix audio with random gains
        mixture_wave, source_gains = self._mix_sources(audio_waveforms)

        # Compute complex spectrograms
        mixture_stft = self.stft(mixture_wave)  # [2, F, T]
        source_stfts = torch.stack([self.stft(w) for w in audio_waveforms])  # [N, 2, F, T]

        # Build output dict
        out = {
            "mixture_stft": mixture_stft,
            "source_stfts": source_stfts,
            "n_sources": self.n_sources,
        }

        # Add video if included
        if self.include_visual and video_batches:
            out["source_videos"] = torch.stack(video_batches)  # [N, V, 3, H, W]

        # Store gains for debugging
        out["source_gains"] = source_gains

        return out

    def _load_source(self, source: dict) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Load one source: returns (audio_waveform, video_frames or None)."""
        # Audio (.pt pre-processed waveform or raw)
        audio_path = source.get("audio_path",source.get("wav_path"))
        wave = torch.load(audio_path) if audio_path.endswith('.pt') else torch.zeros(96000)

        # Video frames (pre-processed tensor)
        video_path = source.get("video_path", source.get("frames_path"))
        if self.include_visual and video_path and os.path.exists(video_path):
            frames = torch.load(video_path)  # [N_frames, 3, H, W]
        else:
            frames = None

        return wave, frames

    def _mix_sources(self, waveforms: List[torch.Tensor],
                     db_range: Tuple[float, float] = (-5.0, 5.0)
                     ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Mix source waveforms with random per-source gains.
        Args:
            waveforms: list of [L] tensors
        Returns:
            mixture: [L], gains: [N]
        """
        n = len(waveforms)
        pad_fn = F.pad

        # Ensure same length (they should be, but be safe)
        max_len = max(w.shape[-1] for w in waveforms)
        padded = []
        for w in waveforms:
            if w.shape[-1] < max_len:
                w = pad_fn(w, (0, max_len - w.shape[-1]))
            padded.append(w)
        stacked = torch.stack(padded)  # [N, L]

        # Random gains
        db = torch.rand(n) * (db_range[1] - db_range[0]) + db_range[0]
        gains = 10 ** (db / 20.0)

        mixture = (stacked * gains.unsqueeze(1)).sum(dim=0)  # [L]
        return mixture, gains


class AudioVisualDataModule:
    """
    Lightning-style DataModule (not a full Lightning module to avoid dependency issues)
    """

    def __init__(self, index_file: str, train_config: dict,
                 batch_size: int = 8, num_workers: int = 4):
        self.index_file = index_file
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.n_sources = train_config.get("n_sources", 2)
        self.include_visual = train_config.get("include_visual", True)

    def train_dataloader(self) -> DataLoader:
        ds = AudioVisualDataset(
            self.index_file, n_sources=self.n_sources,
            split="train", include_visual=self.include_visual,
        )
        return DataLoader(ds, batch_size=self.batch_size,
                          shuffle=True, num_workers=self.num_workers,
                          pin_memory=True)

    def val_dataloader(self) -> DataLoader:
        ds = AudioVisualDataset(
            self.index_file, n_sources=self.n_sources,
            split="val", include_visual=self.include_visual,
        )
        return DataLoader(ds, batch_size=self.batch_size,
                          shuffle=False, num_workers=self.num_workers,
                          pin_memory=True)

    def test_dataloader(self) -> DataLoader:
        ds = AudioVisualDataset(
            self.index_file, n_sources=self.n_sources,
            split="test", include_visual=self.include_visual,
        )
        return DataLoader(ds, batch_size=self.batch_size,
                          shuffle=False, num_workers=self.num_workers,
                          pin_memory=True)
