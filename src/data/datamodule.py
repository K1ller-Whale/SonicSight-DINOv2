"""PyTorch Lightning-compatible DataModule.

Provides train / val / test DataLoaders with per-phase configuration.
"""
import torch
from torch.utils.data import DataLoader
from typing import Optional
import pytorch_lightning as pl

from src.data.preprocessing import AudioPreprocessor, VideoPreprocessor, STFTModule
from src.data.dataset import AudioVisualDataset


class AudioVisualDataModule(pl.LightningDataModule):
    """
    Lightning DataModule for AV source separation.
    Dispenses train / val / test DataLoaders.
    """

    def __init__(self, index_file: str, n_sources: int = 2,
                 batch_size: int = 32, val_batch_size: int = 16,
                 num_workers: int = 4, include_visual: bool = True,
                 seed: int = 42):
        super().__init__()
        self.save_hyperparameters()
        self.index_file = index_file
        self.n_sources = n_sources
        self.batch_size = batch_size
        self.val_batch_size = val_batch_size
        self.num_workers = num_workers
        self.include_visual = include_visual
        self.seed = seed

        self.audio_preproc = AudioPreprocessor()
        self.video_preproc = VideoPreprocessor()
        self.stft = STFTModule()

    def setup(self, stage: Optional[str] = None):
        """Create dataset instances."""
        if stage == "fit" or stage is None:
            self.train_ds = AudioVisualDataset(
                self.index_file, n_sources=self.n_sources,
                split="train", include_visual=self.include_visual,
            )
            self.val_ds = AudioVisualDataset(
                self.index_file, n_sources=self.n_sources,
                split="val", include_visual=self.include_visual,
            )
        if stage == "test" or stage is None:
            self.test_ds = AudioVisualDataset(
                self.index_file, n_sources=self.n_sources,
                split="test", include_visual=self.include_visual,
            )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_ds,
            batch_size=self.val_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_ds,
            batch_size=self.val_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
        )
