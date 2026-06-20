"""PyTorch Lightning-compatible DataModule.

Provides train / val / test DataLoaders with per-phase configuration.
Supports curriculum learning: n_sources increases at configured steps.
"""
import torch
from torch.utils.data import DataLoader
from typing import Optional, List, Tuple
import pytorch_lightning as pl

from src.data.dataset import MixAndSepareDataset


class AudioVisualDataModule(pl.LightningDataModule):
    """
    Lightning DataModule for AV source separation.
    Dispenses train / val / test DataLoaders using MixAndSepareDataset
    (on-the-fly mixing with cached cRM and DINOv2 features).
    Supports curriculum: n_sources increases at configured step thresholds.
    """

    def __init__(self, index_file: str, n_sources: int = 2,
                 batch_size: int = 32, val_batch_size: int = 16,
                 num_workers: int = 4, include_visual: bool = True,
                 seed: int = 42,
                 curriculum_schedule: Optional[List[Tuple[int, int]]] = None,
                 allow_same_category: bool = False):
        super().__init__()
        self.save_hyperparameters()
        self.index_file = index_file
        self.n_sources = n_sources
        self.batch_size = batch_size
        self.val_batch_size = val_batch_size
        self.num_workers = num_workers
        self.include_visual = include_visual
        self.seed = seed
        self.allow_same_category = allow_same_category
        # Curriculum: list of (step_threshold, n_sources)
        # Default to no-op schedule: stay at current n_sources
        self.curriculum_schedule = curriculum_schedule or [
            (0, self.n_sources)
        ]
        self._current_step = 0

    def update_curriculum(self, step: int) -> bool:
        """Update n_sources based on current training step.
        Returns True if n_sources changed."""
        self._current_step = step
        new_n_sources = self.curriculum_schedule[0][1]
        for threshold, n_src in self.curriculum_schedule:
            if step >= threshold:
                new_n_sources = n_src
            else:
                break
        if new_n_sources != self.n_sources:
            self.n_sources = new_n_sources
            # Recreate datasets with new n_sources
            if hasattr(self, 'train_ds'):
                self.setup(stage="fit")
            return True
        return False

    def get_current_n_sources(self) -> int:
        """Return current number of sources based on curriculum."""
        return self.n_sources

    def setup(self, stage: Optional[str] = None):
        """Create dataset instances with current n_sources."""
        if stage == "fit" or stage is None:
            self.train_ds = MixAndSepareDataset(
                self.index_file, n_sources=self.n_sources,
                split="train", include_visual=self.include_visual,
                allow_same_category=self.allow_same_category,
            )
            self.val_ds = MixAndSepareDataset(
                self.index_file, n_sources=self.n_sources,
                split="val", include_visual=self.include_visual,
                allow_same_category=self.allow_same_category,
            )
        if stage == "test" or stage is None:
            self.test_ds = MixAndSepareDataset(
                self.index_file, n_sources=self.n_sources,
                split="test", include_visual=self.include_visual,
                allow_same_category=self.allow_same_category,
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
