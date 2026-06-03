"""Training script for DINOv2-guided audio-visual source separation.

Usage:
    python scripts/train.py train.phase=phase1
    python scripts/train.py train.phase=phase2
    python scripts/train.py train.phase=phase3
"""
import sys
sys.path.insert(0, '.')

import hydra
from omegaconf import DictConfig, OmegaConf
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, GradientAccumulationScheduler
from pytorch_lightning.loggers import TensorBoardLogger

from src.models.separator import SeparatorModule


@hydra.main(config_path="../configs", config_name="config", version_base=None)
def train(cfg: DictConfig):
    """Main training function."""
    print(OmegaConf.to_yaml(cfg))

    # Set precision
    precision = cfg.train.get("precision", "32-true")
    phase = cfg.train.phase

    # Initialize model
    model = SeparatorModule(cfg=OmegaConf.to_container(cfg, resolve=True), phase=phase)

    # Callbacks
    callbacks = [
        LearningRateMonitor(logging_interval="step"),
        ModelCheckpoint(
            dirpath="checkpoints",
            filename="{step}-{val/si_snr_loss:.2f}",
            save_top_k=3,
            monitor="val/si_snr_loss",
            mode="min",
            every_n_train_steps=5000,
        ),
    ]

    # Gradient accumulation for phase 3
    if cfg.train.get("gradient_accumulation_steps", 1) > 1:
        callbacks.append(
            GradientAccumulationScheduler(
                scheduling={0: cfg.train.gradient_accumulation_steps}
            )
        )

    # Logger
    logger = TensorBoardLogger(
        save_dir="logs",
        name=f"{cfg.train.name}_{phase}",
    )

    # Trainer
    trainer = pl.Trainer(
        max_steps=cfg.train.max_steps,
        precision=precision,
        gradient_clip_val=cfg.train.grad_clip,
        gradient_clip_algorithm="norm",
        callbacks=callbacks,
        logger=logger,
        log_every_n_steps=100,
        val_check_interval=5000,
        num_sanity_val_steps=2,
        devices=1,
        accelerator="auto",
    )

    print(f"Model initialized for {phase}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    # Note: Data loading requires preprocessed data index file
    # Run scripts/preprocess_data.py first to generate the index
    print("\nNOTE: Data loading requires preprocessed index file.")
    print("Run: python scripts/preprocess_data.py to generate the data index.")
    print("\nTo test model initialization without data, run:")
    print("  python scripts/test_model.py")


if __name__ == "__main__":
    train()