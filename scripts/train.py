"""Main training script for DINOv2-guided audio-visual source separation.

SPEC 11.4: Three-phase trainer with checkpoint resumption.

Usage:
    python scripts/train.py train=phase1
    python scripts/train.py train=phase2
    python scripts/train.py train=phase3
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import hydra
from omegaconf import DictConfig, OmegaConf
import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
    ModelCheckpoint,
    LearningRateMonitor,
    EarlyStopping,
)
from pytorch_lightning.loggers import TensorBoardLogger

from src.models.separator import SeparatorModule
from src.data.datamodule import AudioVisualDataModule


class CurriculumCallback(pl.Callback):
    """Callback to update DataModule curriculum based on global step."""

    def __init__(self, datamodule: AudioVisualDataModule):
        super().__init__()
        self.datamodule = datamodule

    def on_train_batch_start(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule, batch, batch_idx: int
    ):
        self.datamodule.update_curriculum(trainer.global_step)


@hydra.main(config_path="../configs", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    print("=" * 60)
    print("Configuration:")
    print(OmegaConf.to_yaml(cfg))
    print("=" * 60)

    # ------------------------------------------------------------------ #
    # Resolve paths
    # ------------------------------------------------------------------ #
    index_file = cfg.data.get("index_file")
    if index_file is None:
        # Fallback: look in cache dir
        cache_dir = cfg.data.get("cache_dir", "./cache")
        index_file = os.path.join(cache_dir, "index.json")

    if not os.path.exists(index_file):
        raise FileNotFoundError(
            f"Dataset index not found: {index_file}\n"
            "Run 'python scripts/preprocess_data.py ...' first."
        )

    # ------------------------------------------------------------------ #
    # Model
    # ------------------------------------------------------------------ #
    phase = cfg.train.phase
    model = SeparatorModule(
        cfg=OmegaConf.to_container(cfg, resolve=True),
        phase=phase,
    )

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Phase: {phase}")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # ------------------------------------------------------------------ #
    # DataModule (created early to pass to CurriculumCallback)
    # ------------------------------------------------------------------ #
    include_visual = not cfg.train.get("disable_visual", False)
    n_sources = cfg.model.get("n_sources", cfg.train.get("num_sources", 2))

    # Build curriculum schedule from config
    curriculum_schedule = []
    for item in cfg.train.get("n_sources_schedule", []):
        curriculum_schedule.append((item.step, item.n))

    datamodule = AudioVisualDataModule(
        index_file=index_file,
        n_sources=n_sources,
        batch_size=cfg.train.get("batch_size", cfg.data.get("train_batch_size", 8)),
        val_batch_size=cfg.train.get(
            "val_batch_size", cfg.data.get("val_batch_size", 4)
        ),
        num_workers=cfg.data.get("num_workers", 4),
        include_visual=include_visual,
        seed=cfg.data.get("seed", 42),
        curriculum_schedule=curriculum_schedule if curriculum_schedule else None,
    )

    # ------------------------------------------------------------------ #
    # Callbacks
    # ------------------------------------------------------------------ #
    outputs_cfg = cfg.get("outputs", {})
    logger_cfg = cfg.get("logger", {})
    checkpoint_dir = outputs_cfg.get("checkpoint_dir", "checkpoints")
    log_dir = logger_cfg.get("save_dir") or outputs_cfg.get("log_dir", "logs")

    callbacks = [
        LearningRateMonitor(logging_interval="step"),
        ModelCheckpoint(
            dirpath=checkpoint_dir,
            filename=f"{{step}}-{phase}-" + "{val/sisnri:.2f}",
            save_top_k=3,
            monitor="val/sisnri",
            mode="max",
            every_n_train_steps=cfg.train.get("checkpoint_every_n_steps", 5000),
            save_last=True,
        ),
    ]
    if phase == "phase3":
        callbacks.append(CurriculumCallback(datamodule))

    # Early stopping
    early_stop_patience = cfg.train.get("early_stopping_patience", 0)
    if early_stop_patience > 0:
        callbacks.append(
            EarlyStopping(
                monitor="val/sisnri",
                patience=early_stop_patience,
                mode="max",
                min_delta=0.01,
                verbose=True,
            )
        )

    # ------------------------------------------------------------------ #
    # Logger
    # ------------------------------------------------------------------ #
    logger = TensorBoardLogger(
        save_dir=log_dir,
        name=f"{cfg.train.name}_{phase}",
    )

    # ------------------------------------------------------------------ #
    # Trainer
    # ------------------------------------------------------------------ #
    trainer_cfg = cfg.get("trainer", {})
    trainer_args = {
        "max_steps": cfg.train.max_steps,
        "gradient_clip_val": cfg.train.get("grad_clip", 1.0),
        "gradient_clip_algorithm": "norm",
        "accumulate_grad_batches": cfg.train.get("gradient_accumulation_steps", 1),
        "callbacks": callbacks,
        "logger": logger,
        "log_every_n_steps": cfg.train.get("log_every_n_steps", 100),
        "val_check_interval": cfg.train.get("val_check_interval", 5000),
        "num_sanity_val_steps": cfg.train.get("num_sanity_val_steps", 2),
        "accelerator": trainer_cfg.get("accelerator", "auto"),
        "devices": trainer_cfg.get("devices", 1),
        "num_nodes": trainer_cfg.get("num_nodes", 1),
    }

    strategy = trainer_cfg.get("strategy")
    if strategy:
        trainer_args["strategy"] = strategy
    elif isinstance(trainer_args["devices"], int) and trainer_args["devices"] > 1:
        trainer_args["strategy"] = "ddp_find_unused_parameters_false"

    # Precision
    precision = cfg.train.get("precision")
    if precision is None:
        precision = trainer_cfg.get("precision")
    if precision:
        trainer_args["precision"] = precision

    # Resume from checkpoint (passed to trainer.fit, not Trainer constructor)
    resume_ckpt = cfg.train.get("resume_from_checkpoint")

    trainer = pl.Trainer(**trainer_args)

    # ------------------------------------------------------------------ #
    # Train
    # ------------------------------------------------------------------ #
    print("\n" + "=" * 60)
    print("Starting training...")
    print("=" * 60)
    trainer.fit(
        model, datamodule=datamodule, ckpt_path=resume_ckpt if resume_ckpt else None
    )

    print("\n" + "=" * 60)
    print("Training complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
