"""Entry point for training phases.

Usage:
    python train.py --config-name config \
        model=default data=default train=phase1
"""
import hydra
from omegaconf import DictConfig
import pytorch_lightning as pl
from src.models.separator import SeparatorModule


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig):
    print(f"Training phase: {cfg.train.name}")
    print(f"Model config: {cfg.model}")
    print(f"Data config: {cfg.data}")
    print(f"Train config: {cfg.train}")
    # TODO: instantiate datamodule, model, trainer, and train
    # model = SeparatorModule(cfg)
    # trainer = pl.Trainer(...)
    # trainer.fit(model, datamodule)


if __name__ == "__main__":
    main()
