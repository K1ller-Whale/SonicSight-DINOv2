"""Configuration utilities for Hydra/omegaconf."""
from omegaconf import DictConfig, OmegaConf
from typing import Dict, Any


def instantiate_from_config(cfg: DictConfig):
    """Instantiate a class from a config dict with _target_ key."""
    import hydra
    return hydra.utils.instantiate(cfg)


def merge_with_dotlist(cfg: DictConfig, overrides: Dict[str, Any]) -> DictConfig:
    """Apply dotlist overrides to a config."""
    return OmegaConf.merge(cfg, OmegaConf.create(overrides))
