from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir


CONFIG_DIR = str(Path(__file__).resolve().parents[1] / "configs")


def _compose(*overrides):
    with initialize_config_dir(config_dir=CONFIG_DIR, version_base=None):
        return compose(config_name="config", overrides=list(overrides))


@pytest.mark.parametrize(
    "train_config",
    [
        "phase1",
        "phase1_kaggle",
        "phase2",
        "phase3",
        "phase3_config_a",
        "phase3_config_b",
        "phase3_config_c",
    ],
)
def test_train_configs_compose_without_double_nesting(train_config):
    cfg = _compose(f"train={train_config}")

    assert "train" not in cfg.train
    assert "data" not in cfg.data
    assert "model" not in cfg.model
    assert cfg.train.phase in {"phase1", "phase2", "phase3"}
    assert cfg.train.max_steps > 0


def test_phase1_kaggle_config_has_expected_paths_and_phase():
    cfg = _compose("train=phase1_kaggle")

    assert cfg.train.phase == "phase1"
    assert cfg.train.disable_visual is True
    assert cfg.data.index_file == "/kaggle/working/cache/index.json"
    assert cfg.data.cache_dir == "/kaggle/working/cache"
    assert cfg.outputs.checkpoint_dir == "/kaggle/working/checkpoints"
