import json

import torch

from src.data.dataset import MixAndSepareDataset
from src.data.preprocessing import CLIP_LENGTH


def test_mix_and_separe_dataset_clips_targets_to_model_length(tmp_path):
    src_a = tmp_path / "a.pt"
    src_b = tmp_path / "b.pt"
    torch.save(torch.randn(CLIP_LENGTH + 123), src_a)
    torch.save(torch.randn(CLIP_LENGTH - 77), src_b)

    index = {
        "a": {
            "split": "test",
            "category": "alpha",
            "source_paths": [str(src_a)],
        },
        "b": {
            "split": "test",
            "category": "beta",
            "source_paths": [str(src_b)],
        },
    }
    index_path = tmp_path / "index.json"
    index_path.write_text(json.dumps(index))

    dataset = MixAndSepareDataset(
        str(index_path),
        n_sources=2,
        split="test",
        include_visual=False,
    )

    sample = dataset[0]

    assert sample["target_waveforms"].shape == (2, CLIP_LENGTH)
    assert sample["mixture_stft"].shape[-2:] == (257, 601)
