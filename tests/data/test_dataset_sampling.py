"""Tests for category-aware mix-and-separate sampling."""
import json

import pytest

from src.data.dataset import MixAndSepareDataset


def _write_index(tmp_path, entries):
    index_path = tmp_path / "index.json"
    index_path.write_text(json.dumps(entries))
    return str(index_path)


def test_cross_category_sampler_uses_distinct_categories(tmp_path):
    entries = {
        "violin_a": {
            "split": "train",
            "category": "violin",
            "source_paths": [str(tmp_path / "violin" / "a" / "source_0.pt")],
        },
        "violin_b": {
            "split": "train",
            "category": "violin",
            "source_paths": [str(tmp_path / "violin" / "b" / "source_0.pt")],
        },
        "piano_a": {
            "split": "train",
            "category": "piano",
            "source_paths": [str(tmp_path / "piano" / "a" / "source_0.pt")],
        },
        "flute_a": {
            "split": "train",
            "category": "flute",
            "source_paths": [str(tmp_path / "flute" / "a" / "source_0.pt")],
        },
    }
    dataset = MixAndSepareDataset(
        _write_index(tmp_path, entries),
        n_sources=2,
        split="train",
        include_visual=False,
    )

    for idx in range(20):
        clip_ids = dataset._sample_cross_category_clip_ids(idx)
        categories = [dataset.clip_categories[clip_id] for clip_id in clip_ids]
        assert len(categories) == len(set(categories))


def test_cross_category_sampler_requires_enough_categories(tmp_path):
    entries = {
        "violin_a": {
            "split": "train",
            "category": "violin",
            "source_paths": [str(tmp_path / "violin" / "a" / "source_0.pt")],
        },
        "violin_b": {
            "split": "train",
            "category": "violin",
            "source_paths": [str(tmp_path / "violin" / "b" / "source_0.pt")],
        },
    }

    with pytest.raises(ValueError, match="distinct categories"):
        MixAndSepareDataset(
            _write_index(tmp_path, entries),
            n_sources=2,
            split="train",
            include_visual=False,
        )
