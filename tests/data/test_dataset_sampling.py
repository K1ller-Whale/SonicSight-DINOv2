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


def test_same_category_sampler_uses_single_category(tmp_path):
    """When allow_same_category=True, all sampled clips share one category."""
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
        "violin_c": {
            "split": "train",
            "category": "violin",
            "source_paths": [str(tmp_path / "violin" / "c" / "source_0.pt")],
        },
        "piano_a": {
            "split": "train",
            "category": "piano",
            "source_paths": [str(tmp_path / "piano" / "a" / "source_0.pt")],
        },
        "piano_b": {
            "split": "train",
            "category": "piano",
            "source_paths": [str(tmp_path / "piano" / "b" / "source_0.pt")],
        },
    }
    dataset = MixAndSepareDataset(
        _write_index(tmp_path, entries),
        n_sources=2,
        split="train",
        include_visual=False,
        allow_same_category=True,
    )

    for idx in range(20):
        clip_ids = dataset._sample_same_category_clip_ids(idx)
        categories = [dataset.clip_categories[clip_id] for clip_id in clip_ids]
        # All clips must be from the same category
        assert len(set(categories)) == 1
        # All clip IDs must be distinct
        assert len(clip_ids) == len(set(clip_ids))


def test_same_category_sampler_requires_enough_clips(tmp_path):
    """allow_same_category=True should fail if no category has enough clips."""
    entries = {
        "violin_a": {
            "split": "train",
            "category": "violin",
            "source_paths": [str(tmp_path / "violin" / "a" / "source_0.pt")],
        },
        "piano_a": {
            "split": "train",
            "category": "piano",
            "source_paths": [str(tmp_path / "piano" / "a" / "source_0.pt")],
        },
    }

    with pytest.raises(ValueError, match="allow_same_category"):
        MixAndSepareDataset(
            _write_index(tmp_path, entries),
            n_sources=2,
            split="train",
            include_visual=False,
            allow_same_category=True,
        )
