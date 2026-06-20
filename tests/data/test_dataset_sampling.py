"""Tests for category-aware mix-and-separate sampling."""
import json

import pytest
import torch

from src.data.dataset import MixAndSepareDataset
from src.data.preprocessing import CLIP_LENGTH, N_VIDEO_FRAMES


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


def test_visual_features_use_each_selected_clips_first_visual_path(tmp_path):
    """Each synthetic source should use its selected clip's visual anchor."""
    audio_a = tmp_path / "a.pt"
    audio_b = tmp_path / "b.pt"
    visual_a = tmp_path / "a_visual.pt"
    visual_b = tmp_path / "b_visual.pt"
    torch.save(torch.zeros(CLIP_LENGTH), audio_a)
    torch.save(torch.ones(CLIP_LENGTH), audio_b)
    torch.save(torch.ones(N_VIDEO_FRAMES, 2, 4), visual_a)
    torch.save(torch.full((N_VIDEO_FRAMES, 2, 4), 2.0), visual_b)

    entries = {
        "violin_a": {
            "split": "val",
            "category": "violin",
            "source_paths": [str(audio_a)],
            "visual_paths": [str(visual_a)],
        },
        "piano_a": {
            "split": "val",
            "category": "piano",
            "source_paths": [str(audio_b)],
            "visual_paths": [str(visual_b)],
        },
    }
    dataset = MixAndSepareDataset(
        _write_index(tmp_path, entries),
        n_sources=2,
        split="val",
        include_visual=True,
    )

    sample = dataset[0]

    assert sample["visual_features"].shape == (2, N_VIDEO_FRAMES, 2, 4)
    expected_by_clip = {
        "violin_a": torch.ones(N_VIDEO_FRAMES, 2, 4),
        "piano_a": torch.full((N_VIDEO_FRAMES, 2, 4), 2.0),
    }
    for source_idx, clip_id in enumerate(sample["clip_ids"]):
        torch.testing.assert_close(
            sample["visual_features"][source_idx],
            expected_by_clip[clip_id],
        )
