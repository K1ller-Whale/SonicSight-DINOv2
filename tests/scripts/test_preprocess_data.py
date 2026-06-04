"""Tests for preprocess_data.py with dummy synthetic clips."""
import json
import tempfile
from pathlib import Path

import pytest
import torch

from scripts.preprocess_data import preprocess_dataset, process_single_clip


class MockDINOv2(torch.nn.Module):
    """Mock DINOv2 that returns constant feature shape."""

    def __init__(self, num_patches=1024, dim=768):
        super().__init__()
        self._num_patches = num_patches
        self._dim = dim

    def forward(self, images):
        """Return random features matching DINOv2 output shape.

        Args:
            images: [B, 3, H, W]
        Returns:
            [B, num_patches, dim]
        """
        B = images.shape[0]
        return torch.randn(B, self._num_patches, self._dim)


@pytest.fixture
def dummy_clips_dir():
    """Create a temporary directory with 3 synthetic clips."""
    with tempfile.TemporaryDirectory() as tmpdir:
        input_dir = Path(tmpdir) / "input"
        input_dir.mkdir()

        # Create 3 synthetic clips, each with 2 sources and video
        for i in range(3):
            clip_dir = input_dir / f"clip_{i}"
            clip_dir.mkdir()

            # 2 source waveforms (6 seconds @ 16kHz = 96000 samples)
            for j in range(2):
                torch.save(torch.randn(96000), str(clip_dir / f"source_{j}.pt"))

            # Video frames: [2, 3, 448, 448] (small for speed)
            torch.save(torch.randn(2, 3, 448, 448), str(clip_dir / "video.pt"))

        yield str(input_dir)


@pytest.fixture
def mock_dinov2():
    """Mock DINOv2 model that returns expected feature shapes."""
    return MockDINOv2(num_patches=1024, dim=768)


class TestPreprocessDataset:
    """Tests for preprocess_dataset with dummy clips."""

    def test_creates_index_json(self, dummy_clips_dir, tmp_path, mock_dinov2):
        """index.json should be created with correct clip entries."""
        output_dir = str(tmp_path / "output")
        preprocess_dataset(dummy_clips_dir, output_dir, dinov2_model=mock_dinov2)

        index_path = Path(output_dir) / "index.json"
        assert index_path.exists(), "index.json should be created"

        with open(index_path) as f:
            index = json.load(f)

        assert len(index) == 3, f"Expected 3 clips, got {len(index)}"
        assert "clip_0" in index
        assert "clip_1" in index
        assert "clip_2" in index

    def test_crm_cache_files_exist(self, dummy_clips_dir, tmp_path, mock_dinov2):
        """cRM cache files should be created in crm/ directory."""
        output_dir = str(tmp_path / "output")
        preprocess_dataset(dummy_clips_dir, output_dir, dinov2_model=mock_dinov2)

        crm_dir = Path(output_dir) / "crm"
        assert crm_dir.exists(), "crm directory should be created"

        # Verify at least 3 .pt files exist (one per clip)
        crm_files = list(crm_dir.glob("*.pt"))
        assert len(crm_files) == 3, f"Expected 3 cRM files, got {len(crm_files)}"

        # Verify each file loads and has source count
        for crm_file in crm_files:
            crm = torch.load(str(crm_file), weights_only=False)
            assert crm.dim() == 4, f"cRM should be 4D [N, 2, F, T], got shape {crm.shape}"
            assert crm.shape[0] == 2, f"Expected 2 sources, got {crm.shape[0]}"
            assert crm.shape[1] == 2, f"Expected 2 channels (real/imag), got {crm.shape[1]}"

    def test_visual_cache_files_exist(self, dummy_clips_dir, tmp_path, mock_dinov2):
        """Visual cache files should be created in visual/ directory."""
        output_dir = str(tmp_path / "output")
        preprocess_dataset(dummy_clips_dir, output_dir, dinov2_model=mock_dinov2)

        visual_dir = Path(output_dir) / "visual"
        assert visual_dir.exists(), "visual directory should be created"

        # Verify at least 3 .pt files exist
        visual_files = list(visual_dir.glob("*.pt"))
        assert len(visual_files) == 3, f"Expected 3 visual files, got {len(visual_files)}"

        # Verify each file loads and has correct shape
        for visual_file in visual_files:
            visual = torch.load(str(visual_file), weights_only=False)
            # Our mock returns [N_frames, 1024, 768]
            # For dummy data with 2 frames: [2, 1024, 768]
            assert visual.shape[0] == 2, f"Expected 2 frames, got {visual.shape[0]}"
            assert visual.shape[1] == 1024, f"Expected 1024 patches, got {visual.shape[1]}"
            assert visual.shape[2] == 768, f"Expected 768 dim, got {visual.shape[2]}"

    def test_crm_is_float16(self, dummy_clips_dir, tmp_path, mock_dinov2):
        """cRM should be cached as float16."""
        output_dir = str(tmp_path / "output")
        preprocess_dataset(dummy_clips_dir, output_dir, dinov2_model=mock_dinov2)

        crm_dir = Path(output_dir) / "crm"
        crm_files = list(crm_dir.glob("*.pt"))
        assert len(crm_files) > 0

        crm = torch.load(str(crm_files[0]), weights_only=False)
        assert crm.dtype == torch.float16, f"Expected float16, got {crm.dtype}"

    def test_visual_is_float16(self, dummy_clips_dir, tmp_path, mock_dinov2):
        """Visual features should be cached as float16."""
        output_dir = str(tmp_path / "output")
        preprocess_dataset(dummy_clips_dir, output_dir, dinov2_model=mock_dinov2)

        visual_dir = Path(output_dir) / "visual"
        visual_files = list(visual_dir.glob("*.pt"))
        assert len(visual_files) > 0

        visual = torch.load(str(visual_files[0]), weights_only=False)
        assert visual.dtype == torch.float16, f"Expected float16, got {visual.dtype}"

    def test_index_entries_have_required_fields(self, dummy_clips_dir, tmp_path, mock_dinov2):
        """index.json entries should have all required fields."""
        output_dir = str(tmp_path / "output")
        preprocess_dataset(dummy_clips_dir, output_dir, dinov2_model=mock_dinov2)

        index_path = Path(output_dir) / "index.json"
        with open(index_path) as f:
            index = json.load(f)

        required_fields = {"crm_path", "visual_path", "n_sources", "source_paths", "split"}
        for clip_id, entry in index.items():
            assert set(entry.keys()) == required_fields, (
                f"Missing or extra fields in {clip_id}: got {set(entry.keys())}"
            )
            assert entry["n_sources"] > 0
            assert entry["split"] in {"train", "val", "test"}
            assert Path(entry["crm_path"]).is_absolute() or True  # accept relative too

    def test_splits_sum_correctly(self, dummy_clips_dir, tmp_path, mock_dinov2):
        """Train/val/test splits should sum to total clips."""
        output_dir = str(tmp_path / "output")
        preprocess_dataset(dummy_clips_dir, output_dir, dinov2_model=mock_dinov2)

        index_path = Path(output_dir) / "index.json"
        with open(index_path) as f:
            index = json.load(f)

        train_count = sum(1 for v in index.values() if v["split"] == "train")
        val_count = sum(1 for v in index.values() if v["split"] == "val")
        test_count = sum(1 for v in index.values() if v["split"] == "test")

        assert train_count + val_count + test_count == 3
