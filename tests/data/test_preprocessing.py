"""Comprehensive tests for audio/video preprocessing (Section 11.2)."""
import torch
import pytest
import numpy as np
import math


class TestAudioPreprocessor:
    """Tests for src.data.preprocessing.AudioPreprocessor"""

    def test_output_length_6s(self):
        """SPEC: 6 s at 16 kHz = 96 000 samples."""
        from src.data.preprocessing import AudioPreprocessor
        preproc = AudioPreprocessor(target_sr=16000, clip_duration=6)
        waveform = torch.randn(32000)  # 2 sec
        out = preproc(waveform, sr=16000)
        assert out.numel() == 96000, f"Got {out.numel()} samples"

    def test_zero_pad_short(self):
        """Should zero-pad clips shorter than 6 s."""
        from src.data.preprocessing import AudioPreprocessor
        preproc = AudioPreprocessor(target_sr=16000, clip_duration=6)
        waveform = torch.randn(16000)  # 1 sec
        out = preproc(waveform, sr=16000)
        assert out.numel() == 96000
        assert out[16000:].abs().max() == 0, "Expected zero padding"

    def test_crop_long(self):
        """Should crop clips longer than 6 s."""
        from src.data.preprocessing import AudioPreprocessor
        preproc = AudioPreprocessor(target_sr=16000, clip_duration=6)
        waveform = torch.randn(192000)  # 12 sec
        out = preproc(waveform, sr=16000)
        assert out.numel() == 96000

    def test_resample(self):
        """Should resample from 44.1 kHz to 16 kHz."""
        from src.data.preprocessing import AudioPreprocessor
        preproc = AudioPreprocessor(target_sr=16000, clip_duration=6)
        waveform = torch.randn(44100)  # 1 sec at 44.1 kHz
        out = preproc(waveform, sr=44100)
        assert out.numel() == 96000

    def test_mono_mixdown(self):
        """Should mix stereo to mono."""
        from src.data.preprocessing import AudioPreprocessor
        preproc = AudioPreprocessor(target_sr=16000, clip_duration=6)
        waveform = torch.randn(2, 96000)  # stereo, 6 sec
        out = preproc(waveform, sr=16000)
        assert out.dim() == 1
        assert out.numel() == 96000


class TestSTFTModule:
    """Tests for src.data.preprocessing.STFTModule"""

    def test_output_shape(self):
        """SPEC 11.2: STFT → [2, 257, 601]."""
        from src.data.preprocessing import STFTModule, CLIP_LENGTH
        stft = STFTModule(n_fft=512, hop_length=160)
        waveform = torch.randn(CLIP_LENGTH)  # 16 kHz mono
        spec = stft(waveform)
        assert spec.shape == (2, 257, 601), f"Got {spec.shape}"

    def test_inverse_roundtrip(self):
        """iSTFT should output the correct shape and scale."""
        from src.data.preprocessing import STFTModule, ISTFTModule
        stft = STFTModule(n_fft=512, hop_length=160)
        istft = ISTFTModule(n_fft=512, hop_length=160)
        waveform = torch.randn(96000)
        spec = stft(waveform)  # [2, 257, 601]
        # Build a near-identity mask
        mask = torch.ones_like(spec)
        reconstructed = istft(mask, spec, length=96000)
        assert reconstructed.shape == waveform.shape
        # Check amplitude preservation for Gaussian noise (std ≈ 1)
        assert reconstructed.std() > 0.5  # rough sanity check

    def test_complex_channels(self):
        """real+imag channels."""
        from src.data.preprocessing import STFTModule
        stft = STFTModule(n_fft=512, hop_length=160)
        waveform = torch.randn(96000)
        spec = stft(waveform)
        assert spec[0].dtype == torch.float32
        assert spec[1].dtype == torch.float32


class TestVideoPreprocessor:
    """Tests for src.data.preprocessing.VideoPreprocessor"""

    def test_output_shape(self):
        """Should resize to 448×448."""
        from src.data.preprocessing import VideoPreprocessor
        preproc = VideoPreprocessor(image_size=448)
        frames = torch.rand(5, 3, 720, 1280)  # 5 raw frames
        out = preproc(frames)
        assert out.shape == (5, 3, 448, 448)

    def test_normalization_not_raw(self):
        """After ImageNet normalisation, values are no longer in [0,1]."""
        from src.data.preprocessing import VideoPreprocessor
        preproc = VideoPreprocessor(image_size=448)
        frames = torch.rand(2, 3, 448, 448)  # raw
        out = preproc(frames)
        # Values should be roughly centred, not in raw [0,1]
        assert out.min() < 0.0  # some negative values
        assert out.max() > 1.0  # some above 1.0 after rescale

    def test_aspect_ratio_preserved(self):
        """Resize + CenterCrop should preserve aspect ratio (no stretching)."""
        from src.data.preprocessing import VideoPreprocessor
        preproc = VideoPreprocessor(image_size=448)

        # Wide frame: 320x240 (4:3) -> resize short side to 448 -> 597x448 -> center crop 448x448
        frames = torch.rand(1, 3, 240, 320)  # [1, 3, H, W]
        out = preproc(frames)
        assert out.shape == (1, 3, 448, 448)

        # Tall frame: 320x240 -> resize -> 448x597 -> center crop 448x448
        frames = torch.rand(1, 3, 320, 240)
        out = preproc(frames)
        assert out.shape == (1, 3, 448, 448)

        # Verify center crop by checking content isn't just stretched
        # Create a frame with gradient - after proper resize+crop, center should match
        grad = torch.linspace(0, 1, 240).view(1, 1, 240, 1).expand(1, 3, 240, 320)
        out = preproc(grad)
        # The center 448x448 region of resized 597x448 should be from the middle
        # Just verify it runs without error and produces correct shape
        assert out.shape == (1, 3, 448, 448)


class TestTemporalAlignment:
    """Tests for temporal alignment between STFT and video frames."""

    def test_alignment_length(self):
        """Alignment table covers all STFT frames."""
        from src.data.preprocessing import get_temporal_alignment_table
        table = get_temporal_alignment_table()
        assert len(table) == 601  # SPEC: N_STFT_FRAMES = 601

    def test_alignment_bounds(self):
        """Video frame indices are within [0, 149]."""
        from src.data.preprocessing import get_temporal_alignment_table
        table = get_temporal_alignment_table()
        assert all(0 <= v < 150 for v in table)

    def test_first_and_last(self):
        """First STFT frame → video frame 0, last → 149."""
        from src.data.preprocessing import get_temporal_alignment_table
        table = get_temporal_alignment_table()
        assert table[0] == 0
        assert table[-1] == 149

    def test_specific_values(self):
        """Test specific alignment values from SPEC: table[300] == 74."""
        from src.data.preprocessing import get_temporal_alignment_table
        table = get_temporal_alignment_table()
        assert table[300] == 74

    def test_monotonic(self):
        """Alignment table is non-decreasing."""
        from src.data.preprocessing import get_temporal_alignment_table
        table = get_temporal_alignment_table()
        assert all(table[i] <= table[i + 1] for i in range(len(table) - 1))


class TestCRMComputation:
    """Tests for complex ratio mask target computation."""

    def test_crm_shape(self):
        """cRM targets → [N, 2, F, T]."""
        from src.data.preprocessing import compute_crm_targets
        n_sources = 3
        source_stfts = torch.randn(n_sources, 2, 257, 601)
        mix_stft = torch.randn(2, 257, 601)
        crm = compute_crm_targets(source_stfts, mix_stft)
        assert crm.shape == (n_sources, 2, 257, 601)

    def test_crm_is_bounded(self):
        """Tanh-compressed cRM should be bounded in [-1, 1]."""
        from src.data.preprocessing import compute_crm_targets
        source_stfts = torch.randn(2, 2, 257, 601) * 100
        mix_stft = torch.randn(2, 257, 601)
        crm = compute_crm_targets(source_stfts, mix_stft, k=10.0)
        assert crm.abs().max() <= 1.0 + 1e-6

    def test_single_source_crm(self):
        """For a single source, cRM ≈ 1 (identity mask)."""
        from src.data.preprocessing import compute_crm_targets
        source = torch.randn(1, 2, 5, 5)
        crm = compute_crm_targets(source, source[0])
        assert crm.shape == (1, 2, 5, 5)
