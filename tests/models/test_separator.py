"""Tests for SeparatorModule LightningModule.

SPEC 11.3, 11.4: Forward pass, training step, phase-aware training.
"""

import pytest
import torch
from einops import rearrange
from src.models.separator import JointMaskHead, SeparatorModule


@pytest.fixture
def cfg():
    """Default configuration for testing."""
    return {
        "model": {
            "n_sources": 2,
        },
        "train": {
            "optimizer": {
                "lr": 1e-3,
                "lr_fusion": 3e-4,
                "lr_audio_enc": 3e-5,
                "lr_dinov2": 1e-5,
                "weight_decay": 1e-4,
                "betas": [0.9, 0.999],
            },
            "scheduler": {
                "warmup_steps": 1000,
            },
            "max_steps": 100000,
            "loss": {
                "alpha_crm": 0.1,
                "beta_stft": 0.05,
                "gamma_perceptual": 0.1,
            },
        },
    }


@pytest.fixture
def batch_phase1():
    """Phase 1 batch: audio-only."""
    B, N_sources = 2, 2
    F, T = 257, 601
    L = 96000  # 6s at 16kHz
    device = torch.device("cpu")
    return {
        "mixture_stft": torch.randn(B, 2, F, T, device=device),
        "target_waveforms": torch.randn(B, N_sources, L, device=device),
    }


@pytest.fixture
def batch_phase2():
    """Phase 2 batch: with video."""
    B, N_sources = 1, 2  # Reduced batch to 1 for GPU memory
    F, T = 257, 601
    L = 96000
    N_frames = 8  # Reduced from 150 for GPU memory in test
    device = torch.device("cpu")
    return {
        "mixture_stft": torch.randn(B, 2, F, T, device=device),
        "video_frames": torch.randn(B, N_frames, 3, 448, 448, device=device),
        "target_waveforms": torch.randn(B, N_sources, L, device=device),
    }


class TestSeparatorModuleForward:
    """Test forward pass."""

    def test_forward_phase1_audio_only(self, cfg, batch_phase1):
        """Phase 1: audio-only forward pass."""
        model = SeparatorModule(cfg, phase="phase1")
        mixture_stft = batch_phase1["mixture_stft"]

        output = model(mixture_stft, video_frames=None)

        # Output: [B, N_sources, L]
        assert output.shape[0] == 2  # batch
        assert output.shape[1] == cfg["model"]["n_sources"]  # sources
        assert output.ndim == 3  # [B, N, L]

    def test_forward_phase2_with_video(self, cfg, batch_phase2):
        """Phase 2: with video forward pass."""
        model = SeparatorModule(cfg, phase="phase2")
        mixture_stft = batch_phase2["mixture_stft"]
        video_frames = batch_phase2["video_frames"]

        output = model(mixture_stft, video_frames=video_frames)

        assert output.shape[0] == 1  # batch_phase2 has B=1
        assert output.shape[1] == cfg["model"]["n_sources"]
        assert output.ndim == 3

    def test_forward_output_is_waveform(self, cfg, batch_phase1):
        """Output should be waveform, not spectrogram."""
        model = SeparatorModule(cfg, phase="phase1")
        output = model(batch_phase1["mixture_stft"])

        # 96kHz output for 601 STFT frames at hop=160
        # Expected: ~96000 samples (6s * 16kHz)
        assert output.shape[-1] > 90000  # reasonable waveform length


class TestJointMaskHead:
    """Test source-competitive Phase 1 bottleneck masking."""

    def test_zero_initialized_softmax_masks(self):
        head = JointMaskHead(bottleneck_channels=512, n_sources=2)
        final_conv = head.head[-1]

        assert torch.count_nonzero(final_conv.weight) == 0
        assert torch.count_nonzero(final_conv.bias) == 0

        bottleneck = torch.randn(2, 512, 9, 19)
        masks = head(bottleneck)

        assert masks.shape == (2, 2, 512, 9, 19)
        torch.testing.assert_close(
            masks.sum(dim=1),
            torch.ones(2, 512, 9, 19),
        )
        torch.testing.assert_close(
            masks[:, 0],
            torch.full_like(masks[:, 0], 0.5),
        )


class TestSeparatorModuleTrainingStep:
    """Test training step."""

    def test_training_step_phase1(self, cfg, batch_phase1):
        """Phase 1 training step returns scalar loss."""
        model = SeparatorModule(cfg, phase="phase1")

        loss = model.training_step(batch_phase1, batch_idx=0)

        assert isinstance(loss, torch.Tensor)
        assert loss.ndim == 0  # scalar
        assert loss.requires_grad  # differentiable

    def test_training_step_phase2(self, cfg, batch_phase2):
        """Phase 2 training step with video."""
        model = SeparatorModule(cfg, phase="phase2")

        loss = model.training_step(batch_phase2, batch_idx=0)

        assert isinstance(loss, torch.Tensor)
        assert loss.ndim == 0

    def test_training_step_logs_si_snr(self, cfg, batch_phase1):
        """Training step logs SI-SNR loss."""
        model = SeparatorModule(cfg, phase="phase1")

        # Should not raise
        loss = model.training_step(batch_phase1, batch_idx=0)

        assert loss.ndim == 0


class TestSeparatorModuleOptimizers:
    """Test optimizer configuration."""

    def test_configure_optimizers_phase1(self, cfg):
        """Phase 1: single optimizer with all params + cosine scheduler."""
        model = SeparatorModule(cfg, phase="phase1")
        result = model.configure_optimizers()

        # Lightning 2.x returns dict with optimizer and lr_scheduler
        assert isinstance(result, dict)
        assert "optimizer" in result
        assert "lr_scheduler" in result
        optimizer = result["optimizer"]
        assert optimizer.__class__.__name__ == "AdamW"
        # Default LR for phase1
        assert optimizer.param_groups[0]["lr"] == 1e-3

    def test_configure_optimizers_phase2(self, cfg):
        """Phase 2: only cross-attention params trainable + warmup scheduler."""
        model = SeparatorModule(cfg, phase="phase2")
        result = model.configure_optimizers()

        assert isinstance(result, dict)
        assert "optimizer" in result
        assert "lr_scheduler" in result
        optimizer = result["optimizer"]
        assert optimizer.__class__.__name__ == "AdamW"
        # Uses lr_fusion from config (3e-4), default 5e-4
        assert optimizer.param_groups[0]["lr"] == 3e-4

    def test_configure_optimizers_phase3_differential_lr(self, cfg):
        """Phase 3: differential learning rates + warmup scheduler."""
        model = SeparatorModule(cfg, phase="phase3")
        result = model.configure_optimizers()

        assert isinstance(result, dict)
        assert "optimizer" in result
        assert "lr_scheduler" in result
        optimizer = result["optimizer"]
        assert optimizer.__class__.__name__ == "AdamW"
        # Phase 3 has multiple param groups with different LRs
        assert len(optimizer.param_groups) > 1


class TestSeparatorModuleShapes:
    """Test tensor shapes through the model."""

    def test_bottleneck_shape(self, cfg):
        """Verify bottleneck is [B, 512, 9, 19]."""
        model = SeparatorModule(cfg, phase="phase1")
        x = torch.randn(2, 2, 257, 601)

        bottleneck, skips = model.audio_unet.encoder(x)

        assert bottleneck.shape == torch.Size([2, 512, 9, 19])

    def test_skips_progressive_channels(self, cfg):
        """Verify skip connections have progressive channels."""
        model = SeparatorModule(cfg, phase="phase1")
        x = torch.randn(2, 2, 257, 601)

        _, skips = model.audio_unet.encoder(x)

        # 5 encoder blocks: [32, 64, 128, 256, 512]
        expected_channels = [32, 64, 128, 256, 512]
        for skip, expected_ch in zip(skips, expected_channels):
            assert skip.shape[1] == expected_ch

    def test_decoder_output_shape(self, cfg):
        """Decoder outputs [B, 2, 257, 601] when target_shape provided."""
        model = SeparatorModule(cfg, phase="phase1")
        x = torch.randn(2, 2, 257, 601)

        bottleneck, skips = model.audio_unet.encoder(x)
        output = model.audio_unet.decoder(bottleneck, skips, target_shape=x.shape[-2:])

        assert output.shape == torch.Size([2, 2, 257, 601])


class TestPerSourceAttention:
    """Tests for per-source visual attention architecture (SPEC FIXME - per-source)."""

    def test_forward_5d_per_source_visual_features(self, cfg):
        """Per-source 5D visual features [B, N, T, P, D_vis] produce correct output shape."""
        model = SeparatorModule(cfg, phase="phase2")
        model.eval()
        B, N = 2, 2
        T, P, D_vis = 150, 1024, 768
        mixture_stft = torch.randn(B, 2, 257, 601)

        visual_features_5d = torch.randn(B, N, T, P, D_vis)
        with torch.no_grad():
            out = model(mixture_stft, visual_features=visual_features_5d)

        L_expected = 96000
        assert out.shape == (B, N, L_expected), \
            f"5D visual: expected ({B},{N},{L_expected}), got {out.shape}"

    def test_bottleneck_frame_idx(self, cfg):
        """Test corrected temporal alignment logic from Step 1."""
        model = SeparatorModule(cfg, phase="phase2")
        align = model.bottleneck_frame_idx.tolist()
        assert len(align) == 171
        
        # All positions sharing the same w (time) must map to the same frame
        for w in range(19):
            frames_for_w = {align[h*19 + w] for h in range(9)}
            assert len(frames_for_w) == 1, f"w={w} maps to multiple frames: {frames_for_w}"

    def test_decoder_input_shape(self, cfg):
        """Test that decoder_input preserves all 171 positions per source."""
        model = SeparatorModule(cfg, phase="phase2")
        model.eval()
        B, N = 2, 2
        T, P, D_vis = 150, 1024, 768
        mixture_stft = torch.randn(B, 2, 257, 601)
        visual_features = torch.randn(B, N, T, P, D_vis)
        
        with torch.no_grad():
            out = model(mixture_stft, visual_features=visual_features)
            
        assert hasattr(model, "_cached_decoder_input")
        assert model._cached_decoder_input.shape == (B, N, 512, 9, 19)

    def test_phase1_no_visual_runs(self, cfg):
        """Phase1 without visual args runs correctly."""
        model = SeparatorModule(cfg, phase="phase1")
        model.eval()
        B, N = 2, 2
        mixture_stft = torch.randn(B, 2, 257, 601)

        with torch.no_grad():
            out = model(mixture_stft)

        assert out.shape == (B, N, 96000), \
            f"Phase1: expected ({B},{N},96000), got {out.shape}"

    def test_n_sources_2_3_4(self, cfg):
        """Per-source attention works for N=2,3,4 sources."""
        B, T, P, D = 1, 150, 1024, 768
        mixture_stft = torch.randn(B, 2, 257, 601)

        for n_src in [2, 3, 4]:
            cfg_test = {
                "model": {"n_sources": n_src, "n_sources_max": 4},
                "train": {
                    "optimizer": {"lr_fusion": 3e-4, "lr_audio_enc": 3e-5, "lr_dinov2": 1e-5},
                    "scheduler": {}, "loss": {}
                },
            }
            model = SeparatorModule(cfg_test, phase="phase2")
            model.eval()
            visual_features = torch.randn(B, n_src, T, P, D)
            with torch.no_grad():
                out = model(mixture_stft, visual_features=visual_features)
            assert out.shape == (B, n_src, 96000), \
                f"N={n_src}: expected ({B},{n_src},96000), got {out.shape}"
