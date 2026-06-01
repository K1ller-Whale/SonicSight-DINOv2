"""PyTorch Lightning module for the DINOv2-guided separator.

SPEC 11.3, 11.4: Three-phase training with differential learning rates.
"""
import torch
import torch.nn as nn
import pytorch_lightning as pl
from src.audio.unet import AudioUNet
from src.audio.spectrogram import ISTFTModule
from src.visual.dinov2 import DINOv2FeatureExtractor
from src.fusion.cross_attention import CrossModalAttentionModule
from src.loss.separation import SISNRLoss, CRMLoss, MultiScaleSTFTLoss, PerceptualLoss
from typing import Optional, Dict, Any


class SeparatorModule(pl.LightningModule):
    """
    Phase-aware LightningModule for audio-visual source separation.

    Phase 1: audio-only (visual disabled)
    Phase 2: cross-modal attention warmup (audio frozen, attention only)
    Phase 3: end-to-end fine-tuning (differential learning rates)
    """

    def __init__(self, cfg: Dict[str, Any], phase: str = "phase1"):
        super().__init__()
        self.cfg = cfg
        self.phase = phase
        self.save_hyperparameters()

        # --- Components ---
        self.dinov2 = DINOv2FeatureExtractor()
        self.audio_unet = AudioUNet()
        # Visual projection: D_v=768 → D_a=512, no bias (SPEC 11.3)
        self.visual_proj = nn.Linear(768, 512, bias=False)
        # Temporal alignment: video frame index mapping (see dataset)
        self.cross_attn = CrossModalAttentionModule()
        # Source query tokens: N learnable tokens
        self.source_queries = nn.Parameter(torch.randn(4, 512) * 0.02)
        # Decoder: N independent U-Net decoders (parameter-shared)
        self.istft = ISTFTModule()

        # --- Losses ---
        self.si_snr = SISNRLoss()
        self.crm = CRMLoss()
        self.stft = MultiScaleSTFTLoss()
        self.perceptual = PerceptualLoss()

    def forward(self, mixture_stft: torch.Tensor,
                video_frames: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            mixture_stft: [B, 2, F, T]
            video_frames: [B, N_frames, 3, H, W] or None (phase1)
        Returns:
            separated_waveforms: [B, N_sources, L]
        """
        # TODO: implement full forward pass
        raise NotImplementedError("Forward pass not yet implemented.")

    def training_step(self, batch, batch_idx):
        """Training step with phase-appropriate loss."""
        # TODO: implement
        raise NotImplementedError("Training step not yet implemented.")

    def validation_step(self, batch, batch_idx):
        """Validation step with metrics."""
        # TODO: implement
        pass

    def configure_optimizers(self):
        """Differential learning rates per phase."""
        lr_map = {
            "phase1": {"all": 1e-3},
            "phase2": {"fusion": 5e-4},
            "phase3": {
                "fusion": 3e-4,
                "audio": 3e-5,
                "dinov2": 1e-5,
            },
        }
        # TODO: build optimizer with param groups
        return torch.optim.AdamW(self.parameters(), lr=1e-3)
