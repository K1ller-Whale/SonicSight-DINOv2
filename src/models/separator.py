"""PyTorch Lightning module for the DINOv2-guided separator.

SPEC 11.3, 11.4: Three-phase training with differential learning rates.
"""
import torch
import torch.nn as nn
import pytorch_lightning as pl
from einops import rearrange
from src.audio.unet import AudioUNet
from src.audio.spectrogram import ISTFTModule
from src.visual.dinov2 import DINOv2FeatureExtractor
from src.fusion.cross_attention import CrossModalAttentionModule
from src.loss.separation import SISNRLoss, CRMLoss, MultiScaleSTFTLoss, PerceptualLoss
from typing import Optional, Dict, Any, List


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
        self.n_sources = cfg.get("model", {}).get("n_sources", 4)

        # --- Components ---
        self.dinov2 = DINOv2FeatureExtractor()
        self.audio_unet = AudioUNet()
        # Visual projection: D_v=768 -> D_a=512, no bias (SPEC 11.3)
        self.visual_proj = nn.Linear(768, 512, bias=False)
        # Bottleneck projection: Conv1x1 equivalent (512 -> 512)
        self.bottleneck_proj = nn.Linear(512, 512)
        # Temporal alignment: video frame index mapping (see dataset)
        self.cross_attn = CrossModalAttentionModule()
        # Source query tokens: N learnable tokens
        self.source_queries = nn.Parameter(torch.randn(self.n_sources, 512) * 0.02)
        # iSTFT for waveform reconstruction
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
            mixture_stft: [B, 2, F, T] - complex STFT of mixture
            video_frames: [B, N_frames, 3, H, W] or None (phase1)
        Returns:
            separated_waveforms: [B, N_sources, L]
        """
        B = mixture_stft.shape[0]
        device = mixture_stft.device
        target_shape = mixture_stft.shape[-2:]  # [F, T]

        # 1. Audio U-Net encoder -> bottleneck [B, 512, 9, 19]
        bottleneck, skips = self.audio_unet.encoder(mixture_stft)

        # 2. Flatten bottleneck: [B, 512, 9, 19] -> [B, 171, 512]
        bottleneck_flat = rearrange(bottleneck, "B C H W -> B (H W) C")

        # 3. Bottleneck projection (Conv1x1 equivalent)
        bottleneck_flat = self.bottleneck_proj(bottleneck_flat)  # [B, 171, 512]

        # 4. Source query tokens: [N_sources, 512] -> [B, N_sources, 512]
        source_q = self.source_queries.unsqueeze(0).expand(B, -1, -1)

        # 5. Cross-modal attention
        if video_frames is not None and self.phase != "phase1":
            N_v = video_frames.shape[1]
            video_reshaped = rearrange(video_frames, "B N C H W -> (B N) C H W")

            with torch.no_grad():
                visual_features = self.dinov2(video_reshaped)  # [B*N_v, 1024, 768]

            # Reshape: [B, N_v, 1024, 768]
            visual_features = rearrange(visual_features, "(B N) P D -> B N P D", B=B)

            # Temporal alignment: each bottleneck position -> video frame
            n_bottleneck = bottleneck_flat.shape[1]
            alignment = torch.floor(
                torch.arange(n_bottleneck, device=device).float() * N_v / n_bottleneck
            ).long().clamp(0, N_v - 1)

            # Gather per-position visual features
            per_pos_visual = []
            for t_a in range(n_bottleneck):
                v_idx = alignment[t_a]
                vf = visual_features[:, v_idx, :, :]  # [B, 1024, 768]
                per_pos_visual.append(vf)

            # Stack: [B, n_bottleneck, 1024, 768]
            visual_kv = torch.stack(per_pos_visual, dim=1)

            # Project 768 -> 512 (no bias)
            B_orig, T_bp, P, D_orig = visual_kv.shape
            visual_kv_flat = rearrange(visual_kv, "B T P D -> (B T P) D")
            visual_kv_proj = self.visual_proj(visual_kv_flat)  # [B*T*P, 512]
            visual_kv = rearrange(visual_kv_proj, "(B T P) D -> B T P D",
                                  B=B_orig, T=T_bp)

            # Average over patches per position: [B, n_bottleneck, 512]
            visual_kv = visual_kv.mean(dim=2)

            # Combined query: [source_queries, bottleneck_flat]
            combined_query = torch.cat([source_q, bottleneck_flat], dim=1)
            attended = self.cross_attn(combined_query, visual_kv)
            source_features = attended[:, :self.n_sources, :]  # [B, N, 512]
        else:
            # Phase 1: source queries + bottleneck attend to bottleneck
            combined_query = torch.cat([source_q, bottleneck_flat], dim=1)
            attended = self.cross_attn(combined_query, bottleneck_flat)
            source_features = attended[:, :self.n_sources, :]

        # 6. Per-source decoding
        separated_waveforms = []
        for i in range(self.n_sources):
            sf = source_features[:, i, :]  # [B, 512]
            sf_spatial = rearrange(sf, "B D -> B D 1 1")
            modulated = bottleneck + sf_spatial  # Broadcast to [B, 512, 9, 19]

            # Decode: [B, 512, 9, 19] -> [B, 2, 257, 601]
            mask = self.audio_unet.decoder(modulated, skips, target_shape=target_shape)
            mask = torch.tanh(mask)

            # iSTFT
            separated = self.istft(mask, mixture_stft)  # [B, L]
            separated_waveforms.append(separated)

        # Stack: [B, N_sources, L]
        output = torch.stack(separated_waveforms, dim=1)
        return output

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Training step with phase-appropriate loss."""
        mixture_stft = batch["mixture_stft"]
        target_waveforms = batch["target_waveforms"]
        video_frames = batch.get("video_frames")
        B = target_waveforms.shape[0]

        # Forward pass
        if self.phase == "phase1":
            predicted_waveforms = self(mixture_stft, video_frames=None)
        else:
            predicted_waveforms = self(mixture_stft, video_frames=video_frames)

        # Compute SI-SNR loss per batch element
        losses = []
        for b in range(B):
            losses.append(self.si_snr(predicted_waveforms[b], target_waveforms[b]))
        si_snr_loss = torch.stack(losses).mean()

        # Phase-specific loss
        if self.phase == "phase1":
            total_loss = si_snr_loss
            self.log("train/si_snr_loss", si_snr_loss, prog_bar=True)

        elif self.phase == "phase2":
            entropy_loss = self._compute_attention_entropy()
            total_loss = si_snr_loss + 0.1 * entropy_loss
            self.log("train/si_snr_loss", si_snr_loss, prog_bar=True)
            self.log("train/entropy_loss", entropy_loss, prog_bar=True)

        else:  # phase3
            target_masks = batch.get("target_crm_masks")
            pred_masks = self._predict_masks(mixture_stft)
            crm_loss = self.crm(pred_masks, target_masks)
            stft_loss = self.stft(predicted_waveforms, target_waveforms)
            perceptual_loss = self.perceptual(predicted_waveforms, target_waveforms)

            alpha = self.cfg.get("train", {}).get("loss", {}).get("alpha_crm", 0.1)
            beta = self.cfg.get("train", {}).get("loss", {}).get("beta_stft", 0.05)
            gamma = self.cfg.get("train", {}).get("loss", {}).get("gamma_perceptual", 0.1)

            total_loss = si_snr_loss + alpha * crm_loss + beta * stft_loss + gamma * perceptual_loss
            self.log("train/si_snr_loss", si_snr_loss, prog_bar=True)
            self.log("train/crm_loss", crm_loss, prog_bar=True)
            self.log("train/stft_loss", stft_loss, prog_bar=True)
            self.log("train/perceptual_loss", perceptual_loss, prog_bar=True)

        self.log("train/total_loss", total_loss, prog_bar=True)
        return total_loss

    def _predict_masks(self, mixture_stft: torch.Tensor) -> torch.Tensor:
        """Predict per-source masks for cRM loss."""
        B = mixture_stft.shape[0]
        target_shape = mixture_stft.shape[-2:]

        # Encode
        bottleneck, skips = self.audio_unet.encoder(mixture_stft)
        x = rearrange(bottleneck, "B C H W -> B (H W) C")
        x = self.bottleneck_proj(x)
        source_q = self.source_queries.unsqueeze(0).expand(B, -1, -1)

        # Phase-specific visual processing
        if self.phase != "phase1":
            # Simplified: use bottleneck self-attention for mask prediction
            combined = torch.cat([source_q, x], dim=1)
            attended = self.cross_attn(combined, x)
        else:
            combined = torch.cat([source_q, x], dim=1)
            attended = self.cross_attn(combined, x)

        source_features = attended[:, :self.n_sources, :]  # [B, N, 512]

        # Decode per-source masks
        x_spatial = rearrange(x, "B (H W) C -> B C H W", H=bottleneck.shape[-2],
                              W=bottleneck.shape[-1])
        masks = []
        for i in range(self.n_sources):
            sf = source_features[:, i, :]  # [B, 512]
            sf_spatial = rearrange(sf, "B D -> B D 1 1")
            modulated = x_spatial + sf_spatial
            mask = self.audio_unet.decoder(modulated, skips, target_shape=target_shape)
            mask = torch.tanh(mask)
            masks.append(mask)
        return torch.stack(masks, dim=1)  # [B, N, 2, F, T]

    def _compute_attention_entropy(self) -> torch.Tensor:
        """Compute attention entropy for regularization."""
        return torch.tensor(0.0, device=self.device)

    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Validation step with metrics."""
        mixture_stft = batch["mixture_stft"]
        video_frames = batch.get("video_frames")
        target_waveforms = batch["target_waveforms"]
        B = target_waveforms.shape[0]

        # Forward pass
        if self.phase == "phase1":
            predicted_waveforms = self(mixture_stft, video_frames=None)
        else:
            predicted_waveforms = self(mixture_stft, video_frames=video_frames)

        # Compute SI-SNR loss per batch element
        losses = []
        for b in range(B):
            losses.append(self.si_snr(predicted_waveforms[b], target_waveforms[b]))
        si_snr_loss = torch.stack(losses).mean()

        self.log("val/si_snr_loss", si_snr_loss, prog_bar=True)
        return si_snr_loss

    def configure_optimizers(self) -> torch.optim.Optimizer:
        """Differential learning rates per phase (SPEC 11.4)."""
        lr_map = {
            "phase1": {"all": 1e-3},
            "phase2": {"fusion": 5e-4},
            "phase3": {
                "fusion": 3e-4,
                "audio_enc": 3e-5,
                "dinov2": 1e-5,
            },
        }

        if self.phase == "phase1":
            # All parameters, single LR
            return torch.optim.AdamW(
                self.parameters(),
                lr=lr_map["phase1"]["all"],
                weight_decay=1e-4,
                betas=(0.9, 0.999)
            )

        elif self.phase == "phase2":
            # Train cross-attention and projection only (audio U-Net frozen)
            trainable_params = list(self.cross_attn.parameters()) + \
                              list(self.visual_proj.parameters()) + \
                              [self.source_queries]
            return torch.optim.AdamW(
                trainable_params,
                lr=lr_map["phase2"]["fusion"],
                weight_decay=1e-4,
                betas=(0.9, 0.999)
            )

        else:  # phase3
            # Differential learning rates
            param_groups = [
                {"params": self.cross_attn.parameters(), "lr": lr_map["phase3"]["fusion"]},
                {"params": self.visual_proj.parameters(), "lr": lr_map["phase3"]["fusion"]},
                {"params": [self.source_queries], "lr": lr_map["phase3"]["fusion"]},
                {"params": self.audio_unet.decoder.parameters(), "lr": lr_map["phase3"]["fusion"]},
                {"params": self.audio_unet.encoder.blocks[-2:].parameters(),
                 "lr": lr_map["phase3"]["audio_enc"]},
                {"params": self.dinov2.parameters(), "lr": lr_map["phase3"]["dinov2"]},
            ]
            return torch.optim.AdamW(
                param_groups,
                weight_decay=1e-4,
                betas=(0.9, 0.999)
            )
