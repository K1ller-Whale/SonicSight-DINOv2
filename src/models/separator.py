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
        self._progressive_steps = [20000, 40000]
        self._progressive_sources = [2, 3, 4]

        # --- Components ---
        self.dinov2 = DINOv2FeatureExtractor()
        self.audio_unet = AudioUNet()
        # Visual projection: D_v=768 -> D_a=512, no bias (SPEC 11.3)
        self.visual_proj = nn.Linear(768, 512, bias=False)
        # Bottleneck projection: Conv1x1 equivalent (512 -> 512)
        self.bottleneck_proj = nn.Linear(512, 512)
        # Temporal alignment: video frame index mapping (see dataset)
        self.cross_attn = CrossModalAttentionModule()
        # Source query tokens: N learnable tokens (created on CPU, Lightning moves to device)
        self.source_queries = nn.Parameter(torch.randn(self.n_sources, 512) * 0.02)
        # iSTFT for waveform reconstruction
        self.istft = ISTFTModule()

        # Cache for _predict_masks to avoid re-encoding
        self._cached_bottleneck = None
        self._cached_bottleneck_flat = None
        self._cached_skips = None
        self._cached_source_features = None
        self._cached_attn_weights = None

        # --- Losses ---
        self.si_snr = SISNRLoss()
        self.crm = CRMLoss()
        self.stft = MultiScaleSTFTLoss()
        self.perceptual = PerceptualLoss()

    def _clear_cache(self):
        """Clear cached intermediates."""
        self._cached_bottleneck = None
        self._cached_bottleneck_flat = None
        self._cached_skips = None
        self._cached_source_features = None
        self._cached_attn_weights = None

    def _update_progressive_sources(self) -> bool:
        """Update n_sources and source_queries based on global_step.
        Returns True if n_sources changed."""
        step = self.global_step
        new_n_sources = self._progressive_sources[0]
        for threshold, n_src in zip(self._progressive_steps, self._progressive_sources[1:]):
            if step >= threshold:
                new_n_sources = n_src
            else:
                break
        if new_n_sources != self.n_sources:
            old_queries = self.source_queries.data
            self.n_sources = new_n_sources
            new_queries = torch.randn(new_n_sources, 512, device=old_queries.device) * 0.02
            min_src = min(old_queries.shape[0], new_n_sources)
            new_queries[:min_src] = old_queries[:min_src]
            self.source_queries = nn.Parameter(new_queries)
            return True
        return False

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

        # Clear cache at start of forward
        self._clear_cache()

        # 5. Cross-modal attention
        if video_frames is not None and self.phase != "phase1":
            N_v = video_frames.shape[1]
            video_reshaped = rearrange(video_frames, "B N C H W -> (B N) C H W")

            # Chunk DINOv2 processing to avoid OOM (process in batches of 8 frames)
            chunk_size = 8
            num_chunks = (video_reshaped.shape[0] + chunk_size - 1) // chunk_size
            visual_chunks = []
            for i in range(num_chunks):
                start = i * chunk_size
                end = min((i + 1) * chunk_size, video_reshaped.shape[0])
                chunk = video_reshaped[start:end]
                with torch.no_grad():
                    chunk_features = self.dinov2(chunk)  # [chunk_size, 1024, 768]
                visual_chunks.append(chunk_features)
            visual_features = torch.cat(visual_chunks, dim=0)  # [B*N_v, 1024, 768]

            # Reshape: [B, N_v, 1024, 768]
            visual_features = rearrange(visual_features, "(B N) P D -> B N P D", B=B)

            # Temporal alignment: each bottleneck position -> video frame
            # SPEC: floor(t_a * 150 / 171), hardcoded 150 frames for 10s @ 15fps, 171 bottleneck positions
            n_bottleneck = bottleneck_flat.shape[1]
            alignment = torch.floor(
                torch.arange(n_bottleneck, device=device).float() * 150 / 171
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

            # visual_kv: [B, n_bottleneck, 1024, 512] - full spatial grid preserved
            # Temporal alignment: each bottleneck position t_a maps to video frame alignment[t_a]
            # For cross-attention, flatten visual sequence: [B, n_bottleneck * 1024, 512]
            # Create attention mask: each audio query position attends only to its 1024 patches

            B, T_a, P, D = visual_kv.shape  # [B, 171, 1024, 512]
            visual_kv_flat = rearrange(visual_kv, "B T P D -> B (T P) D")  # [B, 171*1024, 512]

            # Build attention mask: [B, T_q, T_kv]
            # T_q = N_sources + T_a (source queries + bottleneck positions)
            # Each bottleneck position t attends only to its P=1024 patches at [t*P : (t+1)*P]
            # Source queries (first N_sources) attend to all visual patches
            N_sources = self.n_sources
            T_q = N_sources + T_a
            mask = torch.ones(B, T_q, T_a * P, dtype=torch.bool, device=device)
            for t in range(T_a):
                mask[:, N_sources + t, t * P : (t + 1) * P] = False
            # Source queries: no mask (attend to all)
            mask[:, :N_sources, :] = False

            # Combined query: source queries + bottleneck positions [B, N_sources + T_a, 512]
            combined_query = torch.cat([source_q, bottleneck_flat], dim=1)

            # Cross-modal attention with mask - use underlying MultiheadAttention directly
            x = self.cross_attn.pos_enc(combined_query)
            attn_weights_list = []
            for block in self.cross_attn.blocks:
                q = block.pre_norm_query(x)
                attn_out, attn_weights = block.attn(q, visual_kv_flat, visual_kv_flat,
                                          key_padding_mask=None,
                                          attn_mask=mask,
                                          need_weights=True,
                                          average_attn_weights=False)
                attn_weights_list.append(attn_weights)
                x = x + block.dropout1(attn_out)

                ffn_in = block.pre_norm_ffn(x)
                x = x + block.ffn(ffn_in)

            # Store attention weights for entropy loss (phase2)
            self._cached_attn_weights = attn_weights_list  # List of [B, n_heads, T_q, T_kv]

            # Extract source features (first N_sources positions from query)
            source_features = x[:, :self.n_sources, :]

        else:
            # Phase 1: source queries + bottleneck attend to bottleneck (self-attention)
            combined_query = torch.cat([source_q, bottleneck_flat], dim=1)
            attended = self.cross_attn(combined_query, bottleneck_flat)
            source_features = attended[:, :self.n_sources, :]

        # Cache intermediates for _predict_masks
        self._cached_bottleneck = bottleneck
        self._cached_bottleneck_flat = bottleneck_flat
        self._cached_skips = skips
        self._cached_source_features = source_features

        # 6. Per-source decoding with FiLM-style modulation
        separated_waveforms = []
        for i in range(self.n_sources):
            sf = source_features[:, i, :]  # [B, 512]
            sf_spatial = rearrange(sf, "B D -> B D 1 1")
            # FiLM-style multiplicative gating: bottleneck * sigmoid(sf_spatial)
            modulated = bottleneck * sf_spatial.sigmoid()

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
        self._update_progressive_sources()

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
            perceptual_loss = self.perceptual(predicted_waveforms, target_waveforms, self.dinov2)

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
        """Predict per-source masks for cRM loss using cached forward intermediates."""
        B = mixture_stft.shape[0]
        target_shape = mixture_stft.shape[-2:]

        # Use cached forward pass intermediates
        if (self._cached_bottleneck is None
            or self._cached_bottleneck.shape[0] != B
            or self._cached_bottleneck_flat is None
            or self._cached_skips is None
            or self._cached_source_features is None):
            # Fallback: recompute if cache invalid (should not happen in normal training)
            bottleneck, skips = self.audio_unet.encoder(mixture_stft)
            bottleneck_flat = rearrange(bottleneck, "B C H W -> B (H W) C")
            bottleneck_flat = self.bottleneck_proj(bottleneck_flat)
            source_q = self.source_queries.unsqueeze(0).expand(B, -1, -1)

            if self.phase != "phase1":
                combined = torch.cat([source_q, bottleneck_flat], dim=1)
                attended = self.cross_attn(combined, bottleneck_flat)
            else:
                combined = torch.cat([source_q, bottleneck_flat], dim=1)
                attended = self.cross_attn(combined, bottleneck_flat)

            source_features = attended[:, :self.n_sources, :]
        else:
            bottleneck = self._cached_bottleneck
            skips = self._cached_skips
            source_features = self._cached_source_features

        # Decode per-source masks using cached bottleneck + source_features
        masks = []
        for i in range(self.n_sources):
            sf = source_features[:, i, :]
            sf_spatial = rearrange(sf, "B D -> B D 1 1")
            modulated = bottleneck * sf_spatial.sigmoid()
            mask = self.audio_unet.decoder(modulated, skips, target_shape=target_shape)
            mask = torch.tanh(mask)
            masks.append(mask)
        return torch.stack(masks, dim=1)

    def _compute_attention_entropy(self) -> torch.Tensor:
        """Compute attention entropy for regularization.
        Entropy = -sum(p * log(p)) over attention distribution per head.
        High entropy = diffuse attention, low entropy = focused.
        We encourage moderate entropy.
        """
        if not hasattr(self, '_cached_attn_weights') or self._cached_attn_weights is None:
            return torch.tensor(0.0, device=self.device)

        total_entropy = 0.0
        count = 0
        for attn_weights in self._cached_attn_weights:
            # attn_weights: [B, n_heads, T_q, T_kv]
            # Compute entropy per head per query position
            p = attn_weights + 1e-8  # avoid log(0)
            p = p / p.sum(dim=-1, keepdim=True)  # normalize
            entropy = -(p * p.log()).sum(dim=-1)  # [B, n_heads, T_q]
            total_entropy += entropy.mean()
            count += 1

        return total_entropy / count if count > 0 else torch.tensor(0.0, device=self.device)

    def on_before_optimizer_step(self, optimizer):
        """Gradient clipping with max_norm=1.0."""
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)

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
            # Differential learning rates - skip frozen params (requires_grad=False)
            dinov2_params = [p for p in self.dinov2.parameters() if p.requires_grad]
            audio_enc_params = [p for p in self.audio_unet.encoder.blocks[-2:].parameters() if p.requires_grad]
            param_groups = [
                {"params": self.cross_attn.parameters(), "lr": lr_map["phase3"]["fusion"]},
                {"params": self.visual_proj.parameters(), "lr": lr_map["phase3"]["fusion"]},
                {"params": [self.source_queries], "lr": lr_map["phase3"]["fusion"]},
                {"params": self.audio_unet.decoder.parameters(), "lr": lr_map["phase3"]["fusion"]},
                {"params": audio_enc_params, "lr": lr_map["phase3"]["audio_enc"]},
                {"params": dinov2_params, "lr": lr_map["phase3"]["dinov2"]},
            ]
            return torch.optim.AdamW(
                param_groups,
                weight_decay=1e-4,
                betas=(0.9, 0.999)
            )
