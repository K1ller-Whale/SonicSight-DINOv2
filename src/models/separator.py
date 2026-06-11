"""PyTorch Lightning module for the DINOv2-guided separator.

SPEC 11.3, 11.4: Three-phase training with differential learning rates.
"""
import math
import torch
import torch.nn as nn
import pytorch_lightning as pl
from einops import rearrange
from src.audio.unet import AudioUNet
from src.audio.spectrogram import ISTFTModule
from src.visual.dinov2 import DINOv2FeatureExtractor
from src.fusion.cross_attention import CrossModalAttentionModule
from src.loss.separation import SISNRLoss, CRMLoss, MultiScaleSTFTLoss, PerceptualLoss
from src.loss.pit_wrapper import PITLossWrapper
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
        max_sources = cfg.get("model", {}).get("n_sources_max", 4)
        self._progressive_steps = [20000, 40000]
        self._progressive_sources = [2, 3, 4]

        # --- Components ---
        self.dinov2 = DINOv2FeatureExtractor()
        self._apply_dinov2_freeze(cfg)
        self.audio_unet = AudioUNet()
        # Visual projection: D_v=768 -> D_a=512, no bias (SPEC 11.3)
        self.visual_proj = nn.Linear(768, 512, bias=False)
        # Bottleneck projection: Conv1x1 equivalent (512 -> 512)
        self.bottleneck_proj = nn.Linear(512, 512)
        # Temporal alignment: video frame index mapping (see dataset)
        self.cross_attn = CrossModalAttentionModule()
        # Source query tokens: allocate MAX sources upfront, slice in forward
        self.source_queries = nn.Parameter(torch.randn(max_sources, 512) * 0.02)
        # iSTFT for waveform reconstruction
        data_cfg = cfg.get("data", {})
        sr = data_cfg.get("sample_rate", 16000)
        dur = data_cfg.get("clip_duration", 6)
        self.istft = ISTFTModule(length=sr * dur)

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
        self.perceptual = PerceptualLoss(dinov2_extractor=self.dinov2)
        self.pit_wrapper = PITLossWrapper(self.si_snr, self.crm)

    def _apply_dinov2_freeze(self, cfg: Dict[str, Any]):
        """Apply DINOv2 freezing based on config (phase3 only)."""
        if self.phase != "phase3":
            # Phase 1 & 2: DINOv2 always frozen
            for p in self.dinov2.parameters():
                p.requires_grad = False
            return

        train_cfg = cfg.get("train", {})
        dinov2_cfg = train_cfg.get("dinov2", {})
        freeze_all = dinov2_cfg.get("freeze_all", True)
        unfrozen_blocks = dinov2_cfg.get("unfrozen_blocks", 0)

        if freeze_all or unfrozen_blocks == 0:
            for p in self.dinov2.parameters():
                p.requires_grad = False
        else:
            # Unfreeze last N blocks
            # DINOv2 base has 12 blocks (encoder.layer.0 to encoder.layer.11)
            # We need to access the underlying transformer blocks
            for p in self.dinov2.parameters():
                p.requires_grad = False
            # Unfreeze last N blocks
            if hasattr(self.dinov2, 'model') and hasattr(self.dinov2.model, 'encoder'):
                # HF DINOv2 structure
                blocks = self.dinov2.model.encoder.layer
                for block in blocks[-unfrozen_blocks:]:
                    for p in block.parameters():
                        p.requires_grad = True
            elif hasattr(self.dinov2, 'transformer') and hasattr(self.dinov2.transformer, 'blocks'):
                # Alternative structure
                blocks = self.dinov2.transformer.blocks
                for block in blocks[-unfrozen_blocks:]:
                    for p in block.parameters():
                        p.requires_grad = True

    def _clear_cache(self):
        """Clear cached intermediates."""
        self._cached_bottleneck = None
        self._cached_bottleneck_flat = None
        self._cached_skips = None
        self._cached_source_features = None
        self._cached_attn_weights = None

    def _update_progressive_sources(self) -> bool:
        """Update n_sources based on global_step.
        Only runs in Phase 3 (progressive curriculum: 2->3->4).
        Returns True if n_sources changed. Never replaces source_queries Parameter."""
        if self.phase != "phase3":
            return False
        step = self.global_step
        new_n_sources = self._progressive_sources[0]
        for threshold, n_src in zip(self._progressive_steps, self._progressive_sources[1:]):
            if step >= threshold:
                new_n_sources = n_src
            else:
                break
        if new_n_sources != self.n_sources:
            self.n_sources = new_n_sources
            return True
        return False

    def forward(self, mixture_stft: torch.Tensor,
                video_frames: Optional[torch.Tensor] = None,
                visual_features: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            mixture_stft: [B, 2, F, T] - complex STFT of mixture
            video_frames: [B, T, 3, H, W] raw RGB frames (deprecated, use visual_features)
            visual_features: [B, N_sources, T, 1024, 768] or [B, T, 1024, 768] cached DINO features
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
        source_q = self.source_queries[:self.n_sources].unsqueeze(0).expand(B, -1, -1)

        # Clear cache at start of forward
        self._clear_cache()

        # 5. Cross-modal attention
        have_visual = (visual_features is not None or video_frames is not None) and self.phase != "phase1"

        if have_visual:
            visual_kv = None  # [B, T_a, P, D] aligned visual features for bottleneck cross-attention
            source_q_local = None  # [B, N, D] source query features (may be per-source or shared)

            if visual_features is not None:
                # CACHED DINO PATH
                if visual_features.dim() == 5:
                    # Per-source visual features: [B, N_sources, T, 1024, 768]
                    # Each source query attends EXCLUSIVELY to its visual stream
                    source_attended = []
                    for n in range(self.n_sources):
                        vkv_n = visual_features[:, n]  # [B, T, 1024, 768]
                        vkv_n = self.visual_proj(vkv_n)  # [B, T, 1024, 512]
                        vkv_n_flat = rearrange(vkv_n, "B T P D -> B (T P) D")  # [B, T*1024, 512]
                        sq_n = self.source_queries[n:n+1].unsqueeze(0).expand(B, -1, -1)  # [B, 1, 512]
                        attended_n = self.cross_attn(sq_n, vkv_n_flat)  # [B, 1, 512]
                        source_attended.append(attended_n)
                    source_q_local = torch.cat(source_attended, dim=1)  # [B, N, 512]
                    # For bottleneck cross-attention: use averaged visual (both sources see aligned context)
                    N_v = visual_features.shape[2]
                    n_bottleneck = bottleneck_flat.shape[1]
                    align_idx = [int(i * N_v / n_bottleneck) for i in range(n_bottleneck)]
                    visual_avg = visual_features.mean(dim=1)[:, align_idx]  # [B, 171, 1024, 768]
                    # Project for bottleneck cross-attention
                    B_v, T_bp, P, _ = visual_avg.shape
                    visual_flat = rearrange(visual_avg, "B T P D -> (B T P) D")
                    visual_proj_out = self.visual_proj(visual_flat)
                    visual_kv = rearrange(visual_proj_out, "(B T P) D -> B T P D", B=B_v, T=T_bp)  # [B, 171, 1024, 512]
                else:
                    # 4D shared visual features: [B, T, 1024, 768]
                    # All source queries attend to shared visual context
                    visual_kv = self.visual_proj(visual_features)  # [B, T, 1024, 512]
                    source_q_local = self.source_queries[:self.n_sources].unsqueeze(0).expand(B, -1, -1)  # [B, N, 512]
                    # Align visual to bottleneck timesteps
                    N_v = visual_features.shape[1]
                    n_bottleneck = bottleneck_flat.shape[1]
                    align_idx = [int(i * N_v / n_bottleneck) for i in range(n_bottleneck)]
                    visual_aligned = visual_features[:, align_idx]  # [B, 171, 1024, 768]
                    # Project for bottleneck cross-attention
                    B_v, T_bp, P, _ = visual_aligned.shape
                    visual_flat = rearrange(visual_aligned, "B T P D -> (B T P) D")
                    visual_proj_out = self.visual_proj(visual_flat)
                    visual_kv = rearrange(visual_proj_out, "(B T P) D -> B T P D", B=B_v, T=T_bp)  # [B, 171, 1024, 512]
            else:
                # RAW FRAMES PATH: video_frames must be [B, T_frames, 3, H, W]
                if video_frames.dim() != 5 or video_frames.shape[2] != 3:
                    raise ValueError(
                        "video_frames must be [B,T,3,H,W]. "
                        "Pass DINO features as visual_features= instead."
                    )
                N_v = video_frames.shape[1]
                video_reshaped = rearrange(video_frames, "B N C H W -> (B N) C H W")
                # Process in chunks to avoid OOM
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
                visual_features_raw = torch.cat(visual_chunks, dim=0)  # [B*N_v, 1024, 768]
                visual_features_raw = rearrange(visual_features_raw, "(B N) P D -> B N P D", B=B)  # [B, N_v, 1024, 768]
                # Source queries attend to all visual features (shared context)
                source_q_local = self.source_queries[:self.n_sources].unsqueeze(0).expand(B, -1, -1)  # [B, N, 512]
                # Temporal alignment: bottleneck pos -> video frame
                n_bottleneck = bottleneck_flat.shape[1]
                alignment = torch.floor(
                    torch.arange(n_bottleneck, device=device).float() * N_v / n_bottleneck
                ).long().clamp(0, N_v - 1)
                per_pos_visual = []
                for t_a in range(n_bottleneck):
                    vf = visual_features_raw[:, alignment[t_a], :, :]  # [B, 1024, 768]
                    per_pos_visual.append(vf)
                visual_kv = torch.stack(per_pos_visual, dim=1)  # [B, n_bottleneck, 1024, 768]
                # Project for bottleneck cross-attention
                B_v, T_bp, P, _ = visual_kv.shape
                visual_flat = rearrange(visual_kv, "B T P D -> (B T P) D")
                visual_proj_out = self.visual_proj(visual_flat)
                visual_kv = rearrange(visual_proj_out, "(B T P) D -> B T P D", B=B_v, T=T_bp)  # [B, 171, 1024, 512]

            # Cross-attention: combined query (source_q + bottleneck) attends to visual key/value
            B_ca, T_a, P_vis, D_vis = visual_kv.shape
            visual_kv_flat = rearrange(visual_kv, "B T P D -> B (T P) D")  # [B, n_bottleneck * 1024, 512]
            T_q = self.n_sources + T_a
            n_heads = self.cross_attn.blocks[0].attn.num_heads
            mask = torch.ones(B * n_heads, T_q, T_a * P_vis, dtype=torch.bool, device=device)
            for t in range(T_a):
                mask[:, self.n_sources + t, t * P_vis : (t + 1) * P_vis] = False
            mask[:, :self.n_sources, :] = False  # Source queries attend to all
            combined_query = torch.cat([source_q_local, bottleneck_flat], dim=1)  # [B, N + 171, 512]
            # Hook to capture attention weights
            self._cached_attn_weights = []
            def attn_hook(module, input, output):
                if isinstance(output, tuple) and len(output) == 2 and output[1] is not None:
                    self._cached_attn_weights.append(output[1].detach())
            hooks = []
            for block in self.cross_attn.blocks:
                hooks.append(block.attn.register_forward_hook(attn_hook))
            try:
                attended = self.cross_attn(combined_query, visual_kv_flat, attn_mask=mask)
            finally:
                for hook in hooks:
                    hook.remove()
            source_features = attended[:, :self.n_sources, :]  # [B, N, 512]

        else:
            # Phase 1: no visual, source queries + bottleneck (self-attention)
            source_q_local = self.source_queries[:self.n_sources].unsqueeze(0).expand(B, -1, -1)
            combined_query = torch.cat([source_q_local, bottleneck_flat], dim=1)
            attended = self.cross_attn(combined_query, bottleneck_flat)
            source_features = attended[:, :self.n_sources, :]

        # Cache intermediates for _predict_masks
        self._cached_bottleneck = bottleneck
        self._cached_bottleneck_flat = bottleneck_flat
        self._cached_skips = skips
        self._cached_source_features = source_features

        # 6. Per-source decoding: reshape source_features [B, N, 512] -> [B, N, 512, 9, 19]
        # and call decoder N times with shared weights
        # Reshape: [B, N, 512] -> [B, N, 512, 1, 1] -> expand to [B, N, 512, 9, 19]
        B, N, D = source_features.shape
        source_features_spatial = source_features.view(B, N, D, 1, 1).expand(B, N, D, 9, 19)

        separated_waveforms = []
        for i in range(N):
            # Each source gets its own decoding pass with shared decoder weights
            src_feat = source_features_spatial[:, i]  # [B, 512, 9, 19]
            mask = self.audio_unet.decoder(src_feat, skips, target_shape=target_shape)
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
        visual_features = batch.get("visual_features")
        B = target_waveforms.shape[0]

        # Forward pass - route visual input appropriately
        if self.phase == "phase1":
            predicted_waveforms = self(mixture_stft)
        elif visual_features is not None:
            predicted_waveforms = self(mixture_stft, visual_features=visual_features)
        elif video_frames is not None:
            predicted_waveforms = self(mixture_stft, video_frames=video_frames)
        else:
            predicted_waveforms = self(mixture_stft)

        # Phase-specific loss
        if self.phase == "phase1":
            # Phase 1: SI-SNR only, per-batch with PIT
            si_snr_losses = []
            for b in range(B):
                loss, _, _ = self.pit_wrapper(predicted_waveforms[b], target_waveforms[b])
                si_snr_losses.append(loss)
            si_snr_loss = torch.stack(si_snr_losses).mean()
            total_loss = si_snr_loss
            self.log("train/si_snr_loss", si_snr_loss, prog_bar=True)

        elif self.phase == "phase2":
            # Phase 2: SI-SNR + entropy
            si_snr_losses = []
            for b in range(B):
                loss, _, _ = self.pit_wrapper(predicted_waveforms[b], target_waveforms[b])
                si_snr_losses.append(loss)
            si_snr_loss = torch.stack(si_snr_losses).mean()
            entropy_loss = self._compute_attention_entropy()
            total_loss = si_snr_loss + 0.1 * entropy_loss
            self.log("train/si_snr_loss", si_snr_loss, prog_bar=True)
            self.log("train/entropy_loss", entropy_loss, prog_bar=True)

        else:  # phase3
            # Phase 3: SI-SNR + cRM (shared PIT) + STFT + Perceptual
            target_masks = batch.get("target_crm_masks")
            pred_masks = self._predict_masks(mixture_stft)

            alpha = self.cfg.get("train", {}).get("loss", {}).get("alpha_crm", 0.1)
            beta = self.cfg.get("train", {}).get("loss", {}).get("beta_stft", 0.05)
            gamma = self.cfg.get("train", {}).get("loss", {}).get("gamma_perceptual", 0.1)

            si_snr_losses = []
            crm_losses = []
            perms = []
            for b in range(B):
                si_snr_loss, crm_loss, perm = self.pit_wrapper(
                    predicted_waveforms[b], target_waveforms[b],
                    pred_masks[b], target_masks[b],
                    alpha_crm=alpha
                )
                si_snr_losses.append(si_snr_loss)
                crm_losses.append(crm_loss)
                perms.append(perm)

            si_snr_loss = torch.stack(si_snr_losses).mean()
            crm_loss = torch.stack(crm_losses).mean()

            # Apply PIT permutation to predictions for STFT and perceptual losses
            # perms: [B, N] - optimal permutation for each batch element
            aligned_preds = torch.stack([
                predicted_waveforms[b, perms[b]] for b in range(B)
            ])  # [B, N, L]

            # STFT and perceptual losses on PIT-aligned predictions
            stft_loss = self.stft(aligned_preds.view(-1, aligned_preds.shape[-1]),
                                  target_waveforms.view(-1, target_waveforms.shape[-1]))
            perceptual_loss = self.perceptual(aligned_preds, target_waveforms)

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
                # Need to compute visual features for cross-attention
                # This is a fallback - normally cache should be valid
                # For simplicity, use self-attention fallback
                attended = self.cross_attn(combined, bottleneck_flat)
            else:
                combined = torch.cat([source_q, bottleneck_flat], dim=1)
                attended = self.cross_attn(combined, bottleneck_flat)

            source_features = attended[:, :self.n_sources, :]
        else:
            bottleneck = self._cached_bottleneck
            skips = self._cached_skips
            source_features = self._cached_source_features

        # Decode per-source masks using cached source_features
        # Reshape: [B, N, 512] -> [B, N, 512, 9, 19] and call decoder N times
        N = source_features.shape[1]
        source_features_spatial = source_features.view(B, N, 512, 1, 1).expand(B, N, 512, 9, 19)

        masks = []
        for i in range(N):
            src_feat = source_features_spatial[:, i]  # [B, 512, 9, 19]
            mask = self.audio_unet.decoder(src_feat, skips, target_shape=target_shape)
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

    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Validation step with metrics."""
        mixture_stft = batch["mixture_stft"]
        video_frames = batch.get("video_frames")
        visual_features = batch.get("visual_features")
        target_waveforms = batch["target_waveforms"]
        B = target_waveforms.shape[0]

        # Forward pass - route visual input appropriately
        if self.phase == "phase1":
            predicted_waveforms = self(mixture_stft)
        elif visual_features is not None:
            predicted_waveforms = self(mixture_stft, visual_features=visual_features)
        elif video_frames is not None:
            predicted_waveforms = self(mixture_stft, video_frames=video_frames)
        else:
            predicted_waveforms = self(mixture_stft)

        # Compute SI-SNR loss per batch element with PIT
        losses = []
        for b in range(B):
            loss, _, _ = self.pit_wrapper(predicted_waveforms[b], target_waveforms[b])
            losses.append(loss)
        si_snr_loss = torch.stack(losses).mean()

        # Log as sisnri (negative loss = higher is better)
        sisnri = -si_snr_loss
        self.log("val/sisnri", sisnri, prog_bar=True)
        return sisnri

    def configure_optimizers(self):
        """Differential learning rates per phase with schedulers (SPEC 11.4)."""
        cfg = self.hparams.cfg
        train_cfg = cfg.get("train", {})
        opt_cfg = train_cfg.get("optimizer", {})
        sched_cfg = train_cfg.get("scheduler", {})

        max_steps = train_cfg.get("max_steps", 100000)
        warmup_steps = sched_cfg.get("warmup_steps", 1000)

        if self.phase == "phase1":
            lr = opt_cfg.get("lr", 1e-3)
            optimizer = torch.optim.AdamW(
                self.parameters(),
                lr=lr,
                weight_decay=opt_cfg.get("weight_decay", 1e-4),
                betas=opt_cfg.get("betas", [0.9, 0.999])
            )
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=max_steps
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "step",
                }
            }

        elif self.phase == "phase2":
            lr = opt_cfg.get("lr_fusion", 5e-4)
            # Explicitly freeze U-Net (ARCH-06) and enable only intended params
            for p in self.audio_unet.parameters():
                p.requires_grad_(False)
            for p in self.bottleneck_proj.parameters():
                p.requires_grad_(True)
            for p in self.cross_attn.parameters():
                p.requires_grad_(True)
            for p in self.visual_proj.parameters():
                p.requires_grad_(True)
            self.source_queries.requires_grad_(True)

            trainable_params = (
                list(self.bottleneck_proj.parameters()) +
                list(self.cross_attn.parameters()) +
                list(self.visual_proj.parameters()) +
                [self.source_queries]
            )
            optimizer = torch.optim.AdamW(
                trainable_params,
                lr=lr,
                weight_decay=opt_cfg.get("weight_decay", 1e-4),
                betas=opt_cfg.get("betas", [0.9, 0.999])
            )
            # Linear warmup then cosine
            def lr_lambda(step):
                if step < warmup_steps:
                    return step / warmup_steps
                progress = (step - warmup_steps) / max(max_steps - warmup_steps, 1)
                return 0.5 * (1 + math.cos(math.pi * progress))

            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
            # LambdaLR calls step() in constructor. Restore base lr.
            for pg in optimizer.param_groups:
                pg["lr"] = lr
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "step",
                }
            }

        else:  # phase3
            lr_fusion = opt_cfg.get("lr_fusion", 3e-4)
            lr_audio_enc = opt_cfg.get("lr_audio_enc", 3e-5)
            lr_dinov2 = opt_cfg.get("lr_dinov2", 1e-5)

            dinov2_params = [p for p in self.dinov2.parameters() if p.requires_grad]
            audio_enc_params = [p for p in self.audio_unet.encoder.blocks[-2:].parameters() if p.requires_grad]

            param_groups = [
                {"params": list(self.cross_attn.parameters()) +
                        list(self.visual_proj.parameters()) +
                        list(self.bottleneck_proj.parameters()) +
                        [self.source_queries], "lr": lr_fusion},
                {"params": self.audio_unet.decoder.parameters(), "lr": lr_fusion},
                {"params": audio_enc_params, "lr": lr_audio_enc},
                {"params": dinov2_params, "lr": lr_dinov2},
            ]

            optimizer = torch.optim.AdamW(
                param_groups,
                weight_decay=opt_cfg.get("weight_decay", 1e-4),
                betas=opt_cfg.get("betas", [0.9, 0.999])
            )

            # Capture base LRs BEFORE LambdaLR constructor calls step()
            base_lrs = [pg["lr"] for pg in optimizer.param_groups]

            # Linear warmup then cosine (per-param-group)
            def lr_lambda(step):
                if step < warmup_steps:
                    return step / warmup_steps
                progress = (step - warmup_steps) / max(max_steps - warmup_steps, 1)
                return 0.5 * (1 + math.cos(math.pi * progress))

            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
            # LambdaLR calls step() in constructor. Restore base lrs per group.
            for pg, base_lr in zip(optimizer.param_groups, base_lrs):
                pg["lr"] = base_lr
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "step",
                }
            }
