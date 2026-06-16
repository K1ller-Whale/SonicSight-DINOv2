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
        self.audio_unet = AudioUNet()
        # Visual projection: D_v=768 -> D_a=512, no bias (SPEC 11.3)
        self.visual_proj = nn.Linear(768, 512, bias=False)
        # Bottleneck projection: Conv1x1 equivalent (512 -> 512)
        self.bottleneck_proj = nn.Linear(512, 512)
        # Temporal alignment: video frame index mapping (see dataset)
        self.cross_attn = CrossModalAttentionModule()
        
        # CORRECTED TEMPORAL ALIGNMENT
        w_to_frame = [int(w * 150 / 19) for w in range(19)]
        bottleneck_frame_idx = torch.tensor([w_to_frame[i % 19] for i in range(171)], dtype=torch.long)
        self.register_buffer("bottleneck_frame_idx", bottleneck_frame_idx)
        
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
        self._apply_dinov2_freeze(cfg)
        self._apply_phase_trainability()

    @staticmethod
    def _set_trainable(module: nn.Module, trainable: bool) -> None:
        for p in module.parameters():
            p.requires_grad_(trainable)

    def _apply_phase_trainability(self) -> None:
        """Enable gradients only for parameters used by the active phase."""
        if self.phase == "phase1":
            self._set_trainable(self.audio_unet, True)
            self._set_trainable(self.bottleneck_proj, False)
            self._set_trainable(self.visual_proj, False)
            self._set_trainable(self.cross_attn, False)
            self.source_queries.requires_grad_(False)
        elif self.phase == "phase2":
            self._set_trainable(self.audio_unet, False)
            self._set_trainable(self.bottleneck_proj, True)
            self._set_trainable(self.visual_proj, True)
            self._set_trainable(self.cross_attn, True)
            self.source_queries.requires_grad_(True)
        else:
            self._set_trainable(self.audio_unet, False)
            self._set_trainable(self.audio_unet.decoder, True)
            for block in self.audio_unet.encoder.blocks[-2:]:
                self._set_trainable(block, True)
            self._set_trainable(self.bottleneck_proj, True)
            self._set_trainable(self.visual_proj, True)
            self._set_trainable(self.cross_attn, True)
            self.source_queries.requires_grad_(True)

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
        self._cached_decoder_input = None
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

        # Clear cache at start of forward
        self._clear_cache()

        # 5. Cross-modal attention
        have_visual = (visual_features is not None) and self.phase != "phase1"
        bottleneck_flat = None

        if have_visual:
            # 2. Flatten bottleneck: [B, 512, 9, 19] -> [B, 171, 512]
            bottleneck_flat = rearrange(bottleneck, "B C H W -> B (H W) C")

            # 3. Bottleneck projection (Conv1x1 equivalent)
            bottleneck_flat = self.bottleneck_proj(bottleneck_flat)  # [B, 171, 512]
            decoder_inputs = []
            self._cached_attn_weights = []

            def attn_hook(module, input, output):
                if isinstance(output, tuple) and len(output) == 2 and output[1] is not None:
                    self._cached_attn_weights.append(output[1].detach())

            for n in range(self.n_sources):
                vkv_n = self.visual_proj(visual_features[:, n])  # [B, 150, 1024, 512]
                vkv_n_aligned = vkv_n[:, self.bottleneck_frame_idx]  # [B, 171, 1024, 512]

                # Per-source, per-position local attention
                q = rearrange(bottleneck_flat, "B P D -> (B P) 1 D")
                kv = rearrange(vkv_n_aligned, "B P K D -> (B P) K D")
                # For localisation, we also need to capture these weights, so need_weights=True
                attended_tuple = self.cross_attn(q, kv, need_weights=True, average_attn_weights=False)
                attended = attended_tuple[0] if isinstance(attended_tuple, tuple) else attended_tuple
                attended = rearrange(attended, "(B P) 1 D -> B P D", B=B)  # [B, 171, 512]

                bottleneck_out_n = rearrange(attended, "B (H W) D -> B D H W", H=9, W=19)
                decoder_inputs.append(bottleneck_out_n)

                # Source query tokens for entropy / temporal signal
                hooks = []
                for block in self.cross_attn.blocks:
                    hooks.append(block.attn.register_forward_hook(attn_hook))

                try:
                    vkv_n_frame = vkv_n.mean(dim=2)  # [B, 150, 512]
                    sq = self.source_queries[n:n+1].unsqueeze(0).expand(B, -1, -1)  # [B, 1, 512]
                    sq_attended_tuple = self.cross_attn(sq, vkv_n_frame, need_weights=True, average_attn_weights=False)
                finally:
                    for hook in hooks:
                        hook.remove()

            decoder_input = torch.stack(decoder_inputs, dim=1)  # [B, N, 512, 9, 19]
        else:
            # Phase 1 or no visual features: broadcast audio bottleneck to all sources
            decoder_input = bottleneck.unsqueeze(1).expand(-1, self.n_sources, -1, -1, -1)

        # Cache intermediates for _predict_masks
        self._cached_bottleneck = bottleneck
        self._cached_bottleneck_flat = bottleneck_flat
        self._cached_skips = skips
        self._cached_decoder_input = decoder_input

        # 6. Per-source decoding
        separated_waveforms = []
        for i in range(self.n_sources):
            src_feat = decoder_input[:, i]  # [B, 512, 9, 19]
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
            si_snr_db = -si_snr_loss
            total_loss = si_snr_loss
            self.log("train/si_snr_loss", si_snr_loss, prog_bar=False, batch_size=B)
            self.log("train/sisnr_db", si_snr_db, prog_bar=True, batch_size=B)

        elif self.phase == "phase2":
            # Phase 2: SI-SNR + entropy
            si_snr_losses = []
            for b in range(B):
                loss, _, _ = self.pit_wrapper(predicted_waveforms[b], target_waveforms[b])
                si_snr_losses.append(loss)
            si_snr_loss = torch.stack(si_snr_losses).mean()
            si_snr_db = -si_snr_loss
            entropy_loss = self._compute_attention_entropy()
            total_loss = si_snr_loss + 0.1 * entropy_loss
            self.log("train/si_snr_loss", si_snr_loss, prog_bar=False, batch_size=B)
            self.log("train/sisnr_db", si_snr_db, prog_bar=True, batch_size=B)
            self.log("train/entropy_loss", entropy_loss, prog_bar=False, batch_size=B)

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
            si_snr_db = -si_snr_loss
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
            self.log("train/si_snr_loss", si_snr_loss, prog_bar=False, batch_size=B)
            self.log("train/sisnr_db", si_snr_db, prog_bar=True, batch_size=B)
            self.log("train/crm_loss", crm_loss, prog_bar=False, batch_size=B)
            self.log("train/stft_loss", stft_loss, prog_bar=False, batch_size=B)
            self.log("train/perceptual_loss", perceptual_loss, prog_bar=False, batch_size=B)

        self.log("train/total_loss", total_loss, prog_bar=False, batch_size=B)
        self.log("train/loss", total_loss, prog_bar=True, batch_size=B)
        return total_loss

    def _predict_masks(self, mixture_stft: torch.Tensor) -> torch.Tensor:
        """Predict per-source masks for cRM loss using cached forward intermediates."""
        B = mixture_stft.shape[0]
        target_shape = mixture_stft.shape[-2:]

        # Use cached forward pass intermediates
        if (self._cached_bottleneck is None
            or self._cached_bottleneck.shape[0] != B
            or self._cached_skips is None
            or not hasattr(self, '_cached_decoder_input')
            or self._cached_decoder_input is None):
            # Fallback: recompute if cache invalid (should not happen in normal training)
            bottleneck, skips = self.audio_unet.encoder(mixture_stft)
            decoder_input = bottleneck.unsqueeze(1).expand(-1, self.n_sources, -1, -1, -1)
        else:
            skips = self._cached_skips
            decoder_input = self._cached_decoder_input

        # Decode per-source masks using cached decoder_input
        masks = []
        for i in range(self.n_sources):
            src_feat = decoder_input[:, i]  # [B, 512, 9, 19]
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

        # Log as positive SI-SNR dB; val/sisnri is kept for checkpoint compatibility.
        si_snr_db = -si_snr_loss
        self.log("val/sisnri", si_snr_db, prog_bar=False, batch_size=B, sync_dist=True)
        self.log("val/sisnr_db", si_snr_db, prog_bar=True, batch_size=B, sync_dist=True)
        return si_snr_db

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
            trainable_params = [p for p in self.parameters() if p.requires_grad]
            optimizer = torch.optim.AdamW(
                trainable_params,
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
