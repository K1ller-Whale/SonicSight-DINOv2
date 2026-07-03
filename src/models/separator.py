"""PyTorch Lightning module for the DINOv2-guided separator.

SPEC 11.3, 11.4: Three-phase training with differential learning rates.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from einops import rearrange
from src.audio.unet import AudioUNet
from src.audio.spectrogram import ISTFTModule
from src.visual.dinov2 import DINOv2FeatureExtractor
from src.fusion.cross_attention import CrossModalAttentionModule
from src.loss.separation import SISNRLoss, CRMLoss, MultiScaleSTFTLoss, PerceptualLoss
from src.loss.pit_wrapper import PITLossWrapper
from typing import Optional, Dict, Any


class JointMaskHead(nn.Module):
    """Joint, source-competitive bottleneck mask head (softmax over sources)."""

    def __init__(self, bottleneck_channels: int, n_sources: int):
        super().__init__()
        self.bottleneck_channels = bottleneck_channels
        self.n_sources = n_sources
        self.head = nn.Sequential(
            nn.Conv2d(bottleneck_channels, bottleneck_channels, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(
                bottleneck_channels,
                bottleneck_channels * n_sources,
                kernel_size=1,
            ),
        )
        final_conv = self.head[-1]
        nn.init.zeros_(final_conv.weight)
        nn.init.zeros_(final_conv.bias)

    def forward(self, bottleneck: torch.Tensor) -> torch.Tensor:
        B, C, H, W = bottleneck.shape
        masks = self.head(bottleneck)
        masks = masks.view(B, self.n_sources, C, H, W)
        return F.softmax(masks, dim=1)


class SeparatorModule(pl.LightningModule):
    """DINOv2-guided audio-visual source separator (3-phase training)."""

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
        self.joint_mask_head = JointMaskHead(
            bottleneck_channels=512,
            n_sources=self.n_sources,
        )
        # Visual projection: D_v=768 -> D_a=512, no bias (SPEC 11.3)
        self.visual_proj = nn.Linear(768, 512, bias=False)
        # Bottleneck projection: Conv1x1 equivalent (512 -> 512)
        self.bottleneck_proj = nn.Linear(512, 512)
        self.cross_attn = CrossModalAttentionModule()

        # CORRECTED TEMPORAL ALIGNMENT
        # NOTE: kept as a registered buffer for checkpoint compatibility, but the
        # alignment is now computed dynamically in forward() via
        # _frame_alignment_idx() so it adapts to the number of cached video
        # frames (19 for the compact DINOv2 cache, 150 for full features).
        w_to_frame = [int(w * 150 / 19) for w in range(19)]
        bottleneck_frame_idx = torch.tensor(
            [w_to_frame[i % 19] for i in range(171)], dtype=torch.long
        )
        self.register_buffer("bottleneck_frame_idx", bottleneck_frame_idx)
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        self.register_buffer("_imagenet_mean", mean, persistent=False)
        self.register_buffer("_imagenet_std", std, persistent=False)

        # Source query tokens: allocate MAX sources upfront for visual phases.
        self.source_queries = nn.Parameter(torch.randn(max_sources, 512) * 0.02)
        # ReZero-style gate on the visual modulation branch. Warm-started to a
        # small POSITIVE value (not 0) to break the cold-start: the fusion
        # modules (cross_attn / visual_proj) receive gradient proportional to
        # this gate, so at gate==0 they barely learn and the gate never grows
        # (observed: stuck at +0.005 after ~48 epoch-equivalents). A small
        # warm-start is now SAFE because fusion is a bounded multiplicative
        # modulation (gate=1 costs ~1 dB, not the ~-2 dB collapse of the old
        # additive form). Tunable via cfg.model.fusion_gate_init.
        fusion_gate_init = float(cfg.get("model", {}).get("fusion_gate_init", 0.1))
        self.fusion_gate = nn.Parameter(torch.full((1,), fusion_gate_init))

        # --- Cross-modal consistency loss + forced-vision probe (vision-use fix) ---
        loss_cfg = cfg.get("train", {}).get("loss", {})
        # Margin-ranking penalty (dB) requiring each target to be reconstructed
        # better with its MATCHED visual than with a mismatched one. An
        # audio-only / content-independent solution scores matched == mismatched
        # and pays the full margin, so the only way to cut this term is to
        # actually USE the video content to route sources. Off by default
        # (weight 0.0); enable via train.loss.consistency_weight.
        self.consistency_weight = float(loss_cfg.get("consistency_weight", 0.0))
        self.consistency_margin = float(loss_cfg.get("consistency_margin", 2.0))
        # Diagnostic probe: replace the per-source frozen Phase-1 masks with a
        # single source-agnostic base so the per-source VISUAL features are the
        # ONLY signal that can differentiate the outputs. If val still beats the
        # mixture-copy floor, vision CAN separate (earlier failure was an
        # optimization shortcut); if not, the features are uninformative and you
        # need motion/temporal features. Enable via model.force_symmetric_base.
        self.force_symmetric_base = bool(
            cfg.get("model", {}).get("force_symmetric_base", False)
        )

        # Matched-base probe variant (Bug 3 control): the symmetric base above
        # scales every element by a FLAT 1/N, spatially unlike the real masked
        # bottleneck the frozen decoder was trained on, so it may itself pin val
        # at the floor. When BOTH force_symmetric_base and matched_base_probe are
        # on, the base becomes the real, DETACHED Phase-1 per-source masks (in-
        # distribution for the decoder). The detached masks already route the
        # sources, so to isolate VISION run this twice -- once as configured and
        # once with model.fusion_gate_init=0.0 -- and compare val SI-SNR
        # (ON minus OFF = vision contribution). Off by default. Enable via
        # model.matched_base_probe.
        self.matched_base_probe = bool(
            cfg.get("model", {}).get("matched_base_probe", False)
        )

        # --- Motion features (Step B): explicit frame-to-frame token deltas +
        # temporal-window attention. OFF by default (use_motion_features=False)
        # so existing behavior/checkpoints are unaffected. Probe A showed that
        # a SINGLE static frame per audio column -- even at the model's full
        # 19-frame temporal ceiling -- carries no generalizable same-category
        # separation signal (val got WORSE as fusion_gate grew). This adds an
        # explicit motion signal (raw_n[t] - raw_n[t-1], projected separately)
        # and lets each of the 171 audio positions attend over a small
        # TEMPORAL WINDOW of frames (not just its single aligned frame), with a
        # learned per-offset embedding so attention can tell past/current/
        # future apart. Computed on the fly from the existing cache -- no
        # re-precompute needed. Enable via model.use_motion_features (+
        # model.temporal_window, odd int >= 1, default 1 = motion at the
        # aligned frame only, no neighboring frames).
        motion_cfg = cfg.get("model", {})
        self.use_motion_features = bool(motion_cfg.get("use_motion_features", False))
        self.temporal_window = int(motion_cfg.get("temporal_window", 1))
        if self.temporal_window < 1 or self.temporal_window % 2 == 0:
            raise ValueError("model.temporal_window must be a positive odd integer")
        self.motion_proj = nn.Linear(768, 512, bias=False)
        # Same ReZero-with-warm-start reasoning as fusion_gate: at gate==0 the
        # motion pathway gets no gradient signal and never grows.
        motion_gate_init = float(motion_cfg.get("motion_gate_init", 0.1))
        self.motion_gate = nn.Parameter(torch.full((1,), motion_gate_init))
        self.temporal_pos_enc = nn.Parameter(
            torch.randn(self.temporal_window, 512) * 0.02
        )
        self._window_idx_cache: Dict[Any, torch.Tensor] = {}

        self.dinov2_batch_size = int(
            cfg.get("train", {}).get(
                "dinov2_batch_size",
                cfg.get("model", {}).get("visual", {}).get("dinov2_batch_size", 4),
            )
        )
        # iSTFT for waveform reconstruction
        data_cfg = cfg.get("data", {})
        sr = data_cfg.get("sample_rate", 16000)
        dur = data_cfg.get("clip_duration", 6)
        self.istft = ISTFTModule(length=sr * dur)

        # Cache for _predict_masks to avoid re-encoding
        self._cached_bottleneck = None
        self._cached_bottleneck_flat = None
        self._cached_skips = None
        self._cached_decoder_input = None
        self._cached_attn_weights = None

        # Lazily-built map {num_frames: position->frame index tensor}, used to
        # align the 171 bottleneck positions to whatever number of video frames
        # is available (19 for the compact cache, 150 for the full cache).
        self._frame_idx_cache: Dict[int, torch.Tensor] = {}

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
        """Enable gradients only for the parameters used by the active phase."""
        if self.phase == "phase1":
            self._set_trainable(self.audio_unet, True)
            self._set_trainable(self.joint_mask_head, True)
            self._set_trainable(self.bottleneck_proj, False)
            self._set_trainable(self.visual_proj, False)
            self._set_trainable(self.cross_attn, False)
            self.source_queries.requires_grad_(False)
            self.fusion_gate.requires_grad_(False)
            self._set_trainable(self.motion_proj, False)
            self.motion_gate.requires_grad_(False)
            self.temporal_pos_enc.requires_grad_(False)
        elif self.phase == "phase2":
            self._set_trainable(self.audio_unet, False)
            self._set_trainable(self.joint_mask_head, False)
            self._set_trainable(self.bottleneck_proj, True)
            self._set_trainable(self.visual_proj, True)
            self._set_trainable(self.cross_attn, True)
            self.source_queries.requires_grad_(True)
            self.fusion_gate.requires_grad_(True)
            self._set_trainable(self.motion_proj, True)
            self.motion_gate.requires_grad_(True)
            self.temporal_pos_enc.requires_grad_(True)
        else:
            self._set_trainable(self.audio_unet, False)
            self._set_trainable(self.joint_mask_head, False)
            self._set_trainable(self.audio_unet.decoder, True)
            for block in self.audio_unet.encoder.blocks[-2:]:
                self._set_trainable(block, True)
            self._set_trainable(self.bottleneck_proj, True)
            self._set_trainable(self.visual_proj, True)
            self._set_trainable(self.cross_attn, True)
            self.source_queries.requires_grad_(True)
            self.fusion_gate.requires_grad_(True)
            self._set_trainable(self.motion_proj, True)
            self.motion_gate.requires_grad_(True)
            self.temporal_pos_enc.requires_grad_(True)

    def _apply_dinov2_freeze(self, cfg: Dict[str, Any]):
        """Apply DINOv2 freezing based on config (phase3 may unfreeze top blocks)."""
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
            for p in self.dinov2.parameters():
                p.requires_grad = False
            # Unfreeze last N transformer blocks
            if hasattr(self.dinov2, "model") and hasattr(self.dinov2.model, "encoder"):
                blocks = self.dinov2.model.encoder.layer
                for block in blocks[-unfrozen_blocks:]:
                    for p in block.parameters():
                        p.requires_grad = True
            elif hasattr(self.dinov2, "transformer") and hasattr(
                self.dinov2.transformer, "blocks"
            ):
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
        """Increase n_sources at progressive thresholds (phase3 curriculum)."""
        if self.phase != "phase3":
            return False
        step = self.global_step
        new_n_sources = self._progressive_sources[0]
        for threshold, n_src in zip(
            self._progressive_steps, self._progressive_sources[1:]
        ):
            if step >= threshold:
                new_n_sources = n_src
            else:
                break
        if new_n_sources != self.n_sources:
            self.n_sources = new_n_sources
            return True
        return False

    def phase1_forward(self, mixture_spec: torch.Tensor) -> torch.Tensor:
        """Audio-only competitive separation path for Phase 1."""
        target_shape = mixture_spec.shape[-2:]

        bottleneck, skips = self.audio_unet.encoder(mixture_spec)
        self._clear_cache()

        masks = self.joint_mask_head(bottleneck)
        decoder_inputs = []
        source_outputs = []

        for i in range(self.n_sources):
            masked_features = bottleneck * masks[:, i]
            decoder_inputs.append(masked_features)

            mask = self.audio_unet.decoder(
                masked_features,
                skips,
                target_shape=target_shape,
            )
            mask = torch.tanh(mask)
            source_outputs.append(self.istft(mask, mixture_spec))

        decoder_input = torch.stack(decoder_inputs, dim=1)
        self._cached_bottleneck = bottleneck
        self._cached_bottleneck_flat = None
        self._cached_skips = skips
        self._cached_decoder_input = decoder_input
        self._cached_attn_weights = None

        return torch.stack(source_outputs, dim=1)

    def _normalize_video_frames(self, frames: torch.Tensor) -> torch.Tensor:
        """Convert raw RGB frames to ImageNet-normalized DINOv2 input."""
        frames = frames.to(device=self._imagenet_mean.device)
        if not frames.is_floating_point():
            frames = frames.float().div(255.0)
        else:
            frames = frames.float()
            if frames.detach().amax() > 2.0:
                frames = frames.div(255.0)
        if frames.detach().amin() >= 0.0:
            frames = (frames - self._imagenet_mean) / self._imagenet_std
        return frames

    def _video_frames_to_visual_features(
        self, video_frames: torch.Tensor
    ) -> torch.Tensor:
        """Run DINOv2 over raw frames -> [B, N, T, 1024, 768] (in-loop path)."""
        if video_frames.dim() == 5:
            video_frames = video_frames.unsqueeze(1).expand(
                -1, self.n_sources, -1, -1, -1, -1
            )
        if video_frames.dim() != 6:
            raise ValueError(
                "video_frames must be [B,N,T,3,H,W] or [B,T,3,H,W], "
                f"got {tuple(video_frames.shape)}"
            )

        B, N, T, C, H, W = video_frames.shape
        if C != 3:
            raise ValueError(f"video_frames channel dimension must be 3, got {C}")

        grad_enabled = any(p.requires_grad for p in self.dinov2.parameters())
        feature_sources = []
        for n in range(N):
            flat_frames = rearrange(video_frames[:, n], "B T C H W -> (B T) C H W")
            chunks = []
            context = torch.enable_grad() if grad_enabled else torch.no_grad()
            with context:
                for chunk in flat_frames.split(max(1, self.dinov2_batch_size)):
                    chunk = self._normalize_video_frames(chunk)
                    chunks.append(self.dinov2(chunk))
            source_features = torch.cat(chunks, dim=0)
            source_features = rearrange(
                source_features,
                "(B T) P D -> B T P D",
                B=B,
                T=T,
            )
            feature_sources.append(source_features)
        return torch.stack(feature_sources, dim=1)

    def _frame_alignment_idx(self, num_frames: int, device) -> torch.Tensor:
        """Map each of the 171 bottleneck positions to a video-frame index.

        The audio bottleneck is [B, 512, 9, 19], flattened to 171 = 9*19
        positions where column ``w = position % 19``. Each column aligns to one
        video frame: ``frame = int(w * num_frames / 19)``. This works for the
        full 150-frame features AND for the compact 19-frame cache (where it
        reduces to the identity mapping ``w -> w``), so no fixed buffer is
        needed and indices can never overflow the available frames.
        """
        cached = self._frame_idx_cache.get(num_frames)
        if cached is not None:
            return cached.to(device)
        w_to_frame = [min(int(w * num_frames / 19), num_frames - 1) for w in range(19)]
        idx = torch.tensor(
            [w_to_frame[i % 19] for i in range(171)],
            dtype=torch.long,
            device=device,
        )
        self._frame_idx_cache[num_frames] = idx
        return idx

    def _temporal_window_idx(
        self, frame_idx: torch.Tensor, num_frames: int, window: int, device
    ) -> torch.Tensor:
        """For each of the 171 aligned positions, return `window` frame
        indices centered on its single aligned frame (from
        _frame_alignment_idx), clamped to [0, num_frames-1].

        window=1 reduces to frame_idx.unsqueeze(-1) (motion at the aligned
        frame only, no neighboring frames). Cached per (num_frames, window).
        """
        cache_key = (num_frames, window)
        cached = self._window_idx_cache.get(cache_key)
        if cached is not None:
            return cached.to(device)
        half = window // 2
        offsets = torch.arange(-half, half + 1, device=frame_idx.device)
        idx = (frame_idx.unsqueeze(-1) + offsets.unsqueeze(0)).clamp(0, num_frames - 1)
        self._window_idx_cache[cache_key] = idx
        return idx.to(device)

    def forward(
        self,
        mixture_stft: torch.Tensor,
        video_frames: Optional[torch.Tensor] = None,
        visual_features: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Separate sources from a mixture, guided by per-source visual features."""
        if self.phase == "phase1":
            return self.phase1_forward(mixture_stft)

        if visual_features is None:
            if video_frames is None:
                raise RuntimeError(
                    "Phase 2/3 requires visual_features or video_frames. "
                    "Check that the dataset cache was created with visuals."
                )
            visual_features = self._video_frames_to_visual_features(video_frames)

        if visual_features.dim() == 4:
            visual_features = visual_features.unsqueeze(1).expand(
                -1, self.n_sources, -1, -1, -1
            )
        if visual_features.dim() != 5:
            raise ValueError(
                "visual_features must be [B,N,T,1024,768] or [B,T,1024,768], "
                f"got {tuple(visual_features.shape)}"
            )
        if visual_features.shape[1] < self.n_sources:
            raise ValueError(
                f"Need at least {self.n_sources} visual sources, "
                f"got {visual_features.shape[1]}"
            )

        B = mixture_stft.shape[0]
        target_shape = mixture_stft.shape[-2:]  # [F, T]

        # 1. Audio U-Net encoder -> bottleneck [B, 512, 9, 19]
        bottleneck, skips = self.audio_unet.encoder(mixture_stft)

        # Clear cache at start of forward
        self._clear_cache()

        # 2. Flatten bottleneck: [B, 512, 9, 19] -> [B, 171, 512]
        bottleneck_flat = rearrange(bottleneck, "B C H W -> B (H W) C")

        # 3. Bottleneck projection (Conv1x1 equivalent)
        bottleneck_flat = self.bottleneck_proj(bottleneck_flat)  # [B, 171, 512]
        # Frozen, trained source-competitive masks (the Phase-1 path).
        masks = self.joint_mask_head(bottleneck)  # [B, N, 512, 9, 19]
        decoder_inputs = []
        self._cached_attn_weights = []

        def attn_hook(module, input, output):
            if isinstance(output, tuple) and len(output) == 2 and output[1] is not None:
                self._cached_attn_weights.append(output[1].detach())

        for n in range(self.n_sources):
            vkv_n = self.visual_proj(visual_features[:, n])  # [B, V, P, 512]
            # Align each of the 171 bottleneck positions to a cached video frame.
            # Computed dynamically so it works for the full 150-frame features
            # and the compact 19-frame cache (no index overflow either way).
            frame_idx = self._frame_alignment_idx(vkv_n.shape[1], vkv_n.device)

            if self.use_motion_features:
                # STEP B: explicit motion (frame-to-frame token deltas) +
                # temporal-window attention -- see __init__ comment for why.
                raw_n = visual_features[:, n]  # [B, V, P, 768]
                delta_n = raw_n - torch.cat([raw_n[:, :1], raw_n[:, :-1]], dim=1)
                vkv_motion_n = self.motion_proj(delta_n)  # [B, V, P, 512]
                window_idx = self._temporal_window_idx(
                    frame_idx, vkv_n.shape[1], self.temporal_window, vkv_n.device
                )  # [171, W]
                app_window = vkv_n[:, window_idx]  # [B, 171, W, P, 512]
                mot_window = vkv_motion_n[:, window_idx]  # [B, 171, W, P, 512]
                # Learned per-offset embedding so attention can tell WHICH
                # frame in the window a token came from.
                temporal_pe = self.temporal_pos_enc.view(
                    1, 1, self.temporal_window, 1, 512
                )
                combined = (app_window + temporal_pe) + self.motion_gate * mot_window
                kv = rearrange(combined, "B Pos W K D -> (B Pos) (W K) D")
            else:
                vkv_n_aligned = vkv_n[:, frame_idx]  # [B, 171, P, 512]
                kv = rearrange(vkv_n_aligned, "B P K D -> (B P) K D")

            # Per-source, per-position local attention
            q = rearrange(bottleneck_flat, "B P D -> (B P) 1 D")
            # need_weights=True so attention can be inspected for localisation.
            attended_tuple = self.cross_attn(
                q, kv, need_weights=True, average_attn_weights=False
            )
            attended = (
                attended_tuple[0]
                if isinstance(attended_tuple, tuple)
                else attended_tuple
            )
            attended = rearrange(attended, "(B P) 1 D -> B P D", B=B)  # [B, 171, 512]

            bottleneck_out_n = rearrange(attended, "B (H W) D -> B D H W", H=9, W=19)
            if self.force_symmetric_base:
                if self.matched_base_probe:
                    # CONTROL (matched base): in-distribution base = real Phase-1
                    # masks, DETACHED so no gradient flows through audio routing.
                    # Run with fusion_gate_init=0.0 (vision off) vs default
                    # (vision on) and compare val SI-SNR to isolate vision.
                    base_n = bottleneck * masks[:, n].detach()
                else:
                    # PROBE: source-agnostic flat base (1/N) so per-source visual
                    # modulation is the ONLY differentiator. Scaled by 1/N to
                    # match the masked-bottleneck MAGNITUDE the frozen decoder
                    # expects (softmax masks sum to 1 over sources; mean = 1/N).
                    base_n = bottleneck / self.n_sources
            else:
                base_n = bottleneck * masks[:, n]  # Phase-1 masked-bottleneck base
            # Visual fusion as a BOUNDED MULTIPLICATIVE modulation of the masked
            # bottleneck. The Phase-2 decoder is FROZEN, so it can only interpret
            # masked-bottleneck-distributed input. Additive injection of the raw
            # attention output is out-of-distribution for the frozen decoder
            # (gate=1 collapses to ~-2 dB), which forces the ReZero gate to stay
            # ~0 and blocks all learning. tanh bounds the per-element correction
            # to [-1, 1] so the decoder input stays in-distribution; at
            # fusion_gate=0 this is exactly the Phase-1 path (base_n).
            visual_mod = torch.tanh(bottleneck_out_n)
            decoder_inputs.append(base_n * (1.0 + self.fusion_gate * visual_mod))

            # Source query tokens for entropy / temporal signal.
            # vkv_n.mean(dim=2) averages over the 1024 patches and works for any
            # number of cached frames V (19 or 150).
            hooks = []
            for block in self.cross_attn.blocks:
                hooks.append(block.attn.register_forward_hook(attn_hook))

            try:
                vkv_n_frame = vkv_n.mean(dim=2)  # [B, V, 512]
                sq = (
                    self.source_queries[n : n + 1].unsqueeze(0).expand(B, -1, -1)
                )  # [B, 1, 512]
                _ = self.cross_attn(
                    sq, vkv_n_frame, need_weights=True, average_attn_weights=False
                )
            finally:
                for hook in hooks:
                    hook.remove()

        decoder_input = torch.stack(decoder_inputs, dim=1)  # [B, N, 512, 9, 19]

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
            separated = self.istft(mask, mixture_stft)  # [B, L]
            separated_waveforms.append(separated)

        output = torch.stack(separated_waveforms, dim=1)  # [B, N_sources, L]
        return output

    def training_step(
        self, batch: Dict[str, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
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
            si_snr_losses = []
            for b in range(B):
                loss, _, _ = self.pit_wrapper(
                    predicted_waveforms[b], target_waveforms[b]
                )
                si_snr_losses.append(loss)
            si_snr_loss = torch.stack(si_snr_losses).mean()
            si_snr_db = -si_snr_loss
            total_loss = si_snr_loss
            self.log(
                "train/si_snr_loss",
                si_snr_loss,
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                batch_size=B,
            )
            self.log(
                "train/loss_value",
                si_snr_loss,
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                batch_size=B,
            )
            self.log(
                "train/si_snr_raw_db",
                si_snr_db,
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                batch_size=B,
            )
            self.log(
                "train/sisnr_db",
                si_snr_db,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                batch_size=B,
            )

        elif self.phase == "phase2":
            # Visual condition defines source identity -> fixed-order loss, NOT PIT.
            # PIT would relabel outputs by audio similarity and discard the
            # visual-to-source supervision, letting the model ignore the video.
            si_snr_loss = self._fixed_order_si_snr(
                predicted_waveforms, target_waveforms
            )
            si_snr_db = -si_snr_loss
            entropy_loss = self._compute_attention_entropy()
            entropy_weight = (
                self.cfg.get("train", {})
                .get("loss", {})
                .get("attention_entropy_weight", 0.1)
            )
            total_loss = si_snr_loss + entropy_weight * entropy_loss
            if self.consistency_weight > 0:
                consistency_loss = self._cross_modal_consistency_loss(
                    mixture_stft,
                    target_waveforms,
                    si_snr_loss,
                    visual_features=visual_features,
                    video_frames=video_frames,
                )
                total_loss = total_loss + self.consistency_weight * consistency_loss
                self.log(
                    "train/consistency_loss",
                    consistency_loss,
                    on_step=False,
                    on_epoch=True,
                    prog_bar=True,
                    batch_size=B,
                )
            self.log(
                "train/si_snr_loss",
                si_snr_loss,
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                batch_size=B,
            )
            self.log(
                "train/loss_value",
                si_snr_loss,
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                batch_size=B,
            )
            self.log(
                "train/si_snr_raw_db",
                si_snr_db,
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                batch_size=B,
            )
            self.log(
                "train/sisnr_db",
                si_snr_db,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                batch_size=B,
            )
            self.log(
                "train/entropy_loss",
                entropy_loss,
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                batch_size=B,
            )
            self.log(
                "train/fusion_gate",
                self.fusion_gate.detach().float().mean(),
                on_step=True,
                on_epoch=False,
                prog_bar=True,
                batch_size=B,
            )

        else:  # phase3
            target_masks = batch.get("target_crm_masks")
            pred_masks = self._predict_masks(mixture_stft)

            alpha = self.cfg.get("train", {}).get("loss", {}).get("alpha_crm", 0.1)
            beta = self.cfg.get("train", {}).get("loss", {}).get("beta_stft", 0.05)
            gamma = (
                self.cfg.get("train", {}).get("loss", {}).get("gamma_perceptual", 0.1)
            )

            # Visual condition defines source identity -> fixed-order loss, NOT PIT.
            si_snr_loss = self._fixed_order_si_snr(
                predicted_waveforms, target_waveforms
            )
            si_snr_db = -si_snr_loss
            crm_loss = torch.stack(
                [self.crm(pred_masks[b], target_masks[b]) for b in range(B)]
            ).mean()

            # Outputs are already in target order (no permutation to apply).
            stft_loss = self.stft(
                predicted_waveforms.reshape(-1, predicted_waveforms.shape[-1]),
                target_waveforms.reshape(-1, target_waveforms.shape[-1]),
            )
            perceptual_loss = self.perceptual(predicted_waveforms, target_waveforms)

            total_loss = (
                si_snr_loss
                + alpha * crm_loss
                + beta * stft_loss
                + gamma * perceptual_loss
            )
            if self.consistency_weight > 0:
                consistency_loss = self._cross_modal_consistency_loss(
                    mixture_stft,
                    target_waveforms,
                    si_snr_loss,
                    visual_features=visual_features,
                    video_frames=video_frames,
                )
                total_loss = total_loss + self.consistency_weight * consistency_loss
                self.log(
                    "train/consistency_loss",
                    consistency_loss,
                    on_step=False,
                    on_epoch=True,
                    prog_bar=False,
                    batch_size=B,
                )
            self.log(
                "train/si_snr_loss",
                si_snr_loss,
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                batch_size=B,
            )
            self.log(
                "train/loss_value",
                si_snr_loss,
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                batch_size=B,
            )
            self.log(
                "train/si_snr_raw_db",
                si_snr_db,
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                batch_size=B,
            )
            self.log(
                "train/sisnr_db",
                si_snr_db,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                batch_size=B,
            )
            self.log(
                "train/crm_loss",
                crm_loss,
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                batch_size=B,
            )
            self.log(
                "train/stft_loss",
                stft_loss,
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                batch_size=B,
            )
            self.log(
                "train/perceptual_loss",
                perceptual_loss,
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                batch_size=B,
            )

        self.log(
            "train/total_loss",
            total_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            batch_size=B,
        )
        self.log(
            "train/loss",
            total_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            batch_size=B,
        )
        return total_loss

    def _predict_masks(self, mixture_stft: torch.Tensor) -> torch.Tensor:
        """Predict per-source masks for cRM loss using cached forward intermediates."""
        B = mixture_stft.shape[0]
        target_shape = mixture_stft.shape[-2:]

        if (
            self._cached_bottleneck is None
            or self._cached_bottleneck.shape[0] != B
            or self._cached_skips is None
            or not hasattr(self, "_cached_decoder_input")
            or self._cached_decoder_input is None
        ):
            # Fallback: recompute if cache invalid (should not happen normally)
            bottleneck, skips = self.audio_unet.encoder(mixture_stft)
            masks = self.joint_mask_head(bottleneck)
            decoder_input = torch.stack(
                [bottleneck * masks[:, i] for i in range(self.n_sources)],
                dim=1,
            )
        else:
            skips = self._cached_skips
            decoder_input = self._cached_decoder_input

        masks = []
        for i in range(self.n_sources):
            src_feat = decoder_input[:, i]  # [B, 512, 9, 19]
            mask = self.audio_unet.decoder(src_feat, skips, target_shape=target_shape)
            mask = torch.tanh(mask)
            masks.append(mask)
        return torch.stack(masks, dim=1)

    def _fixed_order_si_snr(
        self, predicted_waveforms: torch.Tensor, target_waveforms: torch.Tensor
    ) -> torch.Tensor:
        """Fixed-order (non-PIT) SI-SNR loss for visually-conditioned phases.

        In Phase 2/3 the visual condition for source n defines which target
        must be reconstructed: output[n] is conditioned on visual_features[n],
        which the dataset aligns with target_waveforms[n]. Applying PIT here
        would re-match outputs to targets by audio similarity alone and discard
        the visual-to-source supervision, letting the model ignore the video.
        We therefore supervise output[n] against target[n] directly.
        """
        B, N, _ = predicted_waveforms.shape
        losses = []
        for b in range(B):
            sample_loss = -self.si_snr._si_snr(
                predicted_waveforms[b, 0], target_waveforms[b, 0]
            )
            for n in range(1, N):
                sample_loss = sample_loss - self.si_snr._si_snr(
                    predicted_waveforms[b, n], target_waveforms[b, n]
                )
            losses.append(sample_loss / N)
        return torch.stack(losses).mean()

    def _cross_modal_consistency_loss(
        self,
        mixture_stft: torch.Tensor,
        target_waveforms: torch.Tensor,
        matched_si_snr_loss: torch.Tensor,
        visual_features: Optional[torch.Tensor] = None,
        video_frames: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Margin-ranking penalty forcing outputs to depend on visual CONTENT.

        Re-runs the separator with the per-source visuals shuffled (mismatched)
        and requires the MATCHED visuals to reconstruct the fixed-order targets
        better than the mismatched ones by `consistency_margin` dB:

            relu(margin - (sisnr_db_matched - sisnr_db_mismatched))

        A model that ignores vision scores matched == mismatched and pays the
        full margin; the only way to minimize the term is to use the video
        content to route each source. `matched_si_snr_loss` is the fixed-order
        SI-SNR loss already computed for the matched pass (dB = -loss).
        """
        if visual_features is None:
            if video_frames is None:
                return mixture_stft.new_zeros(())
            visual_features = self._video_frames_to_visual_features(video_frames)
        if visual_features.dim() == 4:
            visual_features = visual_features.unsqueeze(1).expand(
                -1, self.n_sources, -1, -1, -1
            )
        # roll-by-1 along the source dim guarantees no source keeps its own
        # visual (a valid mismatch for any n_sources >= 2).
        mismatched_visual = torch.roll(visual_features, shifts=1, dims=1)
        mismatched_preds = self(mixture_stft, visual_features=mismatched_visual)
        mismatched_si_snr_loss = self._fixed_order_si_snr(
            mismatched_preds, target_waveforms
        )
        matched_db = -matched_si_snr_loss
        mismatched_db = -mismatched_si_snr_loss
        return torch.relu(self.consistency_margin - (matched_db - mismatched_db))

    def _compute_attention_entropy(self) -> torch.Tensor:
        """Mean entropy of cached cross-attention weights (sparsity regularizer)."""
        if (
            not hasattr(self, "_cached_attn_weights")
            or self._cached_attn_weights is None
        ):
            return torch.tensor(0.0, device=self.device)

        total_entropy = 0.0
        count = 0
        for attn_weights in self._cached_attn_weights:
            # attn_weights: [B, n_heads, T_q, T_kv]
            p = attn_weights + 1e-8  # avoid log(0)
            p = p / p.sum(dim=-1, keepdim=True)  # normalize
            entropy = -(p * p.log()).sum(dim=-1)  # [B, n_heads, T_q]
            total_entropy += entropy.mean()
            count += 1

        return (
            total_entropy / count
            if count > 0
            else torch.tensor(0.0, device=self.device)
        )

    def validation_step(
        self, batch: Dict[str, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
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

        # Match training supervision: PIT only in phase1, fixed-order otherwise.
        if self.phase == "phase1":
            losses = []
            for b in range(B):
                loss, _, _ = self.pit_wrapper(
                    predicted_waveforms[b], target_waveforms[b]
                )
                losses.append(loss)
            si_snr_loss = torch.stack(losses).mean()
        else:
            si_snr_loss = self._fixed_order_si_snr(
                predicted_waveforms, target_waveforms
            )

        # Log as positive SI-SNR dB; val/sisnri kept for checkpoint compatibility.
        si_snr_db = -si_snr_loss
        self.log(
            "val/loss_value", si_snr_loss, prog_bar=False, batch_size=B, sync_dist=True
        )
        self.log(
            "val/si_snr_raw_db", si_snr_db, prog_bar=False, batch_size=B, sync_dist=True
        )
        self.log("val/sisnri", si_snr_db, prog_bar=False, batch_size=B, sync_dist=True)
        self.log("val/sisnr_db", si_snr_db, prog_bar=True, batch_size=B, sync_dist=True)
        return si_snr_db

    def configure_optimizers(self):
        """Phase-specific optimizers with differential learning rates."""
        train_cfg = self.cfg.get("train", {})

        if self.phase == "phase1":
            lr = train_cfg.get("lr", 1e-3)
            params = [p for p in self.parameters() if p.requires_grad]
            optimizer = torch.optim.AdamW(
                params, lr=lr, weight_decay=train_cfg.get("weight_decay", 1e-4)
            )
            max_steps = train_cfg.get("max_steps", 100000)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=max_steps
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
            }

        elif self.phase == "phase2":
            lr_fusion = train_cfg.get("lr_fusion", 5e-4)
            params = [p for p in self.parameters() if p.requires_grad]
            optimizer = torch.optim.AdamW(
                params, lr=lr_fusion, weight_decay=train_cfg.get("weight_decay", 1e-4)
            )
            max_steps = train_cfg.get("max_steps", 50000)
            warmup_steps = train_cfg.get("warmup_steps", 1000)

            def lr_lambda(step):
                if step < warmup_steps:
                    return step / max(1, warmup_steps)
                progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
                return 0.5 * (1.0 + math.cos(math.pi * progress))

            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
            }

        else:  # phase3
            lr_fusion = train_cfg.get("lr_fusion", 3e-4)
            lr_decoder = train_cfg.get("lr_decoder", 3e-4)
            lr_encoder = train_cfg.get("lr_encoder", 3e-5)
            lr_dinov2 = train_cfg.get("lr_dinov2", 1e-5)

            fusion_params = (
                list(self.bottleneck_proj.parameters())
                + list(self.visual_proj.parameters())
                + list(self.cross_attn.parameters())
                + [self.source_queries]
            )
            decoder_params = list(self.audio_unet.decoder.parameters())
            encoder_params = []
            for block in self.audio_unet.encoder.blocks[-2:]:
                encoder_params += list(block.parameters())
            dinov2_params = [p for p in self.dinov2.parameters() if p.requires_grad]

            param_groups = [
                {
                    "params": [p for p in fusion_params if p.requires_grad],
                    "lr": lr_fusion,
                },
                {
                    "params": [p for p in decoder_params if p.requires_grad],
                    "lr": lr_decoder,
                },
                {
                    "params": [p for p in encoder_params if p.requires_grad],
                    "lr": lr_encoder,
                },
            ]
            if dinov2_params:
                param_groups.append({"params": dinov2_params, "lr": lr_dinov2})

            optimizer = torch.optim.AdamW(
                param_groups, weight_decay=train_cfg.get("weight_decay", 1e-4)
            )
            max_steps = train_cfg.get("max_steps", 50000)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=max_steps
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
            }
