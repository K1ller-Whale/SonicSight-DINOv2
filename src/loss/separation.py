"""Loss functions for audio source separation.

SPEC 7.3: SI-SNR (permutation-invariant), cRM, multi-scale STFT, perceptual.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from itertools import permutations
from typing import Optional, Tuple
import math
from scipy.optimize import linear_sum_assignment


class SISNRLoss(nn.Module):
    """Permutation-invariant SI-SNR loss."""

    def __init__(self, zero_mean: bool = True, eps: float = 1e-8):
        super().__init__()
        self.zero_mean = zero_mean
        self.eps = eps

    def _si_snr(self, estimated: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            estimated: [L]
            target: [L]
        Returns:
            SI-SNR in dB (scalar)
        """
        if self.zero_mean:
            estimated = estimated - estimated.mean()
            target = target - target.mean()
        dot = (estimated * target).sum()
        norm_target = (target ** 2).sum()
        s_target = dot / (norm_target + self.eps) * target
        e_noise = estimated - s_target
        si_snr = 10 * torch.log10(
            (s_target ** 2).sum() / ((e_noise ** 2).sum() + self.eps) + self.eps
        )
        return si_snr

    def forward(self, estimated: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Standalone PIT path for direct use without PITLossWrapper.
        NOTE: PITLossWrapper calls compute_pairwise_losses() and
        _si_snr() directly and does NOT call this method.
        This is an alternative standalone path kept for reference.
        Args:
            estimated: [N_sources, L]
            target: [N_sources, L]
        Returns:
            PIT loss: minimum SI-SNR loss over all permutations
        """
        N = estimated.shape[0]
        best_loss = float('inf')
        best_perm = None
        for perm in permutations(range(N)):
            perm_est = estimated[list(perm)]
            si_snr_vals = [self._si_snr(perm_est[i], target[i]) for i in range(N)]
            loss = -sum(si_snr_vals) / N  # negative SI-SNR = loss to minimize
            if loss < best_loss:
                best_loss = loss
                best_perm = perm
        return best_loss

    def compute_pairwise_losses(self, estimated: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute pairwise SI-SNR losses for all source pairs.
        Args:
            estimated: [N, L]
            target: [N, L]
        Returns:
            cost_matrix: [N, N] where cost_matrix[i, j] = -SI-SNR(estimated[i], target[j])
        """
        N = estimated.shape[0]
        cost_matrix = torch.zeros(N, N, device=estimated.device)
        for i in range(N):
            for j in range(N):
                si_snr = self._si_snr(estimated[i], target[j])
                cost_matrix[i, j] = -si_snr.item() if si_snr.dim() == 0 else -si_snr  # negative SI-SNR = cost to minimize
        return cost_matrix


class CRMLoss(nn.Module):
    """Complex ratio mask (cRM) L1 loss.

    Per Section 7.3 of the report: L1 loss on compressed complex ratio masks
    is used for robustness on transient frequency-time bins where ideal mask
    values are genuinely hard to predict. MSE quadratically penalises large
    errors and destabilises training on these bins.
    """

    def __init__(self):
        super().__init__()
        self.l1 = nn.L1Loss(reduction="mean")

    def forward(self, pred_mask: torch.Tensor, target_mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred_mask, target_mask: [B, N, 2, F, T]
        Returns:
            scalar loss
        """
        return self.l1(pred_mask, target_mask)

    def compute_pairwise_losses(self, pred_mask: torch.Tensor, target_mask: torch.Tensor) -> torch.Tensor:
        """
        Compute pairwise cRM L1 losses for all source pairs (PIT cost matrix).
        Args:
            pred_mask: [B, N, 2, F, T]
            target_mask: [B, N, 2, F, T]
        Returns:
            cost_matrix: [B, N, N] where cost_matrix[b, i, j] = L1(pred_mask[b,i], target_mask[b,j])
        """
        B, N = pred_mask.shape[0], pred_mask.shape[1]
        cost_matrix = torch.zeros(B, N, N, device=pred_mask.device)
        for b in range(B):
            for i in range(N):
                for j in range(N):
                    cost_matrix[b, i, j] = self.l1(pred_mask[b, i], target_mask[b, j])
        return cost_matrix


class MultiScaleSTFTLoss(nn.Module):
    """Multi-scale STFT loss comparing predicted vs target spectrograms."""

    def __init__(self, n_ffts: list = None, hop_lens: list = None):
        super().__init__()
        if n_ffts is None:
            n_ffts = [512, 1024, 2048]
        if hop_lens is None:
            hop_lens = [128, 256, 512]
        self.n_ffts = n_ffts
        self.hop_lens = hop_lens
        # Pre-compute Hann windows for each scale
        self.register_buffer("_windows", torch.empty(0), persistent=False)
        for n_fft in n_ffts:
            window = torch.hann_window(n_fft)
            self.register_buffer(f"_window_{n_fft}", window, persistent=False)

    def _get_window(self, n_fft: int) -> torch.Tensor:
        return getattr(self, f"_window_{n_fft}")

    def forward(self, pred_wave: torch.Tensor, target_wave: torch.Tensor) -> torch.Tensor:
        """pred_wave, target_wave: [N, L]"""
        total_loss = 0.0
        for n_fft, hop in zip(self.n_ffts, self.hop_lens):
            window = self._get_window(n_fft).to(pred_wave.device)
            pred_spec = torch.stft(pred_wave, n_fft=n_fft, hop_length=hop,
                                   window=window, return_complex=True)
            target_spec = torch.stft(target_wave, n_fft=n_fft, hop_length=hop,
                                     window=window, return_complex=True)
            # L1 on magnitude + L1 on log magnitude
            pred_mag = pred_spec.abs()
            target_mag = target_spec.abs()
            l1 = (pred_mag - target_mag).abs().mean()
            l1_log = (torch.log(pred_mag + 1e-8) - torch.log(target_mag + 1e-8)).abs().mean()
            total_loss += l1 + l1_log
        return total_loss / len(self.n_ffts)


class PerceptualLoss(nn.Module):
    """Perceptual loss: cosine similarity of DINOv2 features on spectrograms."""

    def __init__(self, n_fft: int = 512, hop_length: int = 160, dinov2_extractor=None):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.dinov2 = dinov2_extractor
        if self.dinov2 is not None:
            for p in self.dinov2.parameters():
                p.requires_grad_(False)

    def _waveform_to_dinov2_input(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Convert waveform [B, L] or [B, N, L] to DINOv2 input [B, 3, 448, 448].
        Uses STFT magnitude as grayscale image, repeated to 3 channels.
        """
        if waveform.dim() == 3:
            # [B, N, L] -> flatten batch and sources
            B, N, L = waveform.shape
            waveform = waveform.reshape(B * N, L)
            reshape_back = True
        else:
            reshape_back = False

        # STFT -> [B, 2, F, T] complex
        window = torch.hann_window(self.n_fft, device=waveform.device)
        spec = torch.stft(waveform, n_fft=self.n_fft, hop_length=self.hop_length,
                          window=window, return_complex=True)
        mag = spec.abs()  # [B, F, T]

        # Normalize and resize to 448x448 for DINOv2 (facebook/dinov2-base)
        mag = mag.unsqueeze(1)  # [B, 1, F, T]
        mag = torch.nn.functional.interpolate(mag, size=(448, 448), mode="bilinear", align_corners=False)
        # Repeat to 3 channels
        mag = mag.repeat(1, 3, 1, 1)  # [B, 3, 448, 448]

        # ImageNet normalization
        mean = torch.tensor([0.485, 0.456, 0.406], device=mag.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=mag.device).view(1, 3, 1, 1)
        mag = (mag - mean) / std

        if reshape_back:
            mag = mag.reshape(B, N, 3, 448, 448)

        return mag

    def forward(self, pred_wave: torch.Tensor, target_wave: torch.Tensor,
                dinov2_extractor=None) -> torch.Tensor:
        """
        Args:
            pred_wave: [B, N, L] or [N, L]
            target_wave: [B, N, L] or [N, L]
            dinov2_extractor: Optional DINOv2FeatureExtractor instance (uses self.dinov2 if not provided)
        Returns:
            Cosine similarity loss (1 - cos_sim), mean over batch and sources
        """
        # Use provided extractor or fall back to self.dinov2
        extractor = dinov2_extractor if dinov2_extractor is not None else self.dinov2
        if extractor is None:
            raise ValueError("DINOv2 extractor must be provided either at init or forward call")

        device = pred_wave.device
        # Convert to DINOv2 input images
        pred_img = self._waveform_to_dinov2_input(pred_wave).to(device)
        target_img = self._waveform_to_dinov2_input(target_wave).to(device)

        # Get DINOv2 features [B, 1024, 768] or [B, N, 1024, 768]
        # No torch.no_grad() - gradients flow to pred_wave through frozen DINOv2
        pred_feat = extractor(pred_img.flatten(0, 1)) if pred_img.dim() == 5 else extractor(pred_img)
        target_feat = extractor(target_img.flatten(0, 1)) if target_img.dim() == 5 else extractor(target_img)

        # Ensure features on same device as input
        pred_feat = pred_feat.to(device)
        target_feat = target_feat.to(device)

        # Flatten patch dimension: [B, 1024, 768] -> [B, 1024*768] or pool
        pred_feat = pred_feat.mean(dim=1)  # [B, 768] or [B*N, 768]
        target_feat = target_feat.mean(dim=1)

        # Cosine similarity
        cos_sim = F.cosine_similarity(pred_feat, target_feat, dim=-1)  # [B] or [B*N]
        loss = (1.0 - cos_sim).mean()

        return loss


# PITLossWrapper moved to src/loss/pit_wrapper.py
