"""Loss functions for audio source separation.

SPEC 7.3: SI-SNR (permutation-invariant), cRM, multi-scale STFT, perceptual.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from itertools import permutations
from typing import Optional
import math


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
            estimated = estimated - estimated.mean(dim=-1, keepdim=True)
            target = target - target.mean(dim=-1, keepdim=True)
        dot = (estimated * target).sum(dim=-1, keepdim=True)
        norm_target = (target ** 2).sum(dim=-1, keepdim=True)
        s_target = dot / (norm_target + self.eps) * target
        e_noise = estimated - s_target
        si_snr = 10 * torch.log10(
            (s_target ** 2).sum(dim=-1) / ((e_noise ** 2).sum(dim=-1) + self.eps) + self.eps
        )
        return si_snr

    def forward(self, estimated: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
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


class CRMLoss(nn.Module):
    """Complex ratio mask (cRM) MSE loss."""

    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(self, pred_mask: torch.Tensor, target_mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred_mask, target_mask: [N, 2, F, T]
        """
        return self.mse(pred_mask, target_mask)


class MultiScaleSTFTLoss(nn.Module):
    """Multi-scale STFT loss comparing predicted vs target spectrograms."""

    def __init__(self, n_ffts: list = None, hop_lens: list = None):
        super().__init__()
        if n_ffts is None:
            n_ffts = [256, 512, 1024]
        self.n_ffts = n_ffts

    def forward(self, pred_wave: torch.Tensor, target_wave: torch.Tensor) -> torch.Tensor:
        """pred_wave, target_wave: [N, L]"""
        total_loss = 0.0
        for n_fft in self.n_ffts:
            hop = n_fft // 4
            pred_spec = torch.stft(pred_wave, n_fft=n_fft, hop_length=hop,
                                   return_complex=True)
            target_spec = torch.stft(target_wave, n_fft=n_fft, hop_length=hop,
                                     return_complex=True)
            # L1 on magnitude + L1 on log magnitude
            pred_mag = pred_spec.abs()
            target_mag = target_spec.abs()
            l1 = (pred_mag - target_mag).abs().mean()
            l1_log = (torch.log(pred_mag + 1e-8) - torch.log(target_mag + 1e-8)).abs().mean()
            total_loss += l1 + l1_log
        return total_loss / len(self.n_ffts)


class PerceptualLoss(nn.Module):
    """Perceptual loss: cosine similarity of DINOv2 features on spectrograms."""

    def __init__(self, n_fft: int = 512, hop_length: int = 160):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length

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
        spec = torch.stft(waveform, n_fft=self.n_fft, hop_length=self.hop_length,
                          return_complex=True)
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
                dinov2_extractor) -> torch.Tensor:
        """
        Args:
            pred_wave: [B, N, L] or [N, L]
            target_wave: [B, N, L] or [N, L]
            dinov2_extractor: DINOv2FeatureExtractor instance
        Returns:
            Cosine similarity loss (1 - cos_sim), mean over batch and sources
        """
        device = pred_wave.device
        # Convert to DINOv2 input images
        pred_img = self._waveform_to_dinov2_input(pred_wave).to(device)
        target_img = self._waveform_to_dinov2_input(target_wave).to(device)

        # Get DINOv2 features [B, 1024, 768] or [B, N, 1024, 768]
        with torch.no_grad():
            pred_feat = dinov2_extractor(pred_img.flatten(0, 1)) if pred_img.dim() == 5 else dinov2_extractor(pred_img)
            target_feat = dinov2_extractor(target_img.flatten(0, 1)) if target_img.dim() == 5 else dinov2_extractor(target_img)
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
