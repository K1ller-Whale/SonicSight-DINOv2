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
        best_loss = float('-inf')
        best_perm = None
        for perm in permutations(range(N)):
            perm_est = estimated[list(perm)]
            si_snr_vals = [self._si_snr(perm_est[i], target[i]) for i in range(N)]
            loss = sum(si_snr_vals) / N
            if loss > best_loss:
                best_loss = loss
                best_perm = perm
        return -best_loss  # minimize negative SI-SNR


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
    """Perceptual loss: cosine similarity of DINOv2 features."""

    def __init__(self):
        super().__init__()

    def forward(self, pred_spec: torch.Tensor, target_spec: torch.Tensor,
                dinov2_extractor) -> torch.Tensor:
        """
        pred_spec, target_spec: spectrogram images to feed to DINOv2
        TODO: implement with actual DINOv2 feature comparison
        """
        # Placeholder
        return torch.tensor(0.0)
