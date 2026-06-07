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

    def compute_pairwise_losses(self, pred_mask: torch.Tensor, target_mask: torch.Tensor) -> torch.Tensor:
        """
        Compute pairwise cRM MSE losses for all source pairs.
        Args:
            pred_mask: [N, 2, F, T]
            target_mask: [N, 2, F, T]
        Returns:
            cost_matrix: [N, N] where cost_matrix[i, j] = MSE(pred_mask[i], target_mask[j])
        """
        N = pred_mask.shape[0]
        cost_matrix = torch.zeros(N, N, device=pred_mask.device)
        for i in range(N):
            for j in range(N):
                cost_matrix[i, j] = self.mse(pred_mask[i], target_mask[j])
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


class PITLossWrapper(nn.Module):
    """Permutation-invariant training loss wrapper with shared permutation across losses.

    For N <= 3: enumerates all N! permutations
    For N > 3: uses Hungarian algorithm (linear_sum_assignment)
    """

    def __init__(self, si_snr_loss: SISNRLoss, crm_loss: CRMLoss):
        super().__init__()
        self.si_snr = si_snr_loss
        self.crm = crm_loss

    def _compute_cost_matrix(self, si_snr_cost: torch.Tensor, crm_cost: torch.Tensor,
                              alpha: float = 1.0, beta: float = 1.0) -> torch.Tensor:
        """Combine SI-SNR and cRM cost matrices."""
        return alpha * si_snr_cost + beta * crm_cost

    def _hungarian_permutation(self, cost_matrix: torch.Tensor) -> torch.Tensor:
        """Find optimal permutation using Hungarian algorithm."""
        cost_np = cost_matrix.detach().cpu().numpy()
        row_ind, col_ind = linear_sum_assignment(cost_np)
        # row_ind is already sorted [0, 1, ..., N-1], col_ind gives the permutation
        perm = torch.tensor(col_ind, device=cost_matrix.device, dtype=torch.long)
        return perm

    def _enumerate_permutations(self, cost_matrix: torch.Tensor) -> torch.Tensor:
        """Find optimal permutation by enumerating all permutations (N <= 3)."""
        N = cost_matrix.shape[0]
        best_cost = float('inf')
        best_perm = None
        for perm in permutations(range(N)):
            cost = cost_matrix[torch.arange(N), perm].sum()
            if cost < best_cost:
                best_cost = cost
                best_perm = torch.tensor(perm, device=cost_matrix.device, dtype=torch.long)
        return best_perm

    def find_shared_permutation(self, si_snr_cost: torch.Tensor, crm_cost: torch.Tensor,
                                 alpha: float = 1.0, beta: float = 1.0) -> torch.Tensor:
        """
        Find optimal shared permutation for both SI-SNR and cRM losses.
        Args:
            si_snr_cost: [N, N] pairwise cost matrix for SI-SNR
            crm_cost: [N, N] pairwise cost matrix for cRM
            alpha: weight for SI-SNR cost
            beta: weight for cRM cost
        Returns:
            perm: [N] permutation tensor
        """
        N = si_snr_cost.shape[0]
        combined_cost = self._compute_cost_matrix(si_snr_cost, crm_cost, alpha, beta)

        if N <= 3:
            perm = self._enumerate_permutations(combined_cost)
        else:
            perm = self._hungarian_permutation(combined_cost)

        return perm

    def apply_permutation(self, estimated: torch.Tensor, perm: torch.Tensor) -> torch.Tensor:
        """Apply permutation to estimated sources."""
        return estimated[perm]

    def forward(self, pred_wave: torch.Tensor, target_wave: torch.Tensor,
                pred_mask: torch.Tensor = None, target_mask: torch.Tensor = None,
                alpha_crm: float = 0.1) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute PIT-wrapped SI-SNR and cRM losses with shared permutation.
        Args:
            pred_wave: [N, L] predicted waveforms
            target_wave: [N, L] target waveforms
            pred_mask: [N, 2, F, T] predicted cRM masks (optional)
            target_mask: [N, 2, F, T] target cRM masks (optional)
            alpha_crm: weight for cRM in combined cost matrix
        Returns:
            si_snr_loss: scalar SI-SNR loss with optimal permutation
            crm_loss: scalar cRM loss with same permutation (or 0 if masks not provided)
            perm: [N] the shared permutation used
        """
        N = pred_wave.shape[0]

        # Compute pairwise costs
        si_snr_cost = self.si_snr.compute_pairwise_losses(pred_wave, target_wave)

        if pred_mask is not None and target_mask is not None:
            crm_cost = self.crm.compute_pairwise_losses(pred_mask, target_mask)
        else:
            crm_cost = torch.zeros_like(si_snr_cost)

        # Find shared permutation
        perm = self.find_shared_permutation(si_snr_cost, crm_cost, alpha=1.0, beta=alpha_crm)

        # Apply permutation to compute losses
        perm_pred_wave = self.apply_permutation(pred_wave, perm)
        si_snr_loss = -self.si_snr._si_snr(perm_pred_wave[0], target_wave[0])
        for i in range(1, N):
            si_snr_loss = si_snr_loss - self.si_snr._si_snr(perm_pred_wave[i], target_wave[i])
        si_snr_loss = si_snr_loss / N

        if pred_mask is not None and target_mask is not None:
            perm_pred_mask = self.apply_permutation(pred_mask, perm)
            crm_loss = self.crm(perm_pred_mask, target_mask)
        else:
            crm_loss = torch.tensor(0.0, device=pred_wave.device)

        return si_snr_loss, crm_loss, perm
