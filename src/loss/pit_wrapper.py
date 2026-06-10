"""PIT (Permutation-Invariant Training) loss wrapper with Hungarian algorithm.

SPEC 7.3: PIT for N > 3 using Hungarian algorithm, shared permutation across losses.
"""
import torch
import torch.nn as nn
from typing import Tuple, Optional, List
from scipy.optimize import linear_sum_assignment


class PITLossWrapper(nn.Module):
    """Permutation-invariant training loss wrapper with shared permutation across losses.

    For N <= 3: enumerates all N! permutations
    For N > 3: uses Hungarian algorithm (linear_sum_assignment)
    """

    def __init__(self, si_snr_loss: nn.Module, crm_loss: nn.Module):
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
        from itertools import permutations
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
        Handles both single sample [N, L] and batched [B, N, L] inputs.
        Args:
            pred_wave: [N, L] or [B, N, L] predicted waveforms
            target_wave: [N, L] or [B, N, L] target waveforms
            pred_mask: [N, 2, F, T] or [B, N, 2, F, T] predicted cRM masks (optional)
            target_mask: [N, 2, F, T] or [B, N, 2, F, T] target cRM masks (optional)
            alpha_crm: weight for cRM in combined cost matrix
        Returns:
            si_snr_loss: scalar SI-SNR loss with optimal permutation (mean over batch)
            crm_loss: scalar cRM loss with same permutation (or 0 if masks not provided)
            perm: [N] or [B, N] the shared permutation(s) used
        """
        # Handle both single sample and batched inputs
        is_batched = pred_wave.dim() == 3
        if not is_batched:
            pred_wave = pred_wave.unsqueeze(0)
            target_wave = target_wave.unsqueeze(0)
            if pred_mask is not None:
                pred_mask = pred_mask.unsqueeze(0)
                target_mask = target_mask.unsqueeze(0)

        B, N, L = pred_wave.shape
        si_snr_losses = []
        crm_losses = []
        perms = []

        for b in range(B):
            # Compute pairwise costs for this batch element
            si_snr_cost = self.si_snr.compute_pairwise_losses(pred_wave[b], target_wave[b])

            if pred_mask is not None and target_mask is not None:
                crm_cost = self.crm.compute_pairwise_losses(pred_mask[b], target_mask[b])
            else:
                crm_cost = torch.zeros_like(si_snr_cost)

            # Find shared permutation
            perm = self.find_shared_permutation(si_snr_cost, crm_cost, alpha=1.0, beta=alpha_crm)
            perms.append(perm)

            # Apply permutation to compute losses
            perm_pred_wave = self.apply_permutation(pred_wave[b], perm)
            si_snr_loss = -self.si_snr._si_snr(perm_pred_wave[0], target_wave[b, 0])
            for i in range(1, N):
                si_snr_loss = si_snr_loss - self.si_snr._si_snr(perm_pred_wave[i], target_wave[b, i])
            si_snr_loss = si_snr_loss / N
            si_snr_losses.append(si_snr_loss)

            if pred_mask is not None and target_mask is not None:
                perm_pred_mask = self.apply_permutation(pred_mask[b], perm)
                crm_loss = self.crm(perm_pred_mask, target_mask[b])
                crm_losses.append(crm_loss)

        si_snr_loss = torch.stack(si_snr_losses).mean()
        crm_loss = torch.stack(crm_losses).mean() if crm_losses else torch.tensor(0.0, device=pred_wave.device)
        perm = torch.stack(perms)  # [B, N]

        return si_snr_loss, crm_loss, perm