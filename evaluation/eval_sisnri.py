"""Evaluation: SI-SNRi (Scale-Invariant Signal-to-Noise Ratio improvement) with PIT.

Computes the SI-SNR improvement of separated sources over the mixture
on the test set, using Permutation Invariant Training (PIT) for N>2.

Usage:
    python evaluation/eval_sisnri.py \
        --checkpoint checkpoints/best.ckpt \
        --index_file cache/index.json

Output (JSON):
    {
        "si_snri_mean": 12.34,
        "si_snri_std": 2.1,
        "per_sample": [...]
    }
"""
import argparse
import json
import os
import sys
from typing import Dict, List

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.models.separator import SeparatorModule
from src.data.datamodule import AudioVisualDataModule
from src.audio.spectrogram import ISTFTModule


def compute_si_snr(source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """SI-SNR for single source-target pair. Both [L]."""
    source = source - source.mean()
    target = target - target.mean()
    target_energy = target.pow(2).sum()
    proj = (source * target).sum() / (target_energy + 1e-8) * target
    noise = proj - source
    return 10 * torch.log10(proj.pow(2).sum() / (noise.pow(2).sum() + 1e-8))


def pit_si_snr(pred_waveforms: torch.Tensor, target_waveforms: torch.Tensor,
               mixture_wave: torch.Tensor) -> torch.Tensor:
    """
    PIT SI-SNR: find optimal permutation of sources that maximizes total SI-SNR.

    Args:
        pred_waveforms: [N, L] - separated waveforms
        target_waveforms: [N, L] - target waveforms
        mixture_wave: [L] - mixture waveform for baseline
    Returns:
        sisnri_per_source: [N] - SI-SNR improvement for each source (best permutation)
    """
    N, L = pred_waveforms.shape
    if N == 1:
        sisnr_sep = compute_si_snr(pred_waveforms[0], target_waveforms[0])
        sisnr_mix = compute_si_snr(mixture_wave, target_waveforms[0])
        return torch.tensor([sisnr_sep.item() - sisnr_mix.item()])

    # For N<=3 use brute-force, for N>3 use Hungarian algorithm
    from itertools import permutations
    from scipy.optimize import linear_sum_assignment

    # Compute cost matrix: -SI-SNR for each pred-target pair
    cost_matrix = torch.zeros(N, N)
    for i in range(N):
        for j in range(N):
            sisnr = compute_si_snr(pred_waveforms[i], target_waveforms[j])
            cost_matrix[i, j] = -sisnr.item()  # negative for minimization

    # Find optimal assignment
    if N <= 3:
        # Brute force for small N
        best_perm = None
        best_cost = float('inf')
        for perm in permutations(range(N)):
            cost = sum(cost_matrix[i, perm[i]].item() for i in range(N))
            if cost < best_cost:
                best_cost = cost
                best_perm = perm
        assignment = list(zip(range(N), best_perm))
    else:
        # Hungarian algorithm for N>3
        row_ind, col_ind = linear_sum_assignment(cost_matrix.numpy())
        assignment = list(zip(row_ind.tolist(), col_ind.tolist()))

    # Compute SI-SNRi for each source under optimal assignment
    sisnri_scores = []
    for pred_idx, target_idx in assignment:
        sisnr_sep = compute_si_snr(pred_waveforms[pred_idx], target_waveforms[target_idx])
        sisnr_mix = compute_si_snr(mixture_wave, target_waveforms[target_idx])
        sisnri_scores.append(sisnr_sep.item() - sisnr_mix.item())

    return torch.tensor(sisnri_scores)


def reconstruct_mixture(mixture_stft: torch.Tensor,
                        device: torch.device) -> torch.Tensor:
    """
    Reconstruct mixture waveform from complex STFT for SI-SNRi
    baseline. Uses direct torch.istft with no masking.

    Args:
        mixture_stft: [B, 2, F, T] real+imag channels
        device: target device
    Returns:
        mixture_wave: [B, L]
    """
    complex_mix = torch.complex(
        mixture_stft[:, 0], mixture_stft[:, 1])
    return torch.istft(
        complex_mix,
        n_fft=512,
        hop_length=160,
        win_length=400,
        window=torch.hann_window(400, device=device),
        length=96000,
        return_complex=False,
    )


def evaluate_sisnri(args) -> Dict:
    """Evaluate SI-SNRi with PIT on the test split."""
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")

    # Load model
    print(f"Loading checkpoint from {args.checkpoint} ...")
    model = SeparatorModule.load_from_checkpoint(args.checkpoint)
    model = model.to(device)
    model.eval()

    # DataModule
    dm = AudioVisualDataModule(
        index_file=args.index_file,
        n_sources=args.n_sources,
        batch_size=1,
        num_workers=0,
        include_visual=(model.phase != "phase1"),
    )
    dm.setup("test")
    dataloader = dm.test_dataloader()

    sisnri_scores = []

    # Move ISTFTModule instantiation outside the batch loop
    # (No longer needed since we use reconstruct_mixture directly)

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            mixture_stft = batch["mixture_stft"].to(device)
            target_waveforms = batch["target_waveforms"].to(device)  # [1, N, L]
            visual_features = batch.get("visual_features")
            video_frames = batch.get("video_frames")

            # Forward - route visual input appropriately
            if model.phase == "phase1":
                pred_waveforms = model(mixture_stft)
            elif visual_features is not None:
                pred_waveforms = model(mixture_stft, visual_features=visual_features)
            elif video_frames is not None:
                pred_waveforms = model(mixture_stft, video_frames=video_frames)
            else:
                pred_waveforms = model(mixture_stft)

            # [1, N, L]
            B, N, L = pred_waveforms.shape

            # Reconstruct mixture from STFT for SI-SNRi baseline
            mixture_wave = reconstruct_mixture(mixture_stft, device)

            for b in range(B):
                mix = mixture_wave[b]
                pred = pred_waveforms[b]  # [N, L]
                tgt = target_waveforms[b]  # [N, L]

                # PIT SI-SNRi
                sisnri_per_source = pit_si_snr(pred, tgt, mix)
                sisnri_scores.extend(sisnri_per_source.tolist())

            if (batch_idx + 1) % args.log_every == 0:
                print(f"  Processed {batch_idx + 1}/{len(dataloader)} batches...")

    result = {
        "si_snri_mean": float(np.mean(sisnri_scores)) if sisnri_scores else 0.0,
        "si_snri_std": float(np.std(sisnri_scores)) if sisnri_scores else 0.0,
        "si_snri_median": float(np.median(sisnri_scores)) if sisnri_scores else 0.0,
        "num_samples": len(sisnri_scores),
        "per_sample": sisnri_scores,
    }
    return result


def main():
    parser = argparse.ArgumentParser(description="Evaluate SI-SNRi with PIT on test set")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--index_file", type=str, default="cache/index.json", help="Dataset index")
    parser.add_argument("--n_sources", type=int, default=2, help="Number of sources")
    parser.add_argument("--cpu", action="store_true", help="Force CPU")
    parser.add_argument("--log_every", type=int, default=50, help="Log every N batches")
    parser.add_argument("--output", type=str, default="outputs/eval_sisnri.json", help="Output JSON path")
    args = parser.parse_args()

    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
    if not os.path.exists(args.index_file):
        raise FileNotFoundError(f"Index file not found: {args.index_file}")

    # Ensure outputs directory exists
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    results = evaluate_sisnri(args)

    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {args.output}")
    print(f"  SI-SNRi mean:  {results['si_snri_mean']:.2f} dB")
    print(f"  SI-SNRi std:   {results['si_snri_std']:.2f} dB")
    print(f"  SI-SNRi median:{results['si_snri_median']:.2f} dB")
    print(f"  Samples:       {results['num_samples']}")


if __name__ == "__main__":
    main()