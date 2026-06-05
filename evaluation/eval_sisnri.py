"""Evaluation: SI-SNRi (Scale-Invariant Signal-to-Noise Ratio improvement).

Computes the SI-SNR improvement of separated sources over the mixture
on the test set.

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
    proj = (source * target).sum() / target_energy * target
    noise = proj - source
    return 10 * torch.log10(proj.pow(2).sum() / (noise.pow(2).sum() + 1e-8))


def evaluate_sisnri(args) -> Dict:
    """Evaluate SI-SNRi on the test split."""
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

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            mixture_stft = batch["mixture_stft"].to(device)
            target_waveforms = batch["target_waveforms"].to(device)  # [1, N, L]

            # Forward
            if model.phase == "phase1":
                pred_waveforms = model(mixture_stft, video_frames=None)
            else:
                video = batch.get("video_frames")
                pred_waveforms = model(mixture_stft, video_frames=video)

            # [1, N, L]
            B, N, L = pred_waveforms.shape

            # Reconstruct mixture from STFT for SI-SNRi baseline
            # Use identity mask (1+0j) to recover original mixture waveform
            istft = ISTFTModule().to(device)
            identity_mask = torch.ones_like(mixture_stft)
            identity_mask[:, 1, :, :] = 0  # imaginary part = 0
            mixture_wave = istft(identity_mask, mixture_stft)  # [1, L]

            for b in range(B):
                mix = mixture_wave[b]
                for n in range(N):
                    pred = pred_waveforms[b, n]
                    tgt = target_waveforms[b, n]

                    # SI-SNR of separated vs target
                    sisnr_sep = compute_si_snr(pred, tgt)
                    # SI-SNR of mixture vs target
                    sisnr_mix = compute_si_snr(mix, tgt)
                    # Improvement
                    sisnri = sisnr_sep - sisnr_mix
                    sisnri_scores.append(sisnri.item())

            if (batch_idx + 1) % args.log_every == 0:
                print(f"  Processed {batch_idx + 1}/{len(dataloader)} batches...")

    result = {
        "si_snri_mean": float(torch.tensor(sisnri_scores).mean().item()),
        "si_snri_std": float(torch.tensor(sisnri_scores).std().item()),
        "si_snri_median": float(torch.tensor(sisnri_scores).median().item()),
        "num_samples": len(sisnri_scores),
        "per_sample": sisnri_scores,
    }
    return result


def main():
    parser = argparse.ArgumentParser(description="Evaluate SI-SNRi on test set")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--index_file", type=str, default="cache/index.json", help="Dataset index")
    parser.add_argument("--n_sources", type=int, default=2, help="Number of sources")
    parser.add_argument("--cpu", action="store_true", help="Force CPU")
    parser.add_argument("--log_every", type=int, default=50, help="Log every N batches")
    parser.add_argument("--output", type=str, default="sisnri_results.json", help="Output JSON path")
    args = parser.parse_args()

    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
    if not os.path.exists(args.index_file):
        raise FileNotFoundError(f"Index file not found: {args.index_file}")

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
