"""Evaluation: SDR/SIR/SAR using mir_eval.separation.bss_eval_sources.

Computes BSS Eval metrics (SDR, SIR, SAR) on separated vs target waveforms.

Usage:
    python evaluation/eval_sdr.py \
        --checkpoint checkpoints/best.ckpt \
        --index_file cache/index.json

Output (JSON):
    {
        "SDR_mean": 10.5,
        "SIR_mean": 15.2,
        "SAR_mean": 12.8,
        "SDR_per_source": [...],
        "SIR_per_source": [...],
        "SAR_per_source": [...],
        "num_samples": 100
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


def evaluate_sdr(args) -> Dict:
    """Evaluate SDR/SIR/SAR using mir_eval on the test split."""
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

    # Collect all SDR/SIR/SAR scores
    sdr_scores = []
    sir_scores = []
    sar_scores = []

    try:
        import mir_eval.separation
    except ImportError:
        print("WARNING: mir_eval not installed; returning placeholder values")
        mir_eval = None

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

            for b in range(B):
                # Convert to numpy for mir_eval
                est_sources = pred_waveforms[b].cpu().numpy()  # [N, L]
                ref_sources = target_waveforms[b].cpu().numpy()  # [N, L]

                if mir_eval is not None:
                    # mir_eval expects (n_sources, n_samples) for both
                    sdr, sir, sar, _ = mir_eval.separation.bss_eval_sources(
                        ref_sources, est_sources, compute_permutation=True
                    )
                    sdr_scores.extend(sdr.tolist())
                    sir_scores.extend(sir.tolist())
                    sar_scores.extend(sar.tolist())
                else:
                    # Placeholder
                    sdr_scores.extend([0.0] * N)
                    sir_scores.extend([0.0] * N)
                    sar_scores.extend([0.0] * N)

            if (batch_idx + 1) % args.log_every == 0:
                print(f"  Processed {batch_idx + 1}/{len(dataloader)} batches...")

    result = {
        "SDR_mean": float(np.mean(sdr_scores)) if sdr_scores else 0.0,
        "SDR_std": float(np.std(sdr_scores)) if sdr_scores else 0.0,
        "SIR_mean": float(np.mean(sir_scores)) if sir_scores else 0.0,
        "SIR_std": float(np.std(sir_scores)) if sir_scores else 0.0,
        "SAR_mean": float(np.mean(sar_scores)) if sar_scores else 0.0,
        "SAR_std": float(np.std(sar_scores)) if sar_scores else 0.0,
        "SDR_per_source": sdr_scores,
        "SIR_per_source": sir_scores,
        "SAR_per_source": sar_scores,
        "num_samples": len(sdr_scores),
    }
    return result


def main():
    parser = argparse.ArgumentParser(description="Evaluate SDR/SIR/SAR on test set")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--index_file", type=str, default="cache/index.json", help="Dataset index")
    parser.add_argument("--n_sources", type=int, default=2, help="Number of sources")
    parser.add_argument("--cpu", action="store_true", help="Force CPU")
    parser.add_argument("--log_every", type=int, default=50, help="Log every N batches")
    parser.add_argument("--output", type=str, default="outputs/eval_sdr.json", help="Output JSON path")
    args = parser.parse_args()

    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
    if not os.path.exists(args.index_file):
        raise FileNotFoundError(f"Index file not found: {args.index_file}")

    # Ensure outputs directory exists
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    results = evaluate_sdr(args)

    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {args.output}")
    print(f"  SDR mean: {results['SDR_mean']:.2f} dB (±{results['SDR_std']:.2f})")
    print(f"  SIR mean: {results['SIR_mean']:.2f} dB (±{results['SIR_std']:.2f})")
    print(f"  SAR mean: {results['SAR_mean']:.2f} dB (±{results['SAR_std']:.2f})")
    print(f"  Samples:  {results['num_samples']}")


if __name__ == "__main__":
    main()