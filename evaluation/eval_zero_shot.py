"""Evaluation: Zero-shot generalization on held-out categories.

Loads a checkpoint and evaluates on unseen categories (e.g., Tier 4 animal sounds,
unseen instruments) vs seen categories. Reports the generalization gap.

Usage:
    python evaluation/eval_zero_shot.py \
        --checkpoint checkpoints/best.ckpt \
        --index_file cache/index.json \
        --seen_categories speech,instrument_1,instrument_2 \
        --unseen_categories animal_bird,animal_dog,animal_cat

Output (JSON):
    {
        "seen_sisnri": 12.34,
        "zero_shot_sisnri": 8.56,
        "generalisation_gap": 3.78
    }
"""
import argparse
import json
import os
import sys
from typing import Dict, List, Set

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.models.separator import SeparatorModule
from src.data.datamodule import AudioVisualDataModule
from src.audio.spectrogram import ISTFTModule
from evaluation.common import align_metric_waveforms, maybe_to_device, resolve_n_sources


def compute_si_snr(source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """SI-SNR for single source-target pair. Both [L]."""
    source = source - source.mean()
    target = target - target.mean()
    target_energy = target.pow(2).sum()
    proj = (source * target).sum() / (target_energy + 1e-8) * target
    noise = proj - source
    return 10 * torch.log10(proj.pow(2).sum() / (noise.pow(2).sum() + 1e-8))


def pit_si_snr(pred_waveforms: torch.Tensor, target_waveforms: torch.Tensor,
               mixture_wave: torch.Tensor) -> List[float]:
    """
    PIT SI-SNR: find optimal permutation of sources that maximizes total SI-SNR.
    Returns list of SI-SNRi per source.
    """
    pred_waveforms, target_waveforms, mixture_wave = align_metric_waveforms(
        pred_waveforms, target_waveforms, mixture_wave
    )
    N, L = pred_waveforms.shape
    if N == 1:
        sisnr_sep = compute_si_snr(pred_waveforms[0], target_waveforms[0])
        sisnr_mix = compute_si_snr(mixture_wave, target_waveforms[0])
        return [sisnr_sep.item() - sisnr_mix.item()]

    from itertools import permutations
    from scipy.optimize import linear_sum_assignment

    cost_matrix = torch.zeros(N, N)
    for i in range(N):
        for j in range(N):
            sisnr = compute_si_snr(pred_waveforms[i], target_waveforms[j])
            cost_matrix[i, j] = -sisnr.item()

    if N <= 3:
        best_perm = None
        best_cost = float('inf')
        for perm in permutations(range(N)):
            cost = sum(cost_matrix[i, perm[i]].item() for i in range(N))
            if cost < best_cost:
                best_cost = cost
                best_perm = perm
        assignment = list(zip(range(N), best_perm))
    else:
        row_ind, col_ind = linear_sum_assignment(cost_matrix.numpy())
        assignment = list(zip(row_ind.tolist(), col_ind.tolist()))

    sisnri_scores = []
    for pred_idx, target_idx in assignment:
        sisnr_sep = compute_si_snr(pred_waveforms[pred_idx], target_waveforms[target_idx])
        sisnr_mix = compute_si_snr(mixture_wave, target_waveforms[target_idx])
        sisnri_scores.append(sisnr_sep.item() - sisnr_mix.item())

    return sisnri_scores


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


def evaluate_category(args, category_split: str, clip_ids: List[str]) -> float:
    """Evaluate SI-SNRi on a specific category split."""
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")

    model = SeparatorModule.load_from_checkpoint(args.checkpoint)
    model = model.to(device)
    model.eval()
    n_sources = resolve_n_sources(model, getattr(args, "n_sources", None))

    # Create DataModule with filtered clips
    dm = AudioVisualDataModule(
        index_file=args.index_file,
        n_sources=n_sources,
        batch_size=1,
        num_workers=0,
        include_visual=(model.phase != "phase1"),
    )
    dm.setup("test")

    # Get full test dataloader and filter
    dataloader = dm.test_dataloader()

    sisnri_scores = []
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            clip_id = batch.get("clip_id", [str(batch_idx)])[0] if batch.get("clip_id") else str(batch_idx)
            if clip_id not in clip_ids:
                continue

            mixture_stft = batch["mixture_stft"].to(device)
            target_waveforms = batch["target_waveforms"].to(device)

            visual_features = maybe_to_device(batch.get("visual_features"), device)
            video_frames = maybe_to_device(batch.get("video_frames"), device)

            # Forward - route visual input appropriately
            if model.phase == "phase1":
                pred_waveforms = model(mixture_stft)
            elif visual_features is not None:
                pred_waveforms = model(mixture_stft, visual_features=visual_features)
            elif video_frames is not None:
                pred_waveforms = model(mixture_stft, video_frames=video_frames)
            else:
                pred_waveforms = model(mixture_stft)

            B, N, L = pred_waveforms.shape

            # Reconstruct mixture using direct torch.istft
            mixture_wave = reconstruct_mixture(mixture_stft, device)

            for b in range(B):
                mix = mixture_wave[b]
                pred = pred_waveforms[b]
                tgt = target_waveforms[b]
                sisnri_per_source = pit_si_snr(pred, tgt, mix)
                sisnri_scores.extend(sisnri_per_source)

    return float(np.mean(sisnri_scores)) if sisnri_scores else 0.0


def parse_categories(cat_str: str) -> List[str]:
    """Parse comma-separated category string."""
    return [c.strip() for c in cat_str.split(',') if c.strip()]


def get_clips_by_category(index_file: str, categories: List[str]) -> List[str]:
    """
    Load index.json and return clip_ids in the test split
    that belong to any of the given categories.

    index.json format:
        {clip_id: {"split": "train"|"val"|"test",
                   "category": str, ...}, ...}
    """
    with open(index_file) as f:
        index = json.load(f)

    categories_set = set(categories)
    return [
        clip_id
        for clip_id, entry in index.items()
        if entry.get("split") == "test"
        and entry.get("category") in categories_set
    ]


def evaluate_zero_shot(args) -> Dict:
    """Evaluate zero-shot generalization gap between seen and unseen categories."""
    # Parse category lists
    seen_cats = parse_categories(args.seen_categories)
    unseen_cats = parse_categories(args.unseen_categories)

    print(f"Seen categories: {seen_cats}")
    print(f"Unseen categories: {unseen_cats}")

    # Get clip IDs for each split
    seen_clips = get_clips_by_category(args.index_file, seen_cats)
    unseen_clips = get_clips_by_category(args.index_file, unseen_cats)

    print(f"Seen clips: {len(seen_clips)}")
    print(f"Unseen clips: {len(unseen_clips)}")

    if not seen_clips:
        print("WARNING: No seen clips found")
    if not unseen_clips:
        print("WARNING: No unseen clips found")

    # Evaluate on seen categories
    seen_sisnri = evaluate_category(args, "seen", seen_clips) if seen_clips else 0.0
    print(f"\nSeen SI-SNRi: {seen_sisnri:.2f} dB")

    # Evaluate on unseen categories
    zero_shot_sisnri = evaluate_category(args, "unseen", unseen_clips) if unseen_clips else 0.0
    print(f"Zero-shot SI-SNRi: {zero_shot_sisnri:.2f} dB")

    gap = seen_sisnri - zero_shot_sisnri
    print(f"Generalisation gap: {gap:.2f} dB")

    return {
        "seen_sisnri": seen_sisnri,
        "zero_shot_sisnri": zero_shot_sisnri,
        "generalisation_gap": gap,
        "seen_categories": seen_cats,
        "unseen_categories": unseen_cats,
        "seen_clips": len(seen_clips),
        "unseen_clips": len(unseen_clips),
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate zero-shot generalization gap")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--index_file", type=str, default="cache/index.json")
    parser.add_argument("--n_sources", type=int, default=None)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--seen_categories", type=str,
                        default="speech,music_instrument",
                        help="Comma-separated list of seen categories")
    parser.add_argument("--unseen_categories", type=str,
                        default="animal_bird,animal_dog,animal_cat",
                        help="Comma-separated list of unseen categories")
    parser.add_argument("--output", type=str, default="outputs/eval_zero_shot.json")
    args = parser.parse_args()

    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
    if not os.path.exists(args.index_file):
        raise FileNotFoundError(f"Index file not found: {args.index_file}")

    # Ensure outputs directory exists
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    results = evaluate_zero_shot(args)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {args.output}")
    print(f"  Seen SI-SNRi:       {results['seen_sisnri']:.2f} dB")
    print(f"  Zero-shot SI-SNRi:  {results['zero_shot_sisnri']:.2f} dB")
    print(f"  Generalisation gap: {results['generalisation_gap']:.2f} dB")


if __name__ == "__main__":
    main()
