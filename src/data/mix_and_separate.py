"""Mix-and-separate pipeline for synthetic training data generation.

SPEC 7.1: Random mix of N clean sources with random gain and augmentation.
"""
import torch
import torch.nn.functional as F
from typing import List, Tuple, Optional
import random


def mix_sources(sources: List[torch.Tensor],
                scale_db_range: Tuple[float, float] = (-5.0, 5.0),
                rng: Optional[torch.Generator] = None) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Args:
        sources: List of N waveforms, each [L]
        scale_db_range: (min, max) dB to scale each source
        rng: optional torch RNG for reproducibility
    Returns:
        mixture: [L] sum of scaled sources
        gains: [N] per-source linear gains
    """
    n = len(sources)
    device = sources[0].device
    scales = []
    for i in range(n):
        db = random.uniform(scale_db_range[0], scale_db_range[1])
        gain = 10 ** (db / 20.0)
        scales.append(gain)
    scales = torch.tensor(scales, dtype=torch.float32, device=device).unsqueeze(1)  # [N, 1]

    # Pad all to same length
    max_len = max(s.shape[-1] for s in sources)
    padded = []
    for s in sources:
        if s.shape[-1] < max_len:
            pad = max_len - s.shape[-1]
            s = F.pad(s, (0, pad))
        padded.append(s)
    sources_stacked = torch.stack(padded)  # [N, L]

    # Apply random gains and mix
    scaled = sources_stacked * scales  # [N, L]
    mixture = scaled.sum(dim=0)  # [L]
    return mixture, scales.squeeze(1)


def apply_augmentation(waveform: torch.Tensor,
                       pitch_semitones: Optional[float] = None,
                       stretch_rate: Optional[float] = None,
                       noise_snr_db: Optional[float] = None) -> torch.Tensor:
    """
    Random augmentation: pitch shift, time stretch, additive noise.
    TODO: implement actual augmentation
    """
    # Placeholder: returns waveform unchanged
    return waveform
