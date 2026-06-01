"""Evaluation metrics: SI-SNRi, SDR, SIR, SAR, WER.

SPEC 11.5: torchmetrics SI-SNR, mir_eval bss_eval, Whisper WER.
"""
import torch
import numpy as np
from typing import List, Tuple, Optional


def compute_si_snr_improvement(separated: torch.Tensor, target: torch.Tensor,
                               mixture: torch.Tensor) -> torch.Tensor:
    """
    separated: [N, L]
    target: [N, L]
    mixture: [L]
    Returns: SI-SNR improvement in dB
    """
    from torchmetrics.audio import ScaleInvariantSignalNoiseRatio
    sisnr = ScaleInvariantSignalNoiseRatio()
    sisnr_sep = sisnr(separated, target)
    sisnr_mix = sisnr(mixture.unsqueeze(0).expand_as(target), target)
    # Take mean across sources
    return (sisnr_sep - sisnr_mix).mean()


def compute_bss_eval(separated: np.ndarray, target: np.ndarray):
    """
    separated, target: [N, L] numpy arrays
    Returns: sdr, sir, sar arrays per source
    """
    import mir_eval
    return mir_eval.separation.bss_eval_sources(target, separated)


def compute_wer(reference: str, hypothesis: str) -> float:
    """Compute Word Error Rate between reference and hypothesis."""
    import jiwer
    return jiwer.wer(reference, hypothesis)


def compute_attention_localization_iou(attention_map: torch.Tensor,
                                       bbox: tuple) -> float:
    """Weakly-supervised object localization IoU (Section 11.5)."""
    # TODO: implement
    return 0.0
