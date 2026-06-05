"""Evaluation metrics: SI-SNRi, SDR, SIR, SAR, WER.

SPEC 11.5: torchmetrics SI-SNR, mir_eval bss_eval, Whisper WER.
"""
import torch
import numpy as np
from typing import List, Tuple, Optional, Dict, Any


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
    """Weakly-supervised object localization IoU (Section 11.5).

    Args:
        attention_map: [H, W] or [1, H, W] attention weights (0-1)
        bbox: (x1, y1, x2, y2) in pixel coordinates
    Returns:
        IoU between thresholded attention map and bbox
    """
    if attention_map.dim() == 3:
        attention_map = attention_map.squeeze(0)
    H, W = attention_map.shape

    # Threshold attention at 50% of max
    thresh = attention_map.max() * 0.5
    pred_mask = (attention_map > thresh).float()

    # Create bbox mask
    x1, y1, x2, y2 = bbox
    x1, y1 = max(0, int(x1)), max(0, int(y1))
    x2, y2 = min(W, int(x2)), min(H, int(y2))
    gt_mask = torch.zeros_like(pred_mask)
    gt_mask[y1:y2, x1:x2] = 1.0

    # IoU
    intersection = (pred_mask * gt_mask).sum()
    union = pred_mask.sum() + gt_mask.sum() - intersection
    if union == 0:
        return 0.0
    return (intersection / union).item()


def compute_metrics(predicted: torch.Tensor, target: torch.Tensor,
                    mixture: torch.Tensor) -> Dict[str, float]:
    """
    Compute all evaluation metrics.
    Args:
        predicted: [N, L] or [B, N, L] separated waveforms
        target: [N, L] or [B, N, L] target waveforms
        mixture: [L] or [B, L] mixture waveform
    Returns:
        Dict with si_snri, sdr, sir, sar (if available)
    """
    # Handle batch dimension
    if predicted.dim() == 3:
        B = predicted.shape[0]
        results = []
        for b in range(B):
            res = compute_metrics(predicted[b], target[b], mixture[b] if mixture.dim() == 2 else mixture)
            results.append(res)
        # Average across batch
        avg = {}
        for k in results[0].keys():
            avg[k] = sum(r[k] for r in results) / B
        return avg

    # Single sample: [N, L]
    from torchmetrics.audio import ScaleInvariantSignalNoiseRatio
    sisnr = ScaleInvariantSignalNoiseRatio()
    sisnr_sep = sisnr(predicted, target)
    sisnr_mix = sisnr(mixture.unsqueeze(0).expand_as(target), target)
    si_snri = (sisnr_sep - sisnr_mix).mean().item()

    results = {"si_snri": si_snri}

    # Try to compute BSS eval metrics
    try:
        import mir_eval
        pred_np = predicted.detach().cpu().numpy()
        tgt_np = target.detach().cpu().numpy()
        sdr, sir, sar, _ = mir_eval.separation.bss_eval_sources(tgt_np, pred_np)
        results.update({
            "sdr": float(np.mean(sdr)),
            "sir": float(np.mean(sir)),
            "sar": float(np.mean(sar)),
        })
    except (ImportError, Exception):
        pass

    return results
