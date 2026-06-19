"""Shared helpers for evaluation scripts."""

from typing import Optional, Tuple

import torch


def maybe_to_device(value, device: torch.device):
    """Move tensor-like optional batch values to the evaluation device."""
    return value.to(device) if torch.is_tensor(value) else value


def resolve_n_sources(model, requested_n_sources: Optional[int]) -> int:
    """Choose an evaluation source count that matches the loaded checkpoint."""
    model_n_sources = int(getattr(model, "n_sources", requested_n_sources or 2))
    if requested_n_sources is None or requested_n_sources == model_n_sources:
        return model_n_sources

    phase = getattr(model, "phase", "")
    max_sources = getattr(model, "source_queries", None)
    max_sources = max_sources.shape[0] if max_sources is not None else model_n_sources
    if phase == "phase3" and 1 <= requested_n_sources <= max_sources:
        print(
            f"Using --n_sources={requested_n_sources} for phase3 progressive "
            f"evaluation (checkpoint default is {model_n_sources})."
        )
        model.n_sources = requested_n_sources
        return requested_n_sources

    print(
        f"WARNING: --n_sources={requested_n_sources} does not match the "
        f"checkpoint model.n_sources={model_n_sources}; using the checkpoint "
        "source count."
    )
    return model_n_sources


def align_metric_waveforms(
    pred_waveforms: torch.Tensor,
    target_waveforms: torch.Tensor,
    mixture_wave: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """Validate source count and trim metric inputs to a shared sample length."""
    if pred_waveforms.dim() != 2 or target_waveforms.dim() != 2:
        raise ValueError(
            "Expected predicted and target waveforms with shape [N, L], got "
            f"{tuple(pred_waveforms.shape)} and {tuple(target_waveforms.shape)}."
        )

    pred_sources = pred_waveforms.shape[0]
    target_sources = target_waveforms.shape[0]
    if pred_sources != target_sources:
        raise ValueError(
            "Source-count mismatch during evaluation: model predicted "
            f"{pred_sources} source(s), but the dataloader produced "
            f"{target_sources} target source(s). Use a matching --n_sources "
            "value or evaluate a checkpoint trained with that source count."
        )

    lengths = [pred_waveforms.shape[-1], target_waveforms.shape[-1]]
    if mixture_wave is not None:
        if mixture_wave.dim() != 1:
            raise ValueError(
                f"Expected mixture waveform with shape [L], got {tuple(mixture_wave.shape)}."
            )
        lengths.append(mixture_wave.shape[-1])

    metric_len = min(lengths)
    pred_waveforms = pred_waveforms[..., :metric_len]
    target_waveforms = target_waveforms[..., :metric_len]
    if mixture_wave is not None:
        mixture_wave = mixture_wave[..., :metric_len]

    return pred_waveforms, target_waveforms, mixture_wave
