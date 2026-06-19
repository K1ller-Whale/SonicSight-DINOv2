import pytest
import torch

from evaluation.common import align_metric_waveforms
from evaluation.eval_sisnri import pit_si_snr


def test_pit_si_snr_trims_to_shared_metric_length():
    pred = torch.randn(2, 96000)
    target = torch.randn(2, 96123)
    mixture = target.sum(dim=0)

    scores = pit_si_snr(pred, target, mixture)

    assert scores.shape == (2,)
    assert torch.isfinite(scores).all()


def test_align_metric_waveforms_reports_source_count_mismatch():
    pred = torch.randn(4, 96000)
    target = torch.randn(2, 96000)

    with pytest.raises(ValueError, match="Source-count mismatch"):
        align_metric_waveforms(pred, target)
