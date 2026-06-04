"""Test evaluation scripts."""
import os
import json
import torch


def test_eval_sisnri_script():
    """Verify eval_sisnri.py imports and structure."""
    import sys
    sys.path.insert(0, "evaluation")
    import eval_sisnri
    assert hasattr(eval_sisnri, "evaluate_sisnri")
    assert hasattr(eval_sisnri, "main")


def test_eval_wer_script():
    """Verify eval_wer.py imports and structure."""
    import sys
    sys.path.insert(0, "evaluation")
    import eval_wer
    assert hasattr(eval_wer, "evaluate_wer")
    assert hasattr(eval_wer, "main")


def test_eval_localisation_script():
    """Verify eval_localisation.py imports and structure."""
    import sys
    sys.path.insert(0, "evaluation")
    import eval_localisation
    assert hasattr(eval_localisation, "evaluate_localisation")
    assert hasattr(eval_localisation, "main")
    assert hasattr(eval_localisation, "compute_iou")