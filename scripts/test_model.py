#!/usr/bin/env python
"""Dry-run model validation script.

Validates SeparatorModule works correctly without actual data:
1. Loads Hydra config from configs/config.yaml
2. Initializes SeparatorModule for phase1
3. Creates dummy batch with correct shapes
4. Runs forward pass and training_step
5. Prints parameter counts, output shapes, and loss value
"""

import sys
import torch

from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.models.separator import SeparatorModule


def main():
    print("=" * 60)
    print("Dry-Run Model Validation")
    print("=" * 60)

    # 1. Use minimal config (Hydra config has ${now:...} that fails outside hydra.main)
    print("\n[1] Loading model configuration...")
    cfg = {
        "model": {
            "n_sources": 2,
        },
        "train": {
            "loss": {
                "alpha_crm": 0.1,
                "beta_stft": 0.05,
                "gamma_perceptual": 0.1,
            }
        },
    }
    print(f"    Config loaded: {cfg}")

    # 2. Initialize SeparatorModule for phase1
    print("\n[2] Initializing SeparatorModule for phase1...")
    model = SeparatorModule(cfg, phase="phase1")
    print(f"    Model initialized on device: {next(model.parameters()).device}")

    # 3. Parameter counts
    print("\n[3] Parameter Counts:")
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"    Total parameters:     {total_params:,}")
    print(f"    Trainable parameters: {trainable_params:,}")

    # Breakdown by component
    print("\n    By component:")
    components = [
        ("dinov2", model.dinov2),
        ("audio_unet", model.audio_unet),
        ("visual_proj", model.visual_proj),
        ("cross_attn", model.cross_attn),
        ("source_queries", model.source_queries.numel()),  # Parameter, not Module
    ]
    for name, module in components:
        if isinstance(module, int):
            n_params = module
        else:
            n_params = sum(p.numel() for p in module.parameters())
        print(f"      {name}: {n_params:,}")

    # 4. Create dummy batch with correct shapes
    print("\n[4] Creating dummy batch...")
    B, N_sources = 2, 2
    F, T = 257, 601
    L = 96000  # 6s at 16kHz
    device = torch.device("cpu")

    dummy_batch = {
        "mixture_stft": torch.randn(B, 2, F, T, device=device),
        "target_waveforms": torch.randn(B, N_sources, L, device=device),
    }
    print(f"    mixture_stft:    {dummy_batch['mixture_stft'].shape}")
    print(f"    target_waveforms: {dummy_batch['target_waveforms'].shape}")

    # 5. Run forward pass
    print("\n[5] Running forward pass...")
    with torch.no_grad():
        output = model(dummy_batch["mixture_stft"], video_frames=None)
    print(f"    Output shape: {output.shape}")
    print(f"    Expected:     [B={B}, N_sources={N_sources}, L={L}]")

    # Validate output shape
    assert output.shape == torch.Size(
        [B, N_sources, L]
    ), f"Output shape mismatch: {output.shape} vs expected [{B}, {N_sources}, {L}]"
    print("    Output shape: OK")

    # 6. Run training_step
    print("\n[6] Running training_step...")
    model.train()
    loss = model.training_step(dummy_batch, batch_idx=0)
    print(f"    Loss value: {loss.item():.4f}")
    print(f"    Loss dtype: {loss.dtype}")
    print(f"    Requires grad: {loss.requires_grad}")

    # 7. Summary
    print("\n" + "=" * 60)
    print("VALIDATION SUMMARY")
    print("=" * 60)
    print(f"  Total parameters:   {total_params:,}")
    print(f"  Trainable params:   {trainable_params:,}")
    print(f"  Forward pass:       OK (output shape: {list(output.shape)})")
    print(f"  Training step:      OK (loss: {loss.item():.4f})")
    print("=" * 60)
    print("All checks passed!")


if __name__ == "__main__":
    main()
