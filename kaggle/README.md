# Kaggle Submission for SonicSight-DINOv2

## Quick Start

1. **Upload this folder** to Kaggle as a dataset (private) named `sonicsight-dinov2-code`
2. **Upload your preprocessed cache** to Kaggle as a dataset (private) named `sonicsight-cache` with `index.json` at root
3. **Create a new notebook** on Kaggle:
   - Attach both datasets
   - Set Internet: Off
   - GPU: T4 x2 (or T4 x1)
   - Copy contents of `submission.ipynb` into the notebook

## Directory Structure

```
/kaggle/
├── input/
│   ├── sonicsight-dinov2-code/
│   │   ├── src/
│   │   ├── configs/
│   │   ├── scripts/
│   │   └── evaluation/
│   └── sonicsight-cache/
│       └── index.json
├── working/
│   ├── SonicSightDino/ (copied from code dataset)
│   ├── cache/ (symlink or copy of input cache)
│   └── checkpoints/
```

## Training Phases

| Phase | Steps | Batch Size | LR | Description |
|-------|-------|------------|-----|-------------|
| 1: Audio-only | 10,000 | 8 | 1e-3 | Train U-Net encoder/decoder |
| 2: Attention warmup | 10,000 | 8 | 5e-4 | Freeze audio, train cross-attention |
| 3: End-to-end | 40,000 | 8 | 3e-4/3e-5/1e-5 | Progressive 2→3→4 sources |

## Expected Results

- **SI-SNRi**: 8-12 dB improvement on test set
- **Inference**: ~50ms per 6s clip on T4

## Key Configs

- `configs/kaggle.yaml` - Main training config (T4 optimized)
- `configs/train/phase1.yaml` - Phase 1 specific
- `configs/train/phase2.yaml` - Phase 2 specific
- `configs/train/phase3.yaml` - Phase 3 specific

## Notes

- Uses mixed precision (`16-mixed`) for T4 VRAM efficiency
- Gradient clipping `max_norm=1.0` for stability
- Progressive difficulty: 2 sources → 3 at 20K steps → 4 at 40K steps
- DINOv2 processed in chunks of 8 frames to avoid OOM