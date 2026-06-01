# SonicSight-DINOv2 — Project Handover

## What This Project Is

A PyTorch Lightning implementation of a DINOv2-guided audio-visual source separation system. The model takes a mixed audio signal and a synchronized video, uses a frozen DINOv2-Base ViT to extract patch-level visual features, then fuses those features with a U-Net bottleneck via cross-modal attention to predict per-source complex spectral masks. The goal is to separate up to 4 overlapping audio sources (speech, instruments, environmental sounds) given only the video of the sources.

## Environment Setup

1. **Install Python 3.12** from https://www.python.org/downloads/
2. **Open a terminal in the project root:** `D:\development\python\ai\SonicSightDino`
3. **Create a virtual environment:**
   ```
   C:/Users/H/AppData/Local/Programs/Python/Python312/python.exe -m venv .venv
   .venv\Scripts\activate
   ```
4. **Install dependencies:**
   ```
   pip install -r requirements.txt
   ```
5. **Verify the install by running the tests:**
   ```
   C:/Users/H/AppData/Local/Programs/Python/Python312/python.exe -m pytest tests/ -v
   ```
   All 29 tests should pass.

## What Has Been Implemented

### Section 11.2 — Data Pipeline & Preprocessing
| File | Purpose | Test Passing |
|------|---------|-------------|
| `src/data/preprocessing.py` | AudioPreprocessor (resample to 16kHz, mono, 6s clip), VideoPreprocessor (resize to 448×448, ImageNet normalize), STFTModule (complex STFT → [2,257,601]), ISTFTModule, CRM target computation, temporal alignment | ✅ 16/16 tests pass |
| `src/data/dataset.py` | AudioVisualDataset (torch.utils.data.Dataset with on-the-fly mix-and-separate), AudioVisualDataModule (simple loader wrapper) | Not tested directly |
| `src/data/datamodule.py` | AudioVisualDataModule (PyTorch Lightning DataModule with setup/train/val/test dataloaders) | Not tested directly |
| `src/data/mix_and_separate.py` | mix_sources (random gain mixing), apply_augmentation (skeleton) | Not tested directly |

### Section 11.3 — Model Architecture (Core Components)
| File | Purpose | Test Passing |
|------|---------|-------------|
| `src/audio/spectrogram.py` | STFTModule (torchaudio.transforms.Spectrogram), ISTFTModule (InverseSpectrogram) for complex spectrograms [B, 2, F, T] | ✅ Yes (via test_unet.py) |
| `src/audio/unet.py` | AudioUNetEncoder (5 blocks [2→32→64→128→256→512], stride 2, GroupNorm), AudioUNetDecoder (5 upsampling blocks with skip connections), AudioUNet (full encoder-decoder) | ✅ 6/6 tests pass |
| `src/visual/dinov2.py` | DINOv2FeatureExtractor (frozen facebook/dinov2-base, outputs [B, 1024, 768] patch tokens) | ✅ 3/3 tests pass |
| `src/fusion/positional_encoding.py` | SinusoidalPositionalEncoding (Vaswani-style sinusoids for audio query) | Indirectly via cross_attention test |
| `src/fusion/cross_attention.py` | CrossAttentionBlock (Pre-LN → MultiheadAttention → FFN), CrossModalAttentionModule (2 stacked blocks, n_heads=8, d=512, ffn=2048) | ✅ 3/3 tests pass |
| `src/loss/separation.py` | SISNRLoss (permutation-invariant SI-SNR), CRMLoss (MSE on masks), MultiScaleSTFTLoss ([256,512,1024] STFT scales), PerceptualLoss (skeleton) | Not tested directly |
| `src/models/separator.py` | SeparatorModule (LightningModule skeleton — see Known Issues below) | ❌ Not functional — forward() is NotImplemented |
| `src/utils/config.py` | Hydra config utilities (instantiate_from_config, merge_with_dotlist) | Not tested |
| `src/utils/metrics.py` | compute_si_snr_improvement, compute_bss_eval, compute_wer, compute_attention_localization_iou | Not tested |

### Configuration Files
| File | Purpose |
|------|---------|
| `configs/config.yaml` | Hydra root config (merges model + data + train defaults) |
| `configs/model/default.yaml` | Model hyperparameters (n_sources=4, U-Net channels, DINOv2, fusion dims) |
| `configs/data/default.yaml` | Data parameters (16kHz, 6s clips, STFT settings, batch sizes) |
| `configs/train/phase1.yaml` | Phase 1: audio-only pretraining (200K steps, AdamW lr=1e-3, SI-SNR only) |
| `configs/train/phase2.yaml` | Phase 2: cross-modal attention warmup (30K steps, lr=5e-4, attention entropy regularization) |
| `configs/train/phase3.yaml` | Phase 3: end-to-end fine-tuning (100K steps, differential LR, progressive difficulty 2→3→4 sources) |

### Scripts
| File | Purpose |
|------|---------|
| `scripts/preprocess_data.py` | Skeleton — preprocessing pipeline not yet implemented |
| `scripts/run_evaluation.py` | Skeleton — evaluation runner not yet implemented |

## Current State of the Codebase

All source files under `src/`:
- `src/__init__.py` — Package root marker
- `src/audio/__init__.py` — Audio package marker
- `src/audio/spectrogram.py` — STFT/iSTFT modules for complex spectrograms
- `src/audio/unet.py` — 5-block U-Net encoder/decoder with skip connections
- `src/visual/__init__.py` — Visual package marker
- `src/visual/dinov2.py` — Frozen DINOv2-Base feature extractor (HuggingFace)
- `src/fusion/__init__.py` — Fusion package marker
- `src/fusion/cross_attention.py` — Cross-modal attention module (2 blocks)
- `src/fusion/positional_encoding.py` — Sinusoidal positional encoding
- `src/data/__init__.py` — Data package marker
- `src/data/preprocessing.py` — Preprocessing utilities (audio, video, STFT, cRM, alignment)
- `src/data/dataset.py` — PyTorch Dataset with on-the-fly mix-and-separate
- `src/data/datamodule.py` — Lightning DataModule wrapper
- `src/data/mix_and_separate.py` — Source mixing with random gains
- `src/models/__init__.py` — Models package marker
- `src/models/separator.py` — SeparatorModule (LightningModule skeleton, NOT functional)
- `src/loss/__init__.py` — Loss package marker
- `src/loss/separation.py` — Loss functions (SI-SNR, cRM, multi-scale STFT, perceptual)
- `src/utils/__init__.py` — Utils package marker
- `src/utils/config.py` — Hydra config helper functions
- `src/utils/metrics.py` — Evaluation metrics (SI-SNRi, BSS, WER, IoU)

All test files under `tests/`:
- `tests/conftest.py` — pytest fixtures (seed=42)
- `tests/audio/test_unet.py` — U-Net shape, GroupNorm, einops, STFT tests (6 tests)
- `tests/fusion/test_attention.py` — Cross-attention shape, d_model check, positional encoding (3 tests)
- `tests/visual/test_dinov2.py` — Output shape, frozen params, bias check (3 tests)
- `tests/data/test_preprocessing.py` — Audio preproc, video preproc, STFT/iSTFT, alignment, cRM (16 tests)

## Where We Stopped

Stopped at **Section 11.4: Phase 1 — Audio-Only Pretraining**.

All architectural components for Sections 11.2 and 11.3 are implemented and passing tests. The `SeparatorModule` in `src/models/separator.py` exists as a LightningModule skeleton with:
- Correct component initialization (DINOv2, U-Net, cross-attention, source query tokens)
- Correct loss module initialization
- A `forward()` method that raises `NotImplementedError`
- A `training_step()` method that raises `NotImplementedError`
- A `validation_step()` that is empty
- A `configure_optimizers()` that returns a basic AdamW (not differential LR)

No training script or training execution has been done yet.

## What Needs To Be Done Next

### Immediate Next Steps (in order)

1. **Implement `SeparatorModule.forward()`** — Wire together the full forward pass:
   - Pass `mixture_stft` through `AudioUNet` encoder → bottleneck
   - Flatten bottleneck to `[B, T_a, D_a]` sequence
   - If `video_frames` provided: run DINOv2, project 768→512 (no bias)
   - Run cross-modal attention → output slots
   - Reshape to `[B, N_sources, 512, 9, 19]` → feed per-source decoders
   - Apply tanh activation to masks → complex multiply with mixture → iSTFT
   - Return separated waveforms `[B, N_sources, L]`

2. **Implement `SeparatorModule.training_step()`** — Phase-aware training:
   - Phase 1: audio-only, SI-SNR loss, gradient clipping max_norm=1.0
   - Phase 2: freeze audio U-Net + DINOv2, train cross-attention only, SI-SNR + 0.1×entropy regularization
   - Phase 3: differential LR, combined loss (SI-SNR + α·cRM + β·STFT + γ·perceptual)

3. **Implement `SeparatorModule.configure_optimizers()`** — Differential learning rates per phase.

4. **Create a training script** (e.g., `scripts/train.py`) that loads Hydra config and runs Lightning Trainer.

5. **Implement preprocessing pipeline** in `scripts/preprocess_data.py` — index datasets, cache STFT, compute cRM targets.

### Prompt to resume work

Paste this into Claude Code:

> "Continue implementing the DINOv2 audio-visual source separation project. The SeparatorModule at src/models/separator.py has a skeleton with forward() and training_step() as NotImplementedError. Implement the full forward pass, then implement training_step() for Phase 1 (audio-only pretraining) following the SPEC.md section 11.4 and the configs. Make sure to use `einops.rearrange` for reshaping, not .view(). After implementing, write a test for training_step and run pytest."

## Known Issues and Workarounds

### 1. Windows Path Separator
- **Issue:** Windows paths use `\` but the project convention requires forward slashes `/` in all file paths and bash commands.
- **Workaround:** Always use forward slashes. Python path: `C:/Users/H/AppData/Local/Programs/Python/Python312/python.exe`. Never use `C:\\Users\\H\\...` in any tool call.

### 2. API Timeout / Provider Failure
- **Issue:** Claude Code API provider occasionally times out or fails mid-task.
- **Workaround:** Wait 30 seconds and retry the exact same step. Do not restart from scratch.

### 3. SeparatorModule is a Skeleton
- **Issue:** `src/models/separator.py` has `NotImplementedError` in `forward()` and `training_step()`.
- **Impact:** Cannot train until these are implemented.
- **Workaround:** Implement them as the next task (see What Needs To Be Done Next).

### 4. torchaudio deprecation warning
- **Issue:** `STFTModule` passes `return_complex=True` to `torchaudio.transforms.Spectrogram`, which is deprecated — the transform now always returns complex.
- **Workaround:** Warning is harmless. Could be fixed by removing the deprecated parameter in a future cleanup.

### 5. Preprocessing and evaluation scripts are empty
- **Issue:** `scripts/preprocess_data.py` and `scripts/run_evaluation.py` only print a TODO string.
- **Workaround:** These are not needed for the initial training pipeline but must be implemented before dataset ingestion and evaluation.

## Project Conventions

These must be followed for all new code:

### PyTorch Patterns
- Move tensors to device explicitly: `tensor = tensor.to(self.device)`
- Use `torch.no_grad()` when running frozen modules (DINOv2)
- Prefer `einops.rearrange` over manual `.view()` or `.reshape()`
- Always assert tensor shapes in unit tests: `assert out.shape == torch.Size([...])`
- Use `torch.testing.assert_close` for numerical comparisons in tests
- Complex spectrograms are always 2-channel real/imaginary: shape `[B, 2, F, T]`

### What NOT to do
- NEVER use `torchaudio.functional.spectrogram` — always use `torchaudio.transforms.Spectrogram`
- NEVER batch DINOv2 calls across sources — process each source independently
- NEVER recompute cRM targets on the fly — always load from cache
- NEVER use BatchNorm — always use GroupNorm with `groups=8`
- NEVER skip writing a unit test before moving to the next component
- NEVER use `.view()` for tensor reshaping — use `einops.rearrange` instead
- NEVER use `bias=True` in the visual projection linear layer

### Shell and Path Conventions
- ALWAYS use forward slashes in all paths: `C:/Users/H/...` not `C:\Users\H\...`
- NEVER use backslashes in any bash command or file path
- When running Python use this exact path: `C:/Users/H/AppData/Local/Programs/Python/Python312/python.exe`
- NEVER use the `type` command to read files — use `cat` instead
