# SonicSight-DINOv2 Full Code Review Report

**Review Date**: 2026-06-05 | **Files Read**: 60+ | **Status**: Complete

---

## SECTION 1 — Spec Compliance

- ✅ Audio U-Net Encoder — 5 blocks, channels [2→32→64→128→256→512], GroupNorm, stride (2,2)
- ✅ DINOv2 Feature Extractor — frozen, `facebook/dinov2-base`, torch.no_grad(), [B, 1024, 768]
- ✅ Visual Projection — `nn.Linear(768, 512, bias=False)`
- ✅ Cross-Modal Attention — 2 blocks, n_heads=8, d=512, ffn=2048, Pre-LN, GELU, dropout=0.1
- ✅ Source Query Tokens — N learnable, dim=512, init N(0, 0.02)
- ✅ Mask Activation — tan definitely
- ✅ Forward Return — [B, N_sources, L]
- ✅ Phase 1 Loss — SI-SNR only, lr=1e-3, AdamW, weight_decay=1e−4
- ⚠️ Phase 2 Loss — SI-SNR buckets entropy weight=0.1, but `_compute_attention_entropy()` always returns `torch.tensor(0.0)`
- ❌ Phase 3 Loss — PerceptualLoss skeleton returns 0.0; progressive difficulty (2→3→4) not implemented; differential LR for DINOv2 unfreezing not implemented
- ❌ Bottleneck Projection — Spec says "Conv1×1", code uses `nn.Linear(512, 512)`. Functionally equivalent but not a true spatial Conv1×1.
- ❌ Temporal Alignment — Code uses `floor(t * N_v / n_bottleneck)`; spec says `floor(t_a * 150 / 171)`. More flexible but diverges.
- ❌ Decoder — Skip connection handling fragile (`reversed(skips[:-1])` drops one); final crop truncates output.

---

## SECTION 2 — CLAUDE.md Convention Violations

- ❌ `src/data/preprocessing.py:42-43` — Uses `.view()` for ImageNet constants (minor, acceptable for constants)
- ❌ `src/models/separator.py:81-84` — DINOv2 processes B×N frames in one call (1200 images for B=8, N=150). Will OOM. Should chunk.
- ❌ `src/data/preprocessing.py:103-112` — Uses `torch.stft()` directly instead of `torchaudio.transforms.Spectrogram` (CLAUDE.md rule: "NEVER use torchaudio.functional.spectrogram — always use torchaudio.transforms.Spectrogram"; while this uses `torch.stft` not `torchaudio.functional.spectrogram`, the principle of inconsistency applies)

---

## SECTION 3 — Logic and Bug Review

### 🐛 CRITICAL — PerceptualLoss call/crash
**File**: `src/models/separator.py:179` / `src/loss/separation.py:110-118`
**Problem**: `PerceptualLoss.forward` signature is `(pred_spec, target_spec, dinov2_extractor)` but called as `self.perceptual(predicted_waveforms, target_waveforms)` with 2 args.
**Impact**: **Runtime TypeError** in Phase3 training.
**Fix**: Update call to pass `self.dinov2` or store in loss.

### 🐛 CRITICAL — DINOv2 spatial features destroyed
**File**: `src/models/separator.py:113`
**Problem**: `visual_kv.mean(dim=2)` averages 1024 patches to single 512-dim vector.
**Impact**: **All spatial visual information lost**. Cross-attention cannot localize sources.
**Fix**: Pass full [B, T_a, 1024, 512] or flatten to [B, T_a, 1024×512].

### 🐛 CRITICAL — eval_sisnri baseline is wrong
**File**: `evaluation/eval_sisnri.py:83`
**Problem**: `istft(mixture_stft, mixture_stft)` — passes same tensor twice, computing squared-magnitude not mixture.
**Impact**: SI-SNRi completely wrong.
**Fix**: Use identity mask or call `torch.istft` on complex mixture directly.

### 🐛 HIGH — DataModule/Dataset format mismatch
**File**: `src/data/datamodule.py:42-53`
**Problem**: `AudioVisualDataModule` uses `AudioVisualDataset` which returns keys `"source_stfts"`, `"source_videos"`, but `SeparatorModule.training_step` expects `"target_waveforms"`, `"video_frames"`.
**Impact**: **KeyError at runtime** during training.
**Fix**: Update DataModule to use `MixAndSepareDataset` or fix `AudioVisualDataset.__getitem__`.

### 🐛 HIGH — ImportError on `src.utils`
**File**: `src/utils/__init__.py:3`
**Problem**: `from .metrics import compute_metrics` — function doesn't exist.
**Fix**: Remove import or implement `compute_metrics()`.

### 🐛 HIGH — Dataset NameError
**File**: `src/data/dataset.py:260`
**Problem**: `wave_chunks` undefined (should be `waveforms_chunks`).
**Fix**: Rename parameter reference.

### 🐛 MEDIUM — Duplicate encoding in Phase3
**File**: `src/models/separator.py:194-227`
**Problem**: `_predict_masks()` re-runs full encoder+attention already computed in `forward()`.
**Impact**: 2× compute waste.
**Fix**: Return masks from forward or cache intermediates.

### 🐛 MEDIUM — DINOv2 batch OOM risk
**File**: `src/models/separator.py:81-84`
**Problem**: Passes B×N frames to DINOv2 all at once. For B=8, N=150, that's 1200 frames × 17 GFLOPs each.
**Fix**: Chunk into smaller batches (e.g., B×4 or B×8 at a time).

### 🐛 MEDIUM — No gradient clipping in training_step
**File**: `src/models/separator.py:144-192`
**Problem**: Spec requires `max_norm=1.0` at every step. Not implemented in code.
**Fix**: Add `clip_grad_norm_` or rely on Trainer config.

### 🐛 LOW — `PerceptualLoss` returns wrong device tensor
**File**: `src/loss/separation.py:117`
**Problem**: `return torch.tensor(0.0)` is always CPU.
**Fix**: `return torch.tensor(0.0, device=pred_spec.device)`.

### 🐛 LOW — `run_evaluation.py` empty
**File**: `scripts/run_evaluation.py`
**Problem**: Only prints "TODO".
**Fix**: Implement or remove.

### 🐛 LOW — `eval_localisation.py` can't extract attention
**File**: `evaluation/eval_localisation.py:42-49`
**Problem**: `extract_attention_weights` is no-op.
**Fix**: Register forward hooks on `nn.MultiheadAttention`.

---

## SECTION 4 — Kaggle Compatibility

- 🔴 Missing `kaggle/` directory entirely
- 🔴 Missing `configs/kaggle.yaml`
- ⚠️ `scripts/kaggle_submission.py` uses local path `cache/index.json` not `/kaggle/input/`
- ⚠️ Default `num_workers=4` in DataModule; Kaggle needs `num_workers=2`
- ⚠️ No fp16/bf16 in configs
- ⚠️ Batch sizes too large for T4 (32 for Phase1 → should be 4-8)
- ⚠️ `setup.py` doesn't pin versions (uses `>=` instead of `==`)

---

## SECTION 5 — Missing Pieces

| Missing | Referenced In | Required By |
|---------|--------------|-------------|
| `kaggle/` directory + submission notebook | User prompt | Kaggle submission |
| `configs/kaggle.yaml` | User prompt | Kaggle compat |
| PerceptualLoss implementation | `src/models/separator.py:179` | SPEC 11.3 Phase3 |
| `scripts/run_evaluation.py` | File at line 7-8 | SPEC 11.5 |
| `compute_metrics()` function | `src/utils/__init__.py:3` | Import |
| Progressive difficulty scheduler (2→3→4) | `separator.py:164-183` | SPEC 11.4 Phase3 |
| DINOv2 partial unfreezing (A/B/C experiments) | `separator.py:297` | SPEC 11.4 Phase3 |
| Validation/early stopping in training | `scripts/train.py:83-90` | Phase setup |
| Entropy loss implementation | `separator.py:229-231` | Phase2 regularization |
| Full evaluation pipeline | `scripts/run_evaluation.py` | End-to-end testing |

---

## SECTION 6 — Overall Verdict

### Issue Count

| Category | Count |
|----------|-------|
| Spec violations (divergences) | 7 |
| CLAUDE.md convention violations | 3 |
| Bugs (logic/crash) | 11 |
| Kaggle issues | 7 |
| Missing pieces | 10 |

### Readiness Verdict

🔴 **NOT READY** — Significant problems must be fixed first

The codebase has 3 **CRITICAL** bugs that would cause runtime crashes or produce completely wrong results:
1. PerceptualLoss call signature mismatch (Phase3 crash)
2. DINOv2 spatial features averaged away (defeats core architecture)
3. SI-SNRi evaluation baseline computes wrong mixture signal

### Top 10 Priority Fixes

| # | Issue | Severity | Fix Complexity |
|---|-------|----------|---------------|
| 1 | Fix DINOv2 spatial averaging in `separator.py:113` | CRITICAL | Medium |
| 2 | Fix PerceptualLoss call signature | CRITICAL | Easy |
| 3 | Fix `eval_sisnri.py` baseline ISTFT call | CRITICAL | Easy |
| 4 | Fix DataModule/Dataset format mismatch | HIGH | Medium |
| 5 | Fix `src/utils/__init__.py` ImportError | HIGH | Easy |
| 6 | Chunk DINOv2 batch processing | HIGH | Medium |
| 7 | Implement `MixAndSepareDataset` in DataModule | HIGH | Medium |
| 8 | Remove/avoid duplicate encoding in Phase3 | MEDIUM | Medium |
| 9 | Add gradient clipping to training | MEDIUM | Easy |
| 10 | Implement progressive difficulty (2→3→4 sources) | MEDIUM | Medium |
