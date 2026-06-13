# SonicSight-DINOv2 — Final Idea-by-Idea Review
**Report**: DINOv2_Separation_Report_v4.pdf
**Date**: 2026-06-12
**Total report items extracted**: 14

## Executive Summary
This is the final alignment and quality audit of the SonicSight-DINOv2 project. While the codebase has made significant progress and successfully implemented many complex components (e.g., SI-SNR PIT wrapper, CRMLoss L1 criterion, Multi-Scale STFT, and evaluation scripts), it is **NOT** ready for training.

Several CRITICAL architectural bugs remain in the core cross-modal attention module (`separator.py`) and the data pipeline (`dataset.py`). Specifically, the temporal alignment incorrectly maps frequency bins to time, the visually-attended bottleneck features are completely discarded before the decoder, the per-source visual attention averages features and performs a redundant secondary attention pass, and the cRM target computation scrambles independent sources into complex channels. These issues mathematically break the separation logic and will prevent the model from learning. The verdict is NOT READY until these are fixed.

## Part A — Idea-by-Idea Audit (by report section)
### 3. DINO Model Family
- **R-3.2-1**: DINOv2 frozen patch features yield universal visual representations without task-specific tuning.
  - **Status**: IMPLEMENTED
  - **Evidence**: `src/visual/dinov2.py:28` - `param.requires_grad = False` and `src/models/separator.py:70` applies freezing.

### 5. Proposed Source Classes
- **R-5.1-1**: Tier 1 — Human Speech (AVSpeech, VoxCeleb2).
  - **Status**: IMPLEMENTED
- **R-5.2-1**: Tier 2 — Musical Instruments (21 categories).
  - **Status**: IMPLEMENTED
- **R-5.3-1**: Tier 3 — Environmental and Urban Sounds.
  - **Status**: IMPLEMENTED
- **R-5.4-1**: Tier 4 — Animal Vocalisations.
  - **Status**: IMPLEMENTED
  - **Evidence**: Handled via data preparation and evaluation generalization targets.

### 7. Training Phase Design
- **R-7.2-1**: Phase 1 — Audio-Only Pretraining (SI-SNR, 200K steps).
  - **Status**: IMPLEMENTED
  - **Evidence**: `configs/train/phase1.yaml` and `train.py` handle this correctly.
- **R-7.2-2**: Phase 2 — Cross-Modal Warmup (visual features frozen, attention trained).
  - **Status**: IMPLEMENTED
  - **Evidence**: `configs/train/phase2.yaml` specifies `lr_fusion: 5e-4` and freezing is enforced in `separator.py:526`.
- **R-7.2-3**: Phase 3 — End-to-End Fine-Tuning.
  - **Status**: IMPLEMENTED
  - **Evidence**: `configs/train/phase3.yaml` and `train.py` configure differential learning rates correctly.
- **R-7.2-4**: Temporal alignment: spectrogram frame maps to video frame floor(t*150/601).
  - **Status**: INCORRECT
  - **Evidence**: `src/models/separator.py:186` uses `align_idx = [int(i * N_v / n_bottleneck) for i in range(n_bottleneck)]`. `n_bottleneck` is 171 (flattened 9x19 spatial grid). This aligns the 1D spatial index `h*19 + w` to the video frames, breaking temporal consistency by mapping different frequencies at the same time to different video frames.
  - **Fix**: Compute alignment based on the temporal dimension `w` (19) rather than the flattened sequence `i` (171).

### 7.3 Loss Functions
- **R-7.3-1**: SI-SNR Loss (Permutation Invariant).
  - **Status**: IMPLEMENTED
  - **Evidence**: `src/loss/separation.py:14` and `src/loss/pit_wrapper.py:11`.
- **R-7.3-2**: cRM target compression with tanh, K=10, C=0.1.
  - **Status**: PARTIAL
  - **Evidence**: `src/data/preprocessing.py:138` computes `torch.tanh(k * mag)`. The formula is implemented with K=10, but the constant C=0.1 mentioned in the report is unused.
- **R-7.3-3**: Multi-scale STFT loss.
  - **Status**: IMPLEMENTED
  - **Evidence**: `src/loss/separation.py:124`.
- **R-7.3-4**: Perceptual Loss (Cosine similarity on DINOv2 features).
  - **Status**: IMPLEMENTED
  - **Evidence**: `src/loss/separation.py:162`.

### 7.4 Evaluation Protocol
- **R-7.4-1**: SI-SNRi, SDR, WER, and Localisation evaluation.
  - **Status**: IMPLEMENTED
  - **Evidence**: `evaluation/` directory contains all scripts correctly implemented without mocked masks.

## Part B — Batch 1-4 Verification Summary
| Batch | Claimed Fix | Verified? | Evidence |
|---|---|---|---|
| Batch 1 | CRMLoss uses L1 | YES | `src/loss/separation.py:84` uses `nn.L1Loss()`. No MSE matches found. |
| Batch 1 | SI-SNR PIT / Multi-Scale | YES | `pit_wrapper.py` implements Hungarian algorithm. |
| Batch 2 | Eval scripts reconstruct mixture | YES | `eval_sisnri.py` uses `torch.complex` and `torch.istft` directly. |
| Batch 3 | config lr_fusion, grad clip, early stop | YES | `configs/train/phase2.yaml` uses `lr_fusion`. No manual clipping in codebase. |
| Batch 4 | Visual Projection / Attention | YES | `separator.py:43` uses bias=False. `cross_attention.py` uses Pre-LN. |
| Batch 4 | Per-source Visual Attention | NO | `separator.py:187` uses `.mean(dim=1)` to average visual features, violating the constraint. |

## Part C — End-to-End Shape Trace
**Phase 1 (audio-only, N=2)**
- Raw audio -> STFT: `[B, 2, 257, 601]`
- U-Net encoder -> bottleneck: `[B, 512, 9, 19]` -> flatten -> `[B, 171, 512]`
- Source queries: `[B, 2, 512]`
- Combined query: `[B, 173, 512]`
- Cross-attention (self-attn): `[B, 173, 512]`
- Source features: `[B, 2, 512]` -> expanded to `[B, 2, 512, 9, 19]` (Throws away 171 audio features)
- Decoder -> Mask: `[B, 2, 257, 601]`
- iSTFT -> Waveform: `[B, 2, 96000]`

**Phase 2/3 (visual, N=2)**
- Visual features: `[B, 2, 150, 1024, 768]` -> visual_proj -> `[B, 2, 150, 1024, 512]`
- Per-source loop cross_attn -> `[B, 2, 512]`
- Visual average: `[B, 171, 1024, 768]` -> flatten & proj -> `[B, 175104, 512]`
- Combined query: `[B, 173, 512]` attends to visual_kv `[B, 175104, 512]` -> `[B, 173, 512]`
- Source features: `[B, 2, 512]` -> expanded to `[B, 2, 512, 9, 19]` (Throws away 171 audio features)
- Decoder -> Mask: `[B, 2, 257, 601]`
- iSTFT -> Waveform: `[B, 2, 96000]`

**Phase 2/3 (visual, N=4)**
- Source queries: `[B, 4, 512]`, Combined query: `[B, 175, 512]`. All other shapes identical to N=2.

*Shape Issue*: Expanding the `[B, N, 512]` source tokens to `9x19` discards the 171 audio tokens that contain the semantic spatial/temporal resolution from the bottleneck.

## Part D — Independent Bug Hunt
| Severity | File:Line | Description & Impact | Exact Fix |
|---|---|---|---|
| CRITICAL | `src/data/dataset.py:220` | `unsqueeze(0)` on `source_stfts` causes `compute_crm_targets` to treat the source dimension as the complex channel dimension, merging source 0 as real and source 1 as imaginary. Breaks cRM targets. | Remove `.unsqueeze(0)` from `source_stfts` and `mixture_stft` calls to `compute_crm_targets`. |
| CRITICAL | `src/models/separator.py:290` | Expanding the 1D source query to `9x19` discards the 171 visually-attended bottleneck features, depriving the decoder of all deep audio semantics. | Reshape the `171` attended bottleneck tokens to `[B, N, 512, 9, 19]` for each source and feed them to the decoders. |
| CRITICAL | `src/models/separator.py:187` | Averages visual features across sources (`.mean(dim=1)`) and runs a secondary cross-attention, breaking the per-source visual isolation requirement. | Remove the secondary cross-attention. Use the outputs of the isolated per-source attention loop directly. |
| CRITICAL | `src/models/separator.py:186` | Calculates temporal alignment `align_idx` using `i * 150 / 171`, mapping the 2D flattened spatial index `h*19 + w` to video frames, destroying temporal coherence. | Compute alignment exclusively using the temporal dimension `w` (19). |
| HIGH | `evaluation/eval_wer.py:121` | Returns a WER of `0.0` when no transcript is found, artificially deflating the error rate. | Skip evaluation for clips missing a transcript. |

## Part E — Master Summary Table
| Report Section | Items | Implemented | Partial | Missing | Incorrect | Unverified |
|---|---|---|---|---|---|---|
| 3. DINO Model Family | 1 | 1 | 0 | 0 | 0 | 0 |
| 5. Source Classes | 4 | 4 | 0 | 0 | 0 | 0 |
| 7. Training Design | 4 | 3 | 0 | 0 | 1 | 0 |
| 7.3 Loss Functions | 4 | 3 | 1 | 0 | 0 | 0 |
| 7.4 Evaluation Protocol | 1 | 1 | 0 | 0 | 0 | 0 |
| **TOTAL** | **14** | **12** | **1** | **0** | **1** | **0** |

## Part F — Final Verdict
🔴 NOT READY — at least one MISSING/INCORRECT or CRITICAL/HIGH bug remains

## Part G — Prioritised Fix List
1. **CRITICAL**: `src/data/dataset.py:220` — Remove `unsqueeze(0)` around `source_stfts` and `mixture_stft` so `compute_crm_targets` correctly addresses the `2` dimension for real/imag channels.
2. **CRITICAL**: `src/models/separator.py:174-271` — Refactor cross-modal attention to isolate each source. For each source `n`, pass the `171` audio bottleneck tokens + `1` source query token to `cross_attn` against visual stream `n`. Reshape the resulting `171` audio tokens to `[B, 512, 9, 19]` and pass *that* to the decoder.
3. **CRITICAL**: `src/models/separator.py:186` — Fix temporal alignment. Map the temporal dimension `w` (0 to 18) to the 150 video frames, broadcasting the alignment across the frequency dimension `h`.
4. **HIGH**: `evaluation/eval_wer.py:121` — Add a check `if not ref: continue` to skip WER calculation when a transcript is missing instead of defaulting to `0.0`.
