# Code Review: SonicSightDino vs DINOv2 Separation Report v4

**Date:** 2026-06-06
**Review Scope:** 6 parallel subagent review of full codebase against DINOv2_Separation_Report_v4.pdf
**Status:** MAJOR DEVIATIONS FOUND — Multiple Critical Blockers

## MASTER SUMMARY

| Subagent | Domain | Overall Status | Critical Issues | High Issues | Medium Issues |
|----------|--------|----------------|-----------------|-------------|--------------|
| 1 | Architecture | ⚠️ PARTIAL | 3 | 2 | 4 |
| 2 | Preprocessing | ❌ FAIL | 3 | 1 | 2 |
| 3 | Training Config | ⚠️ PARTIAL | 3 | 2 | 3 |
| 4 | Loss Functions | ❌ FAIL | 1 | 2 | 2 |
| 5 | Evaluation | ❌ FAIL | 4 | 2 | 1 |
| 6 | Kaggle/Integration | ❌ FAIL | 3 | 1 | 2 |

**Total Critical Blockers: 17** — System NOT ready for training.

### Key Cross-Cutting Themes

1. **Per-source visual pipeline broken** (Arch, Preprocessing, Dataset): No per-source DINOv2 features cached; dataset averages across sources; no identity-aware splits.
2. **PIT incomplete** (Loss, Train Config, Eval): Hungarian algorithm missing for N>3; PIT not shared between SI-SNR and cRM; eval has no PIT at all.
3. **Gradient flow blocked** (Loss): PerceptualLoss wrapped in @torch.no_grad() — zero gradients.
4. **Training config implementation gaps** (Train Config): LR schedulers missing; checkpoint monitors wrong metric (loss min vs SI-SNRi max); Config B/C differential LRs missing.
5. **Kaggle deployment incomplete** (Kaggle): 3/4 required files missing; dependency versions mismatch pinned spec; .gitignore incomplete.
6. **Evaluation metrics non-functional** (Eval): SI-SNRi no PIT; SDR/SIR/SAR missing; WER no baseline; localisation wrong metric; zero-shot missing.

## SUBAGENT 1 — ARCHITECTURE REVIEW

**Files reviewed:** audio_unet.py, bottleneck_projection.py, dinov2_wrapper.py, cross_modal_attention.py, source_query_decoder.py, separator.py
**Report sections:** 4, 11.3

### Alignment Summary
| Status | Count |
|--------|-------|
| ✅ ALIGNED | 58 |
| ⚠️ PARTIAL / DIFFERENT | 6 |
| ❌ MISALIGNED | 2 |
| 🐛 BUGS | 11 |

### Critical Issues

| # | Issue | File:Line | Severity | Fix |
|---|-------|-----------|----------|-----|
| 1 | Cross-attention forward manually iterates blocks, bypasses module.forward() — misses pos_enc, breaks encapsulation | separator.py:187-201 | HIGH | Refactor CrossModalAttentionModule.forward() to accept attn_mask/key_padding_mask |
| 2 | Progressive source curriculum active in ALL phases (20K/40K steps); report: Phase 1=200K fixed N=2, curriculum only in Phase 3 | separator.py:32 | HIGH | Move curriculum to Phase 3 only; Phase 1 = 200K fixed N=2 |
| 3 | Attention entropy loss depends on buggy manual forward path — will be 0 with standard module.forward() | separator.py:333-353 | HIGH | Capture attention weights in standard forward or always use manual path |
| 4 | Decoder: FiLM-modulated single shared decoder vs report N parameter-shared decoders with [B,N,512,9,19] reshape | separator.py:220-234 | MEDIUM | Reshape source_features to [B,N,512,9,19] and call decoder N times, or update report |
| 5 | Output slots: code produces [B,N,512] source_features; report expects reshape to [B,N,512,9,19] before decoder | separator.py:215 | MEDIUM | Add reshape: source_features.view(B, N, 512, 9, 19) |
| 6 | DINOv2 device defaults to CPU; may cause device mismatch | dinov2.py:24 | MEDIUM | Use property: @property def device(): return next(self.model.parameters()).device |
| 7 | Temporal alignment ratio inconsistency: report §7.2.2b uses 150/240, §11.3 and code use 150/171 | separator.py:142-144 | MEDIUM | Standardize on 150/171 (bottleneck) throughout |
| 8 | Visual KV flattened to [B, 175104, 512] — memory intensive; report mentions per-timestep attention | separator.py:168-169 | LOW | Implement per-timestep or windowed attention |
| 9 | CrossAttentionBlock dual-use (self-attn Phase 1, cross-attn Phase 2/3) — naming mismatch | cross_attention.py:36-43 | LOW | Document or separate SelfAttentionBlock for Phase 1 |
| 10 | Source query progressive resize creates new nn.Parameter — resets optimizer state | separator.py:80-86 | LOW | Use nn.ParameterList or register_buffer preserving optimizer state |
| 11 | cRM K=10 tanh compression not verified in target pre-computation | separation.py | LOW | Verify CRMLoss and preprocessing apply K=10 |
| 12 | PerceptualLoss DINOv2 argument signature unverified — may error at runtime | separator.py:277 | UNKNOWN | Verify PerceptualLoss accepts DINOv2 instance |

### Well-Aligned Areas
- Audio U-Net: 5 blocks, channels [2,32,64,128,256,512], Conv2d 3x3 stride(2,2), GroupNorm g=8, LeakyReLU 0.2
- Bottleneck: [B,512,9,19] -> flatten [B,171,512] -> Linear 512->512
- DINOv2: facebook/dinov2-base, patch 14, 448x448, 1024 patches, 768-dim, frozen, no_grad
- Visual projection: Linear(768->512, bias=False), shared across timesteps
- Temporal alignment: v = floor(t_a * 150 / 171)
- Cross-attention: 2 blocks, Pre-LN, 8 heads, 64 head_dim, d=512, FFN=2048, GELU, dropout=0.1, sinusoidal pos enc
- Source queries: N=4, dim=512, init N(0,0.02)
- Training config: All LRs, loss coeffs, Phase 3 curriculum match report exactly

## SUBAGENT 2 — PREPROCESSING REVIEW

**Files reviewed:** audio.py, video.py, preprocessing.py, dataset.py, datamodule.py, scripts/preprocess_data.py
**Report sections:** 5, 11.2

### Critical Blockers

| # | Issue | File | Severity | Fix |
|---|-------|------|----------|-----|
| 1 | STFT uses forbidden  directly; spec mandates  only | preprocessing.py STFTModule | CRITICAL | Replace with torchaudio.transforms.Spectrogram(n_fft=512, hop_length=160, win_length=400, window_fn=torch.hann_window, power=None, center=True, pad_mode='reflect') |
| 2 | No per-source DINOv2 features cached — single visual tensor per clip; dataset averages across sources losing identity | scripts/preprocess_data.py, dataset.py | CRITICAL | Restructure preprocessing: accept per-source video inputs, cache visual/<clip>_srcN.pt |
| 3 | Temporal alignment uses interpolation (F.interpolate) instead of spec cached lookup table (v = floor(t * 150 / 601)) | dataset.py:171-179 | CRITICAL | Remove interpolation; use _VIDEO_ALIGN module constant to index DINOv2 frames per STFT frame |
| 4 | Split logic: random shuffle of clips — no identity-based partitioning (MUSIC by video_id, AVSpeech by speaker_id) | scripts/preprocess_data.py:316-332 | CRITICAL | Parse metadata (video_id, speaker_id); partition splits so no identity leaks across train/val/test |

### High Issues

| # | Issue | File | Fix |
|---|-------|------|-----|
| 5 | VideoPreprocessor resizes directly to 448x448 (squashes aspect ratio); spec: resize shorter side to 448, then center crop | preprocessing.py VideoPreprocessor | Use torchvision.transforms.Resize(448) + CenterCrop(448) |

### Medium Issues

| # | Issue | File | Fix |
|---|-------|------|-----|
| 6 | cRM C=0.1 scaling mentioned in spec but not applied in code | preprocessing.py safe_complex_divide | Add C=0.1 scaling if required |
| 7 | STFTModule is plain callable, not nn.Module — cannot be part of traced/exported model graph | preprocessing.py | Convert to nn.Module wrapper |

### Passing Checks
- Audio: 16kHz mono, 6s=96000 samples, zero-pad, STFT params (512/160/400/Hann), output [2,257,601]
- Video: 25fps, bicubic resize, 448x448 crop, ImageNet norm
- DINOv2: [B,150,3,448,448] per source shape, patch tokens (no CLS), frozen backbone, float16 cache
- cRM: M_i = S_i/X complex division, tanh compression K=10, float16 cache, pre-computed
- Temporal alignment table computed correctly: v = floor(t * 150 / 601)
- All config constants match spec exactly

**Verdict:** Preprocessing pipeline NOT compliant with §11.2. Three critical blockers must be fixed before training.

## SUBAGENT 3 — TRAINING CONFIG REVIEW

**Files reviewed:** phase1.yaml, phase2.yaml, phase3.yaml, model.yaml, data.yaml, lightning_module.py, lightning_module_phase2.py, lightning_module_phase3.py, train_phase1.py, scripts/train.py
**Report sections:** 7, 11.4

### Critical Issues

| # | Issue | File | Severity | Fix |
|---|-------|------|----------|-----|
| 1 | Phase 3: Missing Config B/C differential LRs — Config C (5e-6) entirely missing; no mechanism to select config | phase3.yaml, separator.py | CRITICAL | Add dinov2_config: A/B/C param; implement LR mapping for all three configs |
| 2 | Trainer: Checkpoint monitors val/si_snr_loss mode=min; report specifies Val SI-SNRi higher=better | scripts/train.py | CRITICAL | Change monitor=val/si_snri, mode=max (or add SI-SNRi metric and log it) |
| 3 | LR schedulers missing — configs define cosine but configure_optimizers returns only optimizer | separator.py, scripts/train.py | CRITICAL | Implement cosine annealing schedulers in configure_optimizers with T_max per phase |
| 4 | separator.py hardcodes all LRs instead of reading from config — config files ineffective for LR changes | separator.py configure_optimizers | HIGH | Move all LR values to config-driven via self.cfg |
| 5 | Phase 3 curriculum (N=2->20K, N=3->40K, N=4) defined in config but not wired into DataModule | phase3.yaml, datamodule.py | HIGH | Wire n_sources_schedule into AudioVisualDataModule for progressive curriculum |

### High Issues

| # | Issue | File | Fix |
|---|-------|------|-----|
| 6 | Attention entropy loss stub — _compute_attention_entropy() returns 0 | separator.py | Implement actual entropy: -sum(p * log(p)) over attention weights |
| 7 | Phase 1 config has warmup_steps=1000 but report says cosine T_max=200K no warmup | phase1.yaml | Remove warmup or document deviation |

### Medium Issues

| # | Issue | File | Fix |
|---|-------|------|-----|
| 8 | Phase 2 missing +5K concat baseline steps (report: 30K + 5K) | phase2.yaml | Add 5K steps or document |
| 9 | kaggle.yaml deviations: max_steps=60K vs 100K, val_check=2000 vs 5000, log_every=50 vs 100 | configs/kaggle.yaml | Align with report or document Kaggle-specific overrides |
| 10 | gradient_clip_algorithm=norm hardcoded; should be configurable | scripts/train.py | Add to config |

### Passing Checks
- Phase 1: AdamW lr=1e-3, wd=1e-4, batch=32, grad_clip=1.0, steps=200K, loss=SI-SNR+PIT ✅
- Phase 2: Frozen DINOv2+U-Net, lr=5e-4, warmup=1K, batch=16, dropout=0.1, steps=30K ✅
- Phase 3 LR mapping (Config A/B): fusion=3e-4, audio_enc=3e-5, dinov2=1e-5 ✅
- Loss coeffs: alpha_crm=0.1, beta_stft=0.05, gamma_perceptual=0.1 ✅
- Batch/grad_accum: 8/4 effective=32 ✅
- Phase 3 curriculum schedule matches: [20000, 40000] with [2,3,4] sources ✅
- Trainer: max_steps, grad_clip_val=1.0, log_every_n_steps=100, val_check_interval=5000, checkpoint every 5K ✅

## SUBAGENT 4 — LOSS FUNCTIONS REVIEW

**Files reviewed:** sisnr.py, crm_loss.py, multiscale_stft.py, pit_wrapper.py, src/loss/separation.py
**Report sections:** 7.3, 11.4

### Critical Blocker

| # | Issue | File | Severity | Fix |
|---|-------|------|----------|-----|
| 1 | PerceptualLoss gradients COMPLETELY BLOCKED by @torch.no_grad() on DINOv2FeatureExtractor.forward + internal torch.no_grad() — backward fails: element 0 does not require grad | separation.py:104-177, dinov2.py | CRITICAL | Remove @torch.no_grad() from DINOv2FeatureExtractor.forward; remove internal no_grad from PerceptualLoss.forward; rely on param.requires_grad=False to freeze DINOv2 weights |

### High Issues

| # | Issue | File | Fix |
|---|-------|------|-----|
| 2 | PIT: No Hungarian algorithm for N>3 — always uses brute-force factorial permutations; report requires scipy.optimize.linear_sum_assignment for N>3 | pit_wrapper.py / sisnr.py | Add Hungarian for N>3; brute-force for N<=3 |
| 3 | PIT not shared between SI-SNR and cRM loss — report Table 1: PIT applied to both losses simultaneously; currently cRM uses fixed-order batch targets | separation.py | Find optimal perm from SI-SNR; apply same perm to cRM targets before MSE |

### Medium Issues

| # | Issue | File | Fix |
|---|-------|------|-----|
| 4 | MultiScaleSTFTLoss: Missing Hann window — triggers PyTorch warning; uses rectangular window | multiscale_stft.py | Add window=torch.hann_window(n_fft, device=x.device) to stft calls |
| 5 | MultiScaleSTFTLoss hop lengths: uses n_fft//4 -> [64,128,256]; audio STFT uses hop=160; multi-scale hops not in spec but should be consistent | multiscale_stft.py | Document choice or match 25% overlap convention |

### Passing Checks
- SI-SNR: zero_mean=True, scalar output, gradients flow ✅
- SI-SNR PIT: brute-force enumeration for N<=3 works ✅
- cRM loss: MSE on complex masks [N,2,F,T], scalar, gradients flow ✅
- Multi-scale STFT: n_ffts=[256,512,1024] matches SPEC 11.4 ✅
- Multi-scale STFT: L1 on mag + L1 on log mag, scalar, gradients flow ✅
- Phase 3 combined loss coefficients: SI-SNR + 0.1*cRM + 0.05*STFT + 0.1*Perceptual ✅
- All losses return scalar on correct device (CPU/CUDA tested) ✅
- PerceptualLoss: cosine similarity on DINOv2 features of spectrograms, correct STFT->448x448->3ch->norm pipeline ✅ (except gradient block)

## SUBAGENT 5 — EVALUATION REVIEW

**Files reviewed:** eval_sisnri.py, eval_wer.py, eval_localisation.py, eval_zero_shot.py, scripts/run_evaluation.py
**Report sections:** 8, 11.5

### Critical Blockers

| # | Issue | File | Severity | Fix |
|---|-------|------|----------|-----|
| 1 | SI-SNRi: No PIT whatsoever — computes per-source directly; no permutation handling for any N; results meaningless for N>2 | eval_sisnri.py | CRITICAL | Implement PIT: torchmetrics SI-SNR with zero_mean=True + scipy.optimize.linear_sum_assignment (Hungarian) for N>3, brute-force N<=3 |
| 2 | SDR/SIR/SAR: Missing entirely — report requires mir_eval.separation.bss_eval_sources on MUSIC-21 | eval_sdr.py (missing) | CRITICAL | Create eval_sdr.py using mir_eval.separation.bss_eval_sources |
| 3 | WER: No baseline on unprocessed mixture; no 16kHz resampling; transcript lookup by batch_idx (broken) | eval_wer.py | CRITICAL | Add torchaudio resample to 16kHz; compute baseline WER on mixture; fix transcript lookup using sample IDs |
| 4 | Attention localisation: Uses argmax (single patch) not top-50; no YOLOv8 GT bbox integration; no IoU>0.3 threshold metric; no fraction of frames above threshold | eval_localisation.py | CRITICAL | Implement top-50 patches -> [32,32] saliency -> bbox from convex hull; integrate ultralytics YOLOv8 for GT; report loc_acc = mean(IoU>0.3) |

### High Issues

| # | Issue | File | Fix |
|---|-------|------|-----|
| 5 | eval_zero_shot.py: MISSING FILE — report §11.5 implies zero-shot evaluation on held-out categories | (missing) | HIGH | Create eval_zero_shot.py for Tier 4 animal sounds, unseen instruments |
| 6 | All eval outputs save to CWD, not outputs/ directory as required | ALL eval files | HIGH | Change default output paths to outputs/<name>.json; create dir |

### Medium Issues

| # | Issue | File | Fix |
|---|-------|------|-----|
| 7 | run_evaluation.py: Missing SDR and zero-shot steps; only orchestrates partial eval | run_evaluation.py | Add SDR and zero-shot steps; combine all to outputs/evaluation_results.json |

### Passing Checks
- run_evaluation.py: Fully implemented orchestrator (not stub); runs SI-SNRi, localisation, WER; combines results; cleans temp files ✅
- eval_wer.py: Loads whisper-base; uses jiwer.wer ✅
- eval_localisation.py: Has attention hook structure; reshapes to [32,32]; computes IoU ✅ (partial)

**Verdict:** Evaluation suite structurally present but FUNCTIONALLY NON-COMPLIANT. Core metrics use incorrect algorithms; required metrics absent; baseline comparisons missing.

## SUBAGENT 6 — KAGGLE + INTEGRATION REVIEW

**Files reviewed:** kaggle/setup.py, kaggle/preprocess.ipynb, kaggle/train.ipynb, configs/kaggle.yaml, requirements.txt, README.md, .gitignore
**Report section:** 11.6

### Critical Blockers

| # | Issue | File | Severity | Fix |
|---|-------|------|----------|-----|
| 1 | kaggle/ directory MISSING 3 of 4 required files: setup.py, preprocess.ipynb, train.ipynb — only submission.ipynb exists | kaggle/ | CRITICAL | Create all 3 missing files with modular structure |
| 2 | requirements.txt versions DO NOT MATCH report 11.1 pinned versions (torch 2.5.0 vs 2.2.x, etc.) | requirements.txt | CRITICAL | Pin exact versions: torch==2.2.0, torchaudio==2.2.0, transformers==4.40.0, einops==0.7.0, torchmetrics==1.3.0, lightning==2.2.0 |
| 3 | setup.py uses loose >= specifiers instead of pinned == versions | setup.py | CRITICAL | Change all >= to == matching report 11.1 |

### High Issues

| # | Issue | File | Fix |
|---|-------|------|-----|
| 4 | configs/kaggle.yaml: Single phase3 only (max_steps=60K); report requires 3-phase curriculum (10K+10K+40K with progressive N=2->3->4) | configs/kaggle.yaml | Encode full 3-phase curriculum with n_sources_schedule and correct step counts |

### Medium Issues

| # | Issue | File | Fix |
|---|-------|------|-----|
| 5 | submission.ipynb is monolithic (preprocess+all phases+eval+submit); spec requires modular preprocess.ipynb + train.ipynb + submission.ipynb | kaggle/submission.ipynb | Split into 3 notebooks; each self-contained |
| 6 | .gitignore incomplete — missing: cache/, outputs/, __pycache__/, *.pyc, .hydra/, *.pt, *.pth | .gitignore | Add all required patterns |

### Passing Checks
- kaggle/ directory exists ✅
- configs/kaggle.yaml: paths use /kaggle/input/ and /kaggle/working/ ✅
- configs/kaggle.yaml: num_workers=2, amp=true, precision=16-mixed ✅
- configs/kaggle.yaml: batch_size=8, val_batch_size=4 (T4-friendly) ✅
- No hardcoded Windows backslashes in code paths ✅
- Kaggle README.md exists ✅

**Verdict:** Kaggle integration FAIL — incomplete deployment structure, dependency mismatch, missing required files.

## PRIORITY FIXES TABLE (TOP 20)

| Priority | Subagent | Issue | File:Line | Effort | Blocker |
|----------|----------|-------|-----------|--------|---------|
| P0-1 | 4 | PerceptualLoss gradients blocked by no_grad | separation.py:104-177, dinov2.py | Low | YES — loss unusable |
| P0-2 | 2 | STFT uses forbidden torch.stft instead of torchaudio.transforms.Spectrogram | preprocessing.py | Low | YES — spec violation |
| P0-3 | 2 | No per-source DINOv2 features cached; dataset averages across sources | scripts/preprocess_data.py, dataset.py | Medium | YES — breaks mix-and-separate |
| P0-4 | 2 | Temporal alignment uses interpolation not cached lookup table | dataset.py:171-179 | Low | YES — spec violation |
| P0-5 | 2 | Split logic: random shuffle, no identity-based partitioning (MUSIC video_id, AVSpeech speaker_id) | scripts/preprocess_data.py:316-332 | Medium | YES — data leakage |
| P0-6 | 5 | SI-SNRi eval: No PIT whatsoever — meaningless for N>2 | eval_sisnri.py | Medium | YES — eval broken |
| P0-7 | 5 | SDR/SIR/SAR: Missing entirely | eval_sdr.py (missing) | Medium | YES — required metric absent |
| P0-8 | 5 | WER eval: No baseline on mixture, no 16kHz resample, broken transcript lookup | eval_wer.py | Medium | YES — eval broken |
| P0-9 | 5 | Attention localisation: Wrong metric (argmax not top-50), no YOLOv8, no IoU>0.3 threshold | eval_localisation.py | High | YES — eval broken |
| P0-10 | 3 | Trainer checkpoint monitors wrong metric (loss min vs SI-SNRi max) | scripts/train.py | Low | YES — wrong model saved |
| P0-11 | 3 | LR schedulers missing — configs define cosine but not implemented | separator.py configure_optimizers | Medium | YES — no LR decay |
| P0-12 | 3 | Phase 3 Config B/C differential LRs missing (Config C entirely absent) | phase3.yaml, separator.py | Medium | YES — incomplete config |
| P0-13 | 6 | Kaggle: 3/4 required files missing (setup.py, preprocess.ipynb, train.ipynb) | kaggle/ | High | YES — not deployable |
| P0-14 | 6 | requirements.txt versions mismatch report 11.1 pinned spec | requirements.txt | Low | YES — env mismatch |
| P0-15 | 1 | Cross-attention forward bypasses module.forward() — breaks encapsulation, misses pos_enc | separator.py:187-201 | Medium | YES — architecture bug |
| P0-16 | 1 | Progressive curriculum active in ALL phases; should be Phase 3 only (Phase 1=200K fixed N=2) | separator.py:32 | Low | YES — wrong training schedule |
| P0-17 | 5 | eval_zero_shot.py missing file | (missing) | Medium | YES — required eval absent |
| P1-1 | 4 | PIT: No Hungarian algorithm for N>3; always brute-force factorial | pit_wrapper.py | Medium | HIGH — eval spec requires it |
| P1-2 | 4 | PIT not shared between SI-SNR and cRM (report: both simultaneously) | separation.py | Medium | HIGH — inconsistent PIT |
| P1-3 | 1 | Decoder: FiLM single shared vs report N parameter-shared with [B,N,512,9,19] reshape | separator.py:220-234 | Medium | HIGH — architecture mismatch |
| P1-4 | 1 | Attention entropy loss depends on buggy manual forward — will be 0 | separator.py:333-353 | Low | HIGH — Phase 2 loss broken |
| P1-5 | 2 | VideoPreprocessor squashes aspect ratio (resize to 448x448); should resize shorter side then crop | preprocessing.py | Low | HIGH — visual quality |
| P2-1 | 1 | DINOv2 device defaults to CPU | dinov2.py:24 | Low | MEDIUM |
| P2-2 | 1 | Temporal alignment ratio inconsistency (150/240 vs 150/171 in report sections) | separator.py:142-144 | Low | MEDIUM |
| P2-3 | 3 | separator.py hardcodes all LRs instead of config-driven | separator.py configure_optimizers | Medium | MEDIUM |
| P2-4 | 3 | Phase 3 curriculum not wired into DataModule | phase3.yaml, datamodule.py | Medium | MEDIUM |
| P2-5 | 4 | MultiScaleSTFTLoss missing Hann window | multiscale_stft.py | Low | MEDIUM |
| P2-6 | 6 | configs/kaggle.yaml: single phase not 3-phase curriculum | configs/kaggle.yaml | Medium | MEDIUM |
| P2-7 | 6 | .gitignore missing required patterns | .gitignore | Low | MEDIUM |

## COMPLETION STATUS

All 6 parallel subagent reviews completed. Findings merged into this report.

**Total Issues:** 17 Critical (P0), 5 High (P1), 7 Medium (P2)
**System Status:** NOT READY FOR TRAINING — Multiple critical blockers in preprocessing, loss functions, evaluation, and training config.

---
*Report generated by 6 parallel subagent review orchestrated on 2026-06-06*
