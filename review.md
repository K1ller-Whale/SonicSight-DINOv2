# SonicSight-DINOv2 — Code Review

## Executive Summary

The repository has a solid high-level structure and the local pytest suite currently passes (`55 passed, 5 warnings`), but the passing tests do not exercise the real cached-data → model visual path, Lightning checkpoint resume path, progressive-source optimizer behavior, or deterministic evaluation behavior. The most severe issue is a critical visual pipeline contract mismatch: preprocessing caches DINOv2 features, the dataset returns those features under `video_frames`, and `SeparatorModule.forward()` interprets them as raw RGB images and sends them back through DINOv2.

Several training-control bugs will either crash training or silently train the wrong parameters: Lightning 2.x checkpoint resume uses a removed `Trainer` argument, gradient accumulation is configured but ignored, phase-1/2 curricula can change dataset source count independently of the model, and progressive `source_queries` replacement leaves the optimizer pointing at stale parameters. The code also contains duplicated data modules/loss wrappers/STFT wrappers, non-deterministic validation/test sampling, stub augmentation, and evaluation scripts that reconstruct the mixture with a non-identity complex mask.

## Test Run Summary

- **Dependency install**: `C:/Users/H/AppData/Local/Programs/Python/Python312/python.exe -m pip install -r requirements.txt` failed while building/installing `scipy==1.11.0`; no Python 3.12 wheel is available for that pin, so pip attempted a source build and then failed with PyPI DNS/read-timeout errors. `requirements.txt:2` says Python 3.10+, but the prompt/repo target is Python 3.12 and `scipy==1.11.0` is not a good Python 3.12 pin.
- **Pytest command**: `C:/Users/H/AppData/Local/Programs/Python/Python312/python.exe -m pytest tests/ -v --tb=short 2>&1 | Tee-Object -FilePath test_output.txt`
- **Result**: `55 passed, 5 warnings in 157.62s`.
- **Warnings**: `requests` dependency mismatch, torchaudio `return_complex` deprecation, and Lightning `self.log()` warnings because tests call `training_step()` without a `Trainer`.
- **Coverage gap**: tests use raw-shaped dummy video (`tests/models/test_separator.py:63`) and mock DINO features (`tests/scripts/test_preprocess_data.py:20`) separately, so they miss the real cached DINO feature tensor being fed into the model as if it were raw video.

## Shape Invariant Trace

- **Audio path is mostly consistent**: `[B, 2, 257, 601]` enters `AudioUNetEncoder`; `tests/models/test_separator.py:184` confirms bottleneck `[B, 512, 9, 19]`. `SeparatorModule.forward()` flattens it to `[B, 171, 512]` at `src/models/separator.py:150` and projects it at `src/models/separator.py:153`.
- **Decoder output shape is consistent**: the decoder crops to `target_shape` at `src/audio/unet.py:107`, so masks return `[B, 2, 257, 601]`; `ISTFTModule` reconstructs fixed `[B, 96000]` at `src/audio/spectrogram.py:62`.
- **Visual raw-frame path is internally shaped**: `forward()` expects `[B, N_frames, 3, H, W]` (`src/models/separator.py:138`), reshapes to `(B*N_frames, 3, H, W)` (`src/models/separator.py:164`), DINO returns `[B*N_frames, 1024, 768]`, then `visual_proj` maps to 512.
- **Actual dataset path is inconsistent**: preprocessing saves visual cache tensors as `[V, 1024, 768]` (`scripts/preprocess_data.py:280`, `scripts/preprocess_data.py:409`), `MixAndSepareDataset` aligns/stacks them as `[N_sources, 601, 1024, 768]` (`src/data/dataset.py:186`, `src/data/dataset.py:192`), and dataloader collation makes `batch["video_frames"] == [B, N_sources, 601, 1024, 768]`. `forward()` then reads `N_v = video_frames.shape[1]` and treats dimension 2 (`601`) as RGB channels. With the real DINO module this is invalid input; with a permissive mock it silently produces garbage.
- **Attention weights shape differs from comments**: the hook receives `nn.MultiheadAttention` default averaged weights `[B, T_q, T_kv]`, not `[B, n_heads, T_q, T_kv]` as stated at `src/models/separator.py:412`.

## Critical Bugs (blockers for training)

### BUG-01: Cached DINO Features Are Reprocessed As Raw Video
- **File(s)**: `scripts/preprocess_data.py:280`, `scripts/preprocess_data.py:409`, `src/data/dataset.py:186`, `src/data/dataset.py:192`, `src/models/separator.py:138`, `src/models/separator.py:164`, `src/visual/dinov2.py:42`
- **Description**: preprocessing stores DINOv2 features `[V, 1024, 768]`; the dataset returns `[N_sources, 601, 1024, 768]` under `video_frames`; the model expects `[B, N_frames, 3, H, W]` and reshapes that tensor into images for DINO.
- **Impact**: phase 2/3 training with cached data will either crash inside DINO because channels are `601` instead of `3`, or produce nonsensical features if a mock/permissive module is used.
- **Fix**:
  ```python
  # Rename dataset output and pass features explicitly.
  # src/data/dataset.py
  output["visual_features"] = torch.stack(aligned)  # [N, T, 1024, 768]

  # src/models/separator.py
  def forward(self, mixture_stft, video_frames=None, visual_features=None):
      ...
      if visual_features is not None and self.phase != "phase1":
          # Reduce source dimension or attend per-source explicitly.
          # Minimal compatible fix: average selected sources.
          if visual_features.dim() == 5:  # [B, N, T, P, 768]
              visual_features = visual_features.mean(dim=1)
          visual_kv = visual_features.to(bottleneck_flat.device)  # [B, T, P, 768]
          visual_kv = self.visual_proj(visual_kv)
          visual_kv_flat = rearrange(visual_kv, "B T P D -> B (T P) D")
          combined_query = torch.cat([source_q, bottleneck_flat], dim=1)
          source_features = self.cross_attn(combined_query, visual_kv_flat)[:, :self.n_sources]
      elif video_frames is not None and self.phase != "phase1":
          # keep raw-frame extraction path only for true [B, T, 3, H, W]
          if video_frames.dim() != 5 or video_frames.shape[2] != 3:
              raise ValueError("video_frames must be raw [B,T,3,H,W]; use visual_features for cached DINO tokens")
  ```

### BUG-02: Phase-1/2 Dataset Curriculum Changes Source Count Behind The Model
- **File(s)**: `scripts/train.py:83`, `scripts/train.py:96`, `scripts/train.py:113`, `src/data/datamodule.py:37`, `src/data/datamodule.py:42`, `src/models/separator.py:114`
- **Description**: `CurriculumCallback` is always added. Phase 1/2 configs do not define `n_sources_schedule`, so `AudioVisualDataModule` falls back to `[(0,2),(20000,3),(40000,4)]`. The model only updates progressive sources in phase 3.
- **Impact**: after 20k steps in phase 1/2, batches can contain 3 targets while the model still predicts 2 sources, breaking PIT/loss shape assumptions.
- **Fix**:
  ```python
  # scripts/train.py
  callbacks = [LearningRateMonitor(...), ModelCheckpoint(...)]
  if phase == "phase3":
      callbacks.append(CurriculumCallback(datamodule))

  curriculum_schedule = []
  if phase == "phase3":
      for item in cfg.train.get("n_sources_schedule", []):
          curriculum_schedule.append((item.step, item.n))
  ```
  Also change the datamodule default to no curriculum unless explicitly supplied:
  ```python
  self.curriculum_schedule = curriculum_schedule or [(0, n_sources)]
  ```

### BUG-03: Progressive `source_queries` Replacement Leaves Optimizer Stale
- **File(s)**: `src/models/separator.py:124`, `src/models/separator.py:129`, `src/models/separator.py:451`, `src/models/separator.py:521`
- **Description**: `_update_progressive_sources()` assigns a new `nn.Parameter` to `self.source_queries` after the optimizer was created. The optimizer param group still contains the old parameter.
- **Impact**: new source query rows for 3/4-source phase 3 are never optimized.
- **Fix**:
  ```python
  # Allocate max queries once and slice instead of replacing the Parameter.
  self.max_sources = max(cfg.get("train", {}).get("n_sources_schedule", [{"n": self.n_sources}]), key=lambda x: x["n"])["n"]
  self.source_queries = nn.Parameter(torch.randn(self.max_sources, 512) * 0.02)

  # forward()
  source_q = self.source_queries[:self.n_sources].unsqueeze(0).expand(B, -1, -1)

  # _update_progressive_sources()
  if new_n_sources != self.n_sources:
      self.n_sources = new_n_sources
      return True
  ```

### BUG-04: Lightning 2.x Resume Uses Removed `Trainer` Constructor Argument
- **File(s)**: `requirements.txt:15`, `scripts/train.py:156`, `scripts/train.py:158`, `scripts/train.py:168`
- **Description**: `lightning==2.2.0` is pinned, but `resume_from_checkpoint` is set in `trainer_args`. Lightning 2.x expects `ckpt_path` in `trainer.fit()`.
- **Impact**: phase 2/3 resume commands fail before training starts.
- **Fix**:
  ```python
  resume_ckpt = cfg.train.get("resume_from_checkpoint")
  trainer = pl.Trainer(**trainer_args)
  trainer.fit(model, datamodule=datamodule, ckpt_path=resume_ckpt if resume_ckpt else None)
  ```

### BUG-05: Phase 3 Ignores Gradient Accumulation Config
- **File(s)**: `configs/train/phase3.yaml:9`, `configs/train/phase3_config_a.yaml:9`, `configs/train/phase3_config_b.yaml:9`, `configs/train/phase3_config_c.yaml:9`, `scripts/train.py:137`
- **Description**: configs set `gradient_accumulation_steps: 4`, but `Trainer` never receives `accumulate_grad_batches`.
- **Impact**: effective batch size is 4x smaller than intended and training dynamics/memory assumptions differ from the spec.
- **Fix**:
  ```python
  trainer_args = {
      ...
      "accumulate_grad_batches": cfg.train.get("gradient_accumulation_steps", 1),
  }
  ```

### BUG-06: Phase 2 Optimizer Omits `bottleneck_proj`
- **File(s)**: `src/models/separator.py:44`, `src/models/separator.py:480`, `src/models/separator.py:482`
- **Description**: phase 2 trains cross-attention warmup but excludes `bottleneck_proj`, the audio bottleneck projection that feeds attention.
- **Impact**: the new projection layer receives gradients but is not updated, reducing or destabilizing warmup.
- **Fix**:
  ```python
  trainable_params = (
      list(self.bottleneck_proj.parameters()) +
      list(self.cross_attn.parameters()) +
      list(self.visual_proj.parameters()) +
      [self.source_queries]
  )
  ```

### BUG-07: Phase 3 Optimizer Omits Trainable Parameters
- **File(s)**: `src/models/separator.py:515`, `src/models/separator.py:516`, `src/models/separator.py:518`
- **Description**: phase 3 groups include cross-attention, visual projection, source queries, decoder, last two encoder blocks, and DINO. They omit trainable `bottleneck_proj` and the first three encoder blocks still have `requires_grad=True` but are not in the optimizer. `istft` has no parameters, so it is not a real omission.
- **Impact**: trainable tensors accumulate gradients but never update; the intended differential-LR policy is ambiguous.
- **Fix**:
  ```python
  for p in self.audio_unet.encoder.blocks[:-2].parameters():
      p.requires_grad_(False)
  param_groups = [
      {"params": list(self.cross_attn.parameters()) + list(self.visual_proj.parameters()) +
                 list(self.bottleneck_proj.parameters()) + [self.source_queries],
       "lr": lr_fusion},
      {"params": self.audio_unet.decoder.parameters(), "lr": lr_fusion},
      {"params": audio_enc_params, "lr": lr_audio_enc},
  ]
  if dinov2_params:
      param_groups.append({"params": dinov2_params, "lr": lr_dinov2})
  ```

## Architectural Issues (won't crash but produce wrong results)

### ARCH-01: Phase 1 Still Uses Cross-Attention Instead Of Pure Audio U-Net
- **File(s)**: `src/models/separator.py:23`, `src/models/separator.py:246`, `src/models/separator.py:249`
- **Description**: the phase table says "Audio U-Net only", but phase 1 feeds learned source queries through `cross_attn` using the audio bottleneck as key/value.
- **Impact**: phase 1 is not actually audio-U-Net-only pretraining; cross-attention/source-query behavior is entangled from the start.
- **Fix**: either update the spec/config to say phase 1 trains audio self-attention, or bypass fusion:
  ```python
  if self.phase == "phase1":
      # Decode from bottleneck for each source, or add an audio-only source head.
      source_features = self.audio_source_head(bottleneck_flat).view(B, self.n_sources, 512)
  ```

### ARCH-02: Dataset Returns No `clip_id`, Breaking Evaluation Metadata Lookups
- **File(s)**: `src/data/dataset.py:157`, `evaluation/eval_wer.py:93`, `evaluation/eval_localisation.py:205`, `scripts/kaggle_submission.py:57`
- **Description**: evaluators expect `batch["clip_id"]`, but `MixAndSepareDataset.__getitem__()` does not include it.
- **Impact**: WER transcript lookup, localization boxes, zero-shot filtering, and Kaggle output IDs fall back to batch indices and do not match cached metadata.
- **Fix**:
  ```python
  output["clip_ids"] = clip_ids
  output["clip_id"] = clip_ids[0] if self.n_sources == 1 else "+".join(clip_ids)
  ```
  For synthetic mixtures, evaluation metadata should be mixture-aware rather than single-clip-only.

### ARCH-03: Zero-Shot Evaluation Assumes A Different `index.json` Schema
- **File(s)**: `scripts/preprocess_data.py:415`, `evaluation/eval_zero_shot.py:151`, `evaluation/eval_zero_shot.py:155`, `evaluation/eval_zero_shot.py:158`
- **Description**: preprocessing writes a dict keyed by clip ID; zero-shot code expects `index["test"]` to be a list of entries with `category` and `clip_id`.
- **Impact**: zero-shot evaluation will produce no clips or crash depending on the JSON shape.
- **Fix**:
  ```python
  for clip_id, entry in index.items():
      if entry.get("split") == "test" and entry.get("category") in categories:
          clip_ids.append(clip_id)
  ```
  Also add `category` to preprocessing metadata if category splits are required.

### ARCH-04: Cached Visual Features Are Duplicated Per Source Even For Single-Video Clips
- **File(s)**: `scripts/preprocess_data.py:281`, `scripts/preprocess_data.py:282`, `scripts/preprocess_data.py:408`
- **Description**: the same clip-level video features are saved once per source (`clip_src0.pt`, `clip_src1.pt`, ...).
- **Impact**: cache storage is multiplied by `n_sources` without adding information; source-specific visual guidance is not represented.
- **Fix**: save a single `visual_path` per clip unless source-specific crops/tracks exist:
  ```python
  visual_path = visual_dir / f"{clip_name}.pt"
  torch.save(visual_features, str(visual_path))
  index[clip_name]["visual_path"] = str(visual_path)
  ```

### ARCH-05: Validation/Test Dataset Is Stochastic
- **File(s)**: `src/data/dataset.py:128`, `src/data/dataset.py:133`, `src/data/datamodule.py:91`, `src/data/datamodule.py:101`
- **Description**: `MixAndSepareDataset.__getitem__()` ignores `idx` and samples random clip IDs for all splits.
- **Impact**: validation/test metrics are non-reproducible and cannot be compared across checkpoints.
- **Fix**:
  ```python
  if self.split == "train":
      clip_ids = random.sample(self.clips, self.n_sources)
  else:
      start = idx % len(self.clips)
      clip_ids = [self.clips[(start + k) % len(self.clips)] for k in range(self.n_sources)]
  ```

### ARCH-06: Phase 2 "Frozen U-Net" Is Implemented Only By Optimizer Exclusion
- **File(s)**: `src/models/separator.py:480`, `src/models/separator.py:482`
- **Description**: U-Net parameters retain `requires_grad=True` in phase 2 but are excluded from the optimizer.
- **Impact**: gradients are computed and memory is wasted; tests that check `requires_grad` would think U-Net is trainable.
- **Fix**:
  ```python
  if self.phase == "phase2":
      for p in self.audio_unet.parameters():
          p.requires_grad_(False)
      for p in self.bottleneck_proj.parameters():
          p.requires_grad_(True)
  ```

### ARCH-07: Perceptual Loss Uses DINO Under `torch.no_grad()` In The Extractor
- **File(s)**: `src/loss/separation.py:217`, `src/visual/dinov2.py:39`, `src/visual/dinov2.py:48`
- **Description**: the comment says gradients should flow to `pred_wave` through frozen DINO, but DINO parameters are frozen and model code frequently wraps DINO calls in `torch.no_grad()` elsewhere. If the same extractor policy is copied, perceptual gradients can be accidentally cut.
- **Impact**: perceptual loss may become a metric only if no-grad is introduced around extractor calls.
- **Fix**: keep DINO params frozen but do not wrap perceptual extractor calls in `torch.no_grad()`; document this explicitly and add a test that `perceptual_loss.requires_grad` is true for predicted waveforms.

## Logic Errors (incorrect computation or behavior)

### LOGIC-01: Mixture Reconstruction Uses Confusing Mask Indirection
- **File(s)**: `evaluation/eval_sisnri.py:139`, `evaluation/eval_sisnri.py:140`, `evaluation/eval_sisnri.py:141`, `evaluation/eval_wer.py:109`, `evaluation/eval_zero_shot.py:130`
- **Description**: preliminary analysis claimed this zeroes the imaginary STFT, but in the current `ISTFTModule` semantics `(real=1, imag=0)` is the complex identity mask `1+0j`, so it does not zero `mixture_spec[:, 1]`. The implementation is still unnecessarily indirect and duplicated across scripts; the safe baseline is direct inverse STFT of the complex mixture.
- **Impact**: current code is numerically intended to be identity, but the pattern is easy to misread and can become wrong if mask semantics change.
- **Fix**:
  ```python
  complex_mix = torch.complex(mixture_stft[:, 0], mixture_stft[:, 1])
  mixture_wave = torch.istft(
      complex_mix, n_fft=512, hop_length=160, win_length=400,
      window=torch.hann_window(400, device=complex_mix.device),
      length=96000,
  )
  ```

### LOGIC-02: Attention Entropy Treats Averaged Weights As Per-Head Weights
- **File(s)**: `src/fusion/cross_attention.py:42`, `src/models/separator.py:229`, `src/models/separator.py:231`, `src/models/separator.py:412`
- **Description**: `nn.MultiheadAttention` returns averaged attention weights by default. The hook fires because output is `(attn_out, attn_weights)`, but the tensor is `[B, T_q, T_kv]`, not `[B, n_heads, T_q, T_kv]`.
- **Impact**: entropy is computed over averaged attention, contradicting the comment and hiding per-head collapse.
- **Fix**:
  ```python
  attn_out, attn_weights = self.attn(
      q, key, value,
      key_padding_mask=key_padding_mask,
      attn_mask=attn_mask,
      need_weights=True,
      average_attn_weights=False,
  )
  ```

### LOGIC-03: `_predict_masks()` Phase-2/3 Fallback Drops Visual Context
- **File(s)**: `src/models/separator.py:371`, `src/models/separator.py:375`, `src/models/separator.py:376`
- **Description**: if cached forward intermediates are invalid, non-phase1 fallback uses `self.cross_attn(combined, bottleneck_flat)`, not the visual features used in the actual forward pass.
- **Impact**: cRM masks silently differ from waveform predictions in phase 3 whenever the cache is invalid.
- **Fix**:
  ```python
  def _predict_masks(self, mixture_stft, *, video_frames=None, visual_features=None):
      if cache_invalid:
          _ = self.forward(mixture_stft, video_frames=video_frames, visual_features=visual_features)
      source_features = self._cached_source_features
  ```

### LOGIC-04: `compute_pairwise_losses()` Detaches SI-SNR Costs With `.item()`
- **File(s)**: `src/loss/separation.py:72`, `src/loss/separation.py:76`
- **Description**: scalar SI-SNR values are converted through `.item()` when filling the cost matrix.
- **Impact**: the permutation search is non-differentiable anyway, but this pattern makes the pairwise cost matrix unusable for any differentiable relaxation and forces CPU sync per pair.
- **Fix**:
  ```python
  cost_matrix[i, j] = -si_snr
  ```

### LOGIC-05: Phase 3 STFT/Perceptual Losses Ignore PIT Permutation
- **File(s)**: `src/models/separator.py:329`, `src/models/separator.py:341`, `src/models/separator.py:343`
- **Description**: SI-SNR/cRM use PIT per sample, but STFT and perceptual losses compare flattened predicted and target sources in original order.
- **Impact**: phase 3 can penalize a correct separation if the optimal source order differs from the target order.
- **Fix**: use the PIT permutation returned by `PITLossWrapper` to reorder predictions before STFT/perceptual losses.

### LOGIC-06: `target_waveforms` Are Not Truncated/Padded After Source Padding
- **File(s)**: `src/data/dataset.py:146`, `src/data/dataset.py:149`, `src/data/dataset.py:155`
- **Description**: `mixture_wave` is forced to `CLIP_LENGTH`, but `source_waves` are stacked before the same final truncation/padding is applied.
- **Impact**: if source clips are not already 96000 samples, `target_waveforms` can have a different length than model output.
- **Fix**:
  ```python
  source_waves = [F.pad(w[:CLIP_LENGTH], (0, max(0, CLIP_LENGTH - w.shape[-1]))) for w in source_waves]
  target_waveforms = torch.stack(source_waves)
  mixture_wave = mixture_wave[:CLIP_LENGTH]
  ```

### LOGIC-07: `source_gains` Are Not Applied To Targets Or CRM Targets
- **File(s)**: `src/data/dataset.py:115`, `src/data/dataset.py:117`, `src/data/dataset.py:147`, `src/data/dataset.py:155`, `src/data/dataset.py:199`
- **Description**: the mixture is made from gain-scaled sources, but `target_waveforms` are unscaled originals and cached CRM masks come from original clip mixtures, not the on-the-fly scaled synthetic mixture.
- **Impact**: losses train against targets that do not sum to the mixture and CRM targets do not match the generated mixture.
- **Fix**:
  ```python
  scaled_sources = [w * g for w, g in zip(source_waves, source_gains)]
  mixture_wave = torch.stack(scaled_sources).sum(dim=0)
  target_waveforms = torch.stack(scaled_sources)
  target_crm_masks = compute_crm_targets(torch.stack([self._stft(w) for w in scaled_sources]), mixture_stft)
  ```

### LOGIC-08: Tests Validate The Wrong Phase-2 Video Shape
- **File(s)**: `tests/models/test_separator.py:54`, `tests/models/test_separator.py:63`, `tests/scripts/test_preprocess_data.py:112`
- **Description**: separator tests pass raw `[B, N_frames, 3, 448, 448]`; preprocessing tests verify cached `[N_frames, 1024, 768]`; no test passes the real dataloader output to the model.
- **Impact**: BUG-01 remains invisible.
- **Fix**: add an integration test that preprocesses a dummy clip, builds `AudioVisualDataModule`, and calls phase-2 `training_step()` with the dataloader batch.

## Duplicate / Redundant Code (dead code, naming conflicts)

### DUP-01: Duplicate `PITLossWrapper`
- **File(s)**: `src/loss/separation.py:236`, `src/loss/pit_wrapper.py:11`, `src/models/separator.py:15`
- **Description**: two nearly identical classes exist; the model imports `src.loss.pit_wrapper.PITLossWrapper`, making the copy in `separation.py` dead.
- **Impact**: future fixes can be made to one wrapper but not the other.
- **Fix**:
  ```python
  # src/loss/separation.py
  # remove PITLossWrapper class and import it from src.loss.pit_wrapper where needed
  ```

### DUP-02: Duplicate STFT/ISTFT Modules
- **File(s)**: `src/audio/spectrogram.py:10`, `src/audio/spectrogram.py:36`, `src/data/preprocessing.py:88`, `src/data/preprocessing.py:131`
- **Description**: `src/audio` defines `nn.Module` versions; `src/data/preprocessing.py` defines plain callable versions. They use similar parameters but not identical implementation details (`window_fn`, asserts, batch handling).
- **Impact**: preprocessing and model can drift numerically.
- **Fix**: keep `src/audio/spectrogram.py` as canonical and import it from preprocessing, or move shared functional helpers into one file.

### DUP-03: Duplicate `AudioVisualDataModule`
- **File(s)**: `src/data/datamodule.py:14`, `src/data/dataset.py:301`, `scripts/train.py:26`
- **Description**: the real Lightning datamodule is in `src/data/datamodule.py`; `src/data/dataset.py` also defines a non-Lightning datamodule using legacy `AudioVisualDataset`.
- **Impact**: importing `AudioVisualDataModule` from the wrong file returns batches with `source_stfts` (`src/data/dataset.py:255`) instead of `target_waveforms`.
- **Fix**: remove or rename the legacy class:
  ```python
  class LegacyAudioVisualDataModule:
      ...
  ```

### DUP-04: `SISNRLoss.forward()` Is A Dead Standalone PIT Path
- **File(s)**: `src/loss/separation.py:42`, `src/loss/pit_wrapper.py:94`, `src/loss/pit_wrapper.py:106`
- **Description**: the wrapper calls `compute_pairwise_losses()` and `_si_snr()` directly; `SISNRLoss.forward()` is never called by training.
- **Impact**: two PIT implementations can diverge.
- **Fix**: remove `forward()` or document it as a standalone alternative and add tests for both.

### DUP-05: Gradient Clipping Is Done Twice
- **File(s)**: `src/models/separator.py:422`, `src/models/separator.py:424`, `scripts/train.py:139`, `scripts/train.py:140`
- **Description**: the module clips gradients in `on_before_optimizer_step()`, and Trainer also clips by norm.
- **Impact**: redundant work and non-obvious gradient behavior.
- **Fix**: remove the hook and keep Trainer-managed clipping:
  ```python
  # delete SeparatorModule.on_before_optimizer_step()
  ```

## Missing / Stub Implementations

### STUB-01: `apply_augmentation()` Does Nothing
- **File(s)**: `src/data/mix_and_separate.py:48`, `src/data/mix_and_separate.py:57`
- **Description**: the function accepts pitch, stretch, and noise SNR arguments but returns the waveform unchanged.
- **Impact**: no augmentation is applied despite the API/config implication.
- **Fix**:
  ```python
  if noise_snr_db is not None:
      noise = torch.randn_like(waveform)
      signal_power = waveform.pow(2).mean()
      noise_power = signal_power / (10 ** (noise_snr_db / 10))
      waveform = waveform + noise * torch.sqrt(noise_power / (noise.pow(2).mean() + 1e-8))
  # implement pitch/time stretch via torchaudio/sox or remove unsupported args
  return waveform
  ```

### STUB-02: Localisation Depends On Attention Caches That Are Not Source-Localized
- **File(s)**: `evaluation/eval_localisation.py:96`, `src/models/separator.py:228`, `src/models/separator.py:244`
- **Description**: localization extracts cached attention weights, but current attention uses source queries plus all bottleneck positions and averaged heads.
- **Impact**: IoU maps are not clearly tied to one source/object and may be uninterpretable.
- **Fix**: store per-head, per-source attention maps with `average_attn_weights=False` and slice only source-query rows.

## Minor Issues (style, paths, comments)

### MINOR-01: Hardcoded Windows Paths
- **File(s)**: `scripts/test_model.py:15`, `tests/test_fixes.py:4`
- **Description**: both files insert `D:/development/python/ai/SonicSightDino`.
- **Impact**: scripts break outside the original machine.
- **Fix**:
  ```python
  from pathlib import Path
  sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
  ```
  For `tests/test_fixes.py`, use `parents[1]` from `tests/`; for `scripts/test_model.py`, use `parents[1]` from `scripts/`.

### MINOR-02: `tests/models/__init__.py` Is Missing
- **File(s)**: `tests/models/test_separator.py:1`
- **Description**: pytest discovered the file in this run, but `tests/models/` is inconsistent with other test subpackages.
- **Impact**: usually harmless with modern pytest, but inconsistent package layout.
- **Fix**: add empty `tests/models/__init__.py`.

### MINOR-03: Python 3.12 Dependency Pins Are Incompatible
- **File(s)**: `requirements.txt:2`, `requirements.txt:7`, `requirements.txt:11`, `requirements.txt:12`, `requirements.txt:13`
- **Description**: the prompt/repo targets Python 3.12, but `scipy==1.11.0` and the pinned torch stack are old for Python 3.12 on Windows.
- **Impact**: `pip install -r requirements.txt` failed during this review.
- **Fix**: update pins for Python 3.12, e.g. `scipy>=1.12,<1.14` and a PyTorch/torchaudio/torchvision build with Python 3.12 wheels.

### MINOR-04: Torchaudio `return_complex` Is Deprecated
- **File(s)**: `src/audio/spectrogram.py:23`
- **Description**: torchaudio warns that `return_complex` is deprecated/ineffective for `power=None`.
- **Impact**: warning noise in tests.
- **Fix**: remove `return_complex=True`.

### MINOR-05: `ISTFTModule` Is Hardcoded To 96000 Samples
- **File(s)**: `src/audio/spectrogram.py:39`, `src/audio/spectrogram.py:45`, `src/audio/spectrogram.py:62`, `src/data/preprocessing.py:141`
- **Description**: the model ISTFT always reconstructs `length=96000`.
- **Impact**: fine for fixed 6s/16kHz clips; wrong if configs change duration/sample rate.
- **Fix**: pass `length` from config and batch metadata:
  ```python
  self.istft = ISTFTModule(length=cfg["data"]["sample_rate"] * cfg["data"]["clip_duration"])
  ```

### MINOR-06: U-Net Last Decoder Block Bypasses `DecoderBlock.forward()`
- **File(s)**: `src/audio/unet.py:99`, `src/audio/unet.py:105`, `src/audio/unet.py:106`
- **Description**: the last decoder has no skip and calls `block.deconv(x)` directly. Because `is_last=True`, `DecoderBlock.forward()` would only add a skip and skip norm/activation; direct `deconv` is semantically OK for "no skip".
- **Impact**: not a crash; it is non-standard U-Net topology because there is no input-resolution skip. The first encoder output `s0` is used by decoder block 4, not skipped; the fifth block maps `32→2` without a skip, which is expected when encoder has no pre-conv input skip.
- **Fix**: make intent explicit:
  ```python
  if skip is None:
      x = block.deconv(x)  # final upsample, no same-resolution skip exists
  ```

### MINOR-07: `train.num_sources` And `train.n_sources` Are Duplicated
- **File(s)**: `configs/train/phase1.yaml:7`, `configs/train/phase1.yaml:28`, `scripts/train.py:81`
- **Description**: configs contain both names; train code primarily reads `cfg.model.n_sources`.
- **Impact**: changing `train.n_sources` alone does not change the model source count.
- **Fix**: use one key and propagate it:
  ```python
  n_sources = cfg.train.get("n_sources", cfg.model.n_sources)
  cfg.model.n_sources = n_sources
  ```

## Fix Plan (chronological order of changes recommended)

1. Fix BUG-01 by separating raw `video_frames` from cached `visual_features` and adding an integration test.
2. Fix LOGIC-07 by recomputing targets/cRM for the actual on-the-fly scaled mixture.
3. Fix BUG-02 and BUG-03 together so model/datamodule progressive source counts and optimizer parameters stay synchronized.
4. Fix BUG-04 and BUG-05 in `scripts/train.py` before attempting phase handoff training.
5. Fix BUG-06 and BUG-07; explicitly freeze omitted phase-2/3 parameters or include them in optimizer groups.
6. Fix ARCH-05 and ARCH-02 so validation/evaluation are deterministic and metadata-aware.
7. Fix LOGIC-01 across SI-SNRi/WER/zero-shot evaluation scripts with direct inverse STFT.
8. Fix LOGIC-02 and STUB-02 to make attention entropy/localization source/head-aware.
9. Fix LOGIC-03 and LOGIC-05 so cRM/STFT/perceptual losses use the same visual context and PIT permutation.
10. Remove duplicate wrappers/modules/datamodules (DUP-01 through DUP-04) after tests are in place.
11. Implement or remove augmentation API (STUB-01).
12. Apply minor cleanup: portable paths, missing test package marker, Python 3.12-compatible requirements, torchaudio deprecation, and source-count config consolidation.
