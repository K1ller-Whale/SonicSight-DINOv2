# SonicSight-DINOv2 — Training Walkthrough

This guide assumes the blockers in `review.md` are fixed, especially the cached visual-feature/model input contract, Lightning checkpoint resume, gradient accumulation, deterministic validation/test sampling, optimizer source-query handling, and phase-specific curriculum gating.

## Prerequisites

- **Python**: Python 3.12. Use `C:/Users/H/AppData/Local/Programs/Python/Python312/python.exe` on this machine.
- **GPU**: NVIDIA GPU strongly recommended. Minimum practical memory: 12 GB for phase 1, 16–24 GB for phase 2, 24 GB+ for phase 3 with DINO and 448×448 frames/features.
- **CUDA**: install a PyTorch build matching your installed NVIDIA driver/CUDA runtime.
- **Disk**: plan for raw datasets plus cache. Cached DINO features are large; budget hundreds of GB for full MUSIC/AVSpeech/VoxCeleb-style corpora.
- **Network**: first run downloads `facebook/dinov2-base`, Whisper, YOLO if localization is used, and dataset archives.

## Step 1 — Environment Setup

```powershell
git clone <YOUR_REPO_URL> SonicSight-DINOv2
cd SonicSight-DINOv2

C:/Users/H/AppData/Local/Programs/Python/Python312/python.exe -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
```

If Python 3.12 install fails on old pins, update `requirements.txt` as described in `review.md` and install a Python-3.12-compatible PyTorch stack from the official PyTorch selector.

## Step 2 — Verify Installation

```powershell
python -m pytest tests/ -v --tb=short 2>&1 | Tee-Object -FilePath test_output.txt
```

Expected after fixes:

```text
55+ passed
0 failed
```

Warnings about missing Trainer in unit tests should be removed or accepted only if tests intentionally call Lightning hooks directly.

## Step 3 — Data Acquisition

Create this structure:

```text
data/
  raw/
    music/
      <clip_id>/
        source_0.wav|pt
        source_1.wav|pt
        video.mp4|pt
    avspeech/
      <speaker_or_video_id>_<utt_id>/
        source_0.wav|pt
        video.mp4|pt
    voxceleb2/
      <speaker_id>_<utt_id>/
        source_0.wav|pt
        video.mp4|pt
```

- **MUSIC**: download from the MUSIC / Sound of Pixels dataset source used by the project.
- **AVSpeech**: download from the official AVSpeech release or your prepared mirror.
- **VoxCeleb2**: download from the official VoxCeleb2 access portal after accepting its terms.
- Each clip directory must contain one or more `source_*` audio files and optionally one `video*` file.
- Audio will be converted to 16 kHz mono, 6 seconds. Video is expected as raw frames `[T, 3, 448, 448]` in `.pt` or a readable video file.

## Step 4 — Data Preprocessing

Run one dataset at a time:

```powershell
python scripts/preprocess_data.py `
  --input_dir data/raw/music `
  --output_dir cache/music `
  --train_ratio 0.8 `
  --val_ratio 0.1 `
  --test_ratio 0.1 `
  --seed 42 `
  --device cuda `
  --dinov2_batch_size 4 `
  --dataset_type music
```

For AVSpeech:

```powershell
python scripts/preprocess_data.py `
  --input_dir data/raw/avspeech `
  --output_dir cache/avspeech `
  --seed 42 `
  --device cuda `
  --dinov2_batch_size 4 `
  --dataset_type avspeech
```

Flags:

- `--input_dir`: directory containing clip subdirectories.
- `--output_dir`: cache destination.
- `--train_ratio`, `--val_ratio`, `--test_ratio`: MUSIC identity split ratios.
- `--dataset_type`: `music` uses provided ratios; `avspeech` uses 85/7.5/7.5 identity split.
- `--device`: use `cuda` for DINO extraction if available.
- `--dinov2_batch_size`: lower this if GPU memory is insufficient.

Expected cache:

```text
cache/music/
  crm/<clip_id>.pt
  visual/<clip_id>.pt or visual/<clip_id>_src*.pt
  index.json
```

Verify:

```powershell
python - <<'PY'
import json, torch
from pathlib import Path
cache = Path("cache/music")
index = json.loads((cache / "index.json").read_text())
print("clips", len(index))
first = next(iter(index.values()))
crm = torch.load(first["crm_path"], map_location="cpu")
print("crm", crm.shape, crm.dtype)
visual_path = first.get("visual_path") or next(p for p in first["visual_paths"] if p)
visual = torch.load(visual_path, map_location="cpu")
print("visual", visual.shape, visual.dtype)
PY
```

Expected shapes:

- `crm`: `[N, 2, 257, 601]`, usually `float16`.
- cached visual features: `[V, 1024, 768]`, usually `float16`.
- dataloader visual features after fixes: `[B, N_sources, 601, 1024, 768]` or reduced `[B, 601, 1024, 768]`, passed as `visual_features`, not raw `video_frames`.

## Step 5 — Dry-Run Model Validation

```powershell
python scripts/test_model.py
```

Expected output includes:

```text
Dry-Run Model Validation
Total parameters: ...
Trainable parameters: ...
Output shape: torch.Size([2, 2, 96000])
Training step: OK
All checks passed!
```

After the visual pipeline fix, also run a cached-data integration smoke test:

```powershell
python - <<'PY'
from src.data.datamodule import AudioVisualDataModule
from src.models.separator import SeparatorModule
dm = AudioVisualDataModule(index_file="cache/music/index.json", n_sources=2, batch_size=1, val_batch_size=1, num_workers=0, include_visual=True)
dm.setup("fit")
batch = next(iter(dm.train_dataloader()))
model = SeparatorModule({"model": {"n_sources": 2}, "train": {"optimizer": {}, "scheduler": {}, "loss": {}}}, phase="phase2")
print(batch["mixture_stft"].shape, batch.get("visual_features", batch.get("video_frames")).shape)
PY
```

## Step 6 — Phase 1 Training (Audio-Only Pretraining)

```powershell
python scripts/train.py train=phase1 data.cache_dir=cache/music data.index_file=cache/music/index.json
```

Watch:

- `train/si_snr_loss` should trend down.
- `val/sisnri` should trend up.
- no visual/DINO feature loading should be required.
- source count should remain `2` for the entire phase.

Expected checkpoint location:

```text
checkpoints/last.ckpt
checkpoints/step=<step>-phase1-<val_sisnri>.ckpt
```

Expected SI-SNRi depends heavily on data and runtime, but a healthy phase-1 run should improve above the mixture baseline and generally reach low-to-mid single-digit dB SI-SNRi before phase 2.

Verify:

```powershell
python evaluation/eval_sisnri.py `
  --checkpoint checkpoints/last.ckpt `
  --index_file cache/music/index.json `
  --n_sources 2 `
  --output outputs/eval_phase1_sisnri.json
```

## Step 7 — Phase 2 Training (Cross-Modal Attention Warmup)

Resume from the best or last phase-1 checkpoint:

```powershell
python scripts/train.py train=phase2 `
  data.cache_dir=cache/music `
  data.index_file=cache/music/index.json `
  train.resume_from_checkpoint=checkpoints/last.ckpt
```

After the Lightning 2.x fix, the script should call:

```python
trainer.fit(model, datamodule=datamodule, ckpt_path=resume_ckpt)
```

Watch:

- `train/si_snr_loss`
- `train/entropy_loss`
- `val/sisnri`
- attention entropy should be finite and based on non-averaged per-head weights if fixed.
- source count should remain `2`; curriculum must not run in phase 2.

Expected metrics: modest SI-SNRi improvement over phase 1 if visual cues are informative. If SI-SNRi drops sharply, inspect cached visual-feature shape and source/target alignment.

## Step 8 — Phase 3 Training (End-to-End Fine-Tuning)

Use the phase-2 checkpoint:

```powershell
python scripts/train.py train=phase3 `
  data.cache_dir=cache/music `
  data.index_file=cache/music/index.json `
  train.resume_from_checkpoint=checkpoints/last.ckpt
```

For DINO unfreeze variants:

```powershell
python scripts/train.py train=phase3_config_a data.index_file=cache/music/index.json train.resume_from_checkpoint=checkpoints/last.ckpt
python scripts/train.py train=phase3_config_b data.index_file=cache/music/index.json train.resume_from_checkpoint=checkpoints/last.ckpt
python scripts/train.py train=phase3_config_c data.index_file=cache/music/index.json train.resume_from_checkpoint=checkpoints/last.ckpt
```

Watch:

- `train/si_snr_loss`
- `train/crm_loss`
- `train/stft_loss`
- `train/perceptual_loss`
- `val/sisnri`
- learning-rate groups for fusion, decoder, selected encoder blocks, and DINO.

Progressive curriculum:

- steps `0–19999`: `2` sources
- steps `20000–39999`: `3` sources
- steps `40000+`: `4` sources

Gradient accumulation:

- `configs/train/phase3.yaml` sets `gradient_accumulation_steps: 4`.
- after the fix, `Trainer(accumulate_grad_batches=4)` should be active.

Expected final metrics: data-dependent; a healthy model should exceed phase 1/2 SI-SNRi and improve SDR/SIR over the mixture baseline. Track regressions separately for 2-, 3-, and 4-source validation subsets.

## Step 9 — Evaluation

Run SI-SNRi:

```powershell
python evaluation/eval_sisnri.py `
  --checkpoint checkpoints/last.ckpt `
  --index_file cache/music/index.json `
  --n_sources 2 `
  --output outputs/eval_sisnri.json
```

Expected output:

```text
SI-SNRi mean: <value> dB
SI-SNRi std:  <value> dB
Samples:      <count>
```

Run SDR/SIR/SAR:

```powershell
python evaluation/eval_sdr.py `
  --checkpoint checkpoints/last.ckpt `
  --index_file cache/music/index.json `
  --n_sources 2 `
  --output outputs/eval_sdr.json
```

Expected output:

```text
SDR mean: <value> dB
SIR mean: <value> dB
SAR mean: <value> dB
Samples:  <count>
```

Run WER for speech datasets:

```powershell
python evaluation/eval_wer.py `
  --checkpoint checkpoints/last.ckpt `
  --index_file cache/avspeech/index.json `
  --transcripts data/transcripts.json `
  --n_sources 2 `
  --asr_model whisper `
  --sample_rate 16000 `
  --output outputs/eval_wer.json
```

Expected output:

```text
Separated WER mean: <lower is better>
Mixture WER mean:   <baseline>
Samples:            <count>
```

Run localization:

```powershell
python evaluation/eval_localisation.py `
  --checkpoint checkpoints/last.ckpt `
  --index_file cache/music/index.json `
  --n_sources 2 `
  --gt_boxes data/gt_boxes.json `
  --output outputs/eval_localisation.json
```

Optional YOLO pseudo-boxes:

```powershell
python evaluation/eval_localisation.py `
  --checkpoint checkpoints/last.ckpt `
  --index_file cache/music/index.json `
  --n_sources 2 `
  --use_yolo `
  --yolo_model yolov8n.pt `
  --yolo_conf 0.5 `
  --output outputs/eval_localisation_yolo.json
```

Run combined evaluation:

```powershell
python scripts/run_evaluation.py `
  --checkpoint checkpoints/last.ckpt `
  --index_file cache/music/index.json `
  --transcripts data/transcripts.json `
  --gt_boxes data/gt_boxes.json `
  --n_sources 2 `
  --output outputs/evaluation_results.json
```

## Step 10 — Troubleshooting Common Issues

- **OOM during preprocessing**: lower `--dinov2_batch_size` to `1` or run `--device cpu` for DINO extraction.
- **OOM during phase 2/3**: reduce `batch_size`, keep `gradient_accumulation_steps` to preserve effective batch size, and use cached visual features instead of raw frames.
- **STFT shape mismatch**: verify audio length is exactly `96000`, `n_fft=512`, `hop_length=160`, and `win_length=400`; expected STFT is `[2, 257, 601]`.
- **Visual shape error**: cached DINO tokens must be passed as `visual_features`, not `video_frames`; raw `video_frames` must be `[B, T, 3, 448, 448]`.
- **Checkpoint loading error**: with Lightning 2.x, pass resume path to `trainer.fit(..., ckpt_path=...)`, not `Trainer(resume_from_checkpoint=...)`.
- **DINOv2 download issues**: pre-download `facebook/dinov2-base` from HuggingFace, set `HF_HOME`, or run once with network access before offline training.
- **Non-deterministic validation**: ensure validation/test dataset sampling uses `idx` and a fixed mixture list, not `random.sample()`.
- **Bad SI-SNRi baseline**: reconstruct mixture directly with `torch.istft()` from the complex mixture STFT and fixed `length=96000`.
