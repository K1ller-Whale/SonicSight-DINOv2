# SonicSight-DINOv2

Audio-visual source separation guided by DINOv2 visual features.

## Architecture

Replaces the ResNet visual encoder in Sound of Pixels (Zhao et al., 2018) with a frozen DINOv2 backbone, coupled with cross-modal attention fusion.

**Key innovations:**
- Frozen DINOv2-Base provides semantically rich patch features (768-dim)
- Cross-modal attention: audio bottleneck queries visual key/values
- Complex Ratio Mask (cRM) targets with tanh compression
- Three-phase training curriculum

## Setup

```bash
# Python 3.12 required
pip install torch torchaudio pytorch-lightning hydra-core einops
pip install transformers  # DINOv2
pip install whisper jiwer mir_eval  # evaluation (optional)
```

## Data Preparation

```bash
# Download datasets:
# - MUSIC: https://github.com/rlsOrderByMUSIC
# - AVSpeech: https://google.github.io/AVSpeech/
# - VoxCeleb2: https://www.robots.ox.ac.uk/~vgg/data/voxceleb/

# Preprocess and create index
python scripts/preprocess_data.py --data_dir <path> --cache_dir cache/
```

## Training

```bash
# Phase 1: Audio-only pretraining (visual disabled)
python scripts/train.py train=phase1

# Phase 2: Cross-modal attention warmup (audio frozen)
python scripts/train.py train=phase2 --resume_from_checkpoint checkpoints/phase1-best.ckpt

# Phase 3: End-to-end fine-tuning (differential LRs)
python scripts/train.py train=phase3 --resume_from_checkpoint checkpoints/phase2-best.ckpt
```

## Evaluation

```bash
# SI-SNR improvement
python evaluation/eval_sisnri.py --checkpoint checkpoints/best.ckpt --index_file cache/index.json

# Word Error Rate (requires Whisper)
python evaluation/eval_wer.py --checkpoint checkpoints/best.ckpt --index_file cache/index.json --transcripts data/transcripts.json

# Localisation IoU (requires ground-truth boxes)
python evaluation/eval_localisation.py --checkpoint checkpoints/best.ckpt --index_file cache/index.json --gt_boxes data/boxes.json
```

## Project Structure

```
SonicSightDino/
├── configs/          # Hydra configs (model, data, train phases)
├── src/
│   ├── audio/        # U-Net, STFT modules
│   ├── visual/       # DINOv2 wrapper
│   ├── fusion/       # Cross-modal attention
│   ├── data/         # Datasets, preprocessing
│   ├── loss/         # SI-SNR, cRM, perceptual losses
│   └── models/       # LightningModule
├── scripts/          # Training, preprocessing
├── evaluation/       # Evaluation scripts
└── tests/            # Unit tests
```

## Expected Results

| Phase | SI-SNRi (dB) | WER (%) | Localisation IoU |
|-------|-------------|---------|------------------|
| Phase 1 (audio-only) | 8-10 | 25-30 | N/A |
| Phase 2 (attention) | 12-14 | 18-22 | 0.55-0.65 |
| Phase 3 (full) | 14-16 | 12-15 | 0.65-0.75 |

## Citation

```bibtex
@article{quabab2023dinov2,
  title={DINOv2: Learning Robust Visual Features without Supervision},
  author={Oquab, Maxime and Darcet, Timoth{\'e}e and Moutakanni, Theo and others},
  journal={TMLR},
  year={2024}
}

@inproceedings{zhao2018sound,
  title={The Sound of Pixels},
  author={Zhao, Hang and Gan, Chuang and Rouditchenko, Andrew and others},
  booktitle={ECCV},
  year={2018}
}
```

## License

MIT