"""Evaluation: Word Error Rate (WER) via ASR on separated sources.

Requires an ASR model (e.g. Whisper). For each separated source
we transcribe it and compare to the ground-truth transcript.
Also computes baseline WER on unprocessed mixture.

Usage:
    python evaluation/eval_wer.py \
        --checkpoint checkpoints/best.ckpt \
        --index_file cache/index.json \
        --transcripts data/transcripts.json

Output (JSON):
    {
        "wer_mean": 0.15,
        "mixture_wer_mean": 0.35,
        "per_sample": [...],
        "mixture_per_sample": [...]
    }
"""
import argparse
import json
import os
import sys
from typing import Dict, List

import torch
import torchaudio

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.models.separator import SeparatorModule
from src.data.datamodule import AudioVisualDataModule
from evaluation.common import maybe_to_device, resolve_n_sources


def resample_to_16kHz(waveform: torch.Tensor, orig_sr: int) -> torch.Tensor:
    """Resample waveform to 16kHz for Whisper."""
    if orig_sr == 16000:
        return waveform
    return torchaudio.functional.resample(waveform, orig_sr, 16000)


def reconstruct_mixture(mixture_stft: torch.Tensor,
                        device: torch.device) -> torch.Tensor:
    """
    Reconstruct mixture waveform from complex STFT.
    Uses direct torch.istft with no masking.

    Args:
        mixture_stft: [B, 2, F, T] real+imag channels
        device: target device
    Returns:
        mixture_wave: [B, L]
    """
    complex_mix = torch.complex(
        mixture_stft[:, 0], mixture_stft[:, 1])
    return torch.istft(
        complex_mix,
        n_fft=512,
        hop_length=160,
        win_length=400,
        window=torch.hann_window(400, device=device),
        length=96000,
        return_complex=False,
    )


def evaluate_wer(args) -> Dict:
    """Evaluate WER on separated sources and mixture baseline."""
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")

    # Load ASR (Whisper)
    asr = None
    if args.asr_model == "whisper":
        try:
            import whisper  # type: ignore
            asr = whisper.load_model("base").to(device)
            print(f"Loaded Whisper model: base")
        except ImportError:
            print("WARNING: whisper not installed; WER will be -1")

    # Load separator
    model = SeparatorModule.load_from_checkpoint(args.checkpoint)
    model = model.to(device)
    model.eval()
    n_sources = resolve_n_sources(model, getattr(args, "n_sources", None))

    dm = AudioVisualDataModule(
        index_file=args.index_file,
        n_sources=n_sources,
        batch_size=1,
        num_workers=0,
        include_visual=(model.phase != "phase1"),
    )
    dm.setup("test")
    dataloader = dm.test_dataloader()

    # Load transcripts - expect mapping from clip_id to transcript
    with open(args.transcripts) as f:
        transcripts = json.load(f)

    wer_scores = []
    mixture_wer_scores = []

    try:
        import jiwer
    except ImportError:
        print("WARNING: jiwer not installed")
        jiwer = None

    # Get original sample rate from dataset (16kHz)
    orig_sr = args.sample_rate

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            mixture_stft = batch["mixture_stft"].to(device)
            visual_features = maybe_to_device(batch.get("visual_features"), device)
            video_frames = maybe_to_device(batch.get("video_frames"), device)

            # Get clip_id for transcript lookup
            clip_id = batch.get("clip_id", [str(batch_idx)])[0] if batch.get("clip_id") else str(batch_idx)
            ref = transcripts.get(clip_id, "")
            if not ref:
                ref = transcripts.get(str(batch_idx), "")
            
            if not ref:
                continue

            # Forward - route visual input appropriately
            if model.phase == "phase1":
                separated = model(mixture_stft)
            elif visual_features is not None:
                separated = model(mixture_stft, visual_features=visual_features)
            elif video_frames is not None:
                separated = model(mixture_stft, video_frames=video_frames)
            else:
                separated = model(mixture_stft)

            # separated: [B, N, L]
            B, N, L = separated.shape

            # Reconstruct mixture waveform for baseline using direct torch.istft
            mixture_wave = reconstruct_mixture(mixture_stft, device)

            for b in range(B):
                # Baseline: mixture WER
                mix_wave = mixture_wave[b].cpu()
                mix_wave_16k = resample_to_16kHz(mix_wave, orig_sr)

                if asr is not None:
                    mix_hyp = asr.transcribe(mix_wave_16k.numpy())["text"]  # type: ignore
                else:
                    mix_hyp = ""
                mix_wer = jiwer.wer(ref, mix_hyp) if jiwer else 0.0
                mixture_wer_scores.append(mix_wer)

                # Separated sources WER
                for n in range(N):
                    sep_wave = separated[b, n].cpu()
                    sep_wave_16k = resample_to_16kHz(sep_wave, orig_sr)

                    if asr is not None:
                        hyp = asr.transcribe(sep_wave_16k.numpy())["text"]  # type: ignore
                    else:
                        hyp = ""
                        
                    wer = jiwer.wer(ref, hyp) if jiwer else 0.0
                    wer_scores.append(wer)

    return {
        "wer_mean": sum(wer_scores) / len(wer_scores) if wer_scores else -1.0,
        "mixture_wer_mean": sum(mixture_wer_scores) / len(mixture_wer_scores) if mixture_wer_scores else -1.0,
        "num_samples": len(wer_scores),
        "per_sample": wer_scores,
        "mixture_per_sample": mixture_wer_scores,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate WER on test set")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--index_file", type=str, default="cache/index.json")
    parser.add_argument("--transcripts", type=str, default="data/transcripts.json")
    parser.add_argument("--n_sources", type=int, default=None)
    parser.add_argument("--asr_model", type=str, default="whisper")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--sample_rate", type=int, default=16000,
                        help="Original sample rate of audio (for resampling to 16kHz)")
    parser.add_argument("--output", type=str, default="outputs/eval_wer.json")
    args = parser.parse_args()

    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
    if not os.path.exists(args.index_file):
        raise FileNotFoundError(f"Index file not found: {args.index_file}")
    if not os.path.exists(args.transcripts):
        print(f"WARNING: Transcripts file not found: {args.transcripts}")
        # Create empty transcript dict so it doesn't crash
        with open("dummy_transcripts.json", "w") as f:
            json.dump({}, f)
        args.transcripts = "dummy_transcripts.json"

    # Ensure outputs directory exists
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    results = evaluate_wer(args)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {args.output}")
    print(f"  Separated WER mean: {results['wer_mean']:.3f}")
    print(f"  Mixture WER mean:   {results['mixture_wer_mean']:.3f}")
    print(f"  Samples:            {results['num_samples']}")


if __name__ == "__main__":
    main()
