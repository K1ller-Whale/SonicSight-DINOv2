"""Evaluation: Word Error Rate (WER) via ASR on separated sources.

Requires an ASR model (e.g. Whisper).  For each separated source
we transcribe it and compare to the ground-truth transcript.

Usage:
    python evaluation/eval_wer.py \
        --checkpoint checkpoints/best.ckpt \
        --index_file cache/index.json \
        --transcripts data/transcripts.json

Output (JSON):
    {
        "wer_mean": 0.15,
        "per_sample": [...]
    }
"""
import argparse
import json
import os
import sys
from typing import Dict, List

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.models.separator import SeparatorModule
from src.data.datamodule import AudioVisualDataModule


def evaluate_wer(args) -> Dict:
    """Evaluate WER on separated sources."""
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")

    # Load ASR (placeholder – swap for Whisper)
    asr = None
    if args.asr_model == "whisper":
        try:
            import whisper  # type: ignore
            asr = whisper.load_model("base").to(device)
        except ImportError:
            print("WARNING: whisper not installed; WER will be -1")

    # Load separator
    model = SeparatorModule.load_from_checkpoint(args.checkpoint)
    model = model.to(device)
    model.eval()

    dm = AudioVisualDataModule(
        index_file=args.index_file,
        n_sources=args.n_sources,
        batch_size=1,
        num_workers=0,
        include_visual=(model.phase != "phase1"),
    )
    dm.setup("test")
    dataloader = dm.test_dataloader()

    # Load transcripts
    with open(args.transcripts) as f:
        transcripts = json.load(f)

    wer_scores = []
    try:
        import jiwer
    except ImportError:
        print("WARNING: jiwer not installed")
        jiwer = None

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            mixture_stft = batch["mixture_stft"].to(device)

            if model.phase == "phase1":
                separated = model(mixture_stft, video_frames=None)
            else:
                separated = model(mixture_stft, video_frames=batch.get("video_frames"))

            # separated: [B, N, L]
            for b in range(separated.shape[0]):
                for n in range(separated.shape[1]):
                    if asr is not None:
                        hyp = asr.transcribe(separated[b, n].cpu().numpy())["text"]  # type: ignore
                    else:
                        hyp = ""  # placeholder
                    ref = transcripts.get(str(batch_idx), "")
                    wer = jiwer.wer(ref, hyp) if (jiwer and ref) else 0.0
                    wer_scores.append(wer)

    return {
        "wer_mean": sum(wer_scores) / len(wer_scores) if wer_scores else -1.0,
        "num_samples": len(wer_scores),
        "per_sample": wer_scores,
    }

def main():
    parser = argparse.ArgumentParser(description="Evaluate WER on test set")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--index_file", type=str, default="cache/index.json")
    parser.add_argument("--transcripts", type=str, default="data/transcripts.json")
    parser.add_argument("--n_sources", type=int, default=2)
    parser.add_argument("--asr_model", type=str, default="whisper")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--output", type=str, default="wer_results.json")
    args = parser.parse_args()

    results = evaluate_wer(args)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {args.output}")
    print(f"  WER mean: {results['wer_mean']:.3f}")
    print(f"  Samples:  {results['num_samples']}")

if __name__ == "__main__":
    main()
