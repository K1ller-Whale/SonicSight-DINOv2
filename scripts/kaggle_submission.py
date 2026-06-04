"""Kaggle submission script for SonicSight-DINOv2.

Usage:
    python scripts/kaggle_submission.py \
        --checkpoint checkpoints/best.ckpt \
        --index_file cache/index.json \
        --output submission.csv
"""
import argparse
import csv
import json
import os
import sys
from typing import Dict, List

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from src.models.separator import SeparatorModule
from src.data.datamodule import AudioVisualDataModule


def generate_submission(args) -> List[Dict]:
    """Generate Kaggle submission file."""
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")

    # Load model
    print(f"Loading checkpoint from {args.checkpoint} ...")
    model = SeparatorModule.load_from_checkpoint(args.checkpoint)
    model = model.to(device)
    model.eval()

    # DataModule
    dm = AudioVisualDataModule(
        index_file=args.index_file,
        n_sources=args.n_sources,
        batch_size=1,
        num_workers=0,
        include_visual=(model.phase != "phase1"),
    )
    dm.setup("test")
    dataloader = dm.test_dataloader()

    submissions = []
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            mixture_stft = batch["mixture_stft"].to(device)

            if model.phase == "phase1":
                separated = model(mixture_stft, video_frames=None)
            else:
                separated = model(mixture_stft, video_frames=batch.get("video_frames"))

            # separated: [B, N, L]
            for b in range(separated.shape[0]):
                clip_id = batch.get("clip_id", [str(batch_idx)])[b]
                for n in range(separated.shape[1]):
                    submissions.append({
                        "clip_id": clip_id,
                        "source": n + 1,
                        "audio": separated[b, n].cpu().numpy().tobytes().hex(),
                    })

    return submissions


def main():
    parser = argparse.ArgumentParser(description="Generate Kaggle submission")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--index_file", type=str, default="cache/index.json")
    parser.add_argument("--n_sources", type=int, default=2)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--output", type=str, default="submission.csv")
    args = parser.parse_args()

    submissions = generate_submission(args)

    with open(args.output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["clip_id", "source", "audio"])
        writer.writeheader()
        writer.writerows(submissions)

    print(f"\nSubmission saved to {args.output}")
    print(f"  Total samples: {len(submissions)}")


if __name__ == "__main__":
    main()