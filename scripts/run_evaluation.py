"""Full evaluation pipeline runner.

Runs SI-SNRi, localisation (IoU), and WER evaluations in sequence
and saves combined results to a single JSON file.

Usage:
    python scripts/run_evaluation.py \
        --checkpoint checkpoints/best.ckpt \
        --index_file cache/index.json \
        --output evaluation_results.json
"""
import argparse
import json
import os
import sys
from typing import Dict

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from evaluation.eval_sisnri import evaluate_sisnri, compute_si_snr
from evaluation.eval_localisation import evaluate_localisation
from evaluation.eval_wer import evaluate_wer


def evaluate_all(args) -> Dict:
    """Run all evaluations and return combined results."""
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print(f"Using device: {device}")

    # 1. SI-SNRi evaluation
    print("\n" + "=" * 60)
    print("1/3 Running SI-SNRi evaluation...")
    print("=" * 60)
    sisnri_args = argparse.Namespace(
        checkpoint=args.checkpoint,
        index_file=args.index_file,
        n_sources=args.n_sources,
        cpu=args.cpu,
        log_every=args.log_every,
        output="_tmp_sisnri.json",
    )
    sisnri_results = evaluate_sisnri(sisnri_args)
    print(f"  SI-SNRi mean:  {sisnri_results['si_snri_mean']:.2f} dB")
    print(f"  SI-SNRi std:   {sisnri_results['si_snri_std']:.2f} dB")
    print(f"  SI-SNRi median:{sisnri_results['si_snri_median']:.2f} dB")
    print(f"  Samples:       {sisnri_results['num_samples']}")

    # 2. Localisation evaluation
    print("\n" + "=" * 60)
    print("2/3 Running localisation (IoU) evaluation...")
    print("=" * 60)
    loc_args = argparse.Namespace(
        checkpoint=args.checkpoint,
        index_file=args.index_file,
        n_sources=args.n_sources,
        gt_boxes=args.gt_boxes,
        cpu=args.cpu,
        output="_tmp_localisation.json",
    )
    loc_results = evaluate_localisation(loc_args)
    print(f"  IoU mean: {loc_results['iou_mean']:.3f}")
    print(f"  Samples:  {loc_results['num_samples']}")

    # 3. WER evaluation
    print("\n" + "=" * 60)
    print("3/3 Running WER evaluation...")
    print("=" * 60)
    wer_args = argparse.Namespace(
        checkpoint=args.checkpoint,
        index_file=args.index_file,
        transcripts=args.transcripts,
        n_sources=args.n_sources,
        asr_model=args.asr_model,
        cpu=args.cpu,
        output="_tmp_wer.json",
    )
    wer_results = evaluate_wer(wer_args)
    print(f"  WER mean: {wer_results['wer_mean']:.3f}")
    print(f"  Samples:  {wer_results['num_samples']}")

    # Combine results
    combined = {
        "checkpoint": args.checkpoint,
        "index_file": args.index_file,
        "si_snri": {
            "mean": sisnri_results["si_snri_mean"],
            "std": sisnri_results["si_snri_std"],
            "median": sisnri_results["si_snri_median"],
            "num_samples": sisnri_results["num_samples"],
        },
        "localisation": {
            "iou_mean": loc_results["iou_mean"],
            "num_samples": loc_results["num_samples"],
        },
        "wer": {
            "wer_mean": wer_results["wer_mean"],
            "num_samples": wer_results["num_samples"],
        },
    }

    return combined


def main():
    parser = argparse.ArgumentParser(description="Run full evaluation pipeline")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--index_file", type=str, default="cache/index.json", help="Dataset index file")
    parser.add_argument("--transcripts", type=str, default="data/transcripts.json", help="Transcripts JSON for WER")
    parser.add_argument("--gt_boxes", type=str, default=None, help="Ground-truth bounding boxes JSON for localisation")
    parser.add_argument("--n_sources", type=int, default=2, help="Number of sources")
    parser.add_argument("--asr_model", type=str, default="whisper", help="ASR model for WER evaluation")
    parser.add_argument("--cpu", action="store_true", help="Force CPU evaluation")
    parser.add_argument("--log_every", type=int, default=50, help="Log every N batches")
    parser.add_argument("--output", type=str, default="evaluation_results.json", help="Output JSON path")
    args = parser.parse_args()

    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
    if not os.path.exists(args.index_file):
        raise FileNotFoundError(f"Index file not found: {args.index_file}")
    if args.gt_boxes and not os.path.exists(args.gt_boxes):
        print(f"WARNING: GT boxes file not found: {args.gt_boxes} (localisation will use placeholders)")
    if not os.path.exists(args.transcripts):
        print(f"WARNING: Transcripts file not found: {args.transcripts} (WER will be placeholder)")

    results = evaluate_all(args)

    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)

    # Cleanup temp files
    for tmp in ["_tmp_sisnri.json", "_tmp_localisation.json", "_tmp_wer.json"]:
        if os.path.exists(tmp):
            os.remove(tmp)

    print("\n" + "=" * 60)
    print("EVALUATION COMPLETE")
    print("=" * 60)
    print(f"Results saved to {args.output}")
    print(f"\nSummary:")
    print(f"  SI-SNRi:     {results['si_snri']['mean']:.2f} dB (±{results['si_snri']['std']:.2f})")
    print(f"  Localisation IoU: {results['localisation']['iou_mean']:.3f}")
    print(f"  WER:         {results['wer']['wer_mean']:.3f}")


if __name__ == "__main__":
    main()