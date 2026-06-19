"""Full evaluation pipeline runner.

Runs SI-SNRi, SDR/SIR/SAR, localisation (IoU), WER, and zero-shot evaluations
in sequence and saves combined results to a single JSON file.

Usage:
    python scripts/run_evaluation.py \
        --checkpoint checkpoints/best.ckpt \
        --index_file cache/index.json \
        --output outputs/evaluation_results.json
"""
import argparse
import json
import os
import sys
from typing import Dict

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from evaluation.eval_sisnri import evaluate_sisnri
from evaluation.eval_sdr import evaluate_sdr
from evaluation.eval_localisation import evaluate_localisation
from evaluation.eval_wer import evaluate_wer
from evaluation.eval_zero_shot import evaluate_zero_shot


def evaluate_all(args) -> Dict:
    """Run all evaluations and return combined results."""
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print(f"Using device: {device}")

    # Ensure outputs directory exists
    os.makedirs("outputs", exist_ok=True)

    # 1. SI-SNRi evaluation (with PIT)
    print("\n" + "=" * 60)
    print("1/5 Running SI-SNRi evaluation (with PIT)...")
    print("=" * 60)
    sisnri_args = argparse.Namespace(
        checkpoint=args.checkpoint,
        index_file=args.index_file,
        n_sources=args.n_sources,
        cpu=args.cpu,
        log_every=args.log_every,
        output="outputs/eval_sisnri.json",
    )
    sisnri_results = evaluate_sisnri(sisnri_args)
    print(f"  SI-SNRi mean:  {sisnri_results['si_snri_mean']:.2f} dB")
    print(f"  SI-SNRi std:   {sisnri_results['si_snri_std']:.2f} dB")
    print(f"  SI-SNRi median:{sisnri_results['si_snri_median']:.2f} dB")
    print(f"  Samples:       {sisnri_results['num_samples']}")

    # 2. SDR/SIR/SAR evaluation
    print("\n" + "=" * 60)
    print("2/5 Running SDR/SIR/SAR evaluation...")
    print("=" * 60)
    sdr_args = argparse.Namespace(
        checkpoint=args.checkpoint,
        index_file=args.index_file,
        n_sources=args.n_sources,
        cpu=args.cpu,
        log_every=args.log_every,
        output="outputs/eval_sdr.json",
    )
    sdr_results = evaluate_sdr(sdr_args)
    print(f"  SDR mean: {sdr_results['SDR_mean']:.2f} dB (±{sdr_results['SDR_std']:.2f})")
    print(f"  SIR mean: {sdr_results['SIR_mean']:.2f} dB (±{sdr_results['SIR_std']:.2f})")
    print(f"  SAR mean: {sdr_results['SAR_mean']:.2f} dB (±{sdr_results['SAR_std']:.2f})")
    print(f"  Samples:  {sdr_results['num_samples']}")

    # 3. Localisation evaluation (top-50 patches, YOLOv8 GT)
    print("\n" + "=" * 60)
    print("3/5 Running localisation (IoU) evaluation...")
    print("=" * 60)
    loc_args = argparse.Namespace(
        checkpoint=args.checkpoint,
        index_file=args.index_file,
        n_sources=args.n_sources,
        gt_boxes=args.gt_boxes,
        cpu=args.cpu,
        use_yolo=args.use_yolo,
        yolo_model=args.yolo_model,
        yolo_conf=args.yolo_conf,
        output="outputs/eval_localisation.json",
    )
    loc_results = evaluate_localisation(loc_args)
    print(f"  Loc Acc (IoU>0.3): {loc_results['loc_acc']:.3f}")
    print(f"  Mean IoU:          {loc_results['mean_iou']:.3f}")
    print(f"  Frames evaluated:  {loc_results['n_frames_evaluated']}")

    # 4. WER evaluation (with 16kHz resample and mixture baseline)
    print("\n" + "=" * 60)
    print("4/5 Running WER evaluation...")
    print("=" * 60)
    if os.path.exists(args.transcripts):
        wer_args = argparse.Namespace(
            checkpoint=args.checkpoint,
            index_file=args.index_file,
            transcripts=args.transcripts,
            n_sources=args.n_sources,
            asr_model=args.asr_model,
            cpu=args.cpu,
            sample_rate=args.sample_rate,
            output="outputs/eval_wer.json",
        )
        wer_results = evaluate_wer(wer_args)
    else:
        print(f"  Skipping WER: transcripts file not found at {args.transcripts}")
        wer_results = {
            "wer_mean": -1.0,
            "mixture_wer_mean": -1.0,
            "num_samples": 0,
            "per_sample": [],
            "mixture_per_sample": [],
        }
    print(f"  Separated WER mean: {wer_results['wer_mean']:.3f}")
    print(f"  Mixture WER mean:   {wer_results['mixture_wer_mean']:.3f}")
    print(f"  Samples:            {wer_results['num_samples']}")

    # 5. Zero-shot evaluation
    print("\n" + "=" * 60)
    print("5/5 Running zero-shot evaluation...")
    print("=" * 60)
    zs_args = argparse.Namespace(
        checkpoint=args.checkpoint,
        index_file=args.index_file,
        n_sources=args.n_sources,
        cpu=args.cpu,
        seen_categories=args.seen_categories,
        unseen_categories=args.unseen_categories,
        output="outputs/eval_zero_shot.json",
    )
    zs_results = evaluate_zero_shot(zs_args)
    print(f"  Seen SI-SNRi:       {zs_results['seen_sisnri']:.2f} dB")
    print(f"  Zero-shot SI-SNRi:  {zs_results['zero_shot_sisnri']:.2f} dB")
    print(f"  Generalisation gap: {zs_results['generalisation_gap']:.2f} dB")

    # Combine all results
    combined = {
        "checkpoint": args.checkpoint,
        "index_file": args.index_file,
        "si_snri": {
            "mean": sisnri_results["si_snri_mean"],
            "std": sisnri_results["si_snri_std"],
            "median": sisnri_results["si_snri_median"],
            "num_samples": sisnri_results["num_samples"],
        },
        "sdr": {
            "SDR_mean": sdr_results["SDR_mean"],
            "SDR_std": sdr_results["SDR_std"],
            "SIR_mean": sdr_results["SIR_mean"],
            "SIR_std": sdr_results["SIR_std"],
            "SAR_mean": sdr_results["SAR_mean"],
            "SAR_std": sdr_results["SAR_std"],
            "num_samples": sdr_results["num_samples"],
        },
        "localisation": {
            "loc_acc": loc_results["loc_acc"],
            "mean_iou": loc_results["mean_iou"],
            "n_frames_evaluated": loc_results["n_frames_evaluated"],
        },
        "wer": {
            "separated_wer_mean": wer_results["wer_mean"],
            "mixture_wer_mean": wer_results["mixture_wer_mean"],
            "num_samples": wer_results["num_samples"],
        },
        "zero_shot": {
            "seen_sisnri": zs_results["seen_sisnri"],
            "zero_shot_sisnri": zs_results["zero_shot_sisnri"],
            "generalisation_gap": zs_results["generalisation_gap"],
        },
    }

    return combined


def main():
    parser = argparse.ArgumentParser(description="Run full evaluation pipeline")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--index_file", type=str, default="cache/index.json", help="Dataset index file")
    parser.add_argument("--transcripts", type=str, default="data/transcripts.json", help="Transcripts JSON for WER")
    parser.add_argument("--gt_boxes", type=str, default=None, help="Ground-truth bounding boxes JSON for localisation")
    parser.add_argument("--n_sources", type=int, default=None, help="Number of sources")
    parser.add_argument("--asr_model", type=str, default="whisper", help="ASR model for WER evaluation")
    parser.add_argument("--cpu", action="store_true", help="Force CPU evaluation")
    parser.add_argument("--log_every", type=int, default=50, help="Log every N batches")
    parser.add_argument("--output", type=str, default="outputs/evaluation_results.json", help="Output JSON path")
    parser.add_argument("--sample_rate", type=int, default=16000, help="Original sample rate for WER resampling")
    # Zero-shot args
    parser.add_argument("--seen_categories", type=str,
                        default="speech,music_instrument",
                        help="Comma-separated seen categories for zero-shot eval")
    parser.add_argument("--unseen_categories", type=str,
                        default="animal_bird,animal_dog,animal_cat",
                        help="Comma-separated unseen categories for zero-shot eval")
    # Localisation YOLO args
    parser.add_argument("--use_yolo", action="store_true",
                        help="Use YOLOv8 for GT boxes if JSON not provided")
    parser.add_argument("--yolo_model", type=str, default="yolov8n.pt", help="YOLOv8 model name")
    parser.add_argument("--yolo_conf", type=float, default=0.5, help="YOLOv8 confidence threshold")

    args = parser.parse_args()

    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
    if not os.path.exists(args.index_file):
        raise FileNotFoundError(f"Index file not found: {args.index_file}")
    if args.gt_boxes and not os.path.exists(args.gt_boxes):
        print(f"WARNING: GT boxes file not found: {args.gt_boxes} (localisation will use YOLOv8)")
    if not os.path.exists(args.transcripts):
        print(f"WARNING: Transcripts file not found: {args.transcripts} (WER will be placeholder)")

    # Ensure outputs directory exists
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    results = evaluate_all(args)

    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 60)
    print("EVALUATION COMPLETE")
    print("=" * 60)
    print(f"Results saved to {args.output}")
    print(f"\nSummary:")
    print(f"  SI-SNRi:        {results['si_snri']['mean']:.2f} dB (±{results['si_snri']['std']:.2f})")
    print(f"  SDR/SIR/SAR:    {results['sdr']['SDR_mean']:.2f} / {results['sdr']['SIR_mean']:.2f} / {results['sdr']['SAR_mean']:.2f} dB")
    print(f"  Loc Acc:        {results['localisation']['loc_acc']:.3f} (IoU>0.3)")
    print(f"  Mean IoU:       {results['localisation']['mean_iou']:.3f}")
    print(f"  Separated WER:  {results['wer']['separated_wer_mean']:.3f}")
    print(f"  Mixture WER:    {results['wer']['mixture_wer_mean']:.3f}")
    print(f"  Zero-shot gap:  {results['zero_shot']['generalisation_gap']:.2f} dB")


if __name__ == "__main__":
    main()
