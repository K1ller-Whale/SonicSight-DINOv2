#!/usr/bin/env python3
"""Phase-2 fusion diagnostic for SonicSight-DINOv2.

Runs DECISIVE, CHEAP ablations on a few validation batches using a trained
Phase-2 checkpoint, to tell you *why* val SI-SNR is stuck near the Phase-1
baseline -- without training for more epochs.

Usage:
  python scripts/phase2_doctor.py \
      --ckpt 'checkpoints/phase2/last.ckpt' \
      --index-file ./cache/index.json \
      --num-batches 8
"""

import argparse
import torch
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.models.separator import SeparatorModule
from src.data.datamodule import AudioVisualDataModule


def sisnr_db_for_batch(model, batch, device, visual_mode="real"):
    """Mean SI-SNR dB for one batch under a given visual manipulation."""
    mix = batch["mixture_stft"].to(device)
    tgt = batch["target_waveforms"].to(device)
    vf = batch.get("visual_features")
    if vf is None:
        raise RuntimeError(
            "Batch has no 'visual_features' -- the val loader is NOT feeding "
            "visuals. That alone would pin Phase 2 at the audio-only baseline."
        )
    vf = vf.to(device)
    if visual_mode == "swap":  # mismatched vision (flip source dim)
        vf = torch.flip(vf, dims=[1])
    elif visual_mode == "zero":  # vision removed, fusion path still active
        vf = torch.zeros_like(vf)
    elif visual_mode == "noise":  # random vision
        vf = torch.randn_like(vf)
    preds = model(mix, visual_features=vf)
    # FIXED-ORDER (non-PIT) scoring. output[n] is conditioned on
    # visual_features[n], which the dataset aligns with target_waveforms[n], so
    # we must score output[n] against target[n] directly. Using PIT here (the
    # old behavior) re-matches outputs to targets by audio similarity and would
    # silently UNDO the swap ablation -- making swap == baseline no matter what,
    # which is exactly the misleading result we got before this fix.
    loss = model._fixed_order_si_snr(preds, tgt)
    return (-loss).item()


def run_condition(model, batches, device, gate_value=None, visual_mode="real"):
    orig = model.fusion_gate.detach().clone()
    if gate_value is not None:
        with torch.no_grad():
            model.fusion_gate.fill_(float(gate_value))
    vals = []
    with torch.no_grad():
        for batch in batches:
            vals.append(sisnr_db_for_batch(model, batch, device, visual_mode))
    with torch.no_grad():
        model.fusion_gate.copy_(orig)
    return sum(vals) / len(vals)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--index-file", default="./cache/index.json")
    ap.add_argument("--num-batches", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument(
        "--allow-same-category",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Force same-category mixing on/off for the diagnostic. If omitted, "
            "uses the checkpoint's saved cfg value. Pass --allow-same-category "
            "to test same-category routing, or --no-allow-same-category to "
            "force cross-category."
        ),
    )
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ---- rebuild model from the checkpoint's saved hyperparameters ----
    ckpt = torch.load(args.ckpt, map_location="cpu")
    hp = ckpt.get("hyper_parameters", {})
    cfg = hp.get("cfg")
    phase = hp.get("phase", "phase2")
    if cfg is None:
        raise RuntimeError("Checkpoint has no saved cfg; pass it via Hydra instead.")
    if phase != "phase2":
        print(f"[warn] checkpoint phase is '{phase}', expected 'phase2'")

    model = SeparatorModule(cfg=cfg, phase="phase2")
    incompat = model.load_state_dict(ckpt["state_dict"], strict=False)
    print(
        f"missing keys: {len(incompat.missing_keys)} | "
        f"unexpected keys: {len(incompat.unexpected_keys)}"
    )
    model.eval().to(device)

    gate = model.fusion_gate.detach().float().mean().item()
    print("=" * 64)
    print(f"TRAINED fusion_gate value: {gate:+.5f}")
    print("  (init is 0.0; this is how far the gate has moved during Phase 2)")
    print("=" * 64)

    # ---- val data ----
    n_sources = cfg.get("model", {}).get("n_sources", 2)
    if args.allow_same_category is not None:
        allow_same = args.allow_same_category
        _asc_src = "CLI override (--allow-same-category)"
    else:
        allow_same = cfg.get("data", {}).get("allow_same_category", False)
        _asc_src = "checkpoint cfg"
    print(f"allow_same_category: {allow_same}  [{_asc_src}]")
    dm = AudioVisualDataModule(
        cache_to_ram=False,
        index_file=args.index_file,
        n_sources=n_sources,
        batch_size=args.batch_size,
        val_batch_size=args.batch_size,
        num_workers=2,
        include_visual=True,
        seed=42,
        curriculum_schedule=None,
        allow_same_category=allow_same,
    )
    dm.setup(stage="fit")
    loader = dm.val_dataloader()

    batches = []
    for i, b in enumerate(loader):
        batches.append(b)
        if len(batches) >= args.num_batches:
            break
    print(f"Collected {len(batches)} val batches (bs={args.batch_size}).\n")

    # ---- visual feature sanity: do the two sources even differ? ----
    vf0 = batches[0].get("visual_features")
    if vf0 is not None and vf0.shape[1] >= 2:
        d = (vf0[:, 0] - vf0[:, 1]).abs().mean().item()
        m = vf0.abs().mean().item()
        print(f"visual_features shape: {tuple(vf0.shape)}")
        print(f"per-source visual divergence |s0-s1| / |mean| = {d / (m + 1e-9):.4f}")
        print(
            "  (~0 => both sources get identical visuals; vision CANNOT disambiguate)\n"
        )

    # ---- attention entropy on real visuals ----
    with torch.no_grad():
        _ = sisnr_db_for_batch(model, batches[0], device, "real")
        ent = model._compute_attention_entropy().item()
    print(f"mean cross-attention entropy: {ent:.4f}")
    print("  (very high/flat => attention is ~uniform, not localizing)\n")

    # ---- the decisive ablations ----
    base = run_condition(model, batches, device)  # trained gate
    g0 = run_condition(model, batches, device, gate_value=0.0)  # == Phase 1
    ghi = run_condition(model, batches, device, gate_value=1.0)  # force fusion on
    swap = run_condition(
        model, batches, device, visual_mode="swap"
    )  # mismatched vision
    zero = run_condition(model, batches, device, visual_mode="zero")  # vision removed
    noise = run_condition(model, batches, device, visual_mode="noise")  # random vision

    print("-" * 64)
    print(f"{'condition':<34}{'val SI-SNR dB':>14}")
    print("-" * 64)
    print(f"{'baseline (trained gate)':<34}{base:>14.3f}")
    print(f"{'gate = 0  (== Phase-1 path)':<34}{g0:>14.3f}")
    print(f"{'gate = 1  (fusion forced on)':<34}{ghi:>14.3f}")
    print(f"{'real visuals, SWAPPED sources':<34}{swap:>14.3f}")
    print(f"{'visuals ZEROED':<34}{zero:>14.3f}")
    print(f"{'visuals = random NOISE':<34}{noise:>14.3f}")
    print("-" * 64)

    print("\nHOW TO READ THIS:")
    print(
        "* baseline == gate0  -> gate ~ 0: model ignores vision (no learning signal)."
    )
    print("* baseline == swap == zero -> output does not depend on vision content;")
    print("    fusion is wired but visual info is unused/uninformative.")
    print("* gate1 << gate0 (much worse) -> attended features are OOD/misaligned;")
    print("    fusion would hurt if the gate grew -> alignment/feature bug.")
    print("* gate1 > gate0 but baseline ~ gate0 -> fusion HELPS but gate grows too")
    print("    slowly: raise lr_fusion / shorten warmup / warm-start the gate.")
    print("* swap < baseline (clearly worse) -> GOOD: model truly uses vision.")


if __name__ == "__main__":
    main()
