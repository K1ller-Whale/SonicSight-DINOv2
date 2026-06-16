#!/usr/bin/env python3
"""
preprocess_music_v2.py  ──  Kaggle preprocessing script (revised)
DINOv2-Guided Audio-Visual Source Separation — MUSIC Dataset

WHY THIS IS DIFFERENT FROM v1
------------------------------
v1 pre-cached DINOv2 features as [150, 1024, 768] float16 per clip.
That costs 226 MB/clip × ~10,720 clips = ~2.4 TB — does not fit on Kaggle.

v2 fixes this by running in TWO MODES:

  MODE A  audio_only=True  (USE FOR PHASE 1)
    Writes: audio/{clip_id}.pt  [96000] float32
    Disk:   ~380 KB/clip × 10,720 clips ≈ 4 GB total
    Works with: MixAndSepareDataset(include_visual=False)

  MODE B  audio_only=False  (USE FOR PHASE 2 / 3)
    Writes: audio/{clip_id}.pt  [96000] float32
            video/{clip_id}.mp4 6-second 480p H.264 clip (no audio stream)
    Disk:   ~3 MB/clip × 10,720 clips ≈ 32 GB total → fits on Kaggle
    DINOv2 runs at TRAINING TIME inside the model forward pass (~8 ms/sample
    on A100) — not during preprocessing.
    Requires: dataset.py to be updated (see DATASET CHANGES section below).

DISK BUDGET (per mode, full MUSIC solo set)
-------------------------------------------
Mode A  audio only:        ~4 GB     ← Phase 1, works today
Mode B  audio + mp4 clips: ~36 GB    ← Phase 2/3, fits on Kaggle Plus (100 GB)

DATASET CHANGES REQUIRED FOR MODE B
-------------------------------------
dataset.py must detect when visual_paths[0] ends in '.mp4' and return raw
[N_VIDEO_FRAMES, 3, H, W] uint8 frames instead of pre-computed features.
DINOv2 then runs inside the model's forward() method (not in the dataset).
See the stub at the bottom of this file.

CONSTANTS (must match src/data/preprocessing.py)
-------------------------------------------------
SR=16000, CLIP_LENGTH=96000, N_FFT=512, HOP=160, WIN=400
N_STFT_FRAMES=601, N_VIDEO_FRAMES=150, IMAGE_SIZE=448
"""

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 0 — Install dependencies
# ─────────────────────────────────────────────────────────────────────────────
import subprocess, sys


def _pip(*pkgs):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *pkgs])


_pip("yt-dlp", "torchaudio")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1 — Imports
# ─────────────────────────────────────────────────────────────────────────────
import json, logging, math, os, random, tempfile, urllib.request
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
import torchaudio
from tqdm import tqdm

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2 — Constants
# ─────────────────────────────────────────────────────────────────────────────
SR = 16_000
CLIP_LENGTH = 96_000  # 6 s × 16 kHz
CLIP_SECS = 6.0
N_FFT = 512
HOP_LENGTH = 160
WIN_LENGTH = 400
N_STFT_FRAMES = 601  # 1 + CLIP_LENGTH // HOP_LENGTH
N_VIDEO_FRAMES = 150  # 6 s × 25 fps
FPS = 25
IMAGE_SIZE = 448  # DINOv2-Base input resolution

TRAIN_FRAC, VAL_FRAC, TEST_FRAC = 0.80, 0.10, 0.10

MUSIC_CATEGORIES = [
    "accordion",
    "acoustic_guitar",
    "cello",
    "clarinet",
    "erhu",
    "flute",
    "saxophone",
    "trumpet",
    "tuba",
    "violin",
    "xylophone",
]

CACHE_ROOT = Path("/kaggle/working/cache")
AUDIO_DIR = CACHE_ROOT / "audio"
VIDEO_CLIPS_DIR = CACHE_ROOT / "video_clips"  # mp4 clips (Mode B only)
RAW_VIDEOS_DIR = CACHE_ROOT / "raw_videos"  # full-length downloads (temp)
INDEX_FILE = CACHE_ROOT / "index.json"

MUSIC_SOLO_JSON_URL = (
    "https://raw.githubusercontent.com/roudimit/MUSIC_dataset/master/"
    "MUSIC_solo_videos.json"
)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3 — Logging and directory setup
# ─────────────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("preprocess_music_v2")

for _d in [CACHE_ROOT, AUDIO_DIR, VIDEO_CLIPS_DIR, RAW_VIDEOS_DIR]:
    _d.mkdir(parents=True, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4 — Fetch MUSIC solo video IDs from GitHub
# ─────────────────────────────────────────────────────────────────────────────
def fetch_video_ids() -> Dict[str, List[str]]:
    log.info("Fetching MUSIC_solo_videos.json from GitHub ...")
    with urllib.request.urlopen(MUSIC_SOLO_JSON_URL, timeout=30) as resp:
        raw = json.loads(resp.read().decode())
    normalised: Dict[str, List[str]] = {}
    for key, ids in raw.items():
        norm = key.lower().replace(" ", "_")
        if norm != "version":
            normalised[norm] = ids

    normalised = normalised["videos"]
    total = sum(len(v) for v in normalised.values())
    log.info(f"  {total} video IDs across {len(normalised)} categories")
    return normalised


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5 — Download a YouTube video with yt-dlp
# ─────────────────────────────────────────────────────────────────────────────
def download_video(yt_id: str) -> Optional[Path]:
    expected = RAW_VIDEOS_DIR / f"{yt_id}.mp4"
    if expected.exists():
        return expected

    cmd = [
        "yt-dlp",
        "-f",
        "bestvideo[height<=480][ext=mp4]+bestaudio[ext=m4a]/best[height<=480]/best",
        "--no-playlist",
        "--no-warnings",
        "-q",
        "--merge-output-format",
        "mp4",
        "-o",
        str(RAW_VIDEOS_DIR / f"{yt_id}.%(ext)s"),
        f"https://www.youtube.com/watch?v={yt_id}",
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, timeout=180)
        if result.returncode == 0 and expected.exists():
            return expected
        for f in RAW_VIDEOS_DIR.glob(f"{yt_id}.*"):
            if f.suffix in {".mp4", ".mkv", ".webm"}:
                return f
        log.warning(f"  [{yt_id}] download failed")
        return None
    except (subprocess.TimeoutExpired, Exception) as e:
        log.warning(f"  [{yt_id}] download error: {e}")
        return None


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 6 — Get video duration via ffprobe
# ─────────────────────────────────────────────────────────────────────────────
def get_video_duration(video_path: Path) -> float:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(video_path),
    ]
    try:
        return float(
            subprocess.check_output(cmd, stderr=subprocess.DEVNULL, timeout=15).strip()
        )
    except Exception:
        return 0.0


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 7 — Extract audio segment → [CLIP_LENGTH] float32
# ─────────────────────────────────────────────────────────────────────────────
def extract_audio(video_path: Path, start_sec: float) -> Optional[torch.Tensor]:
    """Extracts 6s mono 16kHz audio. Returns [96000] float32 or None."""
    cmd = [
        "ffmpeg",
        "-y",
        "-ss",
        str(start_sec),
        "-t",
        str(CLIP_SECS),
        "-i",
        str(video_path),
        "-vn",
        "-ac",
        "1",
        "-ar",
        str(SR),
        "-f",
        "wav",
        "pipe:1",
        "-loglevel",
        "error",
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, timeout=30)
        if result.returncode != 0 or not result.stdout:
            return None
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tf:
            tf.write(result.stdout)
            tmp = tf.name
        waveform, loaded_sr = torchaudio.load(tmp)
        os.unlink(tmp)
        if waveform.shape[0] > 1:
            waveform = waveform.mean(0, keepdim=True)
        wave = waveform.squeeze(0).float()
        if loaded_sr != SR:
            wave = torchaudio.transforms.Resample(loaded_sr, SR)(
                wave.unsqueeze(0)
            ).squeeze(0)
        if wave.shape[0] < CLIP_LENGTH:
            wave = F.pad(wave, (0, CLIP_LENGTH - wave.shape[0]))
        return wave[:CLIP_LENGTH]
    except Exception as e:
        log.warning(f"  Audio error at {start_sec:.1f}s: {e}")
        return None


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 8 — Extract 6-second mp4 clip (Mode B only)
#
# WHY mp4 AND NOT DINO FEATURES:
#   [150, 1024, 768] float16 = 226 MB per clip → 2.4 TB for full dataset.
#   A 6-second 480p H.264 clip = ~2-4 MB per clip → ~32 GB for full dataset.
#   DINOv2 runs at training time inside the model forward pass (~8 ms/sample
#   on A100). This is the correct architecture — DINOv2 is a frozen model
#   sub-module, not a preprocessing step.
# ─────────────────────────────────────────────────────────────────────────────
def extract_video_clip(
    video_path: Path, start_sec: float, clip_id: str
) -> Optional[Path]:
    """
    Extracts a 6-second mp4 clip at 480p, 25fps, with bicubic scale+crop
    to IMAGE_SIZE × IMAGE_SIZE. Audio stream is stripped (audio saved separately).

    Returns the clip path or None on failure.

    Output file size: ~2-4 MB per clip (H.264 CRF=23).
    Total for ~10,720 clips: ~25-43 GB.
    """
    out_path = VIDEO_CLIPS_DIR / f"{clip_id}.mp4"
    if out_path.exists():
        return out_path

    cmd = [
        "ffmpeg",
        "-y",
        "-ss",
        str(start_sec),
        "-t",
        str(CLIP_SECS),
        "-i",
        str(video_path),
        "-vf",
        (
            f"fps={FPS},"
            # Scale so shortest side = IMAGE_SIZE, then centre-crop to square
            f"scale='if(gt(iw,ih),{IMAGE_SIZE}*iw/ih,{IMAGE_SIZE})':"
            f"'if(gt(iw,ih),{IMAGE_SIZE},{IMAGE_SIZE}*ih/iw)':"
            f"flags=bicubic,"
            f"crop={IMAGE_SIZE}:{IMAGE_SIZE}"
        ),
        "-c:v",
        "libx264",
        "-crf",
        "23",  # quality (lower = larger file; 23 = H.264 default)
        "-preset",
        "fast",
        "-frames:v",
        str(N_VIDEO_FRAMES),  # cap at exactly 150 frames
        "-an",  # no audio stream (audio saved as .pt separately)
        str(out_path),
        "-loglevel",
        "error",
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, timeout=60)
        if result.returncode == 0 and out_path.exists():
            return out_path
        log.warning(f"  [{clip_id}] video clip extraction failed")
        return None
    except Exception as e:
        log.warning(f"  [{clip_id}] video clip error: {e}")
        return None


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 9 — Assign 80/10/10 splits by video identity
# ─────────────────────────────────────────────────────────────────────────────
def assign_splits(yt_ids: List[str], seed: int = 42) -> Dict[str, str]:
    """80/10/10 by VIDEO identity — prevents data leakage across clips."""
    rng = random.Random(seed)
    ids = sorted(yt_ids)
    rng.shuffle(ids)
    n = len(ids)
    n_val = max(1, round(n * VAL_FRAC))
    n_test = max(1, round(n * TEST_FRAC))
    n_train = n - n_val - n_test
    splits = {}
    for i, vid in enumerate(ids):
        if i < n_train:
            splits[vid] = "train"
        elif i < n_train + n_val:
            splits[vid] = "val"
        else:
            splits[vid] = "test"
    return splits


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 10 — Build index.json entry
# ─────────────────────────────────────────────────────────────────────────────
def _build_entry(
    clip_id: str,
    split: str,
    category: str,
    audio_path: Path,
    visual_path: Optional[Path],  # mp4 clip path (Mode B) or None (Mode A)
) -> Dict:
    """
    Builds the index.json entry for one clip.

    Mode A (audio_only=True):
        visual_paths = []  →  dataset.py skips visual loading entirely
        crm_path     = None (cRM computed on-the-fly for all splits in Mode A)

    Mode B (audio_only=False):
        visual_paths = [path_to_mp4]
        dataset.py must detect .mp4 extension and return raw frames instead
        of pre-computed features (see DATASET CHANGES section at bottom).
        crm_path = None  (cRM computed on-the-fly — correct for all splits
        since the mixture is never pre-determined)
    """
    return {
        "split": split,
        "category": category,
        "source_paths": [str(audio_path)],
        "visual_paths": [str(visual_path)] if visual_path is not None else [],
        "crm_path": None,
    }


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 11 — Process one 6-second clip
# ─────────────────────────────────────────────────────────────────────────────
def process_clip(
    clip_id: str,
    video_path: Path,
    start_sec: float,
    split: str,
    category: str,
    audio_only: bool,
) -> Optional[Dict]:
    audio_path = AUDIO_DIR / f"{clip_id}.pt"

    # ── Audio (always) ────────────────────────────────────────────────────
    if not audio_path.exists():
        wave = extract_audio(video_path, start_sec)
        if wave is None:
            log.warning(f"  [{clip_id}] audio failed — skipping")
            return None
        torch.save(wave, audio_path)

    if audio_only:
        return _build_entry(clip_id, split, category, audio_path, None)

    # ── Video clip (Mode B) ───────────────────────────────────────────────
    clip_path = extract_video_clip(video_path, start_sec, clip_id)
    if clip_path is None:
        log.warning(f"  [{clip_id}] video clip failed — skipping")
        return None

    return _build_entry(clip_id, split, category, audio_path, clip_path)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 12 — Main pipeline
# ─────────────────────────────────────────────────────────────────────────────
def main(
    audio_only: bool = True,
    max_clips_per_video: Optional[int] = None,
) -> None:
    """
    Parameters
    ----------
    audio_only : bool
        True  → Mode A: write audio .pt files only. Use for Phase 1.
                Disk: ~4 GB for full MUSIC solo set.
        False → Mode B: write audio .pt + 6-second mp4 clips. Use for Phase 2/3.
                Disk: ~36 GB for full MUSIC solo set.

    max_clips_per_video : int or None
        None  → all non-overlapping 6-second clips per video.
        1     → one clip per video (fastest smoke test, ~540 clips total).
        2     → two clips per video (used in the smoke test that produced
                the disk usage report above).

    Typical usage
    -------------
    # Phase 1 audio-only smoke test (2 clips per video, Mode A):
        main(audio_only=True, max_clips_per_video=2)

    # Phase 1 full run (Mode A):
        main(audio_only=True)

    # Phase 2/3 smoke test (1 clip per video, Mode B):
        main(audio_only=False, max_clips_per_video=1)

    # Phase 2/3 full run (Mode B):
        main(audio_only=False)
    """
    mode_label = "A (audio only)" if audio_only else "B (audio + mp4 clips)"
    log.info(f"Mode: {mode_label}")
    log.info(f"Max clips per video: {max_clips_per_video or 'unlimited'}")

    # Disk estimate
    if audio_only:
        log.info("Estimated disk: ~4 GB for full MUSIC solo set (Mode A)")
    else:
        log.info("Estimated disk: ~36 GB for full MUSIC solo set (Mode B)")
        log.info("  DINOv2 runs at TRAINING TIME — not during preprocessing")

    # Load or resume index
    if INDEX_FILE.exists():
        with open(INDEX_FILE) as f:
            index: Dict = json.load(f)
        log.info(f"Resuming — {len(index)} clips already indexed")
    else:
        index = {}

    category_ids = fetch_video_ids()

    for category in MUSIC_CATEGORIES:
        yt_ids = category_ids.get(category, [])
        if not yt_ids:
            log.warning(f"No IDs for '{category}' — skipping")
            continue

        log.info(f"\n{'─'*64}")
        log.info(f"Category: {category}  ({len(yt_ids)} videos)")

        split_map = assign_splits(yt_ids, seed=42)

        for yt_id in tqdm(yt_ids, desc=category, unit="video"):
            split = split_map[yt_id]

            video_path = download_video(yt_id)
            if video_path is None:
                continue

            duration = get_video_duration(video_path)
            if duration < CLIP_SECS:
                log.warning(f"  [{yt_id}] too short ({duration:.1f}s)")
                continue

            n_clips = int(duration // CLIP_SECS)
            if max_clips_per_video is not None:
                n_clips = min(n_clips, max_clips_per_video)

            for k in range(n_clips):
                clip_id = f"{category}_{yt_id}_{k:03d}"
                if clip_id in index:
                    continue

                entry = process_clip(
                    clip_id=clip_id,
                    video_path=video_path,
                    start_sec=k * CLIP_SECS,
                    split=split,
                    category=category,
                    audio_only=audio_only,
                )
                if entry:
                    index[clip_id] = entry

            # Checkpoint after every video
            with open(INDEX_FILE, "w") as f:
                json.dump(index, f, indent=2)

    # Summary
    split_counts = {}
    for meta in index.values():
        s = meta["split"]
        split_counts[s] = split_counts.get(s, 0) + 1

    log.info(f"\n{'═'*64}")
    log.info(f"Done. Total clips: {len(index)}  splits: {split_counts}")
    log.info(f"index.json → {INDEX_FILE}")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 13 — DATASET CHANGES REQUIRED FOR MODE B
# ─────────────────────────────────────────────────────────────────────────────
# In Mode B, visual_paths[0] ends in '.mp4' instead of '.pt'.
# dataset.py must detect this and return raw frames for the model to process.
#
# The change goes in MixAndSepareDataset.__getitem__ where visual features
# are loaded. Replace the current torch.load(vpath) block with:
#
#   if vpath.endswith('.mp4'):
#       vf = _load_frames_from_mp4(vpath)   # returns [150, 3, 448, 448] uint8
#   else:
#       vf = torch.load(vpath, weights_only=False)   # pre-computed [150, 1024, 768]
#
# And add this helper to dataset.py:
#
#   def _load_frames_from_mp4(path: str) -> torch.Tensor:
#       """Returns [N_VIDEO_FRAMES, 3, H, W] uint8 tensor from a 6s mp4 clip."""
#       import subprocess, numpy as np
#       cmd = [
#           "ffmpeg", "-i", path,
#           "-f", "rawvideo", "-pix_fmt", "rgb24",
#           "pipe:1", "-loglevel", "error",
#       ]
#       result = subprocess.run(cmd, capture_output=True)
#       frame_bytes = IMAGE_SIZE * IMAGE_SIZE * 3
#       n_frames = len(result.stdout) // frame_bytes
#       frames = np.frombuffer(result.stdout[:n_frames*frame_bytes], dtype=np.uint8)
#       frames = frames.reshape(n_frames, IMAGE_SIZE, IMAGE_SIZE, 3)
#       if n_frames < N_VIDEO_FRAMES:
#           pad = np.zeros((N_VIDEO_FRAMES - n_frames, IMAGE_SIZE, IMAGE_SIZE, 3), np.uint8)
#           frames = np.concatenate([frames, pad], axis=0)
#       # [T, H, W, 3] uint8 → [T, 3, H, W] uint8
#       return torch.from_numpy(frames[:N_VIDEO_FRAMES]).permute(0, 3, 1, 2)
#
# The model's forward() method must then run DINOv2 on these raw frames.
# DINOv2 is a frozen nn.Module sub-component of the separator model — it is
# called once per batch at the start of forward(), producing [T, 1024, 768]
# float16 features that feed into the cross-modal attention module.
#
# DINOv2 on 150 frames, A100 FP16: ~8 ms per sample (negligible overhead).
# ─────────────────────────────────────────────────────────────────────────────


if __name__ == "__main__":
    # ── Phase 1 smoke test (audio only, 2 clips per video) ────────────────
    # main(audio_only=True, max_clips_per_video=2)

    # ── Phase 1 full run ──────────────────────────────────────────────────
    main(audio_only=True)

    # ── Phase 2/3 smoke test (audio + mp4, 1 clip per video) ─────────────
    # main(audio_only=False, max_clips_per_video=1)

    # ── Phase 2/3 full run ────────────────────────────────────────────────
    # main(audio_only=False)

    # Default: Phase 1 audio-only (safe starting point)
    # main(audio_only=True, max_clips_per_video=2)
