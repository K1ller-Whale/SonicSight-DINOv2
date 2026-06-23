#!/usr/bin/env python3
"""preprocess_music_v2.py  --  Kaggle preprocessing script (optimized)
DINOv2-Guided Audio-Visual Source Separation -- MUSIC Dataset

WHAT CHANGED vs the previous version
------------------------------------
SPEED:
1. PARALLEL across videos (ProcessPoolExecutor, --workers). Downloads are
   network-bound, ffmpeg is CPU-bound; both overlap across workers.
2. ONE ffmpeg decode per clip instead of two: in Mode B a single invocation
   writes the .mp4 AND emits audio as float32 PCM on stdout.
3. NO temp wav / NO torchaudio / NO Resample: ffmpeg emits 16 kHz mono f32le
   straight to a pipe; numpy reads it.
4. Lean yt-dlp: single progressive download (no bestvideo+bestaudio merge),
   concurrent fragments, retries; only the needed prefix when clips are capped.
5. Faster encode: libx264 -preset veryfast (or NVENC via --nvenc).

CORRECT, FILE-AWARE RESUME (this is the important one for re-runs):
6. A video is skipped on re-run ONLY if its clips' OUTPUT FILES actually exist
   on disk -- verified against index.json, not a blind 'done' flag. So if you
   delete cache/video_clips (or some clips go missing), those exact clips are
   regenerated on the next run while everything else is skipped instantly.
7. When audio is already cached but the mp4 is missing, ffmpeg re-encodes
   VIDEO ONLY (no wasted audio decode) and the existing audio .pt is reused.

The TWO MODES, disk budget, split logic, index.json schema, output formats,
and the Mode-B dataset contract are UNCHANGED.

CONSTANTS (must match src/data/preprocessing.py)
------------------------------------------------
SR=16000, CLIP_LENGTH=96000, N_FFT=512, HOP=160, WIN=400
N_STFT_FRAMES=601, N_VIDEO_FRAMES=150, IMAGE_SIZE=448

Usage
-----
    python preprocess_music_v2.py --mode B --workers 12 --nvenc   # Phase 2/3
    python preprocess_music_v2.py --mode A --workers 12           # Phase 1
    python preprocess_music_v2.py --mode B --max-clips-per-video 1 --workers 8
    python preprocess_music_v2.py --mode B --force                # ignore cache
"""

import argparse
import json
import logging
import multiprocessing as mp
import os
import random
import subprocess
import sys
import urllib.request
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SR = 16_000
CLIP_LENGTH = 96_000  # 6 s x 16 kHz
CLIP_SECS = 6.0
N_FFT = 512
HOP_LENGTH = 160
WIN_LENGTH = 400
N_STFT_FRAMES = 601
N_VIDEO_FRAMES = 150  # 6 s x 25 fps
FPS = 25
IMAGE_SIZE = 448

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

CACHE_ROOT = Path("./cache")
AUDIO_DIR = CACHE_ROOT / "audio"
VIDEO_CLIPS_DIR = CACHE_ROOT / "video_clips"
RAW_VIDEOS_DIR = CACHE_ROOT / "raw_videos"
INDEX_FILE = CACHE_ROOT / "index.json"

MUSIC_SOLO_JSON_URL = (
    "https://raw.githubusercontent.com/roudimit/MUSIC_dataset/master/"
    "MUSIC_solo_videos.json"
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("preprocess_music_v2")


# ---------------------------------------------------------------------------
# Fetch MUSIC solo video IDs
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# Download a YouTube video with yt-dlp
# ---------------------------------------------------------------------------
def _existing_raw(yt_id: str) -> Optional[Path]:
    p = RAW_VIDEOS_DIR / f"{yt_id}.mp4"
    if p.exists() and p.stat().st_size > 0:
        return p
    for f in RAW_VIDEOS_DIR.glob(f"{yt_id}.*"):
        if f.suffix in {".mp4", ".mkv", ".webm"} and f.stat().st_size > 0:
            return f
    return None


def download_video(yt_id: str, max_secs: Optional[float] = None) -> Optional[Path]:
    existing = _existing_raw(yt_id)
    if existing is not None:
        return existing
    cmd = [
        "yt-dlp",
        # Single progressive file -- we re-encode video and re-extract audio,
        # so fetching+merging separate hi-fi streams is wasted work.
        "-f",
        "best[height<=480][ext=mp4]/best[height<=480]/best",
        "--no-playlist",
        "--no-warnings",
        "-q",
        "--no-part",
        "--retries",
        "3",
        "--fragment-retries",
        "3",
        "--concurrent-fragments",
        "4",
        "--merge-output-format",
        "mp4",
        "-o",
        str(RAW_VIDEOS_DIR / f"{yt_id}.%(ext)s"),
    ]
    if max_secs is not None:
        cmd += [
            "--download-sections",
            f"*0-{int(max_secs) + 2}",
            "--force-keyframes-at-cuts",
        ]
    cmd += [f"https://www.youtube.com/watch?v={yt_id}"]
    try:
        result = subprocess.run(cmd, capture_output=True, timeout=300)
        if result.returncode == 0:
            got = _existing_raw(yt_id)
            if got is not None:
                return got
        log.warning(f"  [{yt_id}] download failed")
        return None
    except Exception as e:  # incl. TimeoutExpired
        log.warning(f"  [{yt_id}] download error: {e}")
        return None


# ---------------------------------------------------------------------------
# ffprobe duration
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _pcm_to_wave(pcm: bytes) -> Optional[torch.Tensor]:
    if not pcm:
        return None
    arr = np.frombuffer(pcm, dtype=np.float32).copy()
    if arr.size == 0:
        return None
    wave = torch.from_numpy(arr)
    if wave.numel() < CLIP_LENGTH:
        wave = torch.nn.functional.pad(wave, (0, CLIP_LENGTH - wave.numel()))
    return wave[:CLIP_LENGTH].contiguous()


def _video_filter() -> str:
    # Scale shortest side to IMAGE_SIZE (cover), then centre-crop to square.
    return (
        f"fps={FPS},"
        f"scale={IMAGE_SIZE}:{IMAGE_SIZE}:force_original_aspect_ratio=increase:flags=bicubic,"
        f"crop={IMAGE_SIZE}:{IMAGE_SIZE}"
    )


def _video_codec_args(opts: dict) -> List[str]:
    if opts.get("nvenc"):
        return ["-c:v", "h264_nvenc", "-preset", "p4", "-cq", str(opts["crf"])]
    return [
        "-c:v",
        "libx264",
        "-crf",
        str(opts["crf"]),
        "-preset",
        opts.get("preset", "veryfast"),
    ]


def _audio_path(clip_id: str) -> Path:
    return AUDIO_DIR / f"{clip_id}.pt"


def _clip_path(clip_id: str) -> Path:
    return VIDEO_CLIPS_DIR / f"{clip_id}.mp4"


def outputs_ok(clip_id: str, audio_only: bool) -> bool:
    """True iff every output file this clip needs already exists on disk."""
    if not _audio_path(clip_id).exists():
        return False
    if audio_only:
        return True
    return _clip_path(clip_id).exists()


# ---------------------------------------------------------------------------
# Extract one clip. Only (re)generates the OUTPUTS THAT ARE MISSING:
#   - audio missing + video missing -> one combined ffmpeg decode
#   - video missing only            -> video-only re-encode (your delete case)
#   - audio missing only            -> audio-only decode
# ---------------------------------------------------------------------------
def extract_clip(
    video_path: Path, start_sec: float, clip_id: str, audio_only: bool, opts: dict
) -> None:
    audio_path = _audio_path(clip_id)
    clip_path = _clip_path(clip_id)
    force = bool(opts.get("force"))

    need_audio = force or not audio_path.exists()
    need_video = (not audio_only) and (force or not clip_path.exists())
    if not need_audio and not need_video:
        return

    base = [
        "ffmpeg",
        "-y",
        "-ss",
        f"{start_sec}",
        "-t",
        f"{CLIP_SECS}",
        "-i",
        str(video_path),
    ]
    cmd = list(base)
    if need_video:
        cmd += [
            "-map",
            "0:v:0",
            "-vf",
            _video_filter(),
            *_video_codec_args(opts),
            "-frames:v",
            str(N_VIDEO_FRAMES),
            "-an",
            str(clip_path),
        ]
    if need_audio:
        # audio -> float32 PCM on stdout (works whether or not video is mapped)
        cmd += [
            "-map",
            "0:a:0?" if not audio_only else "0:a:0?",
            "-vn",
            "-ac",
            "1",
            "-ar",
            str(SR),
            "-f",
            "f32le",
            "pipe:1",
        ]
    cmd += ["-loglevel", "error"]

    try:
        result = subprocess.run(cmd, capture_output=True, timeout=120)
    except Exception as e:
        log.warning(f"  [{clip_id}] ffmpeg error: {e}")
        return
    if result.returncode != 0:
        log.warning(f"  [{clip_id}] ffmpeg returncode {result.returncode}")
        return
    if need_audio:
        wave = _pcm_to_wave(result.stdout)
        if wave is None:
            log.warning(f"  [{clip_id}] empty audio")
            return
        torch.save(wave, audio_path)


# ---------------------------------------------------------------------------
# Splits (80/10/10 by video identity)
# ---------------------------------------------------------------------------
def assign_splits(yt_ids: List[str], seed: int = 42) -> Dict[str, str]:
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


def _build_entry(clip_id, split, category, audio_only) -> Dict:
    return {
        "split": split,
        "category": category,
        "source_paths": [str(_audio_path(clip_id))],
        "visual_paths": [] if audio_only else [str(_clip_path(clip_id))],
        "crm_path": None,
    }


def _k_from_clip_id(clip_id: str) -> Optional[int]:
    try:
        return int(clip_id.rsplit("_", 1)[1])
    except (IndexError, ValueError):
        return None


# ---------------------------------------------------------------------------
# Per-video worker (top-level so ProcessPoolExecutor can pickle it)
# ---------------------------------------------------------------------------
def process_video(task: dict) -> List[Tuple[str, dict]]:
    yt_id = task["yt_id"]
    split = task["split"]
    category = task["category"]
    audio_only = task["audio_only"]
    max_clips = task["max_clips"]
    opts = task["opts"]
    known: List[str] = task.get("known_clip_ids") or []
    force = bool(opts.get("force"))

    max_secs = (max_clips * CLIP_SECS) if max_clips else None

    # ---- Fast path: video already known from index -> regenerate only the
    #      clips whose output files are missing (no ffprobe needed). ----------
    if known and not force:
        targets: List[Tuple[str, float]] = []
        for cid in known:
            k = _k_from_clip_id(cid)
            if k is None:
                continue
            targets.append((cid, k * CLIP_SECS))
        missing = [(cid, s) for (cid, s) in targets if not outputs_ok(cid, audio_only)]
        if missing:
            video_path = download_video(yt_id, max_secs=max_secs)
            if video_path is not None:
                for cid, s in missing:
                    extract_clip(video_path, s, cid, audio_only, opts)
                if opts.get("cleanup_raw") and video_path.exists():
                    try:
                        video_path.unlink()
                    except OSError:
                        pass
        return [
            (cid, _build_entry(cid, split, category, audio_only))
            for (cid, _) in targets
            if outputs_ok(cid, audio_only)
        ]

    # ---- New video (or --force): download + probe + generate all clips. -----
    video_path = download_video(yt_id, max_secs=max_secs)
    if video_path is None:
        return []
    duration = get_video_duration(video_path)
    if duration < CLIP_SECS:
        log.warning(f"  [{yt_id}] too short ({duration:.1f}s)")
        return []
    n_clips = int(duration // CLIP_SECS)
    if max_clips is not None:
        n_clips = min(n_clips, max_clips)

    entries: List[Tuple[str, dict]] = []
    for k in range(n_clips):
        clip_id = f"{category}_{yt_id}_{k:03d}"
        extract_clip(video_path, k * CLIP_SECS, clip_id, audio_only, opts)
        if outputs_ok(clip_id, audio_only):
            entries.append(
                (clip_id, _build_entry(clip_id, split, category, audio_only))
            )

    if opts.get("cleanup_raw") and video_path.exists():
        try:
            video_path.unlink()
        except OSError:
            pass
    return entries


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------
def _recover_yt_id(clip_id: str, category: str) -> Optional[str]:
    # clip_id == f"{category}_{yt_id}_{k:03d}"  (yt_id may contain '_' or '-')
    if not clip_id.startswith(category + "_"):
        return None
    last = clip_id.rfind("_")
    if last <= len(category):
        return None
    return clip_id[len(category) + 1 : last]


def main(
    audio_only: bool, workers: int, max_clips_per_video: Optional[int], opts: dict
) -> None:
    force = bool(opts.get("force"))
    mode_label = "A (audio only)" if audio_only else "B (audio + mp4 clips)"
    log.info(
        f"Mode: {mode_label} | workers={workers} | "
        f"max_clips_per_video={max_clips_per_video or 'unlimited'} | force={force}"
    )

    if INDEX_FILE.exists():
        with open(INDEX_FILE) as f:
            index: Dict = json.load(f)
        log.info(f"Loaded index with {len(index)} clips")
    else:
        index = {}

    # Group already-indexed clips by their owning video.
    by_video: Dict[Tuple[str, str], List[str]] = defaultdict(list)
    for cid, meta in index.items():
        cat = meta.get("category")
        if not cat:
            continue
        ytid = _recover_yt_id(cid, cat)
        if ytid is not None:
            by_video[(cat, ytid)].append(cid)

    category_ids = fetch_video_ids()

    tasks: List[dict] = []
    skipped_cached = 0
    for category in MUSIC_CATEGORIES:
        yt_ids = category_ids.get(category, [])
        if not yt_ids:
            log.warning(f"No IDs for '{category}' -- skipping")
            continue
        split_map = assign_splits(yt_ids, seed=42)
        for yt_id in yt_ids:
            known = by_video.get((category, yt_id), [])
            if known and not force:
                # Skip ONLY if every known clip's output files exist on disk.
                if all(outputs_ok(cid, audio_only) for cid in known):
                    skipped_cached += 1
                    continue
            tasks.append(
                {
                    "yt_id": yt_id,
                    "split": split_map[yt_id],
                    "category": category,
                    "audio_only": audio_only,
                    "max_clips": max_clips_per_video,
                    "opts": opts,
                    "known_clip_ids": known,
                }
            )

    log.info(
        f"{skipped_cached} videos fully cached (skipped, no subprocess). "
        f"Dispatching {len(tasks)} videos across {workers} workers ..."
    )
    if not tasks:
        log.info("Nothing to do -- all outputs already present. (Use --force to redo.)")
        return

    def _save_index():
        tmp = INDEX_FILE.with_suffix(".json.tmp")
        with open(tmp, "w") as f:
            json.dump(index, f, indent=2)
        os.replace(tmp, INDEX_FILE)  # atomic

    try:
        from tqdm import tqdm

        pbar = tqdm(total=len(tasks), unit="video")
    except Exception:
        pbar = None

    done = 0
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(process_video, t): t["yt_id"] for t in tasks}
        for fut in as_completed(futures):
            try:
                entries = fut.result()
            except Exception as e:
                log.warning(f"  [{futures[fut]}] worker crashed: {e}")
                entries = []
            for clip_id, entry in entries:
                index[clip_id] = entry
            done += 1
            if pbar is not None:
                pbar.update(1)
            if done % 25 == 0:
                _save_index()
    if pbar is not None:
        pbar.close()

    _save_index()

    split_counts: Dict[str, int] = {}
    for meta in index.values():
        split_counts[meta["split"]] = split_counts.get(meta["split"], 0) + 1
    log.info(f"Done. Total clips: {len(index)}  splits: {split_counts}")
    log.info(f"index.json -> {INDEX_FILE}")


# ---------------------------------------------------------------------------
# DATASET CHANGES REQUIRED FOR MODE B (unchanged contract)
# ---------------------------------------------------------------------------
# In Mode B, visual_paths[0] ends in '.mp4'. dataset.py must detect this and
# return raw [N_VIDEO_FRAMES, 3, H, W] uint8 frames; DINOv2 runs inside the
# model forward() on those frames (frozen sub-module).
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Optimized MUSIC preprocessing.")
    p.add_argument(
        "--mode",
        choices=["A", "B"],
        default="B",
        help="A=audio only (Phase 1); B=audio + mp4 clips (Phase 2/3).",
    )
    p.add_argument(
        "--workers",
        type=int,
        default=min(8, os.cpu_count() or 4),
        help="Parallel videos. 6-12 typical; lower if YouTube throttles.",
    )
    p.add_argument(
        "--max-clips-per-video",
        type=int,
        default=None,
        help="Cap clips per video (also limits the download to that prefix).",
    )
    p.add_argument(
        "--nvenc",
        action="store_true",
        help="Use the GPU H.264 encoder (h264_nvenc) -- great on RTX 4090.",
    )
    p.add_argument("--crf", type=int, default=23, help="x264 CRF / nvenc CQ (quality).")
    p.add_argument("--preset", default="veryfast", help="x264 preset (speed/size).")
    p.add_argument(
        "--cleanup-raw",
        action="store_true",
        help="Delete each raw download after its clips are extracted.",
    )
    p.add_argument(
        "--skip-pip", action="store_true", help="Skip the yt-dlp pip install."
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="Ignore cached outputs and reprocess every video.",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if not args.skip_pip:
        try:
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", "-q", "yt-dlp"]
            )
        except Exception as e:
            log.warning(f"pip install yt-dlp failed ({e}); assuming it is present")
    for _d in [CACHE_ROOT, AUDIO_DIR, VIDEO_CLIPS_DIR, RAW_VIDEOS_DIR]:
        _d.mkdir(parents=True, exist_ok=True)
    try:
        mp.set_start_method("fork")
    except (RuntimeError, ValueError):
        pass
    _opts = {
        "nvenc": args.nvenc,
        "crf": args.crf,
        "preset": args.preset,
        "cleanup_raw": args.cleanup_raw,
        "force": args.force,
    }
    main(
        audio_only=(args.mode == "A"),
        workers=args.workers,
        max_clips_per_video=args.max_clips_per_video,
        opts=_opts,
    )
