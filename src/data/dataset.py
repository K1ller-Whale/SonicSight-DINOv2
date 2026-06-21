"""Audio-Visual dataset with on-the-fly mix-and-separation.

SPEC 11.2, 7.1: Synthetic mixtures from clean (source, video) pairs.
"""
import json
import os
import random
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F

from src.data.preprocessing import (
    AudioPreprocessor,
    STFTModule,
    compute_crm_targets,
    N_VIDEO_FRAMES,
    CLIP_LENGTH,
    IMAGE_SIZE,
)


class MixAndSepareDataset(Dataset):
    """Mix-and-separate dataset with on-the-fly mixing.

    Reads cached per-source audio (and optionally cached DINOv2 visual tokens),
    samples N clips, mixes their first sources, and returns the mixture STFT,
    scaled target waveforms, per-source visual features and cRM targets.
    """

    def __init__(
        self,
        index_file: str,
        n_sources: int = 2,
        split: str = "train",
        cache_dir: Optional[str] = None,
        include_visual: bool = True,
        device: str = "cpu",
        allow_same_category: bool = False,
        cache_to_ram: bool = False,
    ):
        super().__init__()
        self.index_file = index_file
        self.n_sources = n_sources
        self.split = split
        self.include_visual = include_visual
        self.device = torch.device(device)
        self.allow_same_category = allow_same_category
        # When True, precomputed DINOv2 .pt token tensors are loaded into RAM
        # once (in the main process; forked workers share copy-on-write) for
        # faster reads. When False, each access reads the .pt file from disk.
        self.cache_to_ram = cache_to_ram

        with open(index_file, "r") as f:
            self._all_meta = json.load(f)

        self.clips = [
            cid for cid, meta in self._all_meta.items()
            if meta.get("split", "train") == split
        ]
        if not self.clips:
            raise ValueError(f"No clips found for split '{split}'")

        self.clip_categories = {
            cid: self._extract_category(cid, self._all_meta[cid])
            for cid in self.clips
        }
        self.category_to_clips: Dict[str, List[str]] = {}
        for cid in self.clips:
            category = self.clip_categories[cid]
            self.category_to_clips.setdefault(category, []).append(cid)
        self.categories = sorted(self.category_to_clips)

        if self.allow_same_category:
            max_clips_in_cat = max(len(v) for v in self.category_to_clips.values())
            if max_clips_in_cat < self.n_sources:
                raise ValueError(
                    f"allow_same_category=True requires at least one category "
                    f"with {self.n_sources} clips for split '{split}', "
                    f"but the largest category only has {max_clips_in_cat} clips"
                )
            self._eligible_categories = [
                cat for cat, clips in self.category_to_clips.items()
                if len(clips) >= self.n_sources
            ]
        else:
            if len(self.categories) < self.n_sources:
                raise ValueError(
                    f"Need at least {self.n_sources} distinct categories for split "
                    f"'{split}', found {len(self.categories)}"
                )
            self._eligible_categories = self.categories

        self.cache_dir = cache_dir or os.path.dirname(index_file)
        self._stft = STFTModule()

        # Optional in-RAM cache of precomputed DINOv2 token tensors. Built once
        # in the main process so forked DataLoader workers share it copy-on-write.
        # Only .pt feature files are cached (raw mp4 visuals are never preloaded).
        self._ram_visual: Dict[str, torch.Tensor] = {}
        if self.cache_to_ram and self.include_visual:
            self._build_ram_visual_cache()

    def _build_ram_visual_cache(self) -> None:
        """Load every clip's precomputed DINOv2 tokens into memory."""
        loaded = 0
        for cid in self.clips:
            vpaths = self._all_meta[cid].get("visual_paths", []) or []
            if not vpaths:
                continue
            vpath = vpaths[0]
            if (
                vpath
                and Path(vpath).suffix.lower() == ".pt"
                and os.path.exists(vpath)
            ):
                self._ram_visual[cid] = torch.load(vpath, weights_only=False)
                loaded += 1
        print(
            f"[MixAndSepareDataset:{self.split}] cache_to_ram=True: "
            f"loaded {loaded} visual tensors into RAM"
        )

    # ------------------------------------------------------------------ #
    # Loading helpers
    # ------------------------------------------------------------------ #

    @staticmethod
    def _load_waveform(path: str) -> torch.Tensor:
        """Load waveform and enforce [L] float32."""
        if path.endswith(".pt"):
            wave = torch.load(path, weights_only=False)
        else:
            import torchaudio
            waveform, sr = torchaudio.load(path)
            ap = AudioPreprocessor()
            wave = ap(waveform, sr)
        if wave.dim() == 2:
            wave = wave.squeeze(0)
        return wave.float()

    @staticmethod
    def _load_frames_from_mp4(path: str) -> torch.Tensor:
        """Load a 6-second mp4 visual clip as [T, 3, H, W] uint8 frames."""
        cmd = [
            "ffmpeg",
            "-i",
            path,
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "pipe:1",
            "-loglevel",
            "error",
        ]
        result = subprocess.run(cmd, capture_output=True, check=False)
        if result.returncode != 0:
            raise RuntimeError(f"ffmpeg failed while reading visual clip: {path}")

        frame_bytes = IMAGE_SIZE * IMAGE_SIZE * 3
        n_frames = len(result.stdout) // frame_bytes
        if n_frames == 0:
            raise RuntimeError(f"No frames decoded from visual clip: {path}")

        raw = result.stdout[: n_frames * frame_bytes]
        frames = np.frombuffer(raw, dtype=np.uint8).reshape(
            n_frames, IMAGE_SIZE, IMAGE_SIZE, 3
        )
        if n_frames < N_VIDEO_FRAMES:
            pad = np.zeros(
                (N_VIDEO_FRAMES - n_frames, IMAGE_SIZE, IMAGE_SIZE, 3),
                dtype=np.uint8,
            )
            frames = np.concatenate([frames, pad], axis=0)
        else:
            frames = frames[:N_VIDEO_FRAMES]

        return torch.from_numpy(frames.copy()).permute(0, 3, 1, 2)

    @classmethod
    def _load_visual_tensor(cls, path: str) -> torch.Tensor:
        """Load visual data: decode mp4 frames or load a precomputed .pt tensor."""
        suffix = Path(path).suffix.lower()
        if suffix in {".mp4", ".avi", ".mov", ".mkv", ".webm"}:
            return cls._load_frames_from_mp4(path)
        return torch.load(path, weights_only=False)

    def _get_visual_tensor(self, clip_id: str, path: str) -> torch.Tensor:
        """Return a clip's visual tensor, preferring the in-RAM cache.

        When cache_to_ram is True and the clip's precomputed tokens were
        preloaded, the in-memory tensor is returned (no disk read). Otherwise
        the tensor is read from disk via _load_visual_tensor.
        """
        if self.cache_to_ram:
            cached = self._ram_visual.get(clip_id)
            if cached is not None:
                return cached
        return self._load_visual_tensor(path)

    @staticmethod
    def _pad_waveforms(waves: List[torch.Tensor]) -> List[torch.Tensor]:
        max_len = max(w.shape[-1] for w in waves)
        out = []
        for w in waves:
            if w.shape[-1] < max_len:
                w = F.pad(w, (0, max_len - w.shape[-1]))
            out.append(w)
        return out

    @staticmethod
    def _mix(waves: List[torch.Tensor],
             db_range: Tuple[float, float] = (-5.0, 5.0)
             ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Mix sources with random per-source gains; return mixture, gains, scaled."""
        stacked = torch.stack(waves)  # [N, L]
        n = stacked.shape[0]
        db = torch.rand(n) * (db_range[1] - db_range[0]) + db_range[0]
        gains = 10 ** (db / 20.0)
        scaled = stacked * gains.unsqueeze(1)
        mixture = scaled.sum(dim=0)  # [L]
        return mixture, gains, scaled

    # ------------------------------------------------------------------ #
    # Category extraction
    # ------------------------------------------------------------------ #

    @staticmethod
    def _coerce_category(value: Any) -> Optional[str]:
        if value is None:
            return None
        if isinstance(value, (list, tuple)):
            for item in value:
                coerced = MixAndSepareDataset._coerce_category(item)
                if coerced is not None:
                    return coerced
            return None
        if isinstance(value, dict):
            for key in ("category", "instrument", "label", "class", "name"):
                coerced = MixAndSepareDataset._coerce_category(value.get(key))
                if coerced is not None:
                    return coerced
            return None
        text = str(value).strip().lower()
        return text or None

    @staticmethod
    def _category_from_path(path: str) -> Optional[str]:
        parts = Path(path).parts if path else ()
        for part in reversed(parts[:-1]):
            name = Path(part).name.strip().lower()
            if name and not name.startswith("source") and name not in {"audio", "wav", "wavs"}:
                return name.split("_")[0]
        return None

    @classmethod
    def _extract_category(cls, clip_id: str, meta: Dict[str, Any]) -> str:
        category_keys = (
            "category",
            "categories",
            "instrument",
            "instruments",
            "source_category",
            "source_categories",
            "source_type",
            "source_types",
            "class",
            "classes",
            "label",
            "labels",
        )
        for key in category_keys:
            category = cls._coerce_category(meta.get(key))
            if category is not None:
                return category

        source_paths = meta.get("source_paths", [])
        if source_paths:
            category = cls._category_from_path(source_paths[0])
            if category is not None:
                return category

        identity = cls._coerce_category(meta.get("identity"))
        if identity is not None:
            return identity

        return clip_id.strip().lower().split("_")[0]

    # ------------------------------------------------------------------ #
    # Sampling strategies
    # ------------------------------------------------------------------ #

    def _sample_clip_ids(self, idx: int) -> List[str]:
        """Sample clip IDs for mixing, respecting the allow_same_category flag."""
        if self.allow_same_category:
            return self._sample_same_category_clip_ids(idx)
        return self._sample_cross_category_clip_ids(idx)

    def _sample_same_category_clip_ids(self, idx: int) -> List[str]:
        """Sample N distinct clips from the same category."""
        if self.split == "train":
            category = random.choice(self._eligible_categories)
            return random.sample(self.category_to_clips[category], self.n_sources)

        cat_idx = idx % len(self._eligible_categories)
        category = self._eligible_categories[cat_idx]
        pool = self.category_to_clips[category]
        clip_cycle = idx // len(self._eligible_categories)
        return [
            pool[(clip_cycle + i) % len(pool)]
            for i in range(self.n_sources)
        ]

    def _sample_cross_category_clip_ids(self, idx: int) -> List[str]:
        """Sample one clip from each of N distinct categories."""
        if len(self.categories) < self.n_sources:
            raise ValueError(
                f"Need at least {self.n_sources} distinct categories, "
                f"found {len(self.categories)}"
            )

        if self.split == "train":
            categories = random.sample(self.categories, self.n_sources)
            return [
                random.choice(self.category_to_clips[category])
                for category in categories
            ]

        start = idx % len(self.categories)
        categories = [
            self.categories[(start + offset) % len(self.categories)]
            for offset in range(self.n_sources)
        ]
        clip_cycle = idx // len(self.categories)
        return [
            self.category_to_clips[category][
                (clip_cycle + source_idx) % len(self.category_to_clips[category])
            ]
            for source_idx, category in enumerate(categories)
        ]

    # ------------------------------------------------------------------ #
    # Dataset protocol
    # ------------------------------------------------------------------ #

    def __len__(self) -> int:
        return len(self.clips)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        if len(self.clips) < self.n_sources:
            raise ValueError(
                f"Not enough clips ({len(self.clips)}) for n_sources={self.n_sources}"
            )
        clip_ids = self._sample_clip_ids(idx)
        source_categories = [self.clip_categories[clip_id] for clip_id in clip_ids]
        if not self.allow_same_category:
            if len(set(source_categories)) != len(source_categories):
                raise RuntimeError(
                    f"Cross-category sampling failed: {source_categories}"
                )

        # Load waveforms per source
        source_waves = []
        for clip_id in clip_ids:
            meta = self._all_meta[clip_id]
            spaths = meta.get("source_paths", [])
            if not spaths:
                raise RuntimeError(f"No source_paths for clip {clip_id}")
            wave = self._load_waveform(spaths[0])
            source_waves.append(wave)

        # Pad, mix, compute STFT
        source_waves = self._pad_waveforms(source_waves)
        mixture_wave, source_gains, scaled_sources = self._mix(source_waves)

        # Keep targets and mixture on the fixed clip length used by the model's
        # iSTFT. Matters for .pt sources not already trimmed to exactly 6 s.
        if scaled_sources.shape[-1] != CLIP_LENGTH:
            if scaled_sources.shape[-1] < CLIP_LENGTH:
                scaled_sources = F.pad(
                    scaled_sources, (0, CLIP_LENGTH - scaled_sources.shape[-1])
                )
            else:
                scaled_sources = scaled_sources[..., :CLIP_LENGTH]
            mixture_wave = scaled_sources.sum(dim=0)
        mixture_stft = self._stft(mixture_wave).squeeze(0)  # [2, F, T]

        # Targets are SCALED sources so they sum to mixture
        target_waveforms = scaled_sources  # [N, L]

        output: Dict[str, Any] = {
            "mixture_stft": mixture_stft,
            "target_waveforms": target_waveforms,
            "source_gains": source_gains,
            "clip_ids": clip_ids,
            "clip_id": "+".join(clip_ids),
            "source_categories": source_categories,
        }

        # Visual features - load per source separately (no averaging)
        if self.include_visual:
            visual_list = []
            missing_visuals = []
            for clip_id in clip_ids:
                visual_paths = self._all_meta[clip_id].get("visual_paths", [])
                if visual_paths:
                    # Each selected clip contributes its own first source/audio,
                    # so its own first visual path is the matching visual anchor.
                    vpath = visual_paths[0]
                    if vpath and os.path.exists(vpath):
                        vf = self._get_visual_tensor(clip_id, vpath)
                    else:
                        missing_visuals.append(clip_id)
                        vf = None
                else:
                    missing_visuals.append(clip_id)
                    vf = None
                visual_list.append(vf)

            if missing_visuals:
                raise RuntimeError(
                    "Missing visual data for include_visual=True clips: "
                    + ", ".join(missing_visuals)
                )

            first = visual_list[0]
            if first.dim() == 4 and first.shape[1] == 3:
                # Raw frames (DINOv2-in-loop path): [N_sources, 150, 3, 448, 448].
                output["video_frames"] = torch.stack(visual_list)
            else:
                # Precomputed DINOv2 tokens: [V, 1024, 768] per selected clip.
                # The compact cache stores only the 19 frames the model attends
                # to, so do NOT force-pad to 150 here; keep the cached frame
                # count and let the model align positions dynamically. Pad only
                # to the longest tensor among the selected clips so they stack.
                target_v = max(v.shape[0] for v in visual_list)
                features = []
                for v in visual_list:
                    if v.shape[0] < target_v:
                        pad = torch.zeros(
                            target_v - v.shape[0],
                            *v.shape[1:],
                            dtype=v.dtype,
                        )
                        v = torch.cat([v, pad], dim=0)
                    features.append(v[:target_v])
                # Stack as [N_sources, V, 1024, 768]; upcast to float32 so it
                # matches model weights regardless of the cache dtype (fp16).
                output["visual_features"] = torch.stack(features).float()

        # CRM targets
        if self.split == "train":
            # Compute cRM on-the-fly from scaled sources (target_waveforms).
            source_stfts = torch.stack(
                [self._stft(s).squeeze(0) for s in scaled_sources]
            )  # [N, 2, F, T]
            crm_targets = compute_crm_targets(
                source_stfts,  # [N, 2, F, T]
                mixture_stft,  # [2, F, T]
            )  # [N, 2, F, T]
            output["target_crm_masks"] = crm_targets
        else:
            # Validation/test: use cached cRM (deterministic)
            all_crms = []
            for clip_id in clip_ids:
                cpath = self._all_meta[clip_id].get("crm_path")
                if cpath and os.path.exists(cpath):
                    crm = torch.load(cpath, weights_only=False)
                    # crm shape: [n_src, 2, F, T]; take first source per clip
                    all_crms.append(crm[0:1])  # [1, 2, F, T]
            if len(all_crms) == self.n_sources:
                output["target_crm_masks"] = torch.cat(all_crms, dim=0)  # [N, 2, F, T]

        return output
