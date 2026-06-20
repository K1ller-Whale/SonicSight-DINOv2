#!/usr/bin/env python
"""Audit preprocessed dataset files referenced by cache/index.json.

The script writes:
  - a TSV inventory of files referenced by the index
  - a corrupted.txt-style log of missing, unreadable, empty, non-finite, or
    all-zero audio/tensor files

Example:
    python scripts/audit_preprocessed_data.py \
        --index_file cache/index.json \
        --list_output preprocessed_files.txt \
        --corrupted_output corrupted.txt
"""

import argparse
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch


AUDIO_EXTS = {".pt", ".wav", ".flac", ".mp3", ".m4a", ".ogg"}


def resolve_path(path: str, index_dir: Path, cache_dir: Optional[Path]) -> Path:
    """Resolve an index path against cwd, index dir, and cache dir."""
    raw = Path(path).expanduser()
    if raw.is_absolute():
        return raw

    candidates = [Path.cwd() / raw, index_dir / raw]
    if cache_dir is not None:
        candidates.append(cache_dir / raw)

    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def iter_index_entries(index: Dict[str, Any]) -> Iterable[Tuple[str, Dict[str, Any]]]:
    """Yield (clip_id, metadata) for current and legacy index formats."""
    if all(key in index for key in ("train", "val", "test")):
        for split in ("train", "val", "test"):
            samples = index.get(split, [])
            if isinstance(samples, list):
                for i, sample in enumerate(samples):
                    if isinstance(sample, dict):
                        clip_id = str(sample.get("clip_id") or sample.get("id") or f"{split}_{i}")
                        meta = dict(sample)
                        meta.setdefault("split", split)
                        yield clip_id, meta
        return

    for clip_id, meta in index.items():
        if isinstance(meta, dict):
            yield str(clip_id), meta


def extract_audio_paths(meta: Dict[str, Any]) -> List[str]:
    """Collect audio/source paths from known index fields."""
    paths: List[str] = []
    source_paths = meta.get("source_paths", [])
    if isinstance(source_paths, str):
        paths.append(source_paths)
    elif isinstance(source_paths, list):
        paths.extend(str(p) for p in source_paths if p)

    for key in ("audio_path", "wav_path", "source_path"):
        value = meta.get(key)
        if value:
            paths.append(str(value))

    seen = set()
    unique = []
    for path in paths:
        if path not in seen:
            seen.add(path)
            unique.append(path)
    return unique


def extract_cache_paths(meta: Dict[str, Any]) -> List[Tuple[str, Optional[int], str]]:
    """Collect non-audio cached tensor paths from known index fields."""
    out: List[Tuple[str, Optional[int], str]] = []

    crm_path = meta.get("crm_path")
    if crm_path:
        out.append(("crm", None, str(crm_path)))

    visual_paths = meta.get("visual_paths", [])
    if isinstance(visual_paths, str):
        out.append(("visual", None, visual_paths))
    elif isinstance(visual_paths, list):
        for idx, value in enumerate(visual_paths):
            if value:
                out.append(("visual", idx, str(value)))

    return out


def tensor_from_payload(payload: Any) -> Optional[torch.Tensor]:
    """Extract an audio-like tensor from common .pt payload shapes."""
    if torch.is_tensor(payload):
        return payload

    if isinstance(payload, dict):
        for key in ("waveform", "audio", "wav", "source", "samples"):
            value = payload.get(key)
            if torch.is_tensor(value):
                return value

    if isinstance(payload, (list, tuple)):
        tensors = [item for item in payload if torch.is_tensor(item)]
        if tensors:
            return tensors[0]

    return None


def load_audio_tensor(path: Path) -> torch.Tensor:
    """Load a .pt or standard audio file and return a tensor."""
    if path.suffix.lower() == ".pt":
        payload = torch.load(path, map_location="cpu", weights_only=False)
        tensor = tensor_from_payload(payload)
        if tensor is None:
            raise ValueError("torch file does not contain a recognizable audio tensor")
        return tensor

    import torchaudio

    waveform, sample_rate = torchaudio.load(str(path))
    if sample_rate <= 0:
        raise ValueError(f"invalid sample rate: {sample_rate}")
    return waveform


def validate_tensor(tensor: torch.Tensor, silent_eps: float) -> Optional[str]:
    """Return a corruption reason, or None when the tensor looks valid."""
    if tensor.numel() == 0:
        return "empty_tensor"
    if tensor.shape[-1] == 0:
        return "zero_length_audio"

    tensor = tensor.detach().float().cpu()
    if not torch.isfinite(tensor).all():
        return "non_finite_values"
    if tensor.abs().max().item() <= silent_eps:
        return "all_zero_or_silent"

    return None


def file_size(path: Path) -> int:
    try:
        return path.stat().st_size
    except OSError:
        return -1


def inventory_row(
    kind: str,
    clip_id: str,
    split: str,
    source_idx: Optional[int],
    path: str,
    resolved: Path,
) -> str:
    idx = "" if source_idx is None else str(source_idx)
    exists = "yes" if resolved.exists() else "no"
    size = str(file_size(resolved)) if resolved.exists() else ""
    return "\t".join([kind, clip_id, split, idx, path, str(resolved), exists, size])


def corruption_row(
    timestamp: str,
    kind: str,
    clip_id: str,
    split: str,
    source_idx: Optional[int],
    reason: str,
    path: str,
    resolved: Path,
) -> str:
    idx = "" if source_idx is None else str(source_idx)
    return "\t".join([timestamp, kind, clip_id, split, idx, reason, path, str(resolved)])


def audit_audio_file(
    clip_id: str,
    split: str,
    source_idx: int,
    path: str,
    resolved: Path,
    silent_eps: float,
) -> Optional[str]:
    """Validate one audio file and return a corruption reason if found."""
    if not resolved.exists():
        return "missing_file"
    if not resolved.is_file():
        return "not_a_file"
    if file_size(resolved) == 0:
        return "zero_byte_file"
    if resolved.suffix.lower() not in AUDIO_EXTS:
        return f"unsupported_audio_extension:{resolved.suffix.lower()}"

    try:
        tensor = load_audio_tensor(resolved)
    except Exception as exc:
        return f"load_failed:{type(exc).__name__}:{exc}"

    return validate_tensor(tensor, silent_eps)


def audit_cache_tensor(path: Path, silent_eps: float) -> Optional[str]:
    """Lightly validate a cached tensor file, optionally loaded by caller."""
    if not path.exists():
        return "missing_file"
    if not path.is_file():
        return "not_a_file"
    if file_size(path) == 0:
        return "zero_byte_file"

    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as exc:
        return f"load_failed:{type(exc).__name__}:{exc}"

    tensor = tensor_from_payload(payload)
    if tensor is None:
        return "no_tensor_payload"
    return validate_tensor(tensor, silent_eps)


def write_lines(path: Path, lines: List[str], append: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if append else "w"
    with path.open(mode, encoding="utf-8") as f:
        for line in lines:
            f.write(line.rstrip("\n") + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit preprocessed dataset files")
    parser.add_argument("--index_file", type=str, default="cache/index.json", help="Path to cache/index.json")
    parser.add_argument("--list_output", type=str, default="preprocessed_files.txt", help="TSV file inventory output")
    parser.add_argument("--corrupted_output", type=str, default="corrupted.txt", help="TSV file for suspicious/corrupted entries")
    parser.add_argument("--cache_dir", type=str, default=None, help="Cache directory for resolving relative paths")
    parser.add_argument("--silent_eps", type=float, default=1e-8, help="Max abs value treated as silent/all-zero")
    parser.add_argument("--check_cache_tensors", action="store_true", help="Deep-load cRM/visual tensors too")
    parser.add_argument("--overwrite_corrupted", action="store_true", help="Overwrite corrupted output instead of appending")
    parser.add_argument("--overwrite_list", action="store_true", help="Overwrite file inventory instead of appending")
    args = parser.parse_args()

    index_path = Path(args.index_file).expanduser()
    if not index_path.exists():
        raise FileNotFoundError(f"Index file not found: {index_path}")

    index_dir = index_path.resolve().parent
    cache_dir = Path(args.cache_dir).expanduser() if args.cache_dir else index_dir

    with index_path.open("r", encoding="utf-8") as f:
        index = json.load(f)

    timestamp = datetime.now().isoformat(timespec="seconds")
    inventory_lines = [
        "kind\tclip_id\tsplit\tsource_idx\tpath\tresolved_path\texists\tsize_bytes"
    ]
    corrupted_lines = [
        "# timestamp\tkind\tclip_id\tsplit\tsource_idx\treason\tpath\tresolved_path"
    ]

    n_clips = 0
    n_audio = 0
    n_cache = 0
    n_bad = 0

    for clip_id, meta in iter_index_entries(index):
        n_clips += 1
        split = str(meta.get("split", ""))

        for source_idx, path in enumerate(extract_audio_paths(meta)):
            resolved = resolve_path(path, index_dir, cache_dir)
            inventory_lines.append(inventory_row("audio", clip_id, split, source_idx, path, resolved))
            n_audio += 1

            reason = audit_audio_file(
                clip_id, split, source_idx, path, resolved, args.silent_eps
            )
            if reason is not None:
                n_bad += 1
                corrupted_lines.append(
                    corruption_row(timestamp, "audio", clip_id, split, source_idx, reason, path, resolved)
                )

        for kind, source_idx, path in extract_cache_paths(meta):
            resolved = resolve_path(path, index_dir, cache_dir)
            inventory_lines.append(inventory_row(kind, clip_id, split, source_idx, path, resolved))
            n_cache += 1

            reason = None
            if args.check_cache_tensors:
                reason = audit_cache_tensor(resolved, args.silent_eps)
            elif not resolved.exists():
                reason = "missing_file"
            elif file_size(resolved) == 0:
                reason = "zero_byte_file"

            if reason is not None:
                n_bad += 1
                corrupted_lines.append(
                    corruption_row(timestamp, kind, clip_id, split, source_idx, reason, path, resolved)
                )

    write_lines(Path(args.list_output), inventory_lines, append=not args.overwrite_list)
    if len(corrupted_lines) > 1:
        write_lines(Path(args.corrupted_output), corrupted_lines, append=not args.overwrite_corrupted)

    print("Audit complete")
    print(f"  Clips in index:        {n_clips}")
    print(f"  Audio files checked:   {n_audio}")
    print(f"  Cache files listed:    {n_cache}")
    print(f"  Suspicious/corrupted:  {n_bad}")
    print(f"  File inventory:        {args.list_output}")
    print(f"  Corrupted log:         {args.corrupted_output}")
    if n_bad == 0:
        print("  No corrupted entries were appended.")


if __name__ == "__main__":
    main()
