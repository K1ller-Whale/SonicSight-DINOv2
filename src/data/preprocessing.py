"""Audio and video preprocessing utilities.

SPEC 11.2:
- Audio: resample to 16 kHz mono, crop / zero-pad to exactly 6 s (96 000 samples)
- Video: resize to 448×448 (bicubic), ImageNet normalize, 25 fps
- STFT: N_FFT = 512, H = 160, W = 400, Hann → [B, 2, F, T]
- Temporal alignment: spectrogram frame t → video frame v = floor(t × 150 / 601)
- cRM: Mᵢ = Sᵢ / X, tanh-compressed with K = 10, C = 0.1
"""
import torch
import torch.nn.functional as F
import torchaudio.transforms as T
import torchvision.transforms as VT
from typing import Optional, Tuple, List
import numpy as np
import math


# --- Constants ---
TARGET_SR = 16000
CLIP_DURATION = 6  # seconds
CLIP_LENGTH = TARGET_SR * CLIP_DURATION  # 96 000 samples
N_FFT = 512
HOP_LENGTH = 160
WIN_LENGTH = 400
N_MELS = 80
F_MIN = 80
F_MAX = 7600

# Shape after STFT(512, 160, 400) for 96000 samples
# n_frames = ceil(96000 / 160) = 600, but with center padding it's 601
N_FFT_BINS = N_FFT // 2 + 1  # 257 time-domain bins (freq bins through STFT), need num_t frames too
N_STFT_FRAMES = 601  # torch.stft(center=True) → ceil(96000/160)+1 = 601

VIDEO_FPS = 25
N_VIDEO_FRAMES = VIDEO_FPS * CLIP_DURATION  # 150 frames

IMAGE_SIZE = 448
PATCH_SIZE = 14
NUM_PATCHES_SIDE = IMAGE_SIZE // PATCH_SIZE  # 32

# ImageNet normalization
_IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
_IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

# Temporal alignment: spec frame -> video frame index
# precomputed once for efficiency
_ALIGNMENT = torch.arange(N_STFT_FRAMES, dtype=torch.long)
_VIDEO_ALIGN = (_ALIGNMENT.float() * N_VIDEO_FRAMES / N_STFT_FRAMES).long().clamp(0, N_VIDEO_FRAMES - 1)


class AudioPreprocessor:
    """Resample to 16 kHz mono, crop / zero-pad to exactly 6 s."""

    def __init__(self, target_sr: int = TARGET_SR, clip_duration: int = CLIP_DURATION):
        self.target_sr = target_sr
        self.clip_length = target_sr * clip_duration

    def __call__(self, waveform: torch.Tensor, sr: int) -> torch.Tensor:
        """
        Args:
            waveform: [L] or [C, L]
            sr: original sample rate
        Returns:
            waveform: [clip_length] for 6 s
        """
        # Mono mixdown
        if waveform.dim() == 2:
            waveform = waveform.mean(dim=0)
        elif waveform.dim() != 1:
            raise ValueError(f"Expected 1D or 2D waveform, got {waveform.shape}")

        # Resample if needed
        if sr != self.target_sr:
            waveform = T.Resample(sr, self.target_sr)(waveform)

        length = waveform.shape[-1]
        if length >= self.clip_length:
            start = (length - self.clip_length) // 2
            waveform = waveform[start:start + self.clip_length]
        else:
            pad = self.clip_length - length
            waveform = F.pad(waveform, (0, pad))

        return waveform  # [96000]


class STFTModule:
    """Complex STFT → stacked real/imag channels [B, 2, F, T]."""

    def __init__(self, n_fft: int = N_FFT, hop_length: int = HOP_LENGTH,
                 win_length: int = WIN_LENGTH):
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self._spec = T.Spectrogram(
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            window_fn=torch.hann_window,
            power=None,
            center=True,
            pad_mode='reflect',
        )

    def __call__(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Args:
            waveform: [L]
        Returns:
            spec: [2, F, T]  (real + imag in channel dim)
        """
        spec = self._spec(waveform)  # [F, T] complex (torch complex64)
        # Convert complex to real/imag stacked: [2, F, T]
        stacked = torch.stack([spec.real, spec.imag], dim=0)
        assert stacked.shape == (2, N_FFT_BINS, N_STFT_FRAMES), \
            f"STFT shape mismatch: {stacked.shape} (expected (2, {N_FFT_BINS}, {N_STFT_FRAMES}))"
        return stacked

    @staticmethod
    def spectrogram(mag: torch.Tensor) -> torch.Tensor:
        """Convert complex STFT to magnitude spectrogram."""
        return mag.pow(2).sum(dim=0).sqrt()

    @staticmethod
    def phase(mag: torch.Tensor) -> torch.Tensor:
        """Convert complex STFT to phase."""
        return torch.atan2(mag[1], mag[0])  # imag, real


class ISTFTModule:
    """Inverse STFT from complex mask and mixture."""

    def __init__(self, n_fft: int = N_FFT, hop_length: int = HOP_LENGTH,
                 win_length: int = WIN_LENGTH):
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length

    def __call__(self, mask: torch.Tensor, mixture_stft: torch.Tensor,
                 length: int = CLIP_LENGTH) -> torch.Tensor:
        """
        Args:
            mask: [2, F, T]  (real + imag mask channels)
            mixture_stft: [2, F, T]  (real + imag of mixture)
        Returns:
            waveform: [L]
        """
        # Complex multiplication: M * X where both are complex
        mask_c = torch.complex(mask[0], mask[1])
        mix_c = torch.complex(mixture_stft[0], mixture_stft[1])
        separated = mask_c * mix_c

        waveform = torch.istft(
            separated,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=torch.hann_window(self.win_length, device=separated.device),
            length=length,
        )
        return waveform


class VideoPreprocessor:
    """Resize to 448×448 preserving aspect ratio, then center crop. ImageNet normalize. Input is float [0,1]."""

    MEAN = _IMAGENET_MEAN
    STD = _IMAGENET_STD

    def __init__(self, image_size: int = IMAGE_SIZE):
        self.image_size = image_size
        self._resize = VT.Resize(image_size, interpolation=VT.InterpolationMode.BICUBIC, antialias=True)
        self._center_crop = VT.CenterCrop(image_size)

    def __call__(self, frames: torch.Tensor) -> torch.Tensor:
        """
        Args:
            frames: [N, 3, H, W] float in [0, 1]
        Returns:
            normalized: [N, 3, image_size, image_size]
        """
        frames = self._resize(frames)
        frames = self._center_crop(frames)
        frames = (frames - self.MEAN.to(frames.device)) / self.STD.to(frames.device)
        return frames


# --- cRM Target Computation ---

def compute_crm_targets(source_stfts: torch.Tensor, mixture_stft: torch.Tensor,
                        k: float = 10.0) -> torch.Tensor:
    """
    Compute ideal complex ratio mask for each source.

    Args:
        source_stfts: [N, 2, F, T]
        mixture_stft: [2, F, T]
        k: compression factor
    Returns:
        crm: [N, 2, F, T]  (tanh compressed)
    """
    source_c = torch.complex(source_stfts[:, 0], source_stfts[:, 1])
    mix_c = torch.complex(mixture_stft[0], mixture_stft[1])

    # Ideal cRM: M = S / X
    ratio = safe_complex_divide(source_c, mix_c)

    # Compress: M_compressed = tanh(K * |M|) * exp(j * angle(M))
    mag = ratio.abs()
    phase = torch.angle(ratio)
    compressed_mag = torch.tanh(k * mag)
    compressed = compressed_mag * torch.exp(1j * phase)

    # Stack real/imag
    return torch.stack([compressed.real, compressed.imag], dim=1)  # [N, 2, F, T]


def safe_complex_divide(a: torch.Tensor, b: torch.Tensor,
                        eps: float = 1e-8) -> torch.Tensor:
    """Robust complex division. Both a,b can be broadcasted."""
    # b may need broadcasting
    if a.shape != b.shape:
        b = b.unsqueeze(0) if b.dim() < a.dim() else b
    return a / (b + eps)


# --- Temporal Alignment ---

def get_video_alignment() -> torch.Tensor:
    """Return cached temporal alignment: N_STFT_FRAMES → video frame indices."""
    return _VIDEO_ALIGN  # [601]


def stft_frame_to_video_frame(stft_frame: int) -> int:
    """Map a single STFT frame index to video frame index."""
    return _VIDEO_ALIGN[stft_frame].item()


# --- Utility: Pre-computed alignment table for dataset ---

def get_temporal_alignment_table(n_stft_frames: int = N_STFT_FRAMES,
                                 n_video_frames: int = N_VIDEO_FRAMES) -> List[int]:
    """Return a mapping from STFT frame index to video frame index."""
    return [int(stft_f * n_video_frames / n_stft_frames) for stft_f in range(n_stft_frames)]
