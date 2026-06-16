"""STFT / iSTFT utilities for complex spectrogram input / output.

Follows SPEC 11.2 audio preprocessing convention:
    N_FFT = 512, H = 160, W = 400, Hann window
"""
import torch
import torchaudio.transforms as T


class STFTModule(torch.nn.Module):
    """Produces complex spectrograms [B, 2, F, T] from waveform."""

    def __init__(self, n_fft: int = 512, hop_length: int = 160, win_length: int = 400):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.transform = T.Spectrogram(
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            power=None,  # complex output
        )

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        # waveform: [L], [B, L] or [B, 1, L]
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)  # [1, L]
        elif waveform.dim() == 3:
            waveform = waveform.squeeze(1)
        spec = self.transform(waveform)  # [B, F, T] complex
        # Stack real/imag as 2 channels
        spec_stacked = torch.stack([spec.real, spec.imag], dim=1)  # [B, 2, F, T]
        return spec_stacked


class ISTFTModule(torch.nn.Module):
    """Reconstructs waveform from complex spectrogram mask and mixture."""

    def __init__(self, n_fft: int = 512, hop_length: int = 160, win_length: int = 400,
                 length: int = 96000):
        """
        Args:
            n_fft: FFT size
            hop_length: Hop length
            win_length: Window length
            length: output waveform length.
                    Default 96000 = 16000 Hz * 6 seconds.
                    Override via cfg.data.sample_rate * cfg.data.clip_duration
        """
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.length = length
        self.transform = T.InverseSpectrogram(
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
        )

    def forward(self, mask: torch.Tensor, mixture_spec: torch.Tensor, length: int = None) -> torch.Tensor:
        """mask, mixture_spec: [B, 2, F, T] → waveform: [B, L]

        mask is real+imag channels; complex multiply with mixture STFT
        then inverse STFT to waveform.
        """
        # ComplexHalf is poorly supported, so keep iSTFT complex math in fp32
        # even when Lightning runs the surrounding model in mixed precision.
        mask = mask.float()
        mixture_spec = mixture_spec.float()
        complex_mask = torch.complex(mask[:, 0], mask[:, 1])
        complex_mix = torch.complex(mixture_spec[:, 0], mixture_spec[:, 1])
        separated = complex_mask * complex_mix  # element-wise complex multiply
        waveform = self.transform(separated, length=length if length is not None else self.length)
        return waveform
