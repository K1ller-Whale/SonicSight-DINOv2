"""Audio U-Net: 5-block encoder + 5-block decoder with skip connections.

SPEC 11.3: Input [B, 2, F, T], channel dimensions [2, 32, 64, 128, 256, 512]
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List


class EncoderBlock(nn.Module):
    """Conv2d (stride 2,2) → GroupNorm → LeakyReLU."""

    def __init__(self, in_ch: int, out_ch: int, norm_groups: int = 8):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=(2, 2), padding=1)
        self.norm = nn.GroupNorm(num_groups=norm_groups, num_channels=out_ch)
        self.activation = nn.LeakyReLU(negative_slope=0.2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.norm(x)
        x = self.activation(x)
        return x


class DecoderBlock(nn.Module):
    """ConvTranspose2d → [GroupNorm] → [LeakyReLU] (optional for final layer)."""

    def __init__(self, in_ch: int, out_ch: int, norm_groups: int = 8, is_last: bool = False):
        super().__init__()
        self.deconv = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=3, stride=(2, 2),
                                         padding=1, output_padding=(1, 1))
        self.is_last = is_last
        if not is_last:
            self.norm = nn.GroupNorm(num_groups=norm_groups, num_channels=out_ch)
            self.activation = nn.LeakyReLU(negative_slope=0.2)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.deconv(x)
        # Robust skip: crop/pad to match deconv output spatial shape
        _, _, h_out, w_out = x.shape
        _, _, h_skip, w_skip = skip.shape
        if h_skip > h_out:
            skip = skip[:, :, :h_out, :]
        if w_skip > w_out:
            skip = skip[:, :, :, :w_out]
        if h_skip < h_out or w_skip < w_out:
            skip = F.pad(skip, (0, w_out - w_skip, 0, h_out - h_skip))
        x = x + skip
        if not self.is_last:
            x = self.norm(x)
            x = self.activation(x)
        return x


class AudioUNetEncoder(nn.Module):
    """
    5-block encoder.
    Input: [B, 2, 257, 601]
    Bottleneck: [B, 512, 9, 19]
    """

    def __init__(self, channels: List[int] = None, norm_groups: int = 8):
        super().__init__()
        if channels is None:
            channels = [2, 32, 64, 128, 256, 512]
        self.blocks = nn.ModuleList()
        for i in range(len(channels) - 1):
            self.blocks.append(EncoderBlock(channels[i], channels[i + 1], norm_groups))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, List[torch.Tensor]]:
        skips = []
        for block in self.blocks:
            x = block(x)
            skips.append(x)
        return x, skips  # x is bottleneck, skips includes all intermediate activations


class AudioUNetDecoder(nn.Module):
    """
    5-block decoder with skip connections.
    Bottleneck: [B, 512, 9, 19]
    Output: [B, 2, 257, 601]
    """

    def __init__(self, channels: List[int] = None, norm_groups: int = 8):
        super().__init__()
        if channels is None:
            channels = [512, 256, 128, 64, 32, 2]
        self.blocks = nn.ModuleList()
        n_blocks = len(channels) - 1
        for i in range(n_blocks):
            is_last = (i == n_blocks - 1)
            self.blocks.append(DecoderBlock(channels[i], channels[i + 1], norm_groups, is_last=is_last))

    def forward(self, x: torch.Tensor, skips: List[torch.Tensor],
                target_shape: torch.Size = None) -> torch.Tensor:
        skip_iter = iter(reversed(skips[:-1]))
        for block in self.blocks:
            skip = next(skip_iter, None)
            if skip is not None:
                x = block(x, skip)
            else:
                # Last block: no skip connection
                x = block.deconv(x)
        # Crop to target shape if provided
        if target_shape is not None:
            x = x[..., :target_shape[-2], :target_shape[-1]]
        return x


class AudioUNet(nn.Module):
    """Full encoder-decoder U-Net."""

    def __init__(self, channels: List[int] = None, norm_groups: int = 8):
        super().__init__()
        if channels is None:
            channels = [2, 32, 64, 128, 256, 512]
        self.encoder = AudioUNetEncoder(channels, norm_groups)
        decoder_channels = [512, 256, 128, 64, 32, 2]
        self.decoder = AudioUNetDecoder(decoder_channels, norm_groups)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        target_shape = x.shape[-2:]
        bottleneck, skips = self.encoder(x)
        out = self.decoder(bottleneck, skips, target_shape=target_shape)
        return out
