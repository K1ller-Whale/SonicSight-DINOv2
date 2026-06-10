"""Cross-modal attention module for audio-visual fusion.

SPEC 11.3: 2 stacked cross-attention blocks,
Pre-LayerNorm, n_heads=8, d=512, ffn=2048, GELU, dropout=0.1.
"""
import torch
import torch.nn as nn
from src.fusion.positional_encoding import SinusoidalPositionalEncoding
from typing import Optional


class CrossAttentionBlock(nn.Module):
    """Pre-LN cross-attention block: Pre-LN → MultiHead → +resid → FFN → +resid."""

    def __init__(self, d_model: int = 512, n_heads: int = 8,
                 ffn_dim: int = 2048, dropout: float = 0.1):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by n_heads ({n_heads})")
        self.pre_norm_query = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.dropout1 = nn.Dropout(dropout)
        self.pre_norm_ffn = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, query: torch.Tensor, key: torch.Tensor,
                value: torch.Tensor, key_padding_mask=None,
                attn_mask: Optional[torch.Tensor] = None,
                need_weights: bool = False,
                average_attn_weights: bool = False) -> torch.Tensor:
        # Pre-LN
        q = self.pre_norm_query(query)
        attn_out, attn_weights = self.attn(q, key, value,
                                 key_padding_mask=key_padding_mask,
                                 attn_mask=attn_mask,
                                 need_weights=need_weights,
                                 average_attn_weights=average_attn_weights)
        query = query + self.dropout1(attn_out)

        # FFN
        ffn_in = self.pre_norm_ffn(query)
        query = query + self.ffn(ffn_in)

        if need_weights:
            return query, attn_weights
        return query


class CrossModalAttentionModule(nn.Module):
    """
    2 stacked cross-attention blocks + sinusoidal positional encodings.

    Audio query: [B, T_a, D_a] — bottleneck sequence from audio U-Net
    Visual key/value: [B, N_p, D_a] — projected DINOv2 patch tokens
    """

    def __init__(self, d_model: int = 512, n_heads: int = 8,
                 n_layers: int = 2, ffn_dim: int = 2048, dropout: float = 0.1,
                 max_len: int = 5000):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by n_heads ({n_heads})")
        self.pos_enc = SinusoidalPositionalEncoding(d_model, max_len)
        self.blocks = nn.ModuleList([
            CrossAttentionBlock(d_model, n_heads, ffn_dim, dropout)
            for _ in range(n_layers)
        ])

    def forward(self, audio_query: torch.Tensor,
                visual_keyvalue: torch.Tensor,
                attn_mask: Optional[torch.Tensor] = None,
                need_weights: bool = False,
                average_attn_weights: bool = False) -> torch.Tensor:
        """
        Args:
            audio_query: [B, T_a, D_a]
            visual_keyvalue: [B, N_p, D_a]  (projected from 768→512)
            attn_mask: Optional [B*n_heads, T_q, T_kv] or [T_q, T_kv] mask
            need_weights: Whether to return attention weights
            average_attn_weights: If False, returns per-head weights [B, n_heads, T_q, T_kv]
        Returns:
            fused: [B, T_a, D_a] or (fused, attn_weights) if need_weights=True
        """
        # Add sinusoidal positional encoding to audio query only
        x = self.pos_enc(audio_query)

        attn_weights_list = []
        for block in self.blocks:
            if need_weights:
                x, attn_w = block(x, visual_keyvalue, visual_keyvalue,
                                  key_padding_mask=None, attn_mask=attn_mask,
                                  need_weights=need_weights,
                                  average_attn_weights=average_attn_weights)
                attn_weights_list.append(attn_w)
            else:
                x = block(x, visual_keyvalue, visual_keyvalue,
                          key_padding_mask=None, attn_mask=attn_mask,
                          need_weights=False)

        if need_weights:
            # Return weights from last block
            return x, attn_weights_list[-1]
        return x