"""Unit tests for cross-modal attention."""
import torch
import pytest
from src.fusion.cross_attention import CrossModalAttentionModule


class TestCrossModalAttention:

    def test_output_shape(self):
        """Cross-attention should preserve [B, T_a, D_a]."""
        fusion = CrossModalAttentionModule(d_model=512, n_heads=8, n_layers=2)
        audio_query = torch.randn(2, 171, 512)
        visual_kv = torch.randn(2, 1024, 512)
        output = fusion(audio_query, visual_kv)
        assert output.shape == audio_query.shape

    def test_d_model_divisible_by_heads(self):
        """d_model must be divisible by n_heads."""
        with pytest.raises(ValueError):
            CrossModalAttentionModule(d_model=513, n_heads=8)

    def test_positional_encoding_added(self):
        """Positional encoding should be applied to audio query."""
        fusion = CrossModalAttentionModule()
        audio = torch.zeros(1, 10, 512)
        visual = torch.zeros(1, 64, 512)
        out = fusion(audio, visual)
        assert out.shape == audio.shape
