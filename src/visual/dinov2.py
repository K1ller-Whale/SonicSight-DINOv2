"""DINOv2 feature extraction via HuggingFace Transformers.

SPEC 11.3: facebook/dinov2-base, 448×448 frames, 32×32=1024 patches per frame.
DINOv2 patch tokens are frozen; no gradient allowed.
"""
import torch
import torch.nn as nn
from transformers import AutoModel, AutoImageProcessor
from typing import Optional


class DINOv2FeatureExtractor(nn.Module):
    """
    Frozen DINOv2-Base with patch-level feature output.
    Produces [B, N_patches, 768] for each input frame.
    """

    def __init__(self, model_name: str = "facebook/dinov2-base",
                 image_size: int = 448, device: Optional[str] = None):
        super().__init__()
        self.model_name = model_name
        self.image_size = image_size
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Load DINOv2; all parameters frozen
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

        patch_size = self.model.config.patch_size
        self.num_patches_side = image_size // patch_size  # 448 / 14 = 32
        self.embed_dim = self.model.config.hidden_size  # 768

    @torch.no_grad()
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Args:
            images: [B, 3, H, W] normalized ImageNet tensors
        Returns:
            patch_features: [B, N_patches, embed_dim]
        """
        images = images.to(self.device)
        # DINOv2 patch embedding
        outputs = self.model(images, return_dict=True)
        # outputs.last_hidden_state: [B, 1+num_patches, 768] (CLS + patches)
        # Drop CLS token
        patch_tokens = outputs.last_hidden_state[:, 1:, :]  # [B, N_p, 768]
        return patch_tokens
