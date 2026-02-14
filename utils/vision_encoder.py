"""
vision_encoder.py

Reusable ResNet18-based vision encoder.
Outputs a 512-dimensional feature vector.

Designed to be shared across:
    - BC policy
    - Transformer VLA
    - Diffusion policy
"""

import torch
import torch.nn as nn
import torchvision.models as models


class VisionEncoder(nn.Module):
    def __init__(self, pretrained=True, freeze_backbone=False):
        """
        Args:
            pretrained (bool): Use ImageNet pretrained weights
            freeze_backbone (bool): If True, do not update ResNet weights
        """
        super().__init__()

        # Load ResNet18
        resnet = models.resnet18(pretrained=pretrained)

        # Remove final classification layer (fc)
        self.backbone = nn.Sequential(
            *list(resnet.children())[:-1]  # remove final FC
        )

        # Output feature dimension
        self.output_dim = 512

        # Optional freezing
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

    def forward(self, x):
        """
        Args:
            x: Tensor (B, 3, 224, 224)

        Returns:
            features: Tensor (B, 512)
        """

        features = self.backbone(x)        # (B, 512, 1, 1)
        features = features.view(features.size(0), -1)  # flatten to (B, 512)

        return features
