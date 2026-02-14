"""
bc_model.py

Behavior Cloning baseline policy.

Architecture:
    Image → VisionEncoder (ResNet18) → 512
    Text → CLIP embedding → 512
    Concatenate → MLP → 7D action
"""

import torch
import torch.nn as nn
from utils.vision_encoder import VisionEncoder


class BCPolicy(nn.Module):
    def __init__(self):
        super().__init__()

        # Vision backbone
        self.vision_encoder = VisionEncoder(
            pretrained=True,
            freeze_backbone=False  # fine-tuning enabled
        )

        # Fusion + action head
        self.policy_head = nn.Sequential(
            nn.Linear(512 + 512, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 7)
        )

    def forward(self, image, text_embedding):
        """
        image: (B, 3, 224, 224)
        text_embedding: (B, 512)
        """

        img_feat = self.vision_encoder(image)  # (B, 512)

        fused = torch.cat([img_feat, text_embedding], dim=1)

        action = self.policy_head(fused)

        return action
