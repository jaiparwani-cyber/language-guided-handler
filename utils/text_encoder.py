"""
text_encoder.py

Frozen CLIP text encoder for language conditioning.
Converts instruction string → 512-d embedding.
"""

import torch
import torch.nn as nn
from transformers import CLIPTokenizer, CLIPTextModel


class FrozenCLIPTextEncoder(nn.Module):
    def __init__(self, device="cpu"):
        super().__init__()

        self.device = device

        self.tokenizer = CLIPTokenizer.from_pretrained(
            "openai/clip-vit-base-patch32"
        )

        self.model = CLIPTextModel.from_pretrained(
            "openai/clip-vit-base-patch32"
        )

        # Freeze parameters
        for param in self.model.parameters():
            param.requires_grad = False

        self.model.to(self.device)
        self.model.eval()

    def forward(self, text_list):
        """
        text_list: list of strings
        returns: (batch_size, 512) embedding
        """

        tokens = self.tokenizer(
            text_list,
            padding=True,
            truncation=True,
            return_tensors="pt"
        )

        tokens = {k: v.to(self.device) for k, v in tokens.items()}

        with torch.no_grad():
            outputs = self.model(**tokens)

        # Use pooled output (CLS embedding)
        text_embedding = outputs.pooler_output  # shape: (B, 512)

        return text_embedding
