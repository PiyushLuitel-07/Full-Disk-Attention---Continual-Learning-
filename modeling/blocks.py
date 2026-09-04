"""Attention building blocks copied from the original model mathematics."""

import torch
from torch import nn
from torch.nn import functional as F


class ProjectorBlock(nn.Module):
    """Map a 256-channel feature map to the common 512-channel space."""

    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()
        self.op = nn.Conv2d(in_features, out_features, kernel_size=1, bias=False)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.op(inputs)


class LinearAttentionBlock(nn.Module):
    """Learn one spatial weight per pixel, then form a weighted feature sum."""

    def __init__(self, in_features: int, normalize_attention: bool = True) -> None:
        super().__init__()
        self.normalize_attention = normalize_attention
        self.op = nn.Conv2d(in_features, 1, kernel_size=1, bias=False)

    def forward(
        self, local_features: torch.Tensor, global_features: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch, channels, height, width = local_features.shape
        compatibility = self.op(local_features + global_features)

        if self.normalize_attention:
            attention = F.softmax(
                compatibility.reshape(batch, 1, -1), dim=2
            ).reshape(batch, 1, height, width)
        else:
            attention = torch.sigmoid(compatibility)

        weighted = attention.expand_as(local_features) * local_features
        if self.normalize_attention:
            summary = weighted.reshape(batch, channels, -1).sum(dim=2)
        else:
            summary = F.adaptive_avg_pool2d(weighted, (1, 1)).reshape(
                batch, channels
            )
        return compatibility, summary

