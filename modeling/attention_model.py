"""The original full-disk attention architecture with package-safe imports.

The layer widths and operations intentionally match ``fulldiskattention``.
Scientific runs use 256 x 256 inputs, matching the paper and supplied code.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from .blocks import LinearAttentionBlock, ProjectorBlock
from .initialize import initialize_kaiming_uniform


class AttnNet(nn.Module):
    """Predict NF/FL logits and return three interpretable attention maps."""

    def __init__(self, num_classes: int = 2, attention: bool = True) -> None:
        super().__init__()
        self.attention = attention

        self.conv_block1 = self._conv_block(1, 64)
        self.conv_block2 = self._conv_block(64, 128)
        self.conv_block3 = self._conv_block(128, 256)
        self.conv_block4 = self._conv_block(256, 512)
        self.conv_block5 = self._conv_block(512, 512)
        self.conv_block6 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=3, bias=True),
            nn.BatchNorm2d(512, affine=True),
            nn.ReLU(),
        )
        self.dense = nn.Conv2d(512, 512, kernel_size=2, stride=2, bias=True)

        if attention:
            self.projector1 = ProjectorBlock(256, 512)
            self.attn1 = LinearAttentionBlock(512)
            self.attn2 = LinearAttentionBlock(512)
            self.attn3 = LinearAttentionBlock(512)
            self.classify = nn.Linear(512 * 3, num_classes)
        else:
            self.classify = nn.Linear(512, num_classes)

        initialize_kaiming_uniform(self)

    @staticmethod
    def _conv_block(in_channels: int, out_channels: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels, affine=True),
            nn.ReLU(),
        )

    def forward(
        self, image: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
        if image.ndim != 4 or image.shape[1] != 1:
            raise ValueError(
                f"Expected [batch, 1, height, width], received {tuple(image.shape)}"
            )

        x = F.max_pool2d(self.conv_block1(image), 2)
        x = F.max_pool2d(self.conv_block2(x), 2)
        local1 = self.conv_block3(x)
        x = F.max_pool2d(local1, 2)
        local2 = self.conv_block4(x)
        x = F.max_pool2d(local2, 2)
        local3 = self.conv_block5(x)
        x = F.max_pool2d(local3, 2)
        # At 256 x 256 input resolution this reproduces the original
        # 512 x 1 x 1 learned global feature exactly.
        global_feature = self.dense(self.conv_block6(x))

        if self.attention:
            map1, summary1 = self.attn1(self.projector1(local1), global_feature)
            map2, summary2 = self.attn2(local2, global_feature)
            map3, summary3 = self.attn3(local3, global_feature)
            logits = self.classify(torch.cat((summary1, summary2, summary3), dim=1))
            return logits, map1, map2, map3

        logits = self.classify(global_feature.flatten(start_dim=1))
        return logits, None, None, None
