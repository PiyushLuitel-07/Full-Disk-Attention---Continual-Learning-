"""Weight initialization used by the original full-disk attention repository."""

from torch import nn


def initialize_kaiming_uniform(module: nn.Module) -> None:
    """Initialize convolutional, BatchNorm, and linear layers in-place."""

    for layer in module.modules():
        if isinstance(layer, nn.Conv2d):
            nn.init.kaiming_uniform_(layer.weight, mode="fan_in", nonlinearity="relu")
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)
        elif isinstance(layer, nn.BatchNorm2d):
            nn.init.uniform_(layer.weight, a=0.0, b=1.0)
            nn.init.zeros_(layer.bias)
        elif isinstance(layer, nn.Linear):
            nn.init.kaiming_uniform_(layer.weight, mode="fan_in", nonlinearity="relu")
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)

