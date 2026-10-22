"""MobileNetV3-Large for NOAA Stage 1 flare classification."""

from torch import nn
from torchvision.models import MobileNet_V3_Large_Weights, mobilenet_v3_large


def build_model(num_classes=2, pretrained=True, hidden_dim=1280):
    """Return MobileNetV3-Large with an NF/FL classifier and no dropout."""
    # Start from features learned on ImageNet instead of random weights.
    weights = MobileNet_V3_Large_Weights.DEFAULT if pretrained else None
    model = mobilenet_v3_large(weights=weights)

    # Replace the original 1,000-class classifier with NF/FL outputs.
    input_features = model.classifier[0].in_features
    model.classifier = nn.Sequential(
        nn.Linear(input_features, hidden_dim),
        nn.Hardswish(),
        nn.Linear(hidden_dim, num_classes),
    )
    return model
