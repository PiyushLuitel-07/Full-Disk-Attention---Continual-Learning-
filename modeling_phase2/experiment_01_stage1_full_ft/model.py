"""MobileNetV3-Large binary classifier for Phase 2 Experiment 1 (full fine-tuning)."""

from torch import nn
from torchvision.models import MobileNet_V3_Large_Weights, mobilenet_v3_large


def build_model(pretrained=True, freeze_backbone=False, hidden_dim=1280, dropout=0.0):
    """Build MobileNetV3-Large with a 2-class (NF/FL) classifier head."""
    weights = MobileNet_V3_Large_Weights.DEFAULT if pretrained else None
    model = mobilenet_v3_large(weights=weights)

    for parameter in model.features.parameters():
        parameter.requires_grad = not freeze_backbone

    input_features = model.classifier[0].in_features
    layers = [nn.Linear(input_features, hidden_dim), nn.Hardswish()]
    if dropout > 0:
        layers.append(nn.Dropout(dropout))
    layers.append(nn.Linear(hidden_dim, 2))
    model.classifier = nn.Sequential(*layers)
    return model
