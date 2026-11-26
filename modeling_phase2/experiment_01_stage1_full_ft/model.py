"""MobileNetV3-Large binary classifier for Phase 2 Experiment 1 (full fine-tuning)."""

from torch import nn
from torchvision.models import MobileNet_V3_Large_Weights, mobilenet_v3_large


def build_model(pretrained=True, freeze_backbone=False, trainable_backbone_blocks=2, hidden_dim=1280, dropout=0.0):
    """Build MobileNetV3-Large with a 2-class (NF/FL) classifier head."""
    weights = MobileNet_V3_Large_Weights.DEFAULT if pretrained else None
    model = mobilenet_v3_large(weights=weights)

    if freeze_backbone:
        frozen_blocks = len(model.features) - trainable_backbone_blocks
        for idx, block in enumerate(model.features):
            trainable = idx >= frozen_blocks
            for parameter in block.parameters():
                parameter.requires_grad = trainable
        model.frozen_feature_blocks = frozen_blocks
    else:
        for parameter in model.features.parameters():
            parameter.requires_grad = True
        model.frozen_feature_blocks = 0

    input_features = model.classifier[0].in_features
    layers = [nn.Linear(input_features, hidden_dim), nn.Hardswish()]
    if dropout > 0:
        layers.append(nn.Dropout(dropout))
    layers.append(nn.Linear(hidden_dim, 2))
    model.classifier = nn.Sequential(*layers)
    return model
