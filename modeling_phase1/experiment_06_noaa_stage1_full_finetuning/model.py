"""MobileNetV3-Large model for Experiment 6."""

from torch import nn
from torchvision.models import MobileNet_V3_Large_Weights, mobilenet_v3_large


def build_model(
    num_classes=2,
    pretrained=True,
    hidden_dim=1280,
    dropout=0.0,
):
    """Build a fully trainable MobileNetV3-Large NF/FL classifier."""
    # ImageNet weights provide the starting values for MobileNet's parameters.
    weights = MobileNet_V3_Large_Weights.DEFAULT if pretrained else None
    model = mobilenet_v3_large(weights=weights)

    # Replace the original 1,000-class ImageNet classifier with NF/FL outputs.
    input_features = model.classifier[0].in_features
    classifier_layers = [
        nn.Linear(input_features, hidden_dim),
        nn.Hardswish(),
    ]
    if dropout > 0:
        classifier_layers.append(nn.Dropout(p=dropout))
    classifier_layers.append(nn.Linear(hidden_dim, num_classes))
    model.classifier = nn.Sequential(*classifier_layers)

    # Train every feature layer and every classifier layer.
    for parameter in model.parameters():
        parameter.requires_grad = True

    return model
