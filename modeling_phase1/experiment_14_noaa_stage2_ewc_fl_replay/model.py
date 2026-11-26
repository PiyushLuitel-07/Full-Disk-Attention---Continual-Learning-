"""MobileNetV3-Large architecture for Stage 2 EWC with FL replay."""

from torch import nn
from torchvision.models import MobileNet_V3_Large_Weights, mobilenet_v3_large


def build_model(
    num_classes=2,
    pretrained=False,
    hidden_dim=1280,
    dropout=0.0,
):
    """Build the exact architecture used by the selected Stage 1 model.

    Stage 2 constructs the model without downloading ImageNet weights and
    immediately loads the selected Stage 1 EWC anchor. The classifier layout
    must therefore remain identical to the Stage 1 checkpoint.
    """
    weights = MobileNet_V3_Large_Weights.DEFAULT if pretrained else None
    model = mobilenet_v3_large(weights=weights)

    # Selected Stage 1 architecture:
    # 960 pooled backbone features -> 1280 hidden features -> NF/FL logits.
    input_features = model.classifier[0].in_features
    classifier_layers = [
        nn.Linear(input_features, hidden_dim),
        nn.Hardswish(),
    ]
    if dropout > 0:
        classifier_layers.append(nn.Dropout(p=dropout))
    classifier_layers.append(nn.Linear(hidden_dim, num_classes))
    model.classifier = nn.Sequential(*classifier_layers)

    # Replay and EWC regularize learning but do not freeze model parameters.
    # Differential learning rates are assigned later by the optimizer.
    for parameter in model.parameters():
        parameter.requires_grad = True

    return model
