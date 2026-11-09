"""MobileNetV3-Large architecture shared by Stage 1 and Stage 2 EWC."""

from torch import nn
from torchvision.models import MobileNet_V3_Large_Weights, mobilenet_v3_large


def build_model(
    num_classes=2,
    pretrained=False,
    hidden_dim=1280,
    dropout=0.0,
):
    """Build the exact NF/FL MobileNetV3-Large used by the Stage 1 winner.

    Stage 2 normally calls this with ``pretrained=False`` and immediately
    loads the selected Stage 1 checkpoint. ImageNet weights therefore serve
    only as Stage 1 initialization, never as a replacement for learned Stage
    1 parameters.
    """
    weights = MobileNet_V3_Large_Weights.DEFAULT if pretrained else None
    model = mobilenet_v3_large(weights=weights)

    # This replacement exactly matches Experiment 10 / the selected Stage 1
    # checkpoint: 960 pooled features -> 1280 hidden features -> NF/FL logits.
    input_features = model.classifier[0].in_features
    classifier_layers = [
        nn.Linear(input_features, hidden_dim),
        nn.Hardswish(),
    ]
    if dropout > 0:
        classifier_layers.append(nn.Dropout(p=dropout))
    classifier_layers.append(nn.Linear(hidden_dim, num_classes))
    model.classifier = nn.Sequential(*classifier_layers)

    # EWC protects parameters but does not freeze them: every parameter
    # remains trainable and can move when the Stage 2 objective justifies it.
    for parameter in model.parameters():
        parameter.requires_grad = True

    return model
