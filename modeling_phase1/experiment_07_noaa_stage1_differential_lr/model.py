"""MobileNetV3-Large model for Experiment 7."""

from torch import nn
from torchvision.models import MobileNet_V3_Large_Weights, mobilenet_v3_large


def build_model(
    num_classes=2,
    pretrained=True,
    hidden_dim=1280,
    dropout=0.0,
):
    """Build a fully trainable MobileNetV3-Large NF/FL classifier."""
    # ImageNet weights are used only as the starting parameter values.
    # The complete network will still be retrained on the magnetograms.
    weights = MobileNet_V3_Large_Weights.DEFAULT if pretrained else None
    model = mobilenet_v3_large(weights=weights)

    # Replace the original 1,000-class ImageNet classifier with an NF/FL head.
    input_features = model.classifier[0].in_features
    classifier_layers = [
        nn.Linear(input_features, hidden_dim),
        nn.Hardswish(),
    ]

    # Dropout remains configurable, although Experiment 7 sets it to zero.
    if dropout > 0:
        classifier_layers.append(nn.Dropout(p=dropout))

    classifier_layers.append(nn.Linear(hidden_dim, num_classes))
    model.classifier = nn.Sequential(*classifier_layers)

    # No layer is frozen. Differential learning rates will be assigned later:
    # a smaller rate to model.features and a larger rate to model.classifier.
    for parameter in model.parameters():
        parameter.requires_grad = True

    return model
