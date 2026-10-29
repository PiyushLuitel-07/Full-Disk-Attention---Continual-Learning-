"""MobileNetV3-Large model for Experiment 5."""

from torch import nn
from torchvision.models import MobileNet_V3_Large_Weights, mobilenet_v3_large


def build_model(
    num_classes=2,
    pretrained=True,
    hidden_dim=1280,
    dropout=0.0,
    freeze_backbone=False,
    trainable_backbone_blocks=0,
):
    """Build MobileNetV3-Large with a binary NF/FL classifier."""
    weights = MobileNet_V3_Large_Weights.DEFAULT if pretrained else None
    model = mobilenet_v3_large(weights=weights)

    # Replace the original ImageNet classifier with two NF/FL outputs.
    input_features = model.classifier[0].in_features
    classifier_layers = [
        nn.Linear(input_features, hidden_dim),
        nn.Hardswish(),
    ]
    if dropout > 0:
        classifier_layers.append(nn.Dropout(p=dropout))
    classifier_layers.append(nn.Linear(hidden_dim, num_classes))
    model.classifier = nn.Sequential(*classifier_layers)

    if freeze_backbone:
        for parameter in model.features.parameters():
            parameter.requires_grad = False

        # Reopen only the requested final feature block.
        blocks_to_train = int(trainable_backbone_blocks)
        if blocks_to_train > 0:
            for block in model.features[-blocks_to_train:]:
                for parameter in block.parameters():
                    parameter.requires_grad = True

    model.frozen_feature_blocks = (
        len(model.features) - int(trainable_backbone_blocks)
        if freeze_backbone
        else 0
    )
    return model


