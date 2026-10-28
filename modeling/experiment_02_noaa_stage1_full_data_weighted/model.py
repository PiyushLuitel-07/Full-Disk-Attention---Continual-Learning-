"""MobileNetV3-Large model for Experiment 2."""

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
    # Start from features learned from ImageNet rather than random weights.
    weights = MobileNet_V3_Large_Weights.DEFAULT if pretrained else None
    model = mobilenet_v3_large(weights=weights)

    # Replace MobileNet's original 1,000-class head with an NF/FL classifier.
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
        # First freeze the entire feature extractor.
        for parameter in model.features.parameters():
            parameter.requires_grad = False

        # Then reopen only the requested final feature blocks.
        blocks_to_train = int(trainable_backbone_blocks)
        if blocks_to_train > 0:
            for block in model.features[-blocks_to_train:]:
                for parameter in block.parameters():
                    parameter.requires_grad = True

    # The training loop uses this boundary to keep BatchNorm layers inside
    # frozen blocks in evaluation mode while the reopened blocks are trained.
    model.frozen_feature_blocks = (
        len(model.features) - int(trainable_backbone_blocks)
        if freeze_backbone
        else 0
    )
    return model
