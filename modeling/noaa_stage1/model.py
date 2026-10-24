"""MobileNetV3-Large for NOAA Stage 1 flare classification."""

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
    """Return MobileNetV3-Large with a configurable NF/FL classifier."""
    # Start from features learned on ImageNet instead of random weights.
    weights = MobileNet_V3_Large_Weights.DEFAULT if pretrained else None
    model = mobilenet_v3_large(weights=weights)

    # Replace the original 1,000-class classifier with NF/FL outputs.
    input_features = model.classifier[0].in_features
    classifier_layers = [
        nn.Linear(input_features, hidden_dim),
        nn.Hardswish(),
    ]
    # Dropout is applied only during training and regularizes the classifier.
    if dropout > 0:
        classifier_layers.append(nn.Dropout(p=dropout))
    classifier_layers.append(nn.Linear(hidden_dim, num_classes))
    model.classifier = nn.Sequential(*classifier_layers)

    # Freeze the complete feature extractor first.
    if freeze_backbone:
        for parameter in model.features.parameters():
            parameter.requires_grad = False

        # For partial fine-tuning, reopen only the requested final blocks.
        # MobileNetV3-Large's final two entries are its last inverted-residual
        # block and the final convolution before global average pooling.
        blocks_to_train = int(trainable_backbone_blocks)
        if blocks_to_train > 0:
            for block in model.features[-blocks_to_train:]:
                for parameter in block.parameters():
                    parameter.requires_grad = True

    # run_epoch uses this boundary to keep only the frozen BatchNorm layers
    # in evaluation mode while the reopened final blocks continue training.
    model.frozen_feature_blocks = (
        len(model.features) - int(trainable_backbone_blocks)
        if freeze_backbone
        else 0
    )
    return model
