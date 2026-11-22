"""MobileNetV3-Large classifier used by the joint-training baseline."""

from torch import nn
from torchvision.models import MobileNet_V3_Large_Weights, mobilenet_v3_large


def build_model(pretrained=True, hidden_dim=1280, dropout=0.0):
    weights = MobileNet_V3_Large_Weights.DEFAULT if pretrained else None
    model = mobilenet_v3_large(weights=weights)
    input_features = model.classifier[0].in_features
    layers = [nn.Linear(input_features, hidden_dim), nn.Hardswish()]
    if dropout > 0:
        layers.append(nn.Dropout(dropout))
    layers.append(nn.Linear(hidden_dim, 2))
    model.classifier = nn.Sequential(*layers)
    return model
