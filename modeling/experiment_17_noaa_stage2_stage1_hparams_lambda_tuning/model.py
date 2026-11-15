"""Self-contained MobileNetV3-Large definition for Experiment 17."""
from torch import nn
from torchvision.models import MobileNet_V3_Large_Weights, mobilenet_v3_large


def build_model(num_classes=2, pretrained=False, hidden_dim=1280, dropout=0.0):
    weights = MobileNet_V3_Large_Weights.DEFAULT if pretrained else None
    model = mobilenet_v3_large(weights=weights)
    input_features = model.classifier[0].in_features
    layers = [nn.Linear(input_features, hidden_dim), nn.Hardswish()]
    if dropout > 0: layers.append(nn.Dropout(p=dropout))
    layers.append(nn.Linear(hidden_dim, num_classes))
    model.classifier = nn.Sequential(*layers)
    for parameter in model.parameters(): parameter.requires_grad = True
    return model
