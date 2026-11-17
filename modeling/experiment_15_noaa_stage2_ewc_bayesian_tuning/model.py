"""MobileNetV3-Large used by Experiment 15."""
from torch import nn
from torch.optim import AdamW
from torchvision.models import MobileNet_V3_Large_Weights, mobilenet_v3_large


def build_model(hidden_dim=1280, dropout=0.0, pretrained=False):
    model = mobilenet_v3_large(weights=MobileNet_V3_Large_Weights.DEFAULT if pretrained else None)
    layers = [nn.Linear(model.classifier[0].in_features, hidden_dim), nn.Hardswish()]
    if dropout: layers.append(nn.Dropout(dropout))
    model.classifier = nn.Sequential(*layers, nn.Linear(hidden_dim, 2))
    return model


def build_optimizer(model, config):
    """Use separate backbone/head LRs and exclude bias/norm vectors from decay."""
    groups = {(part, decay): [] for part in ("backbone", "classifier") for decay in (0, 1)}
    for name, parameter in model.named_parameters():
        part = "classifier" if name.startswith("classifier.") else "backbone"
        groups[part, int(not (name.endswith(".bias") or parameter.ndim == 1))].append(parameter)
    learning_rates = {"backbone": config["backbone_learning_rate"],
        "classifier": config["classifier_learning_rate"]}
    return AdamW([{"params": groups[part, decay], "lr": learning_rates[part],
                   "weight_decay": config["weight_decay"] if decay else 0.0}
                  for part in ("backbone", "classifier") for decay in (1, 0)])
