"""MobileNetV3-Large model for SciQuality Stage 1 experiments."""

from torch import nn
from torchvision.models import MobileNet_V3_Large_Weights, mobilenet_v3_large
# MobileNet_V3_Large_Weights: the official pretrained ImageNet weights.


'''
This function constructs the model.
- num_classes=2: predicts NF or FL.
- pretrained=True: starts with ImageNet-pretrained features.
- classifier_hidden_dim=1280: number of neurons in the classifier’s hidden layer.
- dropout=0.2: randomly disables 20% of classifier neurons during training to reduce overfitting.
'''

def build_model(
    num_classes=2,
    pretrained=True,
    classifier_hidden_dim=1280,
    dropout=0.2,
):
    """Build MobileNetV3-Large with a configurable binary classifier."""
    weights = MobileNet_V3_Large_Weights.DEFAULT if pretrained else None
    model = mobilenet_v3_large(weights=weights)
# The original model predicts 1,000 ImageNet classes, so its classifier must be replaced for solar-flare prediction.
    input_features = model.classifier[0].in_features #Finds how many features MobileNet sends into its classifier.
    # Replaces the original ImageNet classifier with a new classifier for this project.
    model.classifier = nn.Sequential(
        nn.Linear(input_features, classifier_hidden_dim),
        nn.Hardswish(),
        nn.Dropout(p=dropout),
        nn.Linear(classifier_hidden_dim, num_classes),
    )

    return model


# 960 backbone features → 1280 hidden neurons → 2 classes