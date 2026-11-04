"""Configuration for Experiment 8: differential LR with mild FL weighting."""

from pathlib import Path


# Repository and data locations.
ROOT = Path(__file__).resolve().parents[2]
LABEL_DIR = ROOT / "data_labeling" / "data_labels" / "noaa" / "CL_Labels"
IMAGE_DIR = ROOT / "downloaded_data" / "hmi_jpgs"
RESULTS_DIR = ROOT / "results" / "noaa_stage1_differential_lr_mild_weighting"

TRAIN_CSV = LABEL_DIR / "Stage1_train.csv"
HOLDOUT_CSV = LABEL_DIR / "Stage1_holdout.csv"
VALIDATION_CSV = LABEL_DIR / "validation_2024_2025.csv"


CONFIG = {
    # MobileNetV3-Large starts from ImageNet weights, but every layer is trainable.
    "model": "mobilenet_v3_large",
    "pretrained": True,
    "freeze_backbone": False,
    "classifier_hidden_dim": 1280,

    # Input and training settings remain comparable with Experiment 6.
    "image_size": 512,
    "batch_size": 64,
    "epochs": 40,

    # The pretrained backbone changes slowly, while the new NF/FL classifier
    # learns faster because it begins with randomly initialized weights.
    "backbone_learning_rate": 1e-6,
    "classifier_learning_rate": 1e-5,
    "min_backbone_learning_rate": 1e-8,
    "min_classifier_learning_rate": 1e-7,

    # Moderate regularization is applied to eligible weight parameters.
    # Bias and normalization parameters will be excluded in train.py.
    "weight_decay": 0.1,
    "optimizer": "adamw",

    # Use every original image. Mild weighting makes an FL error 1.5 times as
    # costly as an NF error without the aggressive inverse-frequency weights
    # that produced excessive false alarms in Experiment 2.
    "loss": "mild_weighted_cross_entropy",
    "nf_class_weight": 1.0,
    "fl_class_weight": 1.5,
    "sampling": "full_training_data",
    "augmentation": "none",
    "dropout": 0.0,

    # Holdout CSS controls checkpoint selection, scheduling, and early stopping.
    "early_stopping_patience": 10,
    "early_stopping_min_delta": 0.001,
    "scheduler_patience": 3,
    "scheduler_factor": 0.3,

    "num_workers": 8,
    "random_seed": 42,
}


WANDB_ENTITY = "piyush-luitel-texas-christian-university"
WANDB_PROJECT = "Full disk attention solar flare prediction with continual learning"
WANDB_TAGS = [
    "MobileNetV3-Large",
    "NOAA",
    "Stage-1",
    "experiment-08",
    "full-finetuning",
    "differential-learning-rate",
    "mild-class-weighting",
]

