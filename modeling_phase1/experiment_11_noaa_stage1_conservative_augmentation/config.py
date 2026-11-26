"""Configuration for Experiment 11 conservative augmentation."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
LABEL_DIR = ROOT / "data_labeling" / "data_labels" / "noaa" / "CL_Labels"
IMAGE_DIR = ROOT / "downloaded_data" / "hmi_jpgs"
RESULTS_DIR = ROOT / "results" / "noaa_stage1_conservative_augmentation"

TRAIN_CSV = LABEL_DIR / "Stage1_train.csv"
HOLDOUT_CSV = LABEL_DIR / "Stage1_holdout.csv"
VALIDATION_CSV = LABEL_DIR / "validation_2024_2025.csv"


# Keep the validation-selected Experiment 10 configuration fixed so that
# conservative training augmentation is the only experimental change.
CONFIG = {
    "model": "mobilenet_v3_large",
    "pretrained": True,
    "freeze_backbone": False,
    "classifier_hidden_dim": 1280,

    "image_size": 512,
    "batch_size": 64,
    "epochs": 80,

    # The classifier LR is calculated in train.py as backbone LR multiplied
    # by classifier_lr_multiplier. This prevents unrelated LR combinations.
    "backbone_learning_rate": 1.3649297100111978e-6,
    "classifier_lr_multiplier": 20.0,
    "classifier_learning_rate": 2.7298594200223955e-5,
    "min_backbone_learning_rate": 1e-8,
    "min_classifier_learning_rate": 1e-7,

    "weight_decay": 0.05328805613626618,
    "optimizer": "adamw",

    # Every original image is used. Only the FL loss weight is tuned; NF
    # remains the reference class with weight 1.0.
    "loss": "mild_weighted_cross_entropy",
    "nf_class_weight": 1.0,
    "fl_class_weight": 1.990129625950856,
    "sampling": "full_training_data",
    "augmentation": "one_augmented_copy_per_fl_random_affine",
    "dropout": 0.0,

    "early_stopping_patience": 15,
    "early_stopping_min_delta": 0.001,
    "scheduler_patience": 5,
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
    "experiment-11",
    "conservative-augmentation",
    "full-finetuning",
    "differential-learning-rate",
    "mild-class-weighting",
]
