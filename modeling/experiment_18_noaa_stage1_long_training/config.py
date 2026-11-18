"""Fixed configuration for long Stage 1 training and mature selection."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
LABEL_DIR = ROOT / "data_labeling/data_labels/noaa/CL_Labels"
IMAGE_DIR = ROOT / "downloaded_data/hmi_jpgs"
RESULTS_DIR = ROOT / "results/noaa_stage1_long_training"

TRAIN_CSV = LABEL_DIR / "Stage1_train.csv"
HOLDOUT_CSV = LABEL_DIR / "Stage1_holdout.csv"
VALIDATION_CSV = LABEL_DIR / "validation_2024_2025.csv"

# Candidate 114547 hyperparameters, confirmed with seed 123 in run 223444.
CONFIG = {
    "model": "mobilenet_v3_large",
    "pretrained": True,
    "classifier_hidden_dim": 1280,
    "dropout": 0.0,
    "batch_size": 64,
    "epochs": 100,
    "backbone_learning_rate": 1.3649297100111978e-6,
    "classifier_lr_multiplier": 20.0,
    "weight_decay": 0.05328805613626618,
    "min_backbone_learning_rate": 1e-8,
    "min_classifier_learning_rate": 1e-7,
    # Hold each useful learning rate longer so 100-epoch training can continue
    # learning instead of reaching the minimum rate near epoch 32.
    "scheduler_patience": 15,
    "scheduler_factor": 0.3,
    "mature_holdout_tolerance": 0.02,
    "num_workers": 8,
    "random_seed": 123,
}

WANDB_ENTITY = "piyush-luitel-texas-christian-university"
WANDB_PROJECT = "Full disk attention solar flare prediction with continual learning"
WANDB_TAGS = [
    "MobileNetV3-Large",
    "NOAA",
    "Stage-1",
    "experiment-18",
    "100-epochs",
    "inverse-frequency-weighting",
    "mature-checkpoint-selection",
]
