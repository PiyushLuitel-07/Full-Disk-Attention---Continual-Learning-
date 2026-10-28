"""Configuration for the NOAA Stage 1 full-data weighted-loss experiment."""

from pathlib import Path


# Project paths
ROOT = Path(__file__).resolve().parents[2]
LABEL_DIR = ROOT / "data_labeling/data_labels/noaa/CL_Labels"
IMAGE_DIR = ROOT / "downloaded_data/hmi_jpgs"
RESULTS_DIR = ROOT / "results/noaa_stage1_weighted_full_data"

TRAIN_CSV = LABEL_DIR / "Stage1_train.csv"
HOLDOUT_CSV = LABEL_DIR / "Stage1_holdout.csv"
VALIDATION_CSV = LABEL_DIR / "validation_2024_2025.csv"


# Keep the best sweep trial's settings fixed. The only intended change in this
# experiment is using every training image with a class-weighted loss.
CONFIG = {
    "model": "mobilenet_v3_large",
    "pretrained": True,
    "freeze_backbone": True,
    "trainable_backbone_blocks": 2,
    "classifier_hidden_dim": 1280,
    "image_size": 512,
    "batch_size": 64,
    "epochs": 20,
    "learning_rate": 0.0002302,
    "weight_decay": 0.07529,
    "optimizer": "adamw",
    "dropout": 0.0,
    "augmentation": "none",
    "early_stopping_patience": 10,
    "early_stopping_min_delta": 0.001,
    "scheduler_patience": 3,
    "scheduler_factor": 0.3,
    "min_learning_rate": 1e-7,
    "num_workers": 8,
    "random_seed": 42,
}


WANDB_ENTITY = "piyush-luitel-texas-christian-university"
WANDB_PROJECT = (
    "Full disk attention solar flare prediction with continual learning"
)
