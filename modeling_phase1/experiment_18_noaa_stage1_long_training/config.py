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
    # Regularize the classifier head to reduce the severe train/holdout gap
    # observed in the no-dropout OneCycleLR runs.
    "dropout": 0.2,
    "batch_size": 64,
    "epochs": 60,
    "backbone_learning_rate": 1.3649297100111978e-6,
    "classifier_lr_multiplier": 20.0,
    "weight_decay": 0.05328805613626618,
    "min_backbone_learning_rate": 1e-8,
    "min_classifier_learning_rate": 1e-7,
    "scheduler": "onecycle",
    # Used only by the ReduceLROnPlateau control run.
    "scheduler_patience": 5,
    "scheduler_factor": 0.3,
    # OneCycleLR warms up for the first 10% of optimizer steps and then
    # anneals for the remaining 90%. Configured LRs are treated as peak LRs.
    "onecycle_pct_start": 0.10,
    "onecycle_div_factor": 10.0,
    "onecycle_final_div_factor": 1000.0,
    "mature_holdout_tolerance": 0.02,
    "num_workers": 8,
    "random_seed": 123,
    # Map grayscale values from [0, 1] to approximately [-1, 1], then repeat
    # the identically normalized channel for the pretrained RGB backbone.
    "normalization": "shared_grayscale",
    "grayscale_mean": 0.5,
    "grayscale_std": 0.5,
}

WANDB_ENTITY = "piyush-luitel-texas-christian-university"
WANDB_PROJECT = "Full disk attention solar flare prediction with continual learning"
WANDB_TAGS = [
    "MobileNetV3-Large",
    "NOAA",
    "Stage-1",
    "experiment-18",
    "60-epochs",
    "inverse-frequency-weighting",
    "mature-checkpoint-selection",
]
