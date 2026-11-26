"""Configuration for the NOAA three-stage joint-training baseline."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
LABEL_DIR = ROOT / "data_labeling/data_labels/noaa/CL_Labels"
IMAGE_DIR = ROOT / "downloaded_data/hmi_jpgs"
RESULTS_DIR = ROOT / "results/noaa_joint_training"

JOINT_TRAIN_CSV = LABEL_DIR / "Joint_train.csv"
HOLDOUT_CSVS = {
    "stage1": LABEL_DIR / "Stage1_holdout.csv",
    "stage2": LABEL_DIR / "Stage2_holdout.csv",
    "stage3": LABEL_DIR / "Stage3_holdout.csv",
}
VALIDATION_CSV = LABEL_DIR / "validation_2024_2025.csv"

# Experiment 9 search center with Experiment 18's fixed OneCycleLR protocol.
CONFIG = {
    "model": "mobilenet_v3_large",
    "pretrained": True,
    "classifier_hidden_dim": 1280,
    "dropout": 0.0,
    "batch_size": 64,
    "epochs": 60,
    "backbone_learning_rate": 1.3649297100111978e-6,
    "classifier_lr_multiplier": 20.0,
    "weight_decay": 0.05328805613626618,
    "nf_class_weight": 1.0,
    "fl_class_weight": 2.256075460153488,
    "optimizer": "adamw",
    "scheduler": "onecycle",
    "onecycle_pct_start": 0.10,
    "onecycle_div_factor": 10.0,
    "onecycle_final_div_factor": 1000.0,
    "normalization": "shared_grayscale",
    "grayscale_mean": 0.5,
    "grayscale_std": 0.5,
    "augmentation": "none",
    "num_workers": 8,
    "random_seed": 123,
    "checkpoint_selection": "maximum_macro_stage_holdout_css",
}

WANDB_ENTITY = "piyush-luitel-texas-christian-university"
WANDB_PROJECT = "Full disk attention solar flare prediction with continual learning"
WANDB_TAGS = [
    "MobileNetV3-Large",
    "NOAA",
    "joint-training",
    "experiment-19",
    "offline-upper-bound",
    "bayesian-sweep",
    "onecycle",
    "no-augmentation",
]
