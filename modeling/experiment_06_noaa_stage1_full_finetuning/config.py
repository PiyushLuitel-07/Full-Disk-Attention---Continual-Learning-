"""Configuration for Experiment 6: full MobileNetV3 fine-tuning."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
LABEL_DIR = ROOT / "data_labeling/data_labels/noaa/CL_Labels"
IMAGE_DIR = ROOT / "downloaded_data/hmi_jpgs"
RESULTS_DIR = ROOT / "results/noaa_stage1_full_finetuning"

TRAIN_CSV = LABEL_DIR / "Stage1_train.csv"
HOLDOUT_CSV = LABEL_DIR / "Stage1_holdout.csv"
VALIDATION_CSV = LABEL_DIR / "validation_2024_2025.csv"


# This controlled experiment uses ImageNet weights only as initialization.
# Every MobileNet feature layer and the new NF/FL classifier will be trained.
CONFIG = {
    "model": "mobilenet_v3_large",
    "pretrained": True,
    "freeze_backbone": False,
    "classifier_hidden_dim": 1280,
    "image_size": 512,
    "batch_size": 64,
    "epochs": 40,
    "learning_rate": 1e-5,
    "weight_decay": 0.1,
    "optimizer": "adamw",
    "loss": "ordinary_cross_entropy",
    "dropout": 0.0,
    "augmentation": "none",
    "sampling": "full_training_data",
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
