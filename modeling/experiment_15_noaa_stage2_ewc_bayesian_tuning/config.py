"""Compact configuration for Stage 2 EWC-only Bayesian tuning."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LABELS = ROOT / "data_labeling/data_labels/noaa/CL_Labels"
IMAGE_DIR = ROOT / "downloaded_data/hmi_jpgs"
RESULTS_DIR = ROOT / "results/noaa_stage2_ewc_bayesian_tuning"
ANCHOR = ROOT / "results/noaa_stage2_ewc/anchors/stage1_ewc_ready.pt"
STAGE1_TRAIN_CSV = LABELS / "Stage1_train.csv"
STAGE1_HOLDOUT_CSV = LABELS / "Stage1_holdout.csv"
STAGE2_TRAIN_CSV = LABELS / "Stage2_train.csv"
STAGE2_HOLDOUT_CSV = LABELS / "Stage2_holdout.csv"
VALIDATION_CSV = LABELS / "validation_2024_2025.csv"

CONFIG = {
    "experiment": "noaa_stage2_ewc_bayesian_tuning",
    "model": "mobilenet_v3_large", "classifier_hidden_dim": 1280, "dropout": 0.0,
    "batch_size": 64, "num_workers": 8, "epochs": 40, "random_seed": 123,
    "ewc_lambda": 5000.0, "backbone_learning_rate": 7e-7,
    "classifier_lr_multiplier": 5.0, "weight_decay": 0.08, "label_smoothing": 0.05,
    "adapt_batchnorm_running_stats": True, "nf_class_weight": 1.0, "fl_class_weight": 1.0,
    "min_backbone_learning_rate": 1e-8, "min_classifier_learning_rate": 1e-7,
    "max_gradient_norm": 5.0, "scheduler_factor": 0.3, "scheduler_patience": 3,
    "early_stopping_patience": 10, "early_stopping_min_delta": 0.001,
    "fisher_batch_size": 8, "fisher_num_workers": 8,
    "sampling": "uniform_without_replacement", "augmentation": "none", "replay": False,
    "checkpoint_selection": "maximum_stage2_holdout_css",
}

WANDB_ENTITY = "piyush-luitel-texas-christian-university"
WANDB_PROJECT = "Full disk attention solar flare prediction with continual learning"
WANDB_TAGS = ["NOAA", "Stage-2", "MobileNetV3-Large", "experiment-15", "EWC-only", "bayesian-tuning"]
