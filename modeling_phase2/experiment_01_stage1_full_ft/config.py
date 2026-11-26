"""Configuration for Phase 2 Experiment 1: Stage 1 full fine-tuning with single LR."""

from pathlib import Path


# Repository and Phase 2 data/output locations.
ROOT = Path(__file__).resolve().parents[2]
LABEL_DIR = ROOT / "data_labeling" / "data_labels" / "GOES_2010_2018_NOAA_2019_2026" / "CL_Labels"
IMAGE_DIR = ROOT / "downloaded_data" / "hmi_jpgs"
RESULTS_DIR = ROOT / "results_phase2" / "stage1_full_ft"

# Stage 1 splits (GOES 2010-2014 train, 2015 holdout V1) and untouched 2025-2026 test set.
TRAIN_CSV = LABEL_DIR / "Stage1_train.csv"
HOLDOUT_CSV = LABEL_DIR / "Stage1_holdout.csv"
TEST_CSV = LABEL_DIR / "test_2025_2026_july.csv"


CONFIG = {
    # Architecture: ImageNet weights are used only as the starting point; all layers update.
    "model": "mobilenet_v3_large",
    "pretrained": True,
    "freeze_backbone": False,
    "trainable_backbone_blocks": 2,
    "classifier_hidden_dim": 1280,
    "dropout": 0.0,
    "bn_momentum": 0.1,

    # Input preprocessing: ImageNet statistics matching the pretrained MobileNetV3 weights.
    "image_size": 512,
    "normalization": "imagenet",
    "mean": [0.485, 0.456, 0.406],
    "std": [0.229, 0.224, 0.225],

    # Sampling & loss: full training data once per epoch (matching Phase 1 Experiment 6).
    "sampling": "full_training_data",
    "loss": "ordinary_cross_entropy",
    "augmentation": "none",

    # Optimization: conservative LR=1e-5 and WD=0.1 matching Phase 1 Experiment 6.
    "optimizer": "adamw",
    "learning_rate": 1e-5,
    "min_learning_rate": 1e-7,
    "weight_decay": 0.1,
    "batch_size": 64,

    # Schedule & training length: fixed 40 epochs without premature early stopping.
    "epochs": 40,
    "scheduler": "reduce_lr_on_plateau",
    "scheduler_patience": 3,
    "scheduler_factor": 0.3,

    # Mature checkpoint selection: best V1 holdout CSS after warmup and solid training fit.
    "mature_min_epoch": 5,
    "mature_min_train_css": 0.60,

    "num_workers": 8,
    "random_seed": 42,
}


WANDB_ENTITY = "piyush-luitel-texas-christian-university"
WANDB_PROJECT = "Full disk attention solar flare prediction with continual learning"
WANDB_TAGS = [
    "MobileNetV3-Large",
    "Phase-2",
    "Hybrid-GOES-NOAA",
    "Stage-1",
    "full-finetuning",
    "full-training-data",
    "imagenet-norm",
]
