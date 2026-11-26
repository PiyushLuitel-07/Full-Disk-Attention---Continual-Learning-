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
    "dropout": 0.3,

    # Input preprocessing: symmetric grayscale normalization maps [0, 1] -> [-1, 1]
    # identically across all 3 repeated channels without RGB bias.
    "image_size": 512,
    "normalization": "shared_grayscale",
    "grayscale_mean": 0.5,
    "grayscale_std": 0.5,

    # Sampling & loss: 1:1 FL/NF per epoch, cycling through all unique NF images across epochs.
    "sampling": "cyclic_balanced",
    "loss": "ordinary_cross_entropy",
    "augmentation": "flip_rot5",

    # Optimization: single learning rate in the [1e-5, 1e-4] sweet spot for full fine-tuning.
    "optimizer": "adamw",
    "learning_rate": 3e-5,
    "min_learning_rate": 1e-7,
    "weight_decay": 0.05,
    "batch_size": 64,

    # Schedule & training length: fixed 40 epochs (~11.7 full NF passes) without premature early
    # stopping so we can inspect both peak holdout and mature/final convergence.
    "epochs": 40,
    "scheduler": "reduce_lr_on_plateau",
    "scheduler_patience": 2,
    "scheduler_factor": 0.3,

    # Mature checkpoint selection: best V1 holdout CSS after at least 2 full NF cycles (>= 7 epochs)
    # and strong training fit, alongside overall best holdout and final-epoch checkpoints.
    "mature_min_epoch": 7,
    "mature_min_train_css": 0.90,

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
    "single-lr",
    "cyclic-balanced",
]
