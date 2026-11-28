"""Configuration for Phase 2 Experiment 2: Stage 2 EWC continual learning."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
LABELS_DIR = ROOT / "data_labeling" / "data_labels" / "GOES_2010_2018_NOAA_2019_2026" / "CL_Labels_noaa_dates"
IMAGE_DIR = ROOT / "downloaded_data" / "hmi_jpgs"
RESULTS_DIR = ROOT / "results_phase2" / "stage2_ewc" / "runs"

DEFAULT_ANCHOR = ROOT / "results_phase2" / "stage2_ewc" / "anchors" / "stage1_ewc_ready.pt"

STAGE1_HOLDOUT_CSV = LABELS_DIR / "Stage1_holdout.csv"  # V1 (2017)
STAGE2_TRAIN_CSV = LABELS_DIR / "Stage2_train.csv"      # 2018-01-02 .. 2023-03-08
STAGE2_HOLDOUT_CSV = LABELS_DIR / "Stage2_holdout.csv"  # V2 (2023-03-10 .. 2023-06-30)
TEST_CSV = LABELS_DIR / "test_2025_2026_8m.csv"          # Untouched 8-month test (2025-12-02 .. 2026-07-31)


CONFIG = {
    "experiment": "phase2_stage2_ewc",
    "ewc_lambda": 10.0,

    # Stage 2 optimization (architecture/normalization inherited from Stage 1 anchor).
    "sampling": "full_training_data",
    "augmentation": "none",
    "loss": "sqrt_inverse_weighted_cross_entropy",
    "nf_class_weight": 1.0,
    "fl_class_weight": None,  # Computed as sqrt(NF / FL) unless --fl-weight is provided.

    "optimizer": "adamw",
    "backbone_learning_rate": 2e-6,
    "classifier_learning_rate": 1.5e-5,
    "min_backbone_learning_rate": 1e-8,
    "min_classifier_learning_rate": 1e-7,
    "weight_decay": 0.1,
    "batch_size": 64,
    "max_gradient_norm": 5.0,
    "adapt_batchnorm_running_stats": True,

    "epochs": 35,
    "scheduler": "reduce_lr_on_plateau",
    "scheduler_patience": 5,
    "scheduler_factor": 0.3,

    "num_workers": 8,
    "random_seed": 42,
}

WANDB_ENTITY = "piyush-luitel-texas-christian-university"
WANDB_PROJECT = "Full disk attention solar flare prediction with continual learning"
WANDB_TAGS = [
    "MobileNetV3-Large",
    "Phase-2",
    "Hybrid-GOES-NOAA",
    "Stage-2",
    "EWC",
    "continual-learning",
]
