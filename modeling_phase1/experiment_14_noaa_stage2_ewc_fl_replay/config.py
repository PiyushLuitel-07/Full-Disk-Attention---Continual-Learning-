"""Configuration for NOAA Stage 2 EWC with Stage 1 FL-only replay.

Experiment 14 keeps the corrected Stage 1 Fisher anchor and Stage 2
evaluation protocol from Experiment 13. Its only new training mechanism is a
fixed replay buffer containing temporally thinned Stage 1 flare images.

The defaults define the first controlled comparison. ``train.py`` will expose
the important values as command-line arguments so later runs can change one
factor at a time without editing this file.
"""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
LABEL_DIR = ROOT / "data_labeling" / "data_labels" / "noaa" / "CL_Labels"
REPLAY_DIR = ROOT / "data_labeling" / "data_labels" / "noaa" / "replay_buffers"
IMAGE_DIR = ROOT / "downloaded_data" / "hmi_jpgs"
RESULTS_DIR = ROOT / "results" / "noaa_stage2_ewc_fl_replay"

STAGE1_TRAIN_CSV = LABEL_DIR / "Stage1_train.csv"
STAGE1_HOLDOUT_CSV = LABEL_DIR / "Stage1_holdout.csv"
STAGE2_TRAIN_CSV = LABEL_DIR / "Stage2_train.csv"
STAGE2_HOLDOUT_CSV = LABEL_DIR / "Stage2_holdout.csv"
FUTURE_VALIDATION_CSV = LABEL_DIR / "validation_2024_2025.csv"
STAGE1_FL_REPLAY_CSV = REPLAY_DIR / "stage1_fl_replay.csv"

# Prepared once from the selected Stage 1 model using the corrected
# per-example empirical Fisher estimator.
DEFAULT_EWC_ANCHOR = (
    ROOT
    / "results"
    / "noaa_stage2_ewc"
    / "anchors"
    / "stage1_ewc_ready.pt"
)


CONFIG = {
    "experiment": "noaa_stage2_ewc_stage1_fl_replay",
    "model": "mobilenet_v3_large",
    "pretrained": False,
    "classifier_hidden_dim": 1280,
    "dropout": 0.0,
    "image_size": 512,

    # Each optimization step contains 56 current Stage 2 images and 8 replay
    # images. The total remains 64, matching the earlier experiments.
    "total_batch_size": 64,
    "stage2_batch_size": 56,
    "replay_batch_size": 8,
    "replay_loss_weight": 1.0,
    "replay_source": "stage1_training_fl_only",
    "replay_selection": "all_temporally_thinned_6_hour_windows",
    "replay_sampling": "shuffle_and_cycle_without_replacement",
    "replay_augmentation": "none",
    "stage2_sampling": "uniform_shuffle_without_replacement",
    "stage2_augmentation": "none",
    "num_workers": 8,
    "epochs": 80,
    "random_seed": 123,

    # Best controlled Stage 2 settings identified before adding replay.
    "backbone_learning_rate": 1.3649297100111978e-6,
    "classifier_lr_multiplier": 10.0,
    "classifier_learning_rate": 1.3649297100111978e-5,
    "min_backbone_learning_rate": 1e-8,
    "min_classifier_learning_rate": 1e-7,
    "weight_decay": 0.05328805613626618,
    "optimizer": "adamw",
    "max_gradient_norm": 5.0,

    # BatchNorm running statistics adapt to Stage 2 by default because this
    # improved Stage 2 holdout CSS in the Experiment 13 ablation. A CLI flag
    # will allow the frozen Stage 1 statistics control.
    "freeze_batchnorm_running_stats": False,
    "batchnorm_running_stats_mode": "adapt_on_stage2_training_batches",

    # Unweighted CE gave much lower false-positive rates and better future
    # validation than the earlier mild FL weighting. Replay already increases
    # the effective FL exposure, so the first replay run remains unweighted.
    "loss": "unweighted_cross_entropy",
    "nf_class_weight": 1.0,
    "fl_class_weight": 1.0,
    "ewc_lambda": 5000.0,

    "early_stopping_patience": 15,
    "early_stopping_min_delta": 0.001,
    "scheduler_patience": 5,
    "scheduler_factor": 0.3,

    # Preserve the Experiment 13 protocol. Stage 2 holdout chooses the epoch;
    # Stage 1 metrics are retention diagnostics, and future validation is
    # evaluated only after checkpoint selection.
    "checkpoint_selection": "maximum_stage2_holdout_css",
    "checkpoint_candidates": "stage2_training_epochs_only",
    "checkpoint_diagnostics": [
        "stage2_holdout_recall",
        "stage2_holdout_false_positive_rate",
        "stage2_train_holdout_css_gap",
        "stage1_css_retention_ratio",
        "joint_css",
    ],
}


WANDB_ENTITY = "piyush-luitel-texas-christian-university"
WANDB_PROJECT = "Full disk attention solar flare prediction with continual learning"
WANDB_TAGS = [
    "MobileNetV3-Large",
    "NOAA",
    "Stage-2",
    "experiment-14",
    "EWC",
    "replay-study",
    "continual-learning",
]
