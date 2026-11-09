"""Fixed configuration for the Stage 1 inverse-frequency control."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
LABEL_DIR = ROOT / "data_labeling" / "data_labels" / "noaa" / "CL_Labels"
IMAGE_DIR = ROOT / "downloaded_data" / "hmi_jpgs"
RESULTS_DIR = ROOT / "results" / "noaa_stage1_inverse_frequency_control"

TRAIN_CSV = LABEL_DIR / "Stage1_train.csv"
HOLDOUT_CSV = LABEL_DIR / "Stage1_holdout.csv"
VALIDATION_CSV = LABEL_DIR / "validation_2024_2025.csv"
TEST_CSV = LABEL_DIR / "test_2025_2026_july.csv"


# This is a one-run ablation. It reproduces the selected Candidate 114547
# design (seed 123) and changes only the training-loss class-weight ratio.
#
# Stage 1 has 38,015 NF and 8,623 FL samples. Inverse-frequency weighting
# gives an FL:NF ratio of 38,015 / 8,623 = 4.4085585. CrossEntropyLoss with
# mean reduction is invariant to a common scaling of both weights, therefore
# (1.0, 4.4085585) is equivalent to the older (0.6134, 2.7043) weighting.
CONFIG = {
    "experiment": "stage1_inverse_frequency_control",
    "model": "mobilenet_v3_large",
    "pretrained": True,
    "classifier_hidden_dim": 1280,
    "image_size": 512,
    "batch_size": 64,
    "epochs": 80,
    "backbone_learning_rate": 1.3649297100111978e-6,
    "classifier_lr_multiplier": 20.0,
    "classifier_learning_rate": 2.7298594200223956e-5,
    "min_backbone_learning_rate": 1e-8,
    "min_classifier_learning_rate": 1e-7,
    "weight_decay": 0.05328805613626618,
    "optimizer": "adamw",
    "loss": "inverse_frequency_weighted_cross_entropy",
    "nf_class_weight": 1.0,
    "fl_class_weight": 4.408558506320306,
    "sampling": "full_training_data",
    "augmentation": "none",
    "dropout": 0.0,
    "early_stopping_patience": 15,
    "early_stopping_min_delta": 0.001,
    "scheduler_patience": 5,
    "scheduler_factor": 0.3,
    "num_workers": 8,
    "random_seed": 123,
}


WANDB_ENTITY = "piyush-luitel-texas-christian-university"
WANDB_PROJECT = "Full disk attention solar flare prediction with continual learning"
WANDB_TAGS = [
    "NOAA",
    "Stage-1",
    "MobileNetV3-Large",
    "experiment-12",
    "inverse-frequency-weighting",
    "control-ablation",
    "full-finetuning",
    "differential-learning-rate",
]
