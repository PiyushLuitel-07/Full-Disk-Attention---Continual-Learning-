"""Fixed Stage 1 configuration for the Stage 2 EWC-lambda ablation."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LABELS = ROOT / "data_labeling/data_labels/noaa/CL_Labels"
IMAGE_DIR = ROOT / "downloaded_data/hmi_jpgs"
RESULTS_DIR = ROOT / "results/noaa_stage2_stage1_hparams_lambda_tuning"
ANCHOR = ROOT / "results/noaa_stage2_ewc/anchors/stage1_ewc_ready.pt"
STAGE1_TRAIN_CSV = LABELS / "Stage1_train.csv"
STAGE1_HOLDOUT_CSV = LABELS / "Stage1_holdout.csv"
STAGE2_TRAIN_CSV = LABELS / "Stage2_train.csv"
STAGE2_HOLDOUT_CSV = LABELS / "Stage2_holdout.csv"
VALIDATION_CSV = LABELS / "validation_2024_2025.csv"

# Exact optimization configuration of the selected Stage 1 model. At runtime,
# train.py verifies these values against source_stage1_config in the EWC anchor.
CONFIG = {
    "experiment": "noaa_stage2_stage1_hparams_lambda_tuning",
    "model": "mobilenet_v3_large", "classifier_hidden_dim": 1280, "dropout": 0.0,
    "batch_size": 64, "num_workers": 8, "epochs": 80, "random_seed": 123,
    "backbone_learning_rate": 1.3649297100111978e-6,
    "classifier_lr_multiplier": 20.0,
    "classifier_learning_rate": 2.7298594200223956e-5,
    "min_backbone_learning_rate": 1e-8, "min_classifier_learning_rate": 1e-7,
    "weight_decay": 0.05328805613626618,
    # Stage 2-specific mild inverse-frequency weighting.  Its train split has
    # 40,344 NF and 3,781 FL examples, so w_FL=sqrt(NF/FL)=3.266...
    "nf_class_weight": 1.0, "fl_class_weight": 3.266526147242088,
    "label_smoothing": 0.0, "adapt_batchnorm_running_stats": True,
    "scheduler_factor": 0.3, "scheduler_patience": 5,
    "early_stopping_patience": 15, "early_stopping_min_delta": 0.001,
    "fisher_batch_size": 8, "fisher_num_workers": 8,
    "sampling": "uniform_without_replacement", "augmentation": "none", "replay": False,
    "checkpoint_selection": "maximum_stage2_holdout_css",
    "ewc_lambda": 0.0,
}

STAGE1_KEYS = (
    "classifier_hidden_dim", "batch_size", "num_workers", "epochs", "random_seed",
    "backbone_learning_rate", "classifier_lr_multiplier", "min_backbone_learning_rate",
    "min_classifier_learning_rate", "weight_decay",
    "scheduler_factor", "scheduler_patience", "early_stopping_patience",
    "early_stopping_min_delta", "augmentation", "dropout",
)

WANDB_ENTITY = "piyush-luitel-texas-christian-university"
WANDB_PROJECT = "Full disk attention solar flare prediction with continual learning"
WANDB_TAGS = ["NOAA", "Stage-2", "MobileNetV3-Large", "experiment-17", "EWC-only", "lambda-ablation"]
