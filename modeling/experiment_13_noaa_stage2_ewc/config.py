"""Configuration for corrected NOAA Stage 2 EWC training.

The Stage 1 checkpoint is fixed. Fisher preparation and Stage 2 training are
separate commands so that every EWC-lambda trial uses exactly the same Stage 1
anchor and exactly the same Fisher estimate.
"""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
LABEL_DIR = ROOT / "data_labeling" / "data_labels" / "noaa" / "CL_Labels"
IMAGE_DIR = ROOT / "downloaded_data" / "hmi_jpgs"
RESULTS_DIR = ROOT / "results" / "noaa_stage2_ewc"

# Selected Candidate 114547, seed 123, selected at epoch 10.
STAGE1_CHECKPOINT = (
    ROOT
    / "results"
    / "noaa_stage1_seed_confirmation"
    / "mobilenet_stage1_20261106_223444"
    / "stage1_best.pt"
)

STAGE1_TRAIN_CSV = LABEL_DIR / "Stage1_train.csv"
STAGE1_HOLDOUT_CSV = LABEL_DIR / "Stage1_holdout.csv"
STAGE2_TRAIN_CSV = LABEL_DIR / "Stage2_train.csv"
STAGE2_HOLDOUT_CSV = LABEL_DIR / "Stage2_holdout.csv"
FUTURE_VALIDATION_CSV = LABEL_DIR / "validation_2024_2025.csv"


CONFIG = {
    "experiment": "noaa_stage2_corrected_ewc",
    "model": "mobilenet_v3_large",
    "pretrained": False,
    "classifier_hidden_dim": 1280,
    "dropout": 0.0,
    "image_size": 512,

    # Stage 2 uses every original image. No augmentation is included in this
    # first EWC experiment, keeping EWC as the only continual-learning change.
    "sampling": "full_training_data",
    "augmentation": "none",
    "batch_size": 64,
    "num_workers": 8,
    "epochs": 80,
    "random_seed": 123,

    # Retain the selected Stage 1 optimization configuration initially. This
    # avoids changing optimization and EWC simultaneously.
    "backbone_learning_rate": 1.3649297100111978e-6,
    "classifier_lr_multiplier": 20.0,
    "classifier_learning_rate": 2.7298594200223956e-5,
    "min_backbone_learning_rate": 1e-8,
    "min_classifier_learning_rate": 1e-7,
    "weight_decay": 0.05328805613626618,
    "optimizer": "adamw",
    "max_gradient_norm": 5.0,
    # BatchNorm running statistics are buffers, not parameters, so EWC cannot
    # penalize their drift. Keep those statistics fixed at the Stage 1 values
    # while still allowing BatchNorm affine parameters to learn under EWC.
    "freeze_batchnorm_running_stats": True,

    # Keep the selected Stage 1 mild weighting for the first Stage 2 control.
    # EWC lambda will be chosen only after inspecting the corrected Fisher
    # scale; it is deliberately not guessed here.
    "loss": "mild_weighted_cross_entropy",
    "nf_class_weight": 1.0,
    "fl_class_weight": 1.990129625950856,
    "ewc_lambda": None,

    "early_stopping_patience": 15,
    "early_stopping_min_delta": 0.001,
    "scheduler_patience": 5,
    "scheduler_factor": 0.3,

    # Correct empirical Fisher: mean of per-example squared gradients of
    # log p(y|x, theta), evaluated on original Stage 1 images. The batch size
    # changes throughput/memory only; it must not change the estimate.
    "fisher_estimator": "per_example_empirical_fisher",
    "fisher_split": "complete_original_stage1_training_data",
    "fisher_batch_size": 8,
    "fisher_num_workers": 8,
    "fisher_use_ground_truth_labels": True,
    "fisher_model_mode": "eval",
    "fisher_normalization": "divide_by_number_of_examples",
    # Selection balances current Stage 2 learning with Stage 1 retention.
    # Future validation remains untouched until final evaluation.
    "checkpoint_selection": "geometric_mean_stage1_stage2_holdout_css",
    "checkpoint_candidates": "epoch_0_and_all_stage2_training_epochs",
}


WANDB_ENTITY = "piyush-luitel-texas-christian-university"
WANDB_PROJECT = "Full disk attention solar flare prediction with continual learning"
WANDB_TAGS = [
    "MobileNetV3-Large",
    "NOAA",
    "Stage-2",
    "experiment-13",
    "EWC",
    "corrected-per-example-fisher",
    "continual-learning",
]
