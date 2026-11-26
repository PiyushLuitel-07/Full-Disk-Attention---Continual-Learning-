"""Fixed configuration for Stage 2 EWC seed confirmation."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
LABELS = ROOT / "data_labeling/data_labels/noaa/CL_Labels"
IMAGE_DIR = ROOT / "downloaded_data/hmi_jpgs"
RESULTS_DIR = ROOT / "results/noaa_stage2_ewc_seed_confirmation"
ANCHOR = ROOT / "results/noaa_stage2_ewc/anchors/stage1_ewc_ready.pt"

STAGE1_TRAIN_CSV = LABELS / "Stage1_train.csv"
STAGE1_HOLDOUT_CSV = LABELS / "Stage1_holdout.csv"
STAGE2_TRAIN_CSV = LABELS / "Stage2_train.csv"
STAGE2_HOLDOUT_CSV = LABELS / "Stage2_holdout.csv"
VALIDATION_CSV = LABELS / "validation_2024_2025.csv"
TEST_CSV = LABELS / "test_2025_2026_july.csv"

CANDIDATES = {
    "014029": ROOT / "results/noaa_stage2_ewc_bayesian_tuning/mobilenet_stage2_ewc_tuning_20261114_014029/stage2_best.pt",
    "014124": ROOT / "results/noaa_stage2_ewc_bayesian_tuning/mobilenet_stage2_ewc_tuning_20261115_014124/stage2_best.pt",
    "070421": ROOT / "results/noaa_stage2_ewc_bayesian_tuning/mobilenet_stage2_ewc_tuning_20261114_070421/stage2_best.pt",
}
SEEDS = (42, 123, 2026)

WANDB_ENTITY = "piyush-luitel-texas-christian-university"
WANDB_PROJECT = "Full disk attention solar flare prediction with continual learning"
WANDB_TAGS = ["NOAA", "Stage-2", "MobileNetV3-Large", "experiment-16", "EWC-only", "seed-confirmation"]

