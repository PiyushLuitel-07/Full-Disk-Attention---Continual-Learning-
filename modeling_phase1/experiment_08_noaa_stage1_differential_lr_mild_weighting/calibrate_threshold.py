"""Calibrate Experiment 8's FL threshold on holdout data only.

The selected holdout threshold is then frozen and applied unchanged to the
validation and test periods. No model weights are updated by this script.
"""

import argparse
import csv
import sys
from pathlib import Path

import torch
import wandb


SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.append(str(SCRIPT_DIR.parent))

from config import (
    HOLDOUT_CSV,
    IMAGE_DIR,
    LABEL_DIR,
    VALIDATION_CSV,
    WANDB_ENTITY,
    WANDB_PROJECT,
)
from dataloader import build_evaluation_loader
from evaluation import calculate_classification_metrics
from model import build_model
from train import add_rates


TEST_CSV = LABEL_DIR / "test_2025_2026_july.csv"


def collect_probabilities(model, loader, device):
    """Return FL probabilities and labels without changing the model."""
    model.eval()
    probabilities = []
    targets = []

    with torch.no_grad():
        for images, batch_targets in loader:
            images = images.to(device, non_blocking=True)
            scores = model(images)

            # Column 1 is the model probability assigned to the FL class.
            fl_probabilities = torch.softmax(scores, dim=1)[:, 1]
            probabilities.extend(fl_probabilities.cpu().tolist())
            targets.extend(batch_targets.tolist())

    return probabilities, targets


def metrics_at_threshold(probabilities, targets, threshold):
    """Calculate classification metrics at one fixed FL threshold."""
    predictions = [
        int(probability >= threshold)
        for probability in probabilities
    ]
    return add_rates(calculate_classification_metrics(predictions, targets))


def select_holdout_threshold(probabilities, targets):
    """Find the holdout-CSS optimum without consulting validation or test."""
    rows = []
    for step in range(1, 100):
        threshold = step / 100
        metrics = metrics_at_threshold(probabilities, targets, threshold)
        rows.append({"threshold": threshold, **metrics})

    # Prefer the threshold closest to 0.50 if multiple thresholds have the
    # same CSS. This makes the tie-breaking deterministic and conservative.
    best = max(
        rows,
        key=lambda row: (row["css"], -abs(row["threshold"] - 0.50)),
    )
    return best, rows


def write_rows(path, rows):
    """Write dictionaries with a shared schema to a CSV file."""
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def find_wandb_run_id(checkpoint_path):
    """Find the most recent local W&B run ID beside the checkpoint."""
    run_folders = sorted((checkpoint_path.parent / "wandb").glob("run-*-*"))
    if not run_folders:
        return None
    return run_folders[-1].name.rsplit("-", maxsplit=1)[-1]


def print_metrics(name, metrics):
    """Print the main calibrated metrics in one readable line."""
    print(
        f"{name} | CSS={metrics['css']:.4f} | "
        f"TSS={metrics['tss']:.4f} | HSS={metrics['hss']:.4f} | "
        f"precision={metrics['precision']:.4f} | "
        f"recall={metrics['recall']:.4f} | F1={metrics['f1']:.4f} | "
        f"FPR={metrics['false_positive_rate']:.4f} | "
        f"FNR={metrics['false_negative_rate']:.4f}"
    )


def main():
    """Select on holdout, then evaluate validation and test unchanged."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--wandb-run-id")
    parser.add_argument("--skip-wandb", action="store_true")
    args = parser.parse_args()

    # The checkpoint was created locally by this trusted project.
    checkpoint = torch.load(
        args.checkpoint,
        map_location="cpu",
        weights_only=False,
    )
    config = checkpoint["config"]
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = build_model(
        pretrained=False,
        hidden_dim=int(config["classifier_hidden_dim"]),
        dropout=float(config["dropout"]),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])

    loader_settings = {
        "image_directory": IMAGE_DIR,
        "batch_size": int(config["batch_size"]),
        "num_workers": int(config["num_workers"]),
    }
    holdout_loader = build_evaluation_loader(HOLDOUT_CSV, **loader_settings)
    validation_loader = build_evaluation_loader(
        VALIDATION_CSV,
        **loader_settings,
    )
    test_loader = build_evaluation_loader(TEST_CSV, **loader_settings)

    # Only these holdout predictions participate in threshold selection.
    holdout_probabilities, holdout_targets = collect_probabilities(
        model,
        holdout_loader,
        device,
    )
    best_holdout, threshold_rows = select_holdout_threshold(
        holdout_probabilities,
        holdout_targets,
    )
    selected_threshold = best_holdout["threshold"]

    # Freeze the selected value before looking at validation or test results.
    validation_probabilities, validation_targets = collect_probabilities(
        model,
        validation_loader,
        device,
    )
    test_probabilities, test_targets = collect_probabilities(
        model,
        test_loader,
        device,
    )
    validation_metrics = metrics_at_threshold(
        validation_probabilities,
        validation_targets,
        selected_threshold,
    )
    test_metrics = metrics_at_threshold(
        test_probabilities,
        test_targets,
        selected_threshold,
    )

    curve_path = args.checkpoint.parent / "threshold_calibration_curve.csv"
    metrics_path = args.checkpoint.parent / "calibrated_metrics.csv"
    write_rows(curve_path, threshold_rows)

    result_rows = []
    for split, metrics in (
        ("holdout", best_holdout),
        ("validation", validation_metrics),
        ("test", test_metrics),
    ):
        result_rows.append(
            {
                "split": split,
                "selected_threshold": selected_threshold,
                **{
                    key: value
                    for key, value in metrics.items()
                    if key != "threshold"
                },
            }
        )
    write_rows(metrics_path, result_rows)

    if not args.skip_wandb:
        run_id = args.wandb_run_id or find_wandb_run_id(args.checkpoint)
        if run_id is None:
            raise ValueError(
                "W&B run ID was not found. Pass --wandb-run-id or --skip-wandb."
            )

        run = wandb.init(
            entity=WANDB_ENTITY,
            project=WANDB_PROJECT,
            id=run_id,
            resume="must",
            dir=str(args.checkpoint.parent),
        )
        logged = {
            "calibration/selected_threshold": selected_threshold,
            **{
                f"calibrated_holdout/{key}": value
                for key, value in best_holdout.items()
                if key != "threshold"
            },
            **{
                f"calibrated_validation/{key}": value
                for key, value in validation_metrics.items()
            },
            **{
                f"calibrated_test/{key}": value
                for key, value in test_metrics.items()
            },
        }
        run.log(logged)
        for name, value in logged.items():
            run.summary[name] = value

        artifact = wandb.Artifact(
            f"noaa-stage1-experiment-08-threshold-calibration-{run.id}",
            type="evaluation",
            metadata={
                "selection_split": "holdout",
                "selected_threshold": selected_threshold,
            },
        )
        artifact.add_file(str(curve_path))
        artifact.add_file(str(metrics_path))
        run.log_artifact(artifact)
        run.finish()

    print(f"Selected epoch: {checkpoint['epoch']}")
    print(f"Selected holdout threshold: {selected_threshold:.2f}")
    print_metrics("Calibrated holdout", best_holdout)
    print_metrics("Calibrated validation", validation_metrics)
    print_metrics("Calibrated test", test_metrics)
    print(f"Threshold curve: {curve_path}")
    print(f"Calibrated metrics: {metrics_path}")


if __name__ == "__main__":
    main()
