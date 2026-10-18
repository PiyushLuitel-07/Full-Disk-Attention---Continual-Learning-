"""Evaluate a trained SciQuality Stage 1 MobileNetV3 checkpoint."""

import argparse
import csv
import sys
from pathlib import Path

import torch
from torch import nn


SCRIPT_DIR = Path(__file__).resolve().parent
MODELING_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.append(str(MODELING_DIR))

from dataloader import build_evaluation_loader
from evaluation import calculate_classification_metrics
from model import build_model


ROOT = Path(__file__).resolve().parents[2]
IMAGE_DIR = ROOT / "downloaded_data/hmi_jpgs"
DEFAULT_TEST_CSV = (
    ROOT
    / "data_labeling/data_labels/sciquality/future_evaluation_2024_2026"
    / "test_2025_2026_july.csv"
)


def parse_args():
    """Read checkpoint and evaluation-dataset options."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--csv", type=Path, default=DEFAULT_TEST_CSV)
    parser.add_argument("--dataset-name", default="test_2025_2026_july")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--num-workers", type=int, default=8)
    return parser.parse_args()


def evaluate(model, loader, device):
    """Return loss and binary metrics without changing the model."""
    model.eval()
    criterion = nn.CrossEntropyLoss()
    predictions, targets = [], []
    total_loss = total_images = 0

    with torch.no_grad():
        for images, batch_targets in loader:
            images = images.to(device, non_blocking=True)
            batch_targets = batch_targets.to(device, non_blocking=True)
            scores = model(images)
            loss = criterion(scores, batch_targets)

            size = images.size(0)
            total_loss += loss.item() * size
            total_images += size
            predictions.extend(scores.argmax(1).cpu().tolist())
            targets.extend(batch_targets.cpu().tolist())

    metrics = calculate_classification_metrics(predictions, targets)
    metrics["loss"] = total_loss / total_images
    return metrics


def main():
    args = parse_args()
    checkpoint_path = args.checkpoint.expanduser().resolve()
    csv_path = args.csv.expanduser().resolve()

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    required = {"model_state_dict", "config", "architecture"}
    missing = required - set(checkpoint)
    if missing:
        raise ValueError(f"Checkpoint is missing keys: {sorted(missing)}")
    if checkpoint["architecture"] != "mobilenet_v3_large":
        raise ValueError("This evaluator requires a MobileNetV3-Large checkpoint.")

    config = checkpoint["config"]
    batch_size = args.batch_size or int(config["batch_size"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Pretrained weights are unnecessary because the complete trained state is loaded.
    model = build_model(
        pretrained=False,
        classifier_hidden_dim=int(config["classifier_hidden_dim"]),
        dropout=float(config["dropout"]),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    loader = build_evaluation_loader(
        csv_path, IMAGE_DIR, batch_size, args.num_workers
    )
    metrics = evaluate(model, loader, device)

    output_path = checkpoint_path.parent / f"{args.dataset_name}_metrics.csv"
    with output_path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["dataset", *metrics])
        writer.writeheader()
        writer.writerow({"dataset": args.dataset_name, **metrics})

    print(f"Device: {device}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Dataset: {args.dataset_name}")
    print(f"Samples: {len(loader.dataset):,}")
    for name, value in metrics.items():
        print(f"{name}: {value:.6f}" if isinstance(value, float) else f"{name}: {value}")
    print(f"Saved metrics: {output_path}")


if __name__ == "__main__":
    main()
