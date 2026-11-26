"""Evaluate Experiment 18 strict, mature, and last checkpoints."""

import argparse
import csv
import sys
from pathlib import Path

import torch
from torch import nn

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(HERE), str(ROOT / "modeling"), str(ROOT)]

from config import IMAGE_DIR  # noqa: E402
from dataloader import build_loader  # noqa: E402
from evaluation import calculate_classification_metrics  # noqa: E402
from model import build_model  # noqa: E402

TEST_CSV = ROOT / "data_labeling/data_labels/noaa/CL_Labels/test_2025_2026_july.csv"
VALIDATION_CSV = ROOT / "data_labeling/data_labels/noaa/CL_Labels/validation_2024_2025.csv"
CHECKPOINTS = {
    "strict_best_holdout": "stage1_best_holdout.pt",
    "mature_selected": "stage1_selected_mature.pt",
    "last_epoch": "stage1_last.pt",
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--checkpoint", type=Path, action="append")
    inputs.add_argument(
        "--run-dir",
        type=Path,
        action="append",
        help="Experiment 18 result directory; evaluates all three checkpoint types.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=8)
    return parser.parse_args()


def roc_auc(scores, targets):
    positives, negatives = sum(targets), len(targets) - sum(targets)
    if not positives or not negatives:
        return float("nan")
    ordered = sorted(enumerate(scores), key=lambda item: item[1])
    ranks, start = [0.0] * len(scores), 0
    while start < len(ordered):
        end = start + 1
        while end < len(ordered) and ordered[end][1] == ordered[start][1]:
            end += 1
        rank = ((start + 1) + end) / 2
        for position in range(start, end):
            ranks[ordered[position][0]] = rank
        start = end
    rank_sum = sum(rank for rank, target in zip(ranks, targets) if target == 1)
    return (rank_sum - positives * (positives + 1) / 2) / (positives * negatives)


def average_precision(scores, targets):
    positives = sum(targets)
    if not positives:
        return float("nan")
    ordered = sorted(zip(scores, targets), reverse=True)
    tp = predicted = start = 0
    previous_recall = result = 0.0
    while start < len(ordered):
        end = start + 1
        while end < len(ordered) and ordered[end][0] == ordered[start][0]:
            end += 1
        predicted += end - start
        tp += sum(target for _, target in ordered[start:end])
        recall, precision = tp / positives, tp / predicted
        result += (recall - previous_recall) * precision
        previous_recall, start = recall, end
    return result


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    criterion = nn.CrossEntropyLoss()
    predictions, targets, scores, loss_sum = [], [], [], 0.0
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        logits = model(images)
        loss_sum += criterion(logits, labels).item() * labels.numel()
        predictions.extend(logits.argmax(1).cpu().tolist())
        scores.extend(logits.softmax(1)[:, 1].cpu().tolist())
        targets.extend(labels.cpu().tolist())
    metrics = calculate_classification_metrics(predictions, targets)
    metrics.update({
        "loss": loss_sum / len(targets),
        "false_positive_rate": metrics["fp"] / (metrics["fp"] + metrics["tn"]),
        "false_negative_rate": metrics["fn"] / (metrics["fn"] + metrics["tp"]),
        "predicted_fl_rate": (metrics["tp"] + metrics["fp"]) / len(targets),
        "roc_auc": roc_auc(scores, targets),
        "average_precision": average_precision(scores, targets),
    })
    return metrics


def main():
    args = parse_args()
    if args.batch_size <= 0 or args.num_workers < 0:
        raise ValueError("batch-size must be positive and num-workers nonnegative")
    checkpoints = args.checkpoint or [
        directory / filename
        for directory in args.run_dir
        for filename in CHECKPOINTS.values()
    ]
    if len(set(checkpoints)) != len(checkpoints):
        raise ValueError("Each checkpoint must be listed only once")
    for split, path in {"validation": VALIDATION_CSV, "test": TEST_CSV}.items():
        if not path.is_file():
            raise FileNotFoundError(f"{split.title()} CSV not found: {path}")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    rows = []
    for path in checkpoints:
        if not path.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        saved = torch.load(path, map_location=device, weights_only=False)
        config = saved["config"]
        model = build_model(
            pretrained=False,
            hidden_dim=config.get("classifier_hidden_dim", 1280),
            dropout=config.get("dropout", 0.0),
        ).to(device)
        model.load_state_dict(saved["model_state_dict"], strict=True)
        split_metrics = {}
        for split, csv_path in {"validation": VALIDATION_CSV, "test": TEST_CSV}.items():
            loader = build_loader(
                csv_path,
                IMAGE_DIR,
                args.batch_size,
                args.num_workers,
                augmentation="none",
                normalization=config.get("normalization", "shared_grayscale"),
                grayscale_mean=config.get("grayscale_mean", 0.5),
                grayscale_std=config.get("grayscale_std", 0.5),
            )
            split_metrics[split] = evaluate(model, loader, device)
            split_metrics[split]["samples"] = len(loader.dataset)
        row = {
            "run_name": config.get("run_name", path.parent.name),
            "checkpoint_type": next(
                (name for name, filename in CHECKPOINTS.items() if filename == path.name),
                "custom",
            ),
            "checkpoint": path.name,
            "selected_epoch": saved.get("epoch"),
            **{
                f"{split}_{key}": value
                for split, metrics in split_metrics.items()
                for key, value in metrics.items()
            },
        }
        rows.append(row)
        validation, test = split_metrics["validation"], split_metrics["test"]
        print(
            f"{row['run_name']} | {row['checkpoint_type']} | epoch={row['selected_epoch']} | "
            f"validation CSS={validation['css']:.4f} | test CSS={test['css']:.4f} | "
            f"test TSS={test['tss']:.4f} | test HSS={test['hss']:.4f} | "
            f"test recall={test['recall']:.4f} | test FPR={test['false_positive_rate']:.4f}"
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved comparison: {args.output}")


if __name__ == "__main__":
    main()
