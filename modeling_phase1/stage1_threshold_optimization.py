"""Optimize a Stage 1 decision threshold on holdout CSS without retraining."""

import argparse
import csv
import json
import math
import sys
from datetime import datetime
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
EXP9 = ROOT / "modeling/experiment_09_noaa_stage1_bayesian_tuning"
sys.path[:0] = [str(EXP9), str(ROOT / "modeling"), str(ROOT)]

from dataloader import build_evaluation_loader  # noqa: E402
from model import build_model  # noqa: E402


LABEL_DIR = ROOT / "data_labeling/data_labels/noaa/CL_Labels"
DEFAULT_CHECKPOINT = (
    ROOT
    / "results/noaa_stage1_seed_confirmation"
    / "mobilenet_stage1_20261106_223444/stage1_best.pt"
)
DEFAULT_IMAGE_DIR = ROOT / "downloaded_data/hmi_jpgs"
DEFAULT_OUTPUT_DIR = ROOT / "results/noaa_stage1_threshold_optimization"
SPLITS = {
    "train": LABEL_DIR / "Stage1_train.csv",
    "holdout": LABEL_DIR / "Stage1_holdout.csv",
    "validation": LABEL_DIR / "validation_2024_2025.csv",
    "test": LABEL_DIR / "test_2025_2026_july.csv",
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--image-dir", type=Path, default=DEFAULT_IMAGE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--threshold-step", type=float, default=0.001)
    return parser.parse_args()


def metrics_at_threshold(probabilities, targets, threshold):
    predictions = probabilities >= threshold
    positives = targets == 1
    negatives = ~positives
    tp = int((predictions & positives).sum())
    tn = int((~predictions & negatives).sum())
    fp = int((predictions & negatives).sum())
    fn = int((~predictions & positives).sum())
    total = tp + tn + fp + fn
    recall = tp / (tp + fn) if tp + fn else 0.0
    precision = tp / (tp + fp) if tp + fp else 0.0
    fpr = fp / (fp + tn) if fp + tn else 0.0
    tss = recall - fpr
    hss_denominator = (
        (tp + fn) * (fn + tn) + (tp + fp) * (fp + tn)
    )
    hss = 2 * (tp * tn - fn * fp) / hss_denominator if hss_denominator else 0.0
    css = math.sqrt(tss * hss) if tss > 0 and hss > 0 else 0.0
    return {
        "threshold": float(threshold),
        "accuracy": (tp + tn) / total if total else 0.0,
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        "false_positive_rate": fpr,
        "tss": tss,
        "hss": hss,
        "css": css,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "predicted_fl_rate": (tp + fp) / total if total else 0.0,
    }


@torch.inference_mode()
def predict(model, loader, device):
    model.eval()
    probabilities, targets = [], []
    for images, labels in loader:
        logits = model(images.to(device, non_blocking=True))
        probabilities.append(logits.softmax(dim=1)[:, 1].cpu())
        targets.append(labels.cpu())
    return torch.cat(probabilities), torch.cat(targets).bool()


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()
    if not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)
    if not 0 < args.threshold_step <= 1:
        raise ValueError("threshold-step must be in (0, 1]")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    config = checkpoint.get("config", {})
    model = build_model(
        pretrained=False,
        hidden_dim=int(config.get("classifier_hidden_dim", 1280)),
        dropout=float(config.get("dropout", 0.0)),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = args.output_dir / f"threshold_optimization_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=False)
    predictions = {}

    print(f"Device: {device}")
    print(f"Checkpoint: {args.checkpoint}")
    print("Preprocessing: original Experiment 9 ImageNet RGB normalization")
    for split, csv_path in SPLITS.items():
        loader = build_evaluation_loader(
            csv_path,
            args.image_dir,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )
        probabilities, targets = predict(model, loader, device)
        predictions[split] = (probabilities, targets)
        paths = [path.relative_to(args.image_dir).as_posix() for path in loader.dataset.paths]
        write_csv(
            run_dir / f"{split}_probabilities.csv",
            [
                {"label": path, "target": int(target), "flare_probability": float(probability)}
                for path, target, probability in zip(paths, targets, probabilities)
            ],
        )
        print(f"Predicted {split}: {len(targets):,} images")

    holdout_probabilities, holdout_targets = predictions["holdout"]
    threshold_count = round(1 / args.threshold_step)
    thresholds = [index * args.threshold_step for index in range(threshold_count + 1)]
    curve = [
        metrics_at_threshold(holdout_probabilities, holdout_targets, threshold)
        for threshold in thresholds
    ]
    best = max(curve, key=lambda row: (row["css"], row["tss"], -row["false_positive_rate"]))
    write_csv(run_dir / "holdout_threshold_curve.csv", curve)

    evaluation_rows = []
    for split, (probabilities, targets) in predictions.items():
        for name, threshold in (("default", 0.5), ("holdout_optimized", best["threshold"])):
            row = metrics_at_threshold(probabilities, targets, threshold)
            evaluation_rows.append({"split": split, "threshold_source": name, **row})
    write_csv(run_dir / "threshold_evaluation.csv", evaluation_rows)

    summary = {
        "checkpoint": str(args.checkpoint),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "selection_dataset": "Stage1_holdout",
        "selection_metric": "css",
        "selected_threshold": best["threshold"],
        "selected_holdout_metrics": best,
        "preprocessing": "experiment_09_imagenet_rgb",
    }
    with (run_dir / "summary.json").open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)
        file.write("\n")

    print(f"\nSelected threshold: {best['threshold']:.3f}")
    print(
        f"Holdout at selected threshold | CSS={best['css']:.4f} | "
        f"TSS={best['tss']:.4f} | HSS={best['hss']:.4f} | "
        f"recall={best['recall']:.4f} | FPR={best['false_positive_rate']:.4f}"
    )
    for row in evaluation_rows:
        if row["threshold_source"] == "holdout_optimized" and row["split"] != "holdout":
            print(
                f"{row['split'].title()} | CSS={row['css']:.4f} | "
                f"TSS={row['tss']:.4f} | HSS={row['hss']:.4f} | "
                f"recall={row['recall']:.4f} | FPR={row['false_positive_rate']:.4f}"
            )
    print(f"Results: {run_dir}")


if __name__ == "__main__":
    main()
