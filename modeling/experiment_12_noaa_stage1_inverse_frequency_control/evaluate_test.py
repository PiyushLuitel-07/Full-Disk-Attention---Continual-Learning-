"""Evaluate the Experiment 12 inverse-frequency checkpoint on the untouched test set."""

import argparse
import csv
import sys
from pathlib import Path

import torch
import wandb

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(SCRIPT_DIR.parent))

from config import CONFIG, TEST_CSV, IMAGE_DIR, WANDB_ENTITY, WANDB_PROJECT, WANDB_TAGS  # noqa: E402
from dataloader import build_evaluation_loader  # noqa: E402
from evaluation import calculate_classification_metrics  # noqa: E402
from model import build_model  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--wandb-run-id",
        default=None,
        help="Optional original training run ID to resume when logging test metrics.",
    )
    parser.add_argument("--skip-wandb", action="store_true")
    return parser.parse_args()


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_samples = 0
    predictions, targets = [], []

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        logits = model(images)
        loss = criterion(logits, labels)

        total_loss += loss.item() * labels.size(0)
        total_samples += labels.size(0)
        predictions.extend(logits.argmax(dim=1).cpu().tolist())
        targets.extend(labels.cpu().tolist())

    metrics = calculate_classification_metrics(predictions, targets)
    metrics["loss"] = total_loss / total_samples
    total = metrics["tp"] + metrics["tn"] + metrics["fp"] + metrics["fn"]
    metrics["predicted_fl_rate"] = (metrics["tp"] + metrics["fp"]) / total
    metrics["false_positive_rate"] = metrics["fp"] / (metrics["fp"] + metrics["tn"])
    metrics["false_negative_rate"] = metrics["fn"] / (metrics["fn"] + metrics["tp"])
    return metrics


def main():
    args = parse_args()
    if not args.checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)

    model = build_model(
        num_classes=2,
        pretrained=False,
        hidden_dim=CONFIG["classifier_hidden_dim"],
        dropout=CONFIG["dropout"],
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])

    class_weights = torch.tensor(
        [CONFIG["nf_class_weight"], CONFIG["fl_class_weight"]],
        dtype=torch.float32,
        device=device,
    )
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
    loader = build_evaluation_loader(
        TEST_CSV,
        IMAGE_DIR,
        batch_size=int(CONFIG["batch_size"]),
        num_workers=int(CONFIG["num_workers"]),
    )
    metrics = evaluate(model, loader, criterion, device)

    output_dir = args.checkpoint.parent
    output_path = output_dir / "test_metrics.csv"
    with output_path.open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["metric", "value"])
        writer.writerow(["selected_epoch", checkpoint.get("epoch")])
        for key, value in metrics.items():
            writer.writerow([f"test/{key}", value])

    if not args.skip_wandb:
        wandb.init(
            entity=WANDB_ENTITY,
            project=WANDB_PROJECT,
            id=args.wandb_run_id,
            resume="allow" if args.wandb_run_id else None,
            name=f"{CONFIG['experiment']}_test",
            tags=WANDB_TAGS + ["test-evaluation"],
            config=CONFIG,
        )
        wandb.log({"selected_epoch": checkpoint.get("epoch"), **{f"test/{k}": v for k, v in metrics.items()}})
        wandb.save(str(output_path), policy="now")
        wandb.finish()

    print(f"Selected epoch: {checkpoint.get('epoch')}")
    print(f"Test samples: {len(loader.dataset):,}")
    print(
        "Test | "
        f"loss={metrics['loss']:.4f} | CSS={metrics['css']:.4f} | "
        f"TSS={metrics['tss']:.4f} | HSS={metrics['hss']:.4f} | "
        f"precision={metrics['precision']:.4f} | recall={metrics['recall']:.4f} | "
        f"F1={metrics['f1']:.4f}"
    )
    print(
        "Test confusion matrix | "
        f"TP={metrics['tp']} | TN={metrics['tn']} | FP={metrics['fp']} | FN={metrics['fn']}"
    )
    print(
        "Test rates | "
        f"FPR={metrics['false_positive_rate']:.4f} | "
        f"FNR={metrics['false_negative_rate']:.4f} | "
        f"predicted FL rate={metrics['predicted_fl_rate']:.4f}"
    )
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
