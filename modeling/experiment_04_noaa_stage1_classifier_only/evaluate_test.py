"""Evaluate the selected Experiment 4 checkpoint on the untouched test set."""

import argparse
import csv
import sys
from pathlib import Path

import torch
import wandb
from torch import nn


SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.append(str(SCRIPT_DIR.parent))

from config import IMAGE_DIR, LABEL_DIR, WANDB_ENTITY, WANDB_PROJECT
from dataloader import build_evaluation_loader
from model import build_model
from train import run_epoch


TEST_CSV = LABEL_DIR / "test_2025_2026_july.csv"


def find_wandb_run_id(checkpoint_path):
    """Find the original training run ID in its local W&B directory."""
    run_folders = sorted((checkpoint_path.parent / "wandb").glob("run-*-*"))
    if not run_folders:
        return None
    return run_folders[-1].name.rsplit("-", maxsplit=1)[-1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--wandb-run-id",
        help="Optional W&B run ID if it cannot be detected locally.",
    )
    parser.add_argument(
        "--skip-wandb",
        action="store_true",
        help="Save test metrics locally without updating W&B.",
    )
    args = parser.parse_args()

    checkpoint = torch.load(
        args.checkpoint,
        map_location="cpu",
        weights_only=True,
    )
    config = checkpoint["config"]

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    test_loader = build_evaluation_loader(
        TEST_CSV,
        IMAGE_DIR,
        batch_size=int(config["batch_size"]),
        num_workers=int(config["num_workers"]),
    )

    # Reconstruct the classifier-only architecture before loading its weights.
    model = build_model(
        pretrained=False,
        hidden_dim=int(config["classifier_hidden_dim"]),
        dropout=float(config["dropout"]),
        freeze_backbone=bool(config["freeze_backbone"]),
        trainable_backbone_blocks=int(config["trainable_backbone_blocks"]),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])

    # No balancing, augmentation, shuffling, or parameter updates are applied.
    test_metrics = run_epoch(
        model,
        test_loader,
        nn.CrossEntropyLoss(),
        device,
    )
    results = {
        "selected_epoch": checkpoint["epoch"],
        "test/samples": len(test_loader.dataset),
        **{f"test/{name}": value for name, value in test_metrics.items()},
    }

    metrics_path = args.checkpoint.parent / "test_metrics.csv"
    with metrics_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["metric", "value"])
        writer.writerows(results.items())

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
        run.define_metric("test/*")
        run.log(results)
        for name, value in results.items():
            run.summary[name] = value

        artifact = wandb.Artifact(
            f"noaa-stage1-experiment-04-test-{run.id}",
            type="evaluation",
            metadata={"test_csv": str(TEST_CSV)},
        )
        artifact.add_file(str(metrics_path))
        run.log_artifact(artifact)
        run.finish()

    print(f"Selected epoch: {checkpoint['epoch']}")
    print(f"Test samples: {len(test_loader.dataset):,}")
    print(
        "Test | "
        f"loss={test_metrics['loss']:.4f} | "
        f"CSS={test_metrics['css']:.4f} | "
        f"TSS={test_metrics['tss']:.4f} | "
        f"HSS={test_metrics['hss']:.4f} | "
        f"precision={test_metrics['precision']:.4f} | "
        f"recall={test_metrics['recall']:.4f} | "
        f"F1={test_metrics['f1']:.4f}"
    )
    print(
        "Test confusion matrix | "
        f"TP={test_metrics['tp']} | TN={test_metrics['tn']} | "
        f"FP={test_metrics['fp']} | FN={test_metrics['fn']}"
    )
    print(
        "Test rates | "
        f"FPR={test_metrics['false_positive_rate']:.4f} | "
        f"FNR={test_metrics['false_negative_rate']:.4f} | "
        f"predicted FL rate={test_metrics['predicted_fl_rate']:.4f}"
    )
    print(f"Saved: {metrics_path}")


if __name__ == "__main__":
    main()
