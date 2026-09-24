"""Evaluate a saved continual-learning checkpoint on future data."""

import argparse
import csv
from datetime import datetime
from pathlib import Path

import torch
import wandb
from torch import nn

from attention_model import Attn_Net
from dataloader import build_evaluation_loader
from evaluation import evaluate


PROJECT_ROOT = Path(__file__).resolve().parents[1]
IMAGE_DIRECTORY = PROJECT_ROOT / "downloaded_data" / "hmi_jpgs"

WANDB_ENTITY = "piyush-luitel-texas-christian-university"
WANDB_PROJECT = (
    "Full disk attention solar flare prediction "
    "with continual learning"
)


def parse_arguments():
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate a saved stage checkpoint without training the model."
        )
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Path to stage1.pt, stage2.pt, or stage3.pt.",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        required=True,
        help="Path to a validation or test label CSV.",
    )
    parser.add_argument(
        "--dataset-name",
        default=None,
        help="Metric prefix. Defaults to the CSV filename stem.",
    )
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument(
        "--wandb-mode",
        choices=["online", "offline", "disabled"],
        default="online",
    )
    return parser.parse_args()


def append_metrics(csv_path, row):
    """Append one evaluation result to a local CSV file."""
    write_header = not csv_path.exists()

    with csv_path.open("a", newline="") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=list(row),
        )

        if write_header:
            writer.writeheader()

        writer.writerow(row)


def main():
    args = parse_arguments()
    checkpoint_path = args.checkpoint.resolve()
    label_csv = args.csv.resolve()

    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Checkpoint was not found: {checkpoint_path}"
        )

    if not label_csv.is_file():
        raise FileNotFoundError(
            f"Label CSV was not found: {label_csv}"
        )

    dataset_name = args.dataset_name or label_csv.stem
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )

    training_config = checkpoint.get("config", {})
    image_size = training_config.get("image_size", 256)
    completed_stage = checkpoint.get("completed_stage", "unknown")

    model = Attn_Net(
        im_size=image_size,
        num_classes=2,
        attention=True,
        init="kaimingUniform",
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])

    evaluation_loader = build_evaluation_loader(
        csv_file=label_csv,
        image_directory=IMAGE_DIRECTORY,
        batch_size=args.batch_size,
        image_size=image_size,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    criterion = nn.CrossEntropyLoss()
    metrics = evaluate(
        model,
        evaluation_loader,
        criterion,
        device,
    )

    training_run_directory = checkpoint_path.parent.parent
    metrics_csv = training_run_directory / "future_evaluation_metrics.csv"
    evaluated_at = datetime.now().isoformat(timespec="seconds")

    result = {
        "evaluated_at": evaluated_at,
        "dataset": dataset_name,
        "completed_stage": completed_stage,
        "samples": len(evaluation_loader.dataset),
        "checkpoint": str(checkpoint_path),
        "label_csv": str(label_csv),
        "device": str(device),
        **metrics,
    }

    append_metrics(metrics_csv, result)

    run_name = (
        f"evaluate_stage{completed_stage}_"
        f"{dataset_name}_{datetime.now():%Y%m%d_%H%M%S}"
    )

    run = wandb.init(
        entity=WANDB_ENTITY,
        project=WANDB_PROJECT,
        name=run_name,
        group=training_run_directory.name,
        job_type="evaluation",
        tags=["EWC", "future-evaluation", dataset_name],
        mode=args.wandb_mode,
        config={
            "checkpoint": str(checkpoint_path),
            "completed_stage": completed_stage,
            "evaluation_dataset": dataset_name,
            "label_csv": str(label_csv),
            "samples": len(evaluation_loader.dataset),
            "image_size": image_size,
            "batch_size": args.batch_size,
            "normalization": training_config.get(
                "normalization",
                "0_to_1_scaling",
            ),
            "training_run": training_run_directory.name,
        },
    )

    run.log(
        {
            f"{dataset_name}/{name}": value
            for name, value in metrics.items()
        }
    )
    run.finish()

    print(f"Device: {device}")
    print(f"Checkpoint stage: {completed_stage}")
    print(f"Evaluation dataset: {dataset_name}")
    print(f"Samples: {len(evaluation_loader.dataset):,}")

    for name, value in metrics.items():
        if isinstance(value, float):
            print(f"{name}: {value:.6f}")
        else:
            print(f"{name}: {value}")

    print(f"Local metrics: {metrics_csv}")


if __name__ == "__main__":
    main()
