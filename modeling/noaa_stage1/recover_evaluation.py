"""Finish evaluation after the trusted NOAA Stage 1 run failed to reload."""

import argparse
import sys
from pathlib import Path

import torch
import wandb
from torch import nn


SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from dataloader import build_evaluation_loader
from model import build_model
from train import (
    HOLDOUT_CSV,
    IMAGE_DIR,
    VALIDATION_CSV,
    WANDB_ENTITY,
    WANDB_PROJECT,
    run_epoch,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--wandb-run-id", required=True)
    args = parser.parse_args()

    # This checkpoint was created locally by this project, so it is trusted.
    checkpoint = torch.load(
        args.checkpoint,
        map_location="cpu",
        weights_only=False,
    )
    config = checkpoint["config"]

    # Use only the first GPU made visible by CUDA_VISIBLE_DEVICES.
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    batch_size = int(config["batch_size"])
    workers = int(config["num_workers"])

    holdout_loader = build_evaluation_loader(
        HOLDOUT_CSV, IMAGE_DIR, batch_size, workers
    )
    validation_loader = build_evaluation_loader(
        VALIDATION_CSV, IMAGE_DIR, batch_size, workers
    )

    model = build_model(
        pretrained=False,
        freeze_backbone=bool(config.get("freeze_backbone", False)),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    criterion = nn.CrossEntropyLoss()

    holdout = run_epoch(model, holdout_loader, criterion, device)
    validation = run_epoch(model, validation_loader, criterion, device)
    results = {
        "selected_epoch": checkpoint["epoch"],
        **{f"best_holdout/{key}": value for key, value in holdout.items()},
        **{f"validation/{key}": value for key, value in validation.items()},
    }

    # Resume the interrupted run and add its missing final results.
    run = wandb.init(
        entity=WANDB_ENTITY,
        project=WANDB_PROJECT,
        id=args.wandb_run_id,
        resume="must",
    )
    run.log(results)
    for name, value in results.items():
        run.summary[name] = value

    artifact = wandb.Artifact(
        f"noaa-stage1-{run.id}",
        type="model",
        metadata={key: str(value) for key, value in config.items()},
    )
    artifact.add_file(str(args.checkpoint))
    metrics_path = args.checkpoint.parent / "epoch_metrics.csv"
    artifact.add_file(str(metrics_path))
    run.log_artifact(artifact)
    run.finish()

    print(f"Selected epoch: {checkpoint['epoch']}")
    print(f"Holdout CSS: {holdout['css']:.4f}")
    print(f"Validation CSS: {validation['css']:.4f}")


if __name__ == "__main__":
    main()
