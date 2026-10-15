"""Train and validate MobileNetV3-Large on SciQuality Stage 1."""

import csv
import random
import sys
import time
from datetime import datetime
from pathlib import Path

import torch
import wandb
from torch import nn
from torch.optim import AdamW, SGD
from torch.optim.lr_scheduler import ReduceLROnPlateau


SCRIPT_DIR = Path(__file__).resolve().parent
MODELING_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.append(str(MODELING_DIR))

from evaluation import calculate_classification_metrics  
from dataloader import build_evaluation_loader, build_train_loader 
from model import build_model  


ROOT = Path(__file__).resolve().parents[2]
STAGE_DIR = ROOT / "data_labeling/data_labels/sciquality/continual_stages_2010_2024"
FUTURE_DIR = ROOT / "data_labeling/data_labels/sciquality/future_evaluation_2024_2026"
IMAGE_DIR = ROOT / "downloaded_data/hmi_jpgs"
RESULTS_DIR = ROOT / "results/sciquality_stage1"

TRAIN_CSV = STAGE_DIR / "Stage1_train.csv"
HOLDOUT_CSV = STAGE_DIR / "Stage1_holdout.csv"
VALIDATION_CSV = FUTURE_DIR / "validation_2024_2025.csv"

WANDB_ENTITY = "piyush-luitel-texas-christian-university"
WANDB_PROJECT = "Full disk attention solar flare prediction with continual learning"


DEFAULT_CONFIG = {
    "model": "mobilenet_v3_large",
    "pretrained": True,
    "image_size": 512,
    "epochs": 200,
    "batch_size": 16,
    "learning_rate": 0.0001,
    "weight_decay": 0.0001,
    "optimizer": "adamw",
    "momentum": 0.9,
    "classifier_hidden_dim": 1280,
    "dropout": 0.2,
    "early_stopping_patience": 35,
    "early_stopping_min_delta": 0.001,
    "scheduler_patience": 8,
    "scheduler_factor": 0.3,
    "min_learning_rate": 0.0000001,
    "gradient_clip_norm": 5.0,
    "use_vertical_flip": False,
    "use_rotation_180": False,
    "num_workers": 8,
    "random_seed": 42,
}


def run_epoch(model, loader, criterion, device, optimizer=None, clip_norm=None):
    """Train or evaluate for one epoch and return all binary metrics."""
    training = optimizer is not None
    model.train(training)
    predictions, targets = [], []
    total_loss = total_images = 0

    with torch.set_grad_enabled(training):
        for images, batch_targets in loader:
            images = images.to(device, non_blocking=True)
            batch_targets = batch_targets.to(device, non_blocking=True)

            if training:
                optimizer.zero_grad(set_to_none=True)

            scores = model(images)
            loss = criterion(scores, batch_targets)

            if training:
                loss.backward()
                if clip_norm:
                    nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
                optimizer.step()

            batch_size = images.size(0)
            total_loss += loss.item() * batch_size
            total_images += batch_size
            predictions.extend(scores.argmax(1).detach().cpu().tolist())
            targets.extend(batch_targets.detach().cpu().tolist())

    metrics = calculate_classification_metrics(predictions, targets)
    metrics["loss"] = total_loss / total_images
    return metrics


def save_checkpoint(path, model, optimizer, config, epoch, holdout_metrics):
    """Save the best Stage 1 state and everything needed to reproduce it."""
    saved_model = model.module if isinstance(model, nn.DataParallel) else model
    torch.save(
        {
            "completed_stage": 1,
            "architecture": "mobilenet_v3_large",
            "epoch": epoch,
            "model_state_dict": saved_model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": config,
            "holdout_metrics": holdout_metrics,
        },
        path,
    )


def main():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = RESULTS_DIR / f"mobilenet_stage1_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    run = wandb.init(
        entity=WANDB_ENTITY,
        project=WANDB_PROJECT,
        name=f"sciquality_stage1_mobilenet_{timestamp}",
        config=DEFAULT_CONFIG,
        dir=str(run_dir),
        tags=["SciQuality", "Stage-1", "MobileNetV3-Large"],
        job_type="stage1-hyperparameter-tuning",
    )
    config = dict(run.config)

    # Reproducibility for every sweep trial.
    seed = int(config["random_seed"])
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpu_names = [
        torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())
    ]

    augmentations = ["original", "horizontal_flip", "small_rotation"]
    if config["use_vertical_flip"]:
        augmentations.append("vertical_flip")
    if config["use_rotation_180"]:
        augmentations.append("rotation_180")
    run.config.update({"augmentations": augmentations}, allow_val_change=True)
    config["augmentations"] = augmentations

    batch_size = int(config["batch_size"])
    workers = int(config["num_workers"])
    train_loader = build_train_loader(
        TRAIN_CSV, IMAGE_DIR, augmentations, batch_size, workers
    )
    holdout_loader = build_evaluation_loader(
        HOLDOUT_CSV, IMAGE_DIR, batch_size, workers
    )
    validation_loader = build_evaluation_loader(
        VALIDATION_CSV, IMAGE_DIR, batch_size, workers
    )

    model = build_model(
        pretrained=bool(config["pretrained"]),
        classifier_hidden_dim=int(config["classifier_hidden_dim"]),
        dropout=float(config["dropout"]),
    ).to(device)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    model_for_count = model.module if isinstance(model, nn.DataParallel) else model

    if config["optimizer"] == "adamw":
        optimizer = AdamW(
            model.parameters(),
            lr=float(config["learning_rate"]),
            weight_decay=float(config["weight_decay"]),
        )
    elif config["optimizer"] == "sgd":
        optimizer = SGD(
            model.parameters(),
            lr=float(config["learning_rate"]),
            weight_decay=float(config["weight_decay"]),
            momentum=float(config["momentum"]),
        )
    else:
        raise ValueError(f"Unknown optimizer: {config['optimizer']}")

    criterion = nn.CrossEntropyLoss()
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=float(config["scheduler_factor"]),
        patience=int(config["scheduler_patience"]),
        threshold=float(config["early_stopping_min_delta"]),
        threshold_mode="abs",
        min_lr=float(config["min_learning_rate"]),
    )

    system_config = {
        "train_csv": str(TRAIN_CSV),
        "holdout_csv": str(HOLDOUT_CSV),
        "validation_csv": str(VALIDATION_CSV),
        "train_samples_per_epoch": len(train_loader.dataset),
        "holdout_samples": len(holdout_loader.dataset),
        "validation_samples": len(validation_loader.dataset),
        "total_parameters": sum(p.numel() for p in model_for_count.parameters()),
        "trainable_parameters": sum(
            p.numel() for p in model_for_count.parameters() if p.requires_grad
        ),
        "device": str(device),
        "gpu_names": gpu_names,
        "gpu_count": torch.cuda.device_count(),
        "multi_gpu_method": (
            "DataParallel" if torch.cuda.device_count() > 1 else "single_device"
        ),
        "pytorch_version": torch.__version__,
    }
    run.config.update(system_config, allow_val_change=True)
    config.update(system_config)

    run.define_metric("epoch")
    run.define_metric("training/*", step_metric="epoch")
    run.define_metric("holdout/*", step_metric="epoch")

    checkpoint = run_dir / "stage1_best.pt"
    metrics_csv = run_dir / "epoch_metrics.csv"
    best_css = float("-inf")
    best_epoch = epochs_without_improvement = 0
    best_holdout_metrics = None
    patience = int(config["early_stopping_patience"])
    min_delta = float(config["early_stopping_min_delta"])
    fieldnames = None
    training_start = time.perf_counter()
    stopped_early = False

    print(f"Device: {device} | GPUs: {gpu_names or ['CPU']}")
    print(f"Augmentations: {augmentations}")
    print(f"Results: {run_dir}")

    for epoch in range(1, int(config["epochs"]) + 1):
        start = time.perf_counter()
        train_metrics = run_epoch(
            model,
            train_loader,
            criterion,
            device,
            optimizer,
            float(config["gradient_clip_norm"]),
        )
        holdout_metrics = run_epoch(
            model, holdout_loader, criterion, device
        )
        learning_rate = optimizer.param_groups[0]["lr"]
        scheduler.step(holdout_metrics["css"])

        values = {
            "epoch": epoch,
            **{f"training/{key}": value for key, value in train_metrics.items()},
            **{f"holdout/{key}": value for key, value in holdout_metrics.items()},
            "system/learning_rate": learning_rate,
            "system/epoch_seconds": time.perf_counter() - start,
        }
        run.log(values)

        # Keep a local epoch-by-epoch record in addition to W&B.
        if fieldnames is None:
            fieldnames = list(values)
        with metrics_csv.open("a", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            if file.tell() == 0:
                writer.writeheader()
            writer.writerow(values)

        print(
            f"Epoch {epoch}/{config['epochs']} | "
            f"train loss={train_metrics['loss']:.4f} | "
            f"holdout CSS={holdout_metrics['css']:.4f} | "
            f"TSS={holdout_metrics['tss']:.4f} | "
            f"HSS={holdout_metrics['hss']:.4f}"
        )

        if holdout_metrics["css"] > best_css + min_delta:
            best_css = holdout_metrics["css"]
            best_epoch = epoch
            best_holdout_metrics = holdout_metrics
            epochs_without_improvement = 0
            save_checkpoint(
                checkpoint, model, optimizer, config, epoch, holdout_metrics
            )
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= patience:
            stopped_early = True
            print(f"Early stopping at epoch {epoch}; best epoch was {best_epoch}.")
            break

    # Evaluate validation only after restoring the best holdout checkpoint.
    saved = torch.load(checkpoint, map_location=device, weights_only=False)
    restored_model = model.module if isinstance(model, nn.DataParallel) else model
    restored_model.load_state_dict(saved["model_state_dict"])
    validation_metrics = run_epoch(model, validation_loader, criterion, device)

    final_values = {
        "selection/best_epoch": best_epoch,
        "selection/best_holdout_css": best_css,
        "selection/stopping_epoch": epoch,
        "selection/stopped_early": stopped_early,
        "system/total_training_seconds": time.perf_counter() - training_start,
        **{
            f"best_holdout/{key}": value
            for key, value in best_holdout_metrics.items()
        },
        **{f"validation/{key}": value for key, value in validation_metrics.items()},
    }
    run.log(final_values)
    for key, value in final_values.items():
        run.summary[key] = value

    confusion_table = wandb.Table(
        columns=["Outcome", "Count"],
        data=[
            ["True positive", validation_metrics["tp"]],
            ["True negative", validation_metrics["tn"]],
            ["False positive", validation_metrics["fp"]],
            ["False negative", validation_metrics["fn"]],
        ],
    )
    run.log({
        "validation/confusion_counts": wandb.plot.bar(
            confusion_table, "Outcome", "Count", title="Stage 1 validation"
        )
    })

    artifact = wandb.Artifact(
        f"sciquality-stage1-{run.id}",
        type="model",
        metadata={"best_epoch": best_epoch, "validation_css": validation_metrics["css"]},
    )
    artifact.add_file(str(checkpoint))
    run.log_artifact(artifact)

    print(f"Best checkpoint: {checkpoint}")
    print(f"Validation CSS: {validation_metrics['css']:.4f}")
    run.finish()


if __name__ == "__main__":
    main()
