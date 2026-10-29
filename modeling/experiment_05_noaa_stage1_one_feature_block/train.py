"""Train Experiment 5 with one reopened feature block and ordinary loss."""

import csv
import random
import sys
import time
from datetime import datetime
from pathlib import Path

import torch
import wandb
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau


SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.append(str(SCRIPT_DIR.parent))

from config import (
    CONFIG,
    HOLDOUT_CSV,
    IMAGE_DIR,
    RESULTS_DIR,
    TRAIN_CSV,
    VALIDATION_CSV,
    WANDB_ENTITY,
    WANDB_PROJECT,
)
from dataloader import build_evaluation_loader, build_loaders
from evaluation import calculate_classification_metrics
from model import build_model


def add_rates(metrics):
    """Add prediction and error rates for imbalanced-data interpretation."""
    total = metrics["tp"] + metrics["tn"] + metrics["fp"] + metrics["fn"]
    actual_nf = metrics["tn"] + metrics["fp"]
    actual_fl = metrics["tp"] + metrics["fn"]
    metrics["predicted_fl_rate"] = (metrics["tp"] + metrics["fp"]) / total
    metrics["false_positive_rate"] = metrics["fp"] / actual_nf
    metrics["false_negative_rate"] = metrics["fn"] / actual_fl
    return metrics


def run_epoch(model, loader, criterion, device, optimizer=None):
    """Run one training or evaluation epoch and return classification metrics."""
    training = optimizer is not None
    model.train(training)

    # Do not update BatchNorm statistics inside frozen feature blocks.
    if training:
        for block in model.features[:model.frozen_feature_blocks]:
            block.eval()

    predictions, targets = [], []
    loss_sum = torch.zeros((), device=device)
    image_count = 0

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
                optimizer.step()

            batch_size = images.size(0)
            loss_sum += loss.detach() * batch_size
            image_count += batch_size
            predictions.append(scores.argmax(1).detach())
            targets.append(batch_targets.detach())

    predictions = torch.cat(predictions).cpu().tolist()
    targets = torch.cat(targets).cpu().tolist()
    metrics = calculate_classification_metrics(predictions, targets)
    metrics["loss"] = (loss_sum / image_count).item()
    return add_rates(metrics)


def save_checkpoint(path, model, optimizer, scheduler, epoch, metrics, config):
    """Save the best holdout epoch and its reproducibility information."""
    torch.save(
        {
            "completed_stage": 1,
            "architecture": "mobilenet_v3_large",
            "experiment": "one_feature_block_full_data_unweighted",
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "holdout_metrics": metrics,
            "config": config,
        },
        path,
    )


def write_epoch_metrics(path, rows):
    """Write all epoch-level metrics to a local CSV file."""
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def main():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = RESULTS_DIR / f"mobilenet_stage1_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    run = wandb.init(
        entity=WANDB_ENTITY,
        project=WANDB_PROJECT,
        name=f"noaa_stage1_experiment_05_one_feature_block_{timestamp}",
        config=CONFIG,
        dir=str(run_dir),
        tags=[
            "NOAA", "Stage-1", "MobileNetV3-Large", "experiment-05",
            "full-training-data", "ordinary-loss", "one-feature-block",
        ],
        job_type="stage1-controlled-experiment",
    )
    config = dict(run.config)

    seed = int(config["random_seed"])
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Use only the first visible GPU and intentionally avoid DataParallel.
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(0)

    loaders = build_loaders(
        TRAIN_CSV,
        HOLDOUT_CSV,
        IMAGE_DIR,
        batch_size=int(config["batch_size"]),
        num_workers=int(config["num_workers"]),
    )
    validation_loader = build_evaluation_loader(
        VALIDATION_CSV,
        IMAGE_DIR,
        batch_size=int(config["batch_size"]),
        num_workers=int(config["num_workers"]),
    )

    model = build_model(
        pretrained=bool(config["pretrained"]),
        hidden_dim=int(config["classifier_hidden_dim"]),
        dropout=float(config["dropout"]),
        freeze_backbone=bool(config["freeze_backbone"]),
        trainable_backbone_blocks=int(config["trainable_backbone_blocks"]),
    ).to(device)

    # As in Experiment 4, neither class receives additional loss weight.
    # The controlled change is reopening only the final feature block.
    criterion = nn.CrossEntropyLoss()

    optimizer = AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]),
    )
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=float(config["scheduler_factor"]),
        patience=int(config["scheduler_patience"]),
        threshold=float(config["early_stopping_min_delta"]),
        threshold_mode="abs",
        min_lr=float(config["min_learning_rate"]),
    )

    class_counts = loaders["class_counts"].tolist()
    system_config = {
        "device": str(device),
        "gpu_name": (
            torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU"
        ),
        "gpu_count_used": 1 if device.type == "cuda" else 0,
        "multi_gpu_method": "none",
        "train_samples_per_epoch": len(loaders["train"].dataset),
        "train_nf_samples": int(class_counts[0]),
        "train_fl_samples": int(class_counts[1]),
        "class_weighting": "none",
        "holdout_samples": len(loaders["holdout"].dataset),
        "validation_samples": len(validation_loader.dataset),
        "total_parameters": sum(p.numel() for p in model.parameters()),
        "trainable_parameters": sum(
            p.numel() for p in model.parameters() if p.requires_grad
        ),
        "train_csv": str(TRAIN_CSV),
        "holdout_csv": str(HOLDOUT_CSV),
        "validation_csv": str(VALIDATION_CSV),
        "pytorch_version": str(torch.__version__),
    }
    run.config.update(system_config, allow_val_change=True)
    config.update(system_config)

    run.define_metric("epoch")
    for group in ("train", "holdout"):
        run.define_metric(f"{group}/*", step_metric="epoch")
    run.define_metric("best_holdout/*")
    run.define_metric("validation/*")

    checkpoint_path = run_dir / "stage1_best.pt"
    metrics_path = run_dir / "epoch_metrics.csv"
    final_metrics_path = run_dir / "final_metrics.csv"
    best_css = float("-inf")
    best_epoch = 0
    epochs_without_improvement = 0
    rows = []

    print(f"Device: {device} | {system_config['gpu_name']}")
    print("Multi-GPU: disabled")
    print(f"Results: {run_dir}")
    print(
        "Training data: "
        f"NF={class_counts[0]:,}, FL={class_counts[1]:,}, "
        f"total={sum(class_counts):,}"
    )
    print("Loss: ordinary cross-entropy (no class weights)")

    for epoch in range(1, int(config["epochs"]) + 1):
        epoch_start = time.perf_counter()

        train_metrics = run_epoch(
            model, loaders["train"], criterion, device, optimizer
        )
        holdout_metrics = run_epoch(
            model, loaders["holdout"], criterion, device
        )

        learning_rate = optimizer.param_groups[0]["lr"]
        scheduler.step(holdout_metrics["css"])

        improved = holdout_metrics["css"] > (
            best_css + float(config["early_stopping_min_delta"])
        )
        if improved:
            best_css = holdout_metrics["css"]
            best_epoch = epoch
            epochs_without_improvement = 0
            save_checkpoint(
                checkpoint_path,
                model,
                optimizer,
                scheduler,
                epoch,
                holdout_metrics,
                config,
            )
        else:
            epochs_without_improvement += 1

        epoch_seconds = time.perf_counter() - epoch_start
        logged = {
            "epoch": epoch,
            "learning_rate": learning_rate,
            "best_holdout_css": best_css,
            "best_epoch": best_epoch,
            "epochs_without_improvement": epochs_without_improvement,
            "timing/epoch_seconds": epoch_seconds,
            **{f"train/{key}": value for key, value in train_metrics.items()},
            **{f"holdout/{key}": value for key, value in holdout_metrics.items()},
        }
        run.log(logged)
        rows.append({key.replace("/", "_"): value for key, value in logged.items()})
        write_epoch_metrics(metrics_path, rows)

        print(f"Epoch {epoch:02d}/{int(config['epochs'])}")
        print(
            "  Train   | "
            f"loss={train_metrics['loss']:.4f} | "
            f"CSS={train_metrics['css']:.4f} | "
            f"TSS={train_metrics['tss']:.4f} | "
            f"HSS={train_metrics['hss']:.4f}"
        )
        print(
            "  Holdout | "
            f"loss={holdout_metrics['loss']:.4f} | "
            f"CSS={holdout_metrics['css']:.4f} | "
            f"TSS={holdout_metrics['tss']:.4f} | "
            f"HSS={holdout_metrics['hss']:.4f}"
        )
        print(
            "  Progress | "
            f"best epoch={best_epoch} | best holdout CSS={best_css:.4f} | "
            f"LR={learning_rate:.2e} | time={epoch_seconds:.1f}s"
        )

        if epochs_without_improvement >= int(config["early_stopping_patience"]):
            print(f"Early stopping at epoch {epoch}; best epoch was {best_epoch}.")
            break

    # Restore the selected checkpoint before evaluating the future validation set.
    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=True,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    final_holdout = run_epoch(model, loaders["holdout"], criterion, device)
    validation = run_epoch(model, validation_loader, criterion, device)

    final_results = {
        "selected_epoch": best_epoch,
        **{f"best_holdout/{key}": value for key, value in final_holdout.items()},
        **{f"validation/{key}": value for key, value in validation.items()},
    }
    run.log(final_results)
    for name, value in final_results.items():
        run.summary[name] = value

    with final_metrics_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["metric", "value"])
        writer.writerows(final_results.items())

    artifact = wandb.Artifact(
        f"noaa-stage1-experiment-05-{run.id}", type="model", metadata=config
    )
    artifact.add_file(str(checkpoint_path))
    artifact.add_file(str(metrics_path))
    artifact.add_file(str(final_metrics_path))
    run.log_artifact(artifact)

    print(f"Best epoch: {best_epoch}")
    print(
        "Selected holdout | "
        f"loss={final_holdout['loss']:.4f} | "
        f"CSS={final_holdout['css']:.4f} | "
        f"TSS={final_holdout['tss']:.4f} | "
        f"HSS={final_holdout['hss']:.4f} | "
        f"precision={final_holdout['precision']:.4f} | "
        f"recall={final_holdout['recall']:.4f} | "
        f"F1={final_holdout['f1']:.4f}"
    )
    print(
        "Validation | "
        f"loss={validation['loss']:.4f} | "
        f"CSS={validation['css']:.4f} | "
        f"TSS={validation['tss']:.4f} | "
        f"HSS={validation['hss']:.4f} | "
        f"precision={validation['precision']:.4f} | "
        f"recall={validation['recall']:.4f} | "
        f"F1={validation['f1']:.4f}"
    )
    print(f"Checkpoint: {checkpoint_path}")
    run.finish()


if __name__ == "__main__":
    main()


