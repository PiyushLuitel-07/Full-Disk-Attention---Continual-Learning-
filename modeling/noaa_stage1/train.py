"""Run one controlled NOAA Stage 1 MobileNetV3-Large experiment."""

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

from dataloader import build_evaluation_loader, build_loaders
from evaluation import calculate_classification_metrics
from model import build_model


ROOT = Path(__file__).resolve().parents[2]
LABEL_DIR = ROOT / "data_labeling/data_labels/noaa/CL_Labels"
IMAGE_DIR = ROOT / "downloaded_data/hmi_jpgs"
RESULTS_DIR = ROOT / "results/noaa_stage1"

TRAIN_CSV = LABEL_DIR / "Stage1_train.csv"
HOLDOUT_CSV = LABEL_DIR / "Stage1_holdout.csv"
VALIDATION_CSV = LABEL_DIR / "validation_2024_2025.csv"

WANDB_ENTITY = "piyush-luitel-texas-christian-university"
WANDB_PROJECT = "Full disk attention solar flare prediction with continual learning"


# Fixed diagnostic configuration; 
CONFIG = {
    "model": "mobilenet_v3_large",
    "pretrained": True,
    "image_size": 512,
    "batch_size": 128,
    "epochs": 40,
    "learning_rate": 1e-4,
    "weight_decay": 1e-2,
    "optimizer": "adamw",
    "dropout": 0.0,
    "augmentation": "none",
    "early_stopping_patience": 10,
    "early_stopping_min_delta": 0.001,
    "scheduler_patience": 3,
    "scheduler_factor": 0.3,
    "min_learning_rate": 1e-7,
    "num_workers": 8,
    "random_seed": 42,
}


def add_rates(metrics):
    """Add rates that make class imbalance and false alarms visible."""
    # These rates are easier to interpret than accuracy on imbalanced data.
    total = metrics["tp"] + metrics["tn"] + metrics["fp"] + metrics["fn"]
    actual_nf = metrics["tn"] + metrics["fp"]
    actual_fl = metrics["tp"] + metrics["fn"]
    metrics["predicted_fl_rate"] = (metrics["tp"] + metrics["fp"]) / total
    metrics["false_positive_rate"] = metrics["fp"] / actual_nf
    metrics["false_negative_rate"] = metrics["fn"] / actual_fl
    return metrics


def run_epoch(model, loader, criterion, device, optimizer=None):
    """Train or evaluate once and return loss plus classification metrics."""
    # Supplying an optimizer enables training; otherwise this is evaluation.
    training = optimizer is not None
    model.train(training)
    predictions, targets = [], []
    loss_sum = torch.zeros((), device=device)
    image_count = 0

    with torch.set_grad_enabled(training):
        for images, batch_targets in loader:
            # Pinned memory and non_blocking copies help feed the GPU faster.
            images = images.to(device, non_blocking=True)
            batch_targets = batch_targets.to(device, non_blocking=True)

            if training:
                optimizer.zero_grad(set_to_none=True)

            scores = model(images)
            loss = criterion(scores, batch_targets)

            if training:
                loss.backward()
                optimizer.step()

            # Keep predictions on the GPU until the epoch finishes to avoid
            # synchronizing the CPU and GPU after every batch.
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
    """Save the best epoch and everything needed to reproduce it."""
    torch.save(
        {
            "completed_stage": 1,
            "architecture": "mobilenet_v3_large",
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "holdout_metrics": metrics,
            "config": config,
        },
        path,
    )


def main():
    # Give every run its own folder for checkpoints and local metric files.
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = RESULTS_DIR / f"mobilenet_stage1_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    run = wandb.init(
        entity=WANDB_ENTITY,
        project=WANDB_PROJECT,
        name=f"noaa_stage1_diagnostic_{timestamp}",
        config=CONFIG,
        dir=str(run_dir),
        tags=["NOAA", "Stage-1", "MobileNetV3-Large", "diagnostic"],
        job_type="stage1-diagnostic",
    )
    config = dict(run.config)

    # Make the fixed experiment reproducible.
    seed = int(config["random_seed"])
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Use only the first visible GPU; DataParallel is intentionally not used.
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(0)

    loaders = build_loaders(
        TRAIN_CSV,
        HOLDOUT_CSV,
        IMAGE_DIR,
        batch_size=int(config["batch_size"]),
        num_workers=int(config["num_workers"]),
        seed=seed,
    )
    validation_loader = build_evaluation_loader(
        VALIDATION_CSV,
        IMAGE_DIR,
        batch_size=int(config["batch_size"]),
        num_workers=int(config["num_workers"]),
    )

    # Fine-tune every layer of the ImageNet-pretrained model.
    model = build_model(pretrained=bool(config["pretrained"])).to(device)
    criterion = nn.CrossEntropyLoss()

    # AdamW applies weight decay separately from the adaptive gradient update.
    optimizer = AdamW(
        model.parameters(),
        lr=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]),
    )
    # Reduce the learning rate when holdout CSS stops improving.
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=float(config["scheduler_factor"]),
        patience=int(config["scheduler_patience"]),
        threshold=float(config["early_stopping_min_delta"]),
        threshold_mode="abs",
        min_lr=float(config["min_learning_rate"]),
    )

    sampler = loaders["balanced_sampler"]
    # Save data, hardware and model details with the W&B run.
    system_config = {
        "device": str(device),
        "gpu_name": (
            torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU"
        ),
        "gpu_count_used": 1 if device.type == "cuda" else 0,
        "multi_gpu_method": "none",
        "train_original_samples": len(loaders["natural_train"].dataset),
        "train_samples_per_epoch": len(loaders["balanced_train"].sampler),
        "train_fl_per_epoch": len(sampler.fl_indices),
        "train_nf_per_epoch": len(sampler.fl_indices),
        "holdout_samples": len(loaders["holdout"].dataset),
        "validation_samples": len(validation_loader.dataset),
        "total_parameters": sum(p.numel() for p in model.parameters()),
        "trainable_parameters": sum(
            p.numel() for p in model.parameters() if p.requires_grad
        ),
        "train_csv": str(TRAIN_CSV),
        "holdout_csv": str(HOLDOUT_CSV),
        "validation_csv": str(VALIDATION_CSV),
        "pytorch_version": torch.__version__,
    }
    run.config.update(system_config, allow_val_change=True)
    config.update(system_config)

    run.define_metric("epoch")
    for group in ("balanced_train", "natural_train", "holdout"):
        run.define_metric(f"{group}/*", step_metric="epoch")

    # The holdout CSS determines which epoch is saved as the selected model.
    checkpoint_path = run_dir / "stage1_best.pt"
    metrics_path = run_dir / "epoch_metrics.csv"
    best_css = float("-inf")
    best_epoch = 0
    epochs_without_improvement = 0
    cumulative_nf = set()
    rows = []

    print(f"Device: {device} | {system_config['gpu_name']}")
    print("Multi-GPU: disabled")
    print(f"Results: {run_dir}")

    for epoch in range(1, int(config["epochs"]) + 1):
        epoch_start = time.perf_counter()

        start = time.perf_counter()
        balanced_metrics = run_epoch(
            model, loaders["balanced_train"], criterion, device, optimizer
        )
        balanced_seconds = time.perf_counter() - start

        # Evaluate the same model on the complete natural training distribution.
        start = time.perf_counter()
        natural_metrics = run_epoch(
            model, loaders["natural_train"], criterion, device
        )
        natural_seconds = time.perf_counter() - start

        # The untouched holdout controls scheduling, checkpointing and stopping.
        start = time.perf_counter()
        holdout_metrics = run_epoch(model, loaders["holdout"], criterion, device)
        holdout_seconds = time.perf_counter() - start

        learning_rate = optimizer.param_groups[0]["lr"]
        scheduler.step(holdout_metrics["css"])
        cumulative_nf.update(sampler.last_nf_indices)

        # A change smaller than min_delta does not count as improvement.
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
        # Record how much of the original NF pool has been encountered.
        sampling_metrics = {
            "nf_cycle": sampler.cycle,
            "nf_used_this_epoch": len(sampler.last_nf_indices),
            "nf_unique_this_epoch": len(set(sampler.last_nf_indices)),
            "nf_cumulative_unique": len(cumulative_nf),
            "nf_cumulative_coverage": len(cumulative_nf) / len(sampler.nf_indices),
        }
        timing_metrics = {
            "balanced_train_seconds": balanced_seconds,
            "natural_train_seconds": natural_seconds,
            "holdout_seconds": holdout_seconds,
            "epoch_seconds": epoch_seconds,
        }

        # Prefix metric names so W&B creates separate chart groups.
        logged = {
            "epoch": epoch,
            "learning_rate": learning_rate,
            "best_holdout_css": best_css,
            "best_epoch": best_epoch,
            "epochs_without_improvement": epochs_without_improvement,
            **{f"balanced_train/{k}": v for k, v in balanced_metrics.items()},
            **{f"natural_train/{k}": v for k, v in natural_metrics.items()},
            **{f"holdout/{k}": v for k, v in holdout_metrics.items()},
            **{f"sampling/{k}": v for k, v in sampling_metrics.items()},
            **{f"timing/{k}": v for k, v in timing_metrics.items()},
        }
        run.log(logged)
        rows.append({key.replace("/", "_"): value for key, value in logged.items()})

        # Keep a local CSV copy in addition to W&B tracking.
        with metrics_path.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)

        print(
            f"Epoch {epoch:02d} | "
            f"balanced loss={balanced_metrics['loss']:.4f} | "
            f"natural loss={natural_metrics['loss']:.4f} | "
            f"holdout loss={holdout_metrics['loss']:.4f} | "
            f"CSS={holdout_metrics['css']:.4f} | "
            f"NF coverage={sampling_metrics['nf_cumulative_coverage']:.1%} | "
            f"time={epoch_seconds:.1f}s"
        )

        # Stop after ten consecutive epochs without meaningful CSS improvement.
        if epochs_without_improvement >= int(config["early_stopping_patience"]):
            print(f"Early stopping at epoch {epoch}; best epoch was {best_epoch}.")
            break

    # Restore the selected epoch before final validation.
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    final_holdout = run_epoch(model, loaders["holdout"], criterion, device)
    validation = run_epoch(model, validation_loader, criterion, device)

    final_results = {
        "selected_epoch": best_epoch,
        **{f"best_holdout/{k}": v for k, v in final_holdout.items()},
        **{f"validation/{k}": v for k, v in validation.items()},
    }
    run.log(final_results)
    for name, value in final_results.items():
        run.summary[name] = value

    # Upload the selected checkpoint and epoch history as one W&B artifact.
    artifact = wandb.Artifact(
        f"noaa-stage1-{run.id}", type="model", metadata=config
    )
    artifact.add_file(str(checkpoint_path))
    artifact.add_file(str(metrics_path))
    run.log_artifact(artifact)
    run.finish()

    print(f"Best epoch: {best_epoch} | holdout CSS: {best_css:.4f}")
    print(f"Validation CSS: {validation['css']:.4f}")
    print(f"Checkpoint: {checkpoint_path}")


if __name__ == "__main__":
    main()
