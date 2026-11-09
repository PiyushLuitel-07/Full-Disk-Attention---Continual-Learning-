"""Run the Stage 1 inverse-frequency weighting control experiment.

This is a one-run ablation: it keeps Candidate 114547's architecture,
full-data loading, differential learning rates, regularization, and seed. The
only intentional difference is the stronger inverse-frequency FL:NF loss ratio.
"""

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
    WANDB_TAGS,
)
from dataloader import build_evaluation_loader, build_loaders
from evaluation import calculate_classification_metrics
from model import build_model


def add_rates(metrics):
    """Add rates needed to assess the recall--false-alarm trade-off."""
    total = metrics["tp"] + metrics["tn"] + metrics["fp"] + metrics["fn"]
    actual_nf = metrics["tn"] + metrics["fp"]
    actual_fl = metrics["tp"] + metrics["fn"]
    metrics["predicted_fl_rate"] = (metrics["tp"] + metrics["fp"]) / total
    metrics["false_positive_rate"] = metrics["fp"] / actual_nf
    metrics["false_negative_rate"] = metrics["fn"] / actual_fl
    return metrics


def build_optimizer(model, backbone_lr, classifier_lr, weight_decay):
    """Use Candidate 114547's differential-LR, selective-decay optimizer."""
    groups = {
        "backbone_decay": [],
        "backbone_no_decay": [],
        "classifier_decay": [],
        "classifier_no_decay": [],
    }

    for name, parameter in model.named_parameters():
        section = "classifier" if name.startswith("classifier.") else "backbone"
        decay_type = (
            "no_decay"
            if name.endswith(".bias") or parameter.ndim == 1
            else "decay"
        )
        groups[f"{section}_{decay_type}"].append(parameter)

    return AdamW(
        [
            {
                "params": groups["backbone_decay"],
                "lr": backbone_lr,
                "weight_decay": weight_decay,
            },
            {
                "params": groups["backbone_no_decay"],
                "lr": backbone_lr,
                "weight_decay": 0.0,
            },
            {
                "params": groups["classifier_decay"],
                "lr": classifier_lr,
                "weight_decay": weight_decay,
            },
            {
                "params": groups["classifier_no_decay"],
                "lr": classifier_lr,
                "weight_decay": 0.0,
            },
        ]
    )


def run_epoch(model, loader, criterion, device, optimizer=None):
    """Train or evaluate one epoch and return loss plus classification metrics."""
    training = optimizer is not None
    model.train(training)
    predictions, targets = [], []
    total_loss = 0.0
    total_images = 0

    with torch.set_grad_enabled(training):
        for images, batch_targets in loader:
            images = images.to(device, non_blocking=True)
            batch_targets = batch_targets.to(device, non_blocking=True)

            if training:
                optimizer.zero_grad(set_to_none=True)

            logits = model(images)
            loss = criterion(logits, batch_targets)

            if training:
                loss.backward()
                optimizer.step()

            batch_size = images.size(0)
            total_loss += loss.detach().item() * batch_size
            total_images += batch_size
            predictions.extend(logits.argmax(dim=1).detach().cpu().tolist())
            targets.extend(batch_targets.detach().cpu().tolist())

    metrics = calculate_classification_metrics(predictions, targets)
    metrics["loss"] = total_loss / total_images
    return add_rates(metrics)


def save_checkpoint(path, model, optimizer, scheduler, epoch, metrics, config):
    """Save the checkpoint selected only by chronological holdout CSS."""
    torch.save(
        {
            "completed_stage": 1,
            "architecture": "mobilenet_v3_large",
            "experiment": "inverse_frequency_control",
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "holdout_metrics": metrics,
            "config": config,
        },
        path,
    )


def write_csv(path, rows):
    """Persist epoch-level metrics for later direct comparison."""
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def log_curves(run, history):
    """Overlay training and holdout curves in W&B."""
    for metric in ("loss", "css", "tss", "hss"):
        run.log(
            {
                f"learning_curves/{metric}": wandb.plot.line_series(
                    xs=history["epoch"],
                    ys=[history[f"train/{metric}"], history[f"holdout/{metric}"]],
                    keys=["Training", "Holdout"],
                    title=f"Training and holdout {metric.upper()} across epochs",
                    xname="Epoch",
                )
            }
        )


def main():
    """Run the fixed inverse-frequency ablation on one visible GPU."""
    config = dict(CONFIG)
    config["classifier_learning_rate"] = (
        float(config["backbone_learning_rate"])
        * float(config["classifier_lr_multiplier"])
    )

    seed = int(config["random_seed"])
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # With CUDA_VISIBLE_DEVICES=1, this becomes physical GPU 1 but is named
    # cuda:0 inside the process. This experiment intentionally uses one GPU.
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(0)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = RESULTS_DIR / f"mobilenet_stage1_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=False)

    run = wandb.init(
        entity=WANDB_ENTITY,
        project=WANDB_PROJECT,
        name=f"noaa_stage1_experiment_12_inverse_frequency_{timestamp}",
        config=config,
        dir=str(run_dir),
        tags=WANDB_TAGS + ["single-gpu"],
        job_type="stage1-inverse-frequency-control",
    )
    config = dict(run.config)

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
    ).to(device)

    class_weights = torch.tensor(
        [float(config["nf_class_weight"]), float(config["fl_class_weight"])],
        dtype=torch.float32,
        device=device,
    )
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = build_optimizer(
        model,
        backbone_lr=float(config["backbone_learning_rate"]),
        classifier_lr=float(config["classifier_learning_rate"]),
        weight_decay=float(config["weight_decay"]),
    )
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=float(config["scheduler_factor"]),
        patience=int(config["scheduler_patience"]),
        threshold=float(config["early_stopping_min_delta"]),
        threshold_mode="abs",
        min_lr=[
            float(config["min_backbone_learning_rate"]),
            float(config["min_backbone_learning_rate"]),
            float(config["min_classifier_learning_rate"]),
            float(config["min_classifier_learning_rate"]),
        ],
    )

    counts = loaders["class_counts"].tolist()
    system_config = {
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU",
        "gpu_count_used": 1 if device.type == "cuda" else 0,
        "multi_gpu_method": "none",
        "train_samples_per_epoch": len(loaders["train"].dataset),
        "train_nf_samples": int(counts[0]),
        "train_fl_samples": int(counts[1]),
        "holdout_samples": len(loaders["holdout"].dataset),
        "validation_samples": len(validation_loader.dataset),
        "class_weighting": "inverse_frequency_ratio_control",
        "inverse_frequency_fl_to_nf_ratio": float(config["fl_class_weight"]),
        "optimizer_parameter_groups": 4,
        "weight_decay_exclusions": "bias_and_normalization_parameters",
    }
    run.config.update(system_config, allow_val_change=True)
    config.update(system_config)

    print(f"Device: {device} | {system_config['gpu_name']}")
    print("Multi-GPU: disabled | GPUs=1")
    print(f"Results: {run_dir}")
    print(f"Training data: NF={counts[0]:,}, FL={counts[1]:,}, total={sum(counts):,}")
    print("Model: ImageNet initialization; every layer is trainable")
    print(
        "Loss: inverse-frequency weighted cross-entropy | "
        f"NF={config['nf_class_weight']:.4f} | FL={config['fl_class_weight']:.4f} | "
        f"FL:NF ratio={config['fl_class_weight'] / config['nf_class_weight']:.4f}"
    )
    print(
        "Optimizer: AdamW | "
        f"backbone LR={config['backbone_learning_rate']:.2e} | "
        f"classifier LR={config['classifier_learning_rate']:.2e} | "
        f"weight decay={config['weight_decay']:.4f} "
        "(bias and normalization excluded)"
    )

    run.define_metric("epoch")
    for group in ("train", "holdout"):
        run.define_metric(f"{group}/*", step_metric="epoch")
    run.define_metric("best_holdout/*")
    run.define_metric("validation/*")

    best_css = float("-inf")
    best_epoch = 0
    without_improvement = 0
    checkpoint_path = run_dir / "stage1_best.pt"
    rows = []
    history = {"epoch": []}
    for group in ("train", "holdout"):
        for metric in ("loss", "css", "tss", "hss"):
            history[f"{group}/{metric}"] = []

    for epoch in range(1, int(config["epochs"]) + 1):
        started = time.perf_counter()
        train_metrics = run_epoch(model, loaders["train"], criterion, device, optimizer)
        holdout_metrics = run_epoch(model, loaders["holdout"], criterion, device)

        backbone_lr = optimizer.param_groups[0]["lr"]
        classifier_lr = optimizer.param_groups[2]["lr"]
        scheduler.step(holdout_metrics["css"])

        improved = holdout_metrics["css"] > (
            best_css + float(config["early_stopping_min_delta"])
        )
        if improved:
            best_css = holdout_metrics["css"]
            best_epoch = epoch
            without_improvement = 0
            save_checkpoint(
                checkpoint_path, model, optimizer, scheduler, epoch,
                holdout_metrics, config,
            )
        else:
            without_improvement += 1

        elapsed = time.perf_counter() - started
        logged = {
            "epoch": epoch,
            "learning_rate/backbone": backbone_lr,
            "learning_rate/classifier": classifier_lr,
            "best_holdout_css": best_css,
            "best_epoch": best_epoch,
            "epochs_without_improvement": without_improvement,
            "timing/epoch_seconds": elapsed,
            **{f"train/{key}": value for key, value in train_metrics.items()},
            **{f"holdout/{key}": value for key, value in holdout_metrics.items()},
        }
        run.log(logged)
        rows.append({key.replace("/", "_"): value for key, value in logged.items()})
        write_csv(run_dir / "epoch_metrics.csv", rows)

        history["epoch"].append(epoch)
        for group, metrics in (("train", train_metrics), ("holdout", holdout_metrics)):
            for metric in ("loss", "css", "tss", "hss"):
                history[f"{group}/{metric}"].append(metrics[metric])

        print(f"Epoch {epoch:02d}/{int(config['epochs'])}")
        print(
            "  Train   | "
            f"loss={train_metrics['loss']:.4f} | CSS={train_metrics['css']:.4f} | "
            f"TSS={train_metrics['tss']:.4f} | HSS={train_metrics['hss']:.4f}"
        )
        print(
            "  Holdout | "
            f"loss={holdout_metrics['loss']:.4f} | CSS={holdout_metrics['css']:.4f} | "
            f"TSS={holdout_metrics['tss']:.4f} | HSS={holdout_metrics['hss']:.4f}"
        )
        print(
            "  Progress | "
            f"best epoch={best_epoch} | best holdout CSS={best_css:.4f} | "
            f"backbone LR={backbone_lr:.2e} | classifier LR={classifier_lr:.2e} | "
            f"time={elapsed:.1f}s"
        )

        if without_improvement >= int(config["early_stopping_patience"]):
            print(f"Early stopping at epoch {epoch}; best epoch was {best_epoch}.")
            break

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    best_holdout = run_epoch(model, loaders["holdout"], criterion, device)
    validation = run_epoch(model, validation_loader, criterion, device)

    final_results = {
        "selected_epoch": checkpoint["epoch"],
        **{f"best_holdout/{key}": value for key, value in best_holdout.items()},
        **{f"validation/{key}": value for key, value in validation.items()},
    }
    run.log(final_results)
    for name, value in final_results.items():
        run.summary[name] = value
    log_curves(run, history)

    with (run_dir / "final_metrics.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["metric", "value"])
        writer.writerows(final_results.items())

    artifact = wandb.Artifact(
        f"noaa-stage1-inverse-frequency-control-{run.id}",
        type="model",
        metadata=config,
    )
    artifact.add_file(str(checkpoint_path))
    artifact.add_file(str(run_dir / "epoch_metrics.csv"))
    artifact.add_file(str(run_dir / "final_metrics.csv"))
    run.log_artifact(artifact)

    print(f"Best epoch: {checkpoint['epoch']}")
    print(
        "Selected holdout | "
        f"loss={best_holdout['loss']:.4f} | CSS={best_holdout['css']:.4f} | "
        f"TSS={best_holdout['tss']:.4f} | HSS={best_holdout['hss']:.4f} | "
        f"precision={best_holdout['precision']:.4f} | recall={best_holdout['recall']:.4f} | "
        f"FPR={best_holdout['false_positive_rate']:.4f}"
    )
    print(
        "Validation | "
        f"loss={validation['loss']:.4f} | CSS={validation['css']:.4f} | "
        f"TSS={validation['tss']:.4f} | HSS={validation['hss']:.4f} | "
        f"precision={validation['precision']:.4f} | recall={validation['recall']:.4f} | "
        f"FPR={validation['false_positive_rate']:.4f}"
    )
    print(f"Checkpoint: {checkpoint_path}")
    run.finish()


if __name__ == "__main__":
    main()
