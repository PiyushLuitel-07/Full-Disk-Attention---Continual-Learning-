"""Run Experiment 7: full fine-tuning with differential learning rates."""

import csv
import os
import random
import sys
import time
from datetime import datetime
from pathlib import Path

import torch
import torch.distributed as dist
import wandb
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP
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
    """Add rates that help interpret predictions on imbalanced data."""
    total = metrics["tp"] + metrics["tn"] + metrics["fp"] + metrics["fn"]
    actual_nf = metrics["tn"] + metrics["fp"]
    actual_fl = metrics["tp"] + metrics["fn"]
    metrics["predicted_fl_rate"] = (metrics["tp"] + metrics["fp"]) / total
    metrics["false_positive_rate"] = metrics["fp"] / actual_nf
    metrics["false_negative_rate"] = metrics["fn"] / actual_fl
    return metrics


def unwrap_model(model):
    """Return the underlying model when DDP is active."""
    return model.module if isinstance(model, DDP) else model


def build_optimizer(
    model,
    backbone_learning_rate,
    classifier_learning_rate,
    weight_decay,
):
    """Create AdamW groups for differential learning and selective decay."""
    groups = {
        "backbone_decay": [],
        "backbone_no_decay": [],
        "classifier_decay": [],
        "classifier_no_decay": [],
    }

    # One-dimensional parameters are normalization scales or biases. They are
    # not decayed. All other convolutional and linear weights are decayed.
    for name, parameter in unwrap_model(model).named_parameters():
        if not parameter.requires_grad:
            continue

        section = "classifier" if name.startswith("classifier.") else "backbone"
        decay_type = (
            "no_decay"
            if name.endswith(".bias") or parameter.ndim == 1
            else "decay"
        )
        groups[f"{section}_{decay_type}"].append(parameter)

    # The pretrained feature extractor changes slowly. The new classifier is
    # allowed to learn ten times faster because it starts from random weights.
    return AdamW(
        [
            {
                "params": groups["backbone_decay"],
                "lr": backbone_learning_rate,
                "weight_decay": weight_decay,
                "group_name": "backbone_decay",
            },
            {
                "params": groups["backbone_no_decay"],
                "lr": backbone_learning_rate,
                "weight_decay": 0.0,
                "group_name": "backbone_no_decay",
            },
            {
                "params": groups["classifier_decay"],
                "lr": classifier_learning_rate,
                "weight_decay": weight_decay,
                "group_name": "classifier_decay",
            },
            {
                "params": groups["classifier_no_decay"],
                "lr": classifier_learning_rate,
                "weight_decay": 0.0,
                "group_name": "classifier_no_decay",
            },
        ]
    )


def run_epoch(model, loader, criterion, device, optimizer=None,
              distributed=False):
    """Run one training or evaluation epoch and calculate all metrics."""
    training = optimizer is not None
    model.train(training)
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

    # Combine all GPU partitions before calculating global training metrics.
    if distributed:
        gathered_predictions = [None] * dist.get_world_size()
        gathered_targets = [None] * dist.get_world_size()
        dist.all_gather_object(gathered_predictions, predictions)
        dist.all_gather_object(gathered_targets, targets)
        predictions = [value for part in gathered_predictions for value in part]
        targets = [value for part in gathered_targets for value in part]

        totals = torch.tensor(
            [loss_sum.item(), image_count], device=device, dtype=torch.float64
        )
        dist.all_reduce(totals, op=dist.ReduceOp.SUM)
        loss_sum = totals[0]
        image_count = int(totals[1].item())

    metrics = calculate_classification_metrics(predictions, targets)
    metrics["loss"] = (loss_sum / image_count).item()
    return add_rates(metrics)


def save_checkpoint(path, model, optimizer, scheduler, epoch, metrics, config):
    """Save the checkpoint with the best chronological holdout CSS."""
    torch.save(
        {
            "completed_stage": 1,
            "architecture": "mobilenet_v3_large",
            "experiment": "full_finetuning_differential_lr",
            "epoch": epoch,
            "model_state_dict": unwrap_model(model).state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "holdout_metrics": metrics,
            "config": config,
        },
        path,
    )


def write_epoch_metrics(path, rows):
    """Write the accumulated epoch metrics to CSV."""
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def log_combined_curves(run, history):
    """Log explicit train-versus-holdout plots to W&B."""
    epochs = history["epoch"]
    for metric in ("loss", "css", "tss", "hss"):
        chart = wandb.plot.line_series(
            xs=epochs,
            ys=[history[f"train/{metric}"], history[f"holdout/{metric}"]],
            keys=["Training", "Holdout"],
            title=f"Training and holdout {metric.upper()} across epochs",
            xname="Epoch",
        )
        run.log({f"learning_curves/{metric}": chart})


def main():
    """Train, select on holdout CSS, then evaluate validation once."""
    # torchrun supplies one process per GPU. A normal Python launch uses one
    # GPU and follows the same training logic.
    distributed = "LOCAL_RANK" in os.environ
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    is_main = rank == 0

    if distributed:
        if not torch.cuda.is_available():
            raise RuntimeError("NCCL distributed training requires CUDA GPUs.")
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl", init_method="env://")

    timestamp_holder = [
        datetime.now().strftime("%Y%m%d_%H%M%S") if is_main else None
    ]
    if distributed:
        dist.broadcast_object_list(timestamp_holder, src=0)
    timestamp = timestamp_holder[0]
    run_dir = RESULTS_DIR / f"mobilenet_stage1_{timestamp}"
    if is_main:
        run_dir.mkdir(parents=True, exist_ok=True)
    if distributed:
        dist.barrier()

    # Only rank 0 writes W&B data, checkpoints, CSV files, and console logs.
    run = None
    if is_main:
        run = wandb.init(
            entity=WANDB_ENTITY,
            project=WANDB_PROJECT,
            name=f"noaa_stage1_experiment_07_differential_lr_{timestamp}",
            config=CONFIG,
            dir=str(run_dir),
            tags=[
                "NOAA", "Stage-1", "MobileNetV3-Large", "experiment-07",
                "full-finetuning", "full-training-data", "ordinary-loss",
                "differential-learning-rate", "strong-weight-decay",
                "ddp" if distributed else "single-gpu",
            ],
            job_type="stage1-controlled-experiment",
        )
        config = dict(run.config)
    else:
        config = dict(CONFIG)

    seed = int(config["random_seed"])
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    device = torch.device(
        f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"
    )
    if device.type == "cuda":
        torch.cuda.set_device(local_rank)

    global_batch_size = int(config["batch_size"])
    if global_batch_size % world_size != 0:
        raise ValueError("Global batch size must be divisible by GPU count.")
    per_gpu_batch_size = global_batch_size // world_size
    per_process_workers = max(1, int(config["num_workers"]) // world_size)

    loaders = build_loaders(
        TRAIN_CSV,
        HOLDOUT_CSV,
        IMAGE_DIR,
        batch_size=per_gpu_batch_size,
        num_workers=per_process_workers,
        distributed=distributed,
        rank=rank,
        world_size=world_size,
        seed=seed,
    )

    validation_loader = None
    if is_main:
        validation_loader = build_evaluation_loader(
            VALIDATION_CSV,
            IMAGE_DIR,
            batch_size=global_batch_size,
            num_workers=int(config["num_workers"]),
        )

    model = build_model(
        pretrained=bool(config["pretrained"]),
        hidden_dim=int(config["classifier_hidden_dim"]),
        dropout=float(config["dropout"]),
    ).to(device)

    # Synchronize BatchNorm statistics because every feature layer is trained.
    if distributed:
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    criterion = nn.CrossEntropyLoss()
    optimizer = build_optimizer(
        model,
        backbone_learning_rate=float(config["backbone_learning_rate"]),
        classifier_learning_rate=float(config["classifier_learning_rate"]),
        weight_decay=float(config["weight_decay"]),
    )
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=float(config["scheduler_factor"]),
        patience=int(config["scheduler_patience"]),
        threshold=float(config["early_stopping_min_delta"]),
        threshold_mode="abs",
        # One lower bound is supplied for each optimizer parameter group.
        min_lr=[
            float(config["min_backbone_learning_rate"]),
            float(config["min_backbone_learning_rate"]),
            float(config["min_classifier_learning_rate"]),
            float(config["min_classifier_learning_rate"]),
        ],
    )

    class_counts = loaders["class_counts"].tolist()
    base_model = unwrap_model(model)
    system_config = {
        "device": str(device),
        "gpu_name": (
            torch.cuda.get_device_name(local_rank)
            if device.type == "cuda" else "CPU"
        ),
        "gpu_count_used": world_size if device.type == "cuda" else 0,
        "multi_gpu_method": "DDP" if distributed else "none",
        "global_batch_size": global_batch_size,
        "per_gpu_batch_size": per_gpu_batch_size,
        "train_samples_per_epoch": len(loaders["train"].dataset),
        "train_nf_samples": int(class_counts[0]),
        "train_fl_samples": int(class_counts[1]),
        "class_weighting": "none",
        "holdout_samples": len(loaders["holdout"].dataset),
        "validation_samples": len(validation_loader.dataset) if is_main else 0,
        "total_parameters": sum(p.numel() for p in base_model.parameters()),
        "trainable_parameters": sum(
            p.numel() for p in base_model.parameters() if p.requires_grad
        ),
        "weight_decay_exclusions": "bias_and_normalization_parameters",
        "optimizer_parameter_groups": 4,
        "train_csv": str(TRAIN_CSV),
        "holdout_csv": str(HOLDOUT_CSV),
        "validation_csv": str(VALIDATION_CSV),
        "pytorch_version": str(torch.__version__),
    }
    if is_main:
        run.config.update(system_config, allow_val_change=True)
    config.update(system_config)

    if is_main:
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
    history = {
        "epoch": [],
        **{
            f"{group}/{metric}": []
            for group in ("train", "holdout")
            for metric in ("loss", "css", "tss", "hss")
        },
    }

    if is_main:
        print(f"Device: {device} | {system_config['gpu_name']}")
        print(
            f"Multi-GPU: {'DDP' if distributed else 'disabled'} | "
            f"GPUs={world_size} | global batch={global_batch_size} | "
            f"per-GPU batch={per_gpu_batch_size}"
        )
        print(f"Results: {run_dir}")
        print(
            "Training data: "
            f"NF={class_counts[0]:,}, FL={class_counts[1]:,}, "
            f"total={sum(class_counts):,}"
        )
        print("Model: ImageNet initialization; every layer is trainable")
        print("Loss: ordinary cross-entropy (no class weights)")
        print(
            "Optimizer: AdamW | "
            f"backbone LR={float(config['backbone_learning_rate']):.2e} | "
            f"classifier LR={float(config['classifier_learning_rate']):.2e} | "
            f"weight decay={float(config['weight_decay']):.4f} "
            "(bias and normalization excluded)"
        )

    for epoch in range(1, int(config["epochs"]) + 1):
        epoch_start = time.perf_counter()
        if loaders["train_sampler"] is not None:
            loaders["train_sampler"].set_epoch(epoch)

        train_metrics = run_epoch(
            model,
            loaders["train"],
            criterion,
            device,
            optimizer,
            distributed=distributed,
        )

        # The complete chronological holdout is evaluated on rank 0. Its
        # metrics are shared so all DDP processes make identical decisions.
        holdout_holder = [None]
        if is_main:
            holdout_holder[0] = run_epoch(
                unwrap_model(model), loaders["holdout"], criterion, device
            )
        if distributed:
            dist.broadcast_object_list(holdout_holder, src=0)
        holdout_metrics = holdout_holder[0]

        backbone_learning_rate = optimizer.param_groups[0]["lr"]
        classifier_learning_rate = optimizer.param_groups[2]["lr"]
        scheduler.step(holdout_metrics["css"])

        improved = holdout_metrics["css"] > (
            best_css + float(config["early_stopping_min_delta"])
        )
        if improved:
            best_css = holdout_metrics["css"]
            best_epoch = epoch
            epochs_without_improvement = 0
            if is_main:
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
            "learning_rate/backbone": backbone_learning_rate,
            "learning_rate/classifier": classifier_learning_rate,
            "best_holdout_css": best_css,
            "best_epoch": best_epoch,
            "epochs_without_improvement": epochs_without_improvement,
            "timing/epoch_seconds": epoch_seconds,
            **{f"train/{key}": value for key, value in train_metrics.items()},
            **{
                f"holdout/{key}": value
                for key, value in holdout_metrics.items()
            },
        }

        if is_main:
            run.log(logged)
            rows.append({
                key.replace("/", "_"): value for key, value in logged.items()
            })
            write_epoch_metrics(metrics_path, rows)

            history["epoch"].append(epoch)
            for group, values in (
                ("train", train_metrics),
                ("holdout", holdout_metrics),
            ):
                for metric in ("loss", "css", "tss", "hss"):
                    history[f"{group}/{metric}"].append(values[metric])

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
                f"best epoch={best_epoch} | "
                f"best holdout CSS={best_css:.4f} | "
                f"backbone LR={backbone_learning_rate:.2e} | "
                f"classifier LR={classifier_learning_rate:.2e} | "
                f"time={epoch_seconds:.1f}s"
            )

        if epochs_without_improvement >= int(config["early_stopping_patience"]):
            if is_main:
                print(
                    f"Early stopping at epoch {epoch}; "
                    f"best epoch was {best_epoch}."
                )
            break

    if distributed:
        dist.barrier()

    if not is_main:
        dist.barrier()
        dist.destroy_process_group()
        return

    # The checkpoint is locally created and trusted, so its optimizer metadata
    # can safely be loaded with weights_only=False.
    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )
    evaluation_model = unwrap_model(model)
    evaluation_model.load_state_dict(checkpoint["model_state_dict"])
    final_holdout = run_epoch(
        evaluation_model, loaders["holdout"], criterion, device
    )
    validation = run_epoch(
        evaluation_model, validation_loader, criterion, device
    )

    final_results = {
        "selected_epoch": best_epoch,
        **{
            f"best_holdout/{key}": value
            for key, value in final_holdout.items()
        },
        **{f"validation/{key}": value for key, value in validation.items()},
    }
    run.log(final_results)
    for name, value in final_results.items():
        run.summary[name] = value

    # These four W&B panels explicitly overlay training and holdout curves.
    log_combined_curves(run, history)

    with final_metrics_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["metric", "value"])
        writer.writerows(final_results.items())

    artifact = wandb.Artifact(
        f"noaa-stage1-experiment-07-{run.id}",
        type="model",
        metadata=config,
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
        "Validation (final selected checkpoint only) | "
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

    if distributed:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
