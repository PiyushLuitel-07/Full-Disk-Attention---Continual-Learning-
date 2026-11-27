"""Train Stage 1 MobileNetV3-Large with full fine-tuning, single LR, and CyclicBalancedSampler."""

import argparse
import csv
import json
import logging
import math
import random
import sys
import time
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import torch
import wandb
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parents[1]
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.append(str(ROOT / "modeling_phase1"))

from config import CONFIG, HOLDOUT_CSV, IMAGE_DIR, RESULTS_DIR, TEST_CSV, TRAIN_CSV, WANDB_ENTITY, WANDB_PROJECT, WANDB_TAGS
from dataloader import build_evaluation_loader, build_loaders
from evaluation import calculate_classification_metrics
from model import build_model


def setup_logger(log_path):
    """Configure logger writing detailed timestamps to both console and training.log."""
    logger = logging.getLogger("phase2_stage1")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger


def format_metrics(label, m, include_loss=True):
    """Format classification metrics on one readable line."""
    loss_str = f"Loss={m['loss']:.4f} | " if include_loss and "loss" in m else ""
    return (
        f"  {label:<15} | {loss_str}"
        f"CSS={m['css']:.4f} TSS={m['tss']:.4f} HSS={m['hss']:.4f} F1={m['f1']:.4f} | "
        f"Prec={m['precision']:.4f} Rec={m['recall']:.4f} Acc={m['accuracy']:.4f}"
    )


def add_rates(metrics):
    """Add predicted flare rate, FPR, and FNR for imbalanced evaluation diagnostics."""
    total = metrics["tp"] + metrics["tn"] + metrics["fp"] + metrics["fn"]
    actual_nf = metrics["tn"] + metrics["fp"]
    actual_fl = metrics["tp"] + metrics["fn"]
    metrics["predicted_fl_rate"] = (metrics["tp"] + metrics["fp"]) / total if total else 0.0
    metrics["false_positive_rate"] = metrics["fp"] / actual_nf if actual_nf else 0.0
    metrics["false_negative_rate"] = metrics["fn"] / actual_fl if actual_fl else 0.0
    return metrics


def build_optimizer(model, config):
    """Build AdamW with differential learning rates for backbone and classifier."""
    groups = {
        "backbone_decay": [],
        "backbone_no_decay": [],
        "classifier_decay": [],
        "classifier_no_decay": [],
    }
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        section = "classifier" if name.startswith("classifier.") else "backbone"
        decay_type = "no_decay" if name.endswith(".bias") or param.ndim == 1 else "decay"
        groups[f"{section}_{decay_type}"].append(param)

    backbone_lr = float(config["backbone_learning_rate"])
    classifier_lr = float(config["classifier_learning_rate"])
    wd = float(config["weight_decay"])
    return AdamW([
        {"params": groups["backbone_decay"], "lr": backbone_lr, "weight_decay": wd},
        {"params": groups["backbone_no_decay"], "lr": backbone_lr, "weight_decay": 0.0},
        {"params": groups["classifier_decay"], "lr": classifier_lr, "weight_decay": wd},
        {"params": groups["classifier_no_decay"], "lr": classifier_lr, "weight_decay": 0.0},
    ])


def run_epoch(model, loader, criterion, device, optimizer=None):
    """Run one training or evaluation pass and return loss + classification metrics."""
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
            predictions.append(scores.argmax(dim=1).detach())
            targets.append(batch_targets.detach())

    all_preds = torch.cat(predictions).cpu().tolist()
    all_targets = torch.cat(targets).cpu().tolist()
    metrics = calculate_classification_metrics(all_preds, all_targets)
    metrics["loss"] = (loss_sum / image_count).item()
    return add_rates(metrics)


def save_checkpoint(path, model, optimizer, scheduler, epoch, holdout_metrics, train_metrics, config):
    """Save model, optimizer, scheduler, and metrics to disk."""
    torch.save(
        {
            "completed_stage": 1,
            "architecture": config["model"],
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "holdout_metrics": holdout_metrics,
            "train_metrics": train_metrics,
            "config": config,
        },
        path,
    )


def evaluate_checkpoint(ckpt_path, model, loaders, test_loader, criterion, device):
    """Restore a saved checkpoint and evaluate it on Holdout V1 and Test Set (read-only)."""
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    v1_metrics = run_epoch(model, loaders["holdout"], criterion, device)
    test_metrics = run_epoch(model, test_loader, criterion, device)
    return ckpt["epoch"], v1_metrics, test_metrics


def build_training_curves_figure(metrics_path, best_holdout_epoch, best_mature_epoch):
    """Build 6-panel matplotlib figure of train and V1 holdout metrics for W&B logging."""
    df = pd.read_csv(metrics_path)
    epochs = df["epoch"]

    fig, axes = plt.subplots(2, 3, figsize=(18, 10), dpi=300)
    panels = [
        ("css", "CSS (Composite Skill Score)", (0.0, 1.05)),
        ("tss", "TSS (True Skill Statistic)", (-0.1, 1.05)),
        ("hss", "HSS (Heidke Skill Score)", (-0.1, 1.05)),
        ("f1", "F1 Score", (0.0, 1.05)),
        ("recall", "Recall (POD) & Precision", (0.0, 1.05)),
        ("loss", "Loss & Learning Rate", None),
    ]

    for ax, (metric, title, ylim) in zip(axes.flat, panels):
        if metric == "loss":
            ax.plot(epochs, df["train_loss"], label="Train Loss", color="#1f77b4", lw=2)
            ax.plot(epochs, df["holdout_loss"], label="Holdout V1 Loss", color="#d62728", lw=2)
            ax_lr = ax.twinx()
            lr_col = "classifier_learning_rate" if "classifier_learning_rate" in df.columns else "learning_rate"
            ax_lr.plot(epochs, df[lr_col], label="Classifier LR", color="#7f7f7f", ls=":", lw=1.8)
            ax_lr.set_yscale("log")
            ax_lr.set_ylabel("Learning Rate", color="#7f7f7f")
            ax.set_ylabel("Cross-Entropy Loss")
            lines1, labels1 = ax.get_legend_handles_labels()
            lines2, labels2 = ax_lr.get_legend_handles_labels()
            ax.legend(lines1 + lines2, labels1 + labels2, loc="upper right", fontsize=8)
        elif metric == "recall":
            ax.plot(epochs, df["train_recall"], label="Train Recall", color="#1f77b4", lw=2)
            ax.plot(epochs, df["holdout_recall"], label="Holdout V1 Recall", color="#d62728", lw=2)
            ax.plot(epochs, df["train_precision"], label="Train Prec", color="#1f77b4", ls="--", lw=1.6)
            ax.plot(epochs, df["holdout_precision"], label="Holdout V1 Prec", color="#d62728", ls="--", lw=1.6)
            ax.legend(loc="best", fontsize=8)
        else:
            ax.plot(epochs, df[f"train_{metric}"], label="Train", color="#1f77b4", lw=2)
            if f"natural_train_{metric}" in df.columns:
                ax.plot(epochs, df[f"natural_train_{metric}"], label="Natural Train", color="#2ca02c", ls="--", lw=1.8)
            ax.plot(epochs, df[f"holdout_{metric}"], label="Holdout V1", color="#d62728", lw=2)
            ax.legend(loc="best", fontsize=8)

        if best_holdout_epoch:
            ax.axvline(best_holdout_epoch, color="#d62728", ls="-.", lw=1.1, alpha=0.6)
        if best_mature_epoch:
            ax.axvline(best_mature_epoch, color="#9467bd", ls="-.", lw=1.1, alpha=0.6)

        ax.set_title(title, fontweight="bold")
        ax.set_xlabel("Epoch")
        ax.grid(True, alpha=0.25)
        if ylim is not None:
            ax.set_ylim(*ylim)

    fig.suptitle(
        f"Stage 1 Full Fine-Tuning Curves (Best V1: ep {best_holdout_epoch} | Mature: ep {best_mature_epoch})",
        fontsize=14,
        fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    return fig


def parse_args():
    """Parse optional CLI overrides on top of config.py defaults."""
    parser = argparse.ArgumentParser(description="Phase 2 Stage 1 MobileNetV3-Large Training")
    parser.add_argument("--sampling", choices=["full_training_data", "cyclic_balanced"], default=None)
    parser.add_argument("--augmentation", choices=["none", "both_flip_rot5", "fl_flip_rot5"], default=None)
    parser.add_argument("--normalization", choices=["imagenet", "shared_grayscale"], default=None)
    parser.add_argument(
        "--loss",
        choices=["sqrt_inverse_weighted_cross_entropy", "mild_weighted_cross_entropy", "ordinary_cross_entropy"],
        default=None,
    )
    parser.add_argument("--nf-weight", type=float, default=None)
    parser.add_argument("--fl-weight", type=float, default=None)
    parser.add_argument("--backbone-lr", type=float, default=None)
    parser.add_argument("--classifier-lr", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument("--dropout", type=float, default=None)
    parser.add_argument("--bn-momentum", type=str, default=None, help="Float (e.g. 0.1) or 'none' for torchvision default (0.01)")
    parser.add_argument("--freeze-backbone", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--trainable-backbone-blocks", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--scheduler-patience", type=int, default=None)
    parser.add_argument("--scheduler-factor", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--run-name", type=str, default=None, help="Optional custom suffix for run dir & W&B run name")
    return parser.parse_args()


def apply_cli_overrides(base_config, args):
    """Apply CLI overrides to a copy of CONFIG."""
    cfg = dict(base_config)
    direct_map = {
        "sampling": args.sampling,
        "augmentation": args.augmentation,
        "loss": args.loss,
        "nf_class_weight": args.nf_weight,
        "backbone_learning_rate": args.backbone_lr,
        "classifier_learning_rate": args.classifier_lr,
        "weight_decay": args.weight_decay,
        "dropout": args.dropout,
        "freeze_backbone": args.freeze_backbone,
        "trainable_backbone_blocks": args.trainable_backbone_blocks,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "scheduler_patience": args.scheduler_patience,
        "scheduler_factor": args.scheduler_factor,
        "random_seed": args.seed,
        "num_workers": args.num_workers,
    }
    for key, val in direct_map.items():
        if val is not None:
            cfg[key] = val

    if args.fl_weight is not None:
        cfg["fl_class_weight"] = args.fl_weight
        if args.loss is None:
            cfg["loss"] = "mild_weighted_cross_entropy"

    if args.normalization is not None:
        cfg["normalization"] = args.normalization
        if args.normalization == "imagenet":
            cfg["mean"] = [0.485, 0.456, 0.406]
            cfg["std"] = [0.229, 0.224, 0.225]
        else:
            cfg["mean"] = [0.5, 0.5, 0.5]
            cfg["std"] = [0.5, 0.5, 0.5]

    if args.bn_momentum is not None:
        cfg["bn_momentum"] = None if args.bn_momentum.lower() == "none" else float(args.bn_momentum)

    return cfg


def main():
    args = parse_args()
    resolved_config = apply_cli_overrides(CONFIG, args)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = f"_{args.run_name}" if args.run_name else ""
    run_dir = RESULTS_DIR / f"mobilenet_stage1_full_ft_{timestamp}{suffix}"
    run_dir.mkdir(parents=True, exist_ok=True)

    run = wandb.init(
        entity=WANDB_ENTITY,
        project=WANDB_PROJECT,
        name=f"phase2_stage1_full_ft_{timestamp}{suffix}",
        config=resolved_config,
        dir=str(run_dir),
        tags=WANDB_TAGS,
        job_type="phase2-stage1-baseline",
    )
    config = dict(run.config)

    seed = int(config["random_seed"])
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(0)

    mean = tuple(config["mean"])
    std = tuple(config["std"])
    sampling = config.get("sampling", "full_training_data")

    loaders = build_loaders(
        TRAIN_CSV,
        HOLDOUT_CSV,
        IMAGE_DIR,
        batch_size=int(config["batch_size"]),
        num_workers=int(config["num_workers"]),
        seed=seed,
        mean=mean,
        std=std,
        augmentation=config.get("augmentation", "none"),
        sampling=sampling,
    )
    test_loader = build_evaluation_loader(
        TEST_CSV,
        IMAGE_DIR,
        batch_size=int(config["batch_size"]),
        num_workers=int(config["num_workers"]),
        mean=mean,
        std=std,
    )

    model = build_model(
        pretrained=bool(config["pretrained"]),
        freeze_backbone=bool(config["freeze_backbone"]),
        trainable_backbone_blocks=int(config.get("trainable_backbone_blocks", 2)),
        hidden_dim=int(config["classifier_hidden_dim"]),
        dropout=float(config["dropout"]),
        bn_momentum=config.get("bn_momentum", 0.1),
    ).to(device)
    train_targets = loaders["targets"]
    fl_count = sum(1 for t in train_targets if t == 1)
    nf_count = len(train_targets) - fl_count

    nf_weight = float(config.get("nf_class_weight", 1.0))
    if config.get("loss") == "ordinary_cross_entropy":
        nf_weight, fl_weight = 1.0, 1.0
    elif config.get("loss") == "sqrt_inverse_weighted_cross_entropy" or config.get("fl_class_weight") is None:
        fl_weight = math.sqrt(nf_count / fl_count)
    else:
        fl_weight = float(config["fl_class_weight"])
    config["fl_class_weight"] = fl_weight
    class_weights = torch.tensor([nf_weight, fl_weight], dtype=torch.float32, device=device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = build_optimizer(model, config)
    min_backbone_lr = float(config["min_backbone_learning_rate"])
    min_classifier_lr = float(config["min_classifier_learning_rate"])
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=float(config["scheduler_factor"]),
        patience=int(config["scheduler_patience"]),
        threshold=1e-3,
        threshold_mode="abs",
        min_lr=[min_backbone_lr, min_backbone_lr, min_classifier_lr, min_classifier_lr],
    )

    sampler = loaders["balanced_sampler"]
    system_config = {
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU",
        "train_original_samples": len(train_targets),
        "train_samples_per_epoch": len(sampler) if sampler is not None else len(train_targets),
        "holdout_samples": len(loaders["holdout"].dataset),
        "test_samples": len(test_loader.dataset),
        "total_parameters": sum(p.numel() for p in model.parameters()),
        "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "train_csv": str(TRAIN_CSV),
        "holdout_csv": str(HOLDOUT_CSV),
        "test_csv": str(TEST_CSV),
        "pytorch_version": str(torch.__version__),
        "fl_class_weight": fl_weight,
    }
    run.config.update(system_config, allow_val_change=True)
    config.update(system_config)

    with (run_dir / "config.json").open("w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    run.define_metric("epoch")
    for group in ("train", "natural_train", "holdout"):
        run.define_metric(f"{group}/*", step_metric="epoch")

    best_holdout_ckpt = run_dir / "stage1_best_holdout.pt"
    mature_ckpt = run_dir / "stage1_selected_mature.pt"
    last_ckpt = run_dir / "stage1_last.pt"
    metrics_path = run_dir / "epoch_metrics.csv"
    final_metrics_path = run_dir / "final_metrics.csv"

    best_holdout_css = float("-inf")
    best_holdout_epoch = 0
    best_mature_css = float("-inf")
    best_mature_epoch = 0
    cumulative_nf = set()
    rows = []

    logger = setup_logger(run_dir / "training.log")
    logger.info("=" * 85)
    logger.info(f"Phase 2 Stage 1 Full Fine-Tuning | Run Dir: {run_dir}")
    logger.info(f"Device: {device} ({system_config['gpu_name']}) | PyTorch: {system_config['pytorch_version']} | Seed: {seed}")
    logger.info(
        f"Model: {config['model']} | freeze_backbone={config['freeze_backbone']} | "
        f"dropout={config['dropout']} | bn_momentum={config.get('bn_momentum', 0.1)} | "
        f"Params: {system_config['trainable_parameters']:,}/{system_config['total_parameters']:,}"
    )
    logger.info(
        f"Optimizer: {config['optimizer']} | Backbone LR={config['backbone_learning_rate']} | "
        f"Classifier LR={config['classifier_learning_rate']} | WD={config['weight_decay']} | "
        f"Batch={config['batch_size']} | Norm={config['normalization']} | Aug={config['augmentation']} | "
        f"LossWeights=(NF={nf_weight:.2f}, FL={fl_weight:.4f})"
    )
    logger.info(
        f"Scheduler: ReduceLROnPlateau(factor={config['scheduler_factor']}, patience={config['scheduler_patience']}, "
        f"min_backbone_lr={config['min_backbone_learning_rate']}, min_classifier_lr={config['min_classifier_learning_rate']}) | Epochs={config['epochs']}"
    )
    logger.info(
        f"Data: Train={system_config['train_original_samples']:,} "
        f"(FL={fl_count:,}, NF={nf_count:,}, Sampling={sampling}, PerEpoch={system_config['train_samples_per_epoch']:,}) | "
        f"Holdout V1={system_config['holdout_samples']:,} | Test={system_config['test_samples']:,}"
    )
    logger.info("=" * 85)

    total_epochs = int(config["epochs"])
    for epoch in range(1, total_epochs + 1):
        epoch_start = time.perf_counter()

        t0 = time.perf_counter()
        train_metrics = run_epoch(model, loaders["train"], criterion, device, optimizer)
        train_seconds = time.perf_counter() - t0

        natural_metrics = None
        natural_seconds = 0.0
        if loaders["natural_train"] is not None:
            t0 = time.perf_counter()
            natural_metrics = run_epoch(model, loaders["natural_train"], criterion, device)
            natural_seconds = time.perf_counter() - t0

        t0 = time.perf_counter()
        holdout_metrics = run_epoch(model, loaders["holdout"], criterion, device)
        holdout_seconds = time.perf_counter() - t0

        backbone_lr = optimizer.param_groups[0]["lr"]
        classifier_lr = optimizer.param_groups[2]["lr"]
        scheduler.step(holdout_metrics["css"])
        new_backbone_lr = optimizer.param_groups[0]["lr"]
        new_classifier_lr = optimizer.param_groups[2]["lr"]
        if sampler is not None:
            cumulative_nf.update(sampler.last_nf_indices)

        saved_tags = []
        if holdout_metrics["css"] > best_holdout_css:
            best_holdout_css = holdout_metrics["css"]
            best_holdout_epoch = epoch
            save_checkpoint(best_holdout_ckpt, model, optimizer, scheduler, epoch, holdout_metrics, train_metrics, config)
            saved_tags.append("best_holdout")

        is_mature = (
            epoch >= int(config["mature_min_epoch"])
            and train_metrics["css"] >= float(config["mature_min_train_css"])
        )
        if is_mature and holdout_metrics["css"] > best_mature_css:
            best_mature_css = holdout_metrics["css"]
            best_mature_epoch = epoch
            save_checkpoint(mature_ckpt, model, optimizer, scheduler, epoch, holdout_metrics, train_metrics, config)
            saved_tags.append("best_mature")

        if epoch == total_epochs:
            save_checkpoint(last_ckpt, model, optimizer, scheduler, epoch, holdout_metrics, train_metrics, config)
            saved_tags.append("last_epoch")

        epoch_seconds = time.perf_counter() - epoch_start
        timing_metrics = {
            "train_seconds": train_seconds,
            "holdout_seconds": holdout_seconds,
            "epoch_seconds": epoch_seconds,
        }
        logged = {
            "epoch": epoch,
            "backbone_learning_rate": backbone_lr,
            "classifier_learning_rate": classifier_lr,
            "best_holdout_css": best_holdout_css,
            "best_holdout_epoch": best_holdout_epoch,
            "best_mature_css": best_mature_css if best_mature_epoch else 0.0,
            "best_mature_epoch": best_mature_epoch,
            **{f"train/{k}": v for k, v in train_metrics.items()},
            **{f"holdout/{k}": v for k, v in holdout_metrics.items()},
            **{f"timing/{k}": v for k, v in timing_metrics.items()},
        }
        if natural_metrics is not None and sampler is not None:
            logged.update({
                **{f"natural_train/{k}": v for k, v in natural_metrics.items() if k != "loss"},
                "sampling/nf_cycle": sampler.cycle,
                "sampling/nf_cumulative_coverage": len(cumulative_nf) / len(sampler.nf_indices),
                "timing/natural_train_seconds": natural_seconds,
            })

        run.log(logged)
        rows.append({k.replace("/", "_"): v for k, v in logged.items()})

        with metrics_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)

        cycle_info = (
            f" | NF_cycle={sampler.cycle} ({len(cumulative_nf) / len(sampler.nf_indices):.1%})"
            if sampler is not None
            else ""
        )
        logger.info(
            f"Epoch {epoch:02d}/{total_epochs} | BackboneLR={backbone_lr:.2e} | ClassifierLR={classifier_lr:.2e}{cycle_info} | "
            f"Time={epoch_seconds:.1f}s (train={train_seconds:.1f}s, v1={holdout_seconds:.1f}s)"
        )
        logger.info(format_metrics("Train", train_metrics, include_loss=True))
        logger.info(format_metrics("Holdout V1", holdout_metrics, include_loss=True))
        if saved_tags:
            logger.info(f"  [CHECKPOINT SAVED] {', '.join(saved_tags)} (Best V1 CSS={best_holdout_css:.4f} @ ep {best_holdout_epoch})")
        if new_classifier_lr < classifier_lr or new_backbone_lr < backbone_lr:
            logger.info(
                f"  [LR DECAY] Backbone {backbone_lr:.2e} -> {new_backbone_lr:.2e} | "
                f"Classifier {classifier_lr:.2e} -> {new_classifier_lr:.2e}"
            )

    logger.info("=" * 85)
    logger.info("FINAL CHECKPOINT EVALUATIONS (Holdout V1 & Untouched Test Set)")
    final_rows = []
    for ckpt_label, ckpt_path in [
        ("best_holdout", best_holdout_ckpt),
        ("selected_mature", mature_ckpt),
        ("last_epoch", last_ckpt),
    ]:
        if not ckpt_path.exists():
            continue
        ckpt_epoch, v1_m, test_m = evaluate_checkpoint(ckpt_path, model, loaders, test_loader, criterion, device)
        logger.info(f"Checkpoint [{ckpt_label}] (Epoch {ckpt_epoch}):")
        logger.info(format_metrics("  Holdout V1", v1_m, include_loss=True))
        logger.info(format_metrics("  Test Set", test_m, include_loss=True))
        run.log({
            **{f"{ckpt_label}_v1/{k}": v for k, v in v1_m.items()},
            **{f"{ckpt_label}_test/{k}": v for k, v in test_m.items()},
        })
        final_rows.append({"checkpoint": ckpt_label, "epoch": ckpt_epoch, "split": "holdout_v1", **v1_m})
        final_rows.append({"checkpoint": ckpt_label, "epoch": ckpt_epoch, "split": "test", **test_m})

    with final_metrics_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=final_rows[0].keys())
        writer.writeheader()
        writer.writerows(final_rows)

    fig = build_training_curves_figure(metrics_path, best_holdout_epoch, best_mature_epoch)
    run.log({"training_curves": wandb.Image(fig)})
    plt.close(fig)
    logger.info("Logged training curves figure to W&B.")
    logger.info("=" * 85)

    run.finish()


if __name__ == "__main__":
    main()
