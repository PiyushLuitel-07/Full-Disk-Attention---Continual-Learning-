"""Train Stage 1 for 100 epochs and select a mature near-best checkpoint."""

import argparse
import csv
import json
import random
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path

import torch
import wandb
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(HERE), str(ROOT / "modeling"), str(ROOT)]

from config import (  # noqa: E402
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
from dataloader import build_loader, class_counts  # noqa: E402
from evaluation import calculate_classification_metrics  # noqa: E402
from model import build_model  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=CONFIG["epochs"])
    parser.add_argument(
        "--mature-holdout-tolerance",
        type=float,
        default=CONFIG["mature_holdout_tolerance"],
    )
    parser.add_argument("--random-seed", type=int, default=CONFIG["random_seed"])
    parser.add_argument(
        "--scheduler-patience",
        type=int,
        default=CONFIG["scheduler_patience"],
    )
    parser.add_argument(
        "--fl-class-weight",
        type=float,
        default=None,
        help="Override the default exact inverse-frequency FL weight.",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=CONFIG["weight_decay"],
        help="AdamW weight decay applied to weight tensors (bias/norm excluded).",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=CONFIG["dropout"],
        help="Dropout probability between the two classifier linear layers.",
    )
    parser.add_argument(
        "--augmentation",
        choices=(
            "none",
            "rotation5",
            "rotation5_fl_copy",
            "rotation5_hflip_fl_copy",
            "rotation5_and_hflip_fl_copies",
        ),
        default="none",
        help=(
            "Training-only augmentation; rotation5 applies a -5 to +5 degree "
            "rotation to each training sample with probability 0.5; "
            "rotation5_fl_copy appends one rotated copy of every FL sample; "
            "rotation5_hflip_fl_copy additionally flips each appended copy "
            "horizontally with probability 0.5; rotation5_and_hflip_fl_copies "
            "appends two FL copies: one rotation-only and one flip-only."
        ),
    )
    parser.add_argument(
        "--lr-restart-epoch",
        type=int,
        default=0,
        help=(
            "Reset the learning rates and ReduceLROnPlateau state at the start of this "
            "epoch. Use 0 to disable (the default)."
        ),
    )
    parser.add_argument(
        "--lr-restart-scale",
        type=float,
        default=1.0,
        help=(
            "Multiplier on the initial backbone and classifier learning rates used at "
            "the optional restart."
        ),
    )
    parser.add_argument("--run-name", default=None)
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def add_rates(metrics):
    nf = metrics["tn"] + metrics["fp"]
    fl = metrics["tp"] + metrics["fn"]
    total = nf + fl
    metrics["predicted_fl_rate"] = (metrics["tp"] + metrics["fp"]) / total
    metrics["false_positive_rate"] = metrics["fp"] / nf
    metrics["false_negative_rate"] = metrics["fn"] / fl
    return metrics


def run_epoch(model, loader, criterion, device, optimizer=None):
    training = optimizer is not None
    model.train(training)
    predictions, targets, loss_sum, count = [], [], 0.0, 0

    with torch.set_grad_enabled(training):
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            if training:
                optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, labels)
            if training:
                loss.backward()
                optimizer.step()
            count += labels.numel()
            loss_sum += loss.item() * labels.numel()
            predictions.extend(logits.argmax(1).detach().cpu().tolist())
            targets.extend(labels.cpu().tolist())

    metrics = calculate_classification_metrics(predictions, targets)
    metrics["loss"] = loss_sum / count
    return add_rates(metrics)


def build_optimizer(model, config):
    groups = {(part, decay): [] for part in ("backbone", "classifier") for decay in (0, 1)}
    for name, parameter in model.named_parameters():
        part = "classifier" if name.startswith("classifier.") else "backbone"
        decay = int(not (name.endswith(".bias") or parameter.ndim == 1))
        groups[part, decay].append(parameter)

    backbone_lr = config["backbone_learning_rate"]
    classifier_lr = backbone_lr * config["classifier_lr_multiplier"]
    rates = {"backbone": backbone_lr, "classifier": classifier_lr}
    parameters = [
        {
            "params": groups[part, decay],
            "lr": rates[part],
            "weight_decay": config["weight_decay"] if decay else 0.0,
        }
        for part in ("backbone", "classifier")
        for decay in (1, 0)
    ]
    return AdamW(parameters), classifier_lr


def build_scheduler(optimizer, config):
    """Create a fresh plateau scheduler, including after an optional LR restart."""
    return ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=config["scheduler_factor"],
        patience=config["scheduler_patience"],
        threshold=0.001,
        threshold_mode="abs",
        min_lr=[
            config["min_backbone_learning_rate"],
            config["min_backbone_learning_rate"],
            config["min_classifier_learning_rate"],
            config["min_classifier_learning_rate"],
        ],
    )


def restart_learning_rates(optimizer, config):
    """Restore group LRs to a specified fraction of their initial values."""
    backbone_lr = config["backbone_learning_rate"] * config["lr_restart_scale"]
    classifier_lr = (
        config["backbone_learning_rate"]
        * config["classifier_lr_multiplier"]
        * config["lr_restart_scale"]
    )
    for group_index, group in enumerate(optimizer.param_groups):
        group["lr"] = backbone_lr if group_index < 2 else classifier_lr
    return backbone_lr, classifier_lr


def checkpoint_payload(model, epoch, train_metrics, holdout_metrics, config):
    """Store weights and selection metadata only, not large optimizer states."""
    return {
        "completed_stage": 1,
        "architecture": "mobilenet_v3_large",
        "experiment": "stage1_long_training",
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "train_metrics": train_metrics,
        "holdout_metrics": holdout_metrics,
        "config": config,
    }


def update_pareto(directory, candidates, model, epoch, train_metrics, holdout_metrics, config):
    """Retain only checkpoints not dominated on both train and holdout CSS."""
    train_css, holdout_css = train_metrics["css"], holdout_metrics["css"]
    dominated = any(
        item["train_css"] >= train_css and item["holdout_css"] >= holdout_css
        for item in candidates
    )
    if dominated:
        return candidates

    survivors = []
    for item in candidates:
        new_dominates = train_css >= item["train_css"] and holdout_css >= item["holdout_css"]
        if new_dominates:
            item["path"].unlink(missing_ok=True)
        else:
            survivors.append(item)

    path = directory / f"epoch_{epoch:03d}.pt"
    torch.save(checkpoint_payload(model, epoch, train_metrics, holdout_metrics, config), path)
    survivors.append(
        {"epoch": epoch, "train_css": train_css, "holdout_css": holdout_css, "path": path}
    )
    return survivors


def write_rows(path, rows):
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def log_learning_curves(run, rows):
    """Create W&B charts with final-model train metrics versus holdout metrics."""
    epochs = [row["epoch"] for row in rows]
    metrics = (
        "css", "tss", "hss", "loss", "precision", "recall", "f1",
        "false_positive_rate", "false_negative_rate",
    )
    for metric in metrics:
        chart = wandb.plot.line_series(
            xs=epochs,
            ys=[
                [row[f"train_eval_{metric}"] for row in rows],
                [row[f"holdout_{metric}"] for row in rows],
            ],
            keys=["Training (end-of-epoch evaluation)", "Holdout"],
            title=f"Training and holdout {metric.upper()} across epochs",
            xname="Epoch",
        )
        run.log({f"learning_curves/{metric}": chart})


def load_metrics(model, checkpoint_path, loader, criterion, device):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    return run_epoch(model, loader, criterion, device)


def main():
    args = parse_args()
    config = dict(CONFIG)
    config.update(
        epochs=args.epochs,
        mature_holdout_tolerance=args.mature_holdout_tolerance,
        random_seed=args.random_seed,
        scheduler_patience=args.scheduler_patience,
        weight_decay=args.weight_decay,
        dropout=args.dropout,
        augmentation=args.augmentation,
        lr_restart_epoch=args.lr_restart_epoch,
        lr_restart_scale=args.lr_restart_scale,
    )
    if config["epochs"] != 100:
        print(f"Warning: requested {config['epochs']} epochs; the planned experiment uses 100.")
    if not 0 <= config["mature_holdout_tolerance"] <= 1:
        raise ValueError("mature_holdout_tolerance must be between 0 and 1")
    if config["scheduler_patience"] < 0:
        raise ValueError("scheduler_patience must be non-negative")
    if config["weight_decay"] < 0:
        raise ValueError("weight_decay must be non-negative")
    if not 0 <= config["dropout"] < 1:
        raise ValueError("dropout must be in the interval [0, 1)")
    if not 0 <= config["lr_restart_epoch"] <= config["epochs"]:
        raise ValueError("lr_restart_epoch must be between 0 and epochs")
    if config["lr_restart_epoch"] == 1:
        raise ValueError("lr_restart_epoch must be 0 or at least 2")
    if config["lr_restart_scale"] <= 0:
        raise ValueError("lr_restart_scale must be positive")

    set_seed(config["random_seed"])
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = args.run_name or f"noaa_stage1_long_training_{timestamp}"
    run_dir = RESULTS_DIR / f"mobilenet_stage1_long_{timestamp}"
    candidate_dir = run_dir / "pareto_candidates"
    candidate_dir.mkdir(parents=True, exist_ok=False)

    batch_size, workers = config["batch_size"], config["num_workers"]
    train_loader = build_loader(
        TRAIN_CSV,
        IMAGE_DIR,
        batch_size,
        workers,
        shuffle=True,
        augmentation=config["augmentation"],
    )
    train_eval_loader = build_loader(TRAIN_CSV, IMAGE_DIR, batch_size, workers)
    holdout_loader = build_loader(HOLDOUT_CSV, IMAGE_DIR, batch_size, workers)
    validation_loader = build_loader(VALIDATION_CSV, IMAGE_DIR, batch_size, workers)

    counts = class_counts(train_loader)
    nf_count, fl_count = int(counts[0]), int(counts[1])
    exact_fl_weight = nf_count / fl_count
    config["nf_class_weight"] = 1.0
    config["fl_class_weight"] = (
        args.fl_class_weight
        if args.fl_class_weight is not None
        else exact_fl_weight
    )
    if config["fl_class_weight"] <= 0:
        raise ValueError("fl_class_weight must be positive")
    config["class_weighting"] = (
        "manual_fl_weight_override" if args.fl_class_weight is not None
        else "exact_inverse_frequency_ratio"
    )
    config["exact_inverse_frequency_fl_weight"] = exact_fl_weight
    config["classifier_learning_rate"] = (
        config["backbone_learning_rate"] * config["classifier_lr_multiplier"]
    )
    config["run_name"] = run_name
    config["checkpoint_selection"] = "highest_train_css_within_best_holdout_minus_tolerance"
    config["early_stopping"] = False
    with (run_dir / "resolved_config.json").open("w", encoding="utf-8") as file:
        json.dump(config, file, indent=2, sort_keys=True)
        file.write("\n")

    criterion = nn.CrossEntropyLoss(
        weight=torch.tensor([1.0, config["fl_class_weight"]], device=device)
    )
    model = build_model(
        pretrained=config["pretrained"],
        hidden_dim=config["classifier_hidden_dim"],
        dropout=config["dropout"],
    ).to(device)
    optimizer, classifier_lr = build_optimizer(model, config)
    scheduler = build_scheduler(optimizer, config)

    run = wandb.init(
        entity=WANDB_ENTITY,
        project=WANDB_PROJECT,
        name=run_name,
        tags=WANDB_TAGS,
        config=config,
        dir=str(run_dir),
    )
    run.define_metric("epoch")
    for group in ("train_online", "train_eval", "holdout"):
        run.define_metric(f"{group}/*", step_metric="epoch")

    best_path = run_dir / "stage1_best_holdout.pt"
    last_path = run_dir / "stage1_last.pt"
    mature_path = run_dir / "stage1_selected_mature.pt"
    rows, pareto, best_holdout, best_epoch = [], [], float("-inf"), 0

    print(f"Device: {device}")
    print(f"Results: {run_dir}")
    print(f"Training data: NF={nf_count:,} | FL={fl_count:,} | total={nf_count + fl_count:,}")
    print(f"Loss weights: NF=1.0000 | FL={config['fl_class_weight']:.6f}")
    print(f"AdamW weight decay: {config['weight_decay']:.6f}")
    print(f"Classifier dropout: {config['dropout']:.3f}")
    print(f"Training augmentation: {config['augmentation']}")
    print(f"Training: fixed {config['epochs']} epochs | early stopping disabled")
    print(
        "Scheduler: ReduceLROnPlateau | "
        f"patience={config['scheduler_patience']} | "
        f"factor={config['scheduler_factor']:.2f}"
    )
    if config["lr_restart_epoch"]:
        print(
            "LR restart: start of epoch "
            f"{config['lr_restart_epoch']} | scale={config['lr_restart_scale']:.3f}"
        )

    for epoch in range(1, config["epochs"] + 1):
        restart_applied = epoch == config["lr_restart_epoch"]
        if restart_applied:
            restart_backbone_lr, restart_classifier_lr = restart_learning_rates(optimizer, config)
            scheduler = build_scheduler(optimizer, config)
            print(
                f"Restarted learning rates at epoch {epoch}: "
                f"backbone={restart_backbone_lr:.2e} | "
                f"classifier={restart_classifier_lr:.2e}"
            )
        started = time.perf_counter()
        train_online = run_epoch(model, train_loader, criterion, device, optimizer)
        # This is the stable train CSS attached to the saved end-of-epoch model.
        train_eval = run_epoch(model, train_eval_loader, criterion, device)
        holdout = run_epoch(model, holdout_loader, criterion, device)
        scheduler.step(holdout["css"])

        payload = checkpoint_payload(model, epoch, train_eval, holdout, config)
        if holdout["css"] > best_holdout:
            best_holdout, best_epoch = holdout["css"], epoch
            torch.save(payload, best_path)
        if epoch == config["epochs"]:
            torch.save(payload, last_path)
        pareto = update_pareto(
            candidate_dir, pareto, model, epoch, train_eval, holdout, config
        )

        elapsed = time.perf_counter() - started
        logged = {
            "epoch": epoch,
            "learning_rate/backbone": optimizer.param_groups[0]["lr"],
            "learning_rate/classifier": optimizer.param_groups[2]["lr"],
            "selection/best_holdout_css": best_holdout,
            "selection/best_holdout_epoch": best_epoch,
            "schedule/restart_applied": int(restart_applied),
            "timing/epoch_seconds": elapsed,
            **{f"train_online/{key}": value for key, value in train_online.items()},
            **{f"train_eval/{key}": value for key, value in train_eval.items()},
            **{f"holdout/{key}": value for key, value in holdout.items()},
        }
        run.log(logged)
        rows.append({key.replace("/", "_"): value for key, value in logged.items()})
        write_rows(run_dir / "epoch_metrics.csv", rows)

        print(f"Epoch {epoch:03d}/{config['epochs']}")
        print(
            f"  Train(eval) | loss={train_eval['loss']:.4f} | "
            f"CSS={train_eval['css']:.4f} | TSS={train_eval['tss']:.4f} | "
            f"HSS={train_eval['hss']:.4f}"
        )
        print(
            f"              | recall={train_eval['recall']:.4f} | "
            f"precision={train_eval['precision']:.4f} | F1={train_eval['f1']:.4f} | "
            f"FPR={train_eval['false_positive_rate']:.4f} | "
            f"FNR={train_eval['false_negative_rate']:.4f}"
        )
        print(
            f"  Holdout     | loss={holdout['loss']:.4f} | CSS={holdout['css']:.4f} | "
            f"TSS={holdout['tss']:.4f} | HSS={holdout['hss']:.4f}"
        )
        print(
            f"              | recall={holdout['recall']:.4f} | "
            f"precision={holdout['precision']:.4f} | F1={holdout['f1']:.4f} | "
            f"FPR={holdout['false_positive_rate']:.4f} | "
            f"FNR={holdout['false_negative_rate']:.4f}"
        )
        print(
            f"  Progress    | best={best_holdout:.4f} at epoch {best_epoch} | "
            f"gap={train_eval['css'] - holdout['css']:+.4f} | "
            f"backbone LR={optimizer.param_groups[0]['lr']:.2e} | "
            f"classifier LR={optimizer.param_groups[2]['lr']:.2e} | time={elapsed:.1f}s"
        )

    cutoff = best_holdout - config["mature_holdout_tolerance"]
    eligible = [item for item in pareto if item["holdout_css"] >= cutoff]
    selected = max(eligible, key=lambda item: (item["train_css"], item["holdout_css"]))
    shutil.copy2(selected["path"], mature_path)
    log_learning_curves(run, rows)

    best_holdout_metrics = load_metrics(model, best_path, holdout_loader, criterion, device)
    best_validation = load_metrics(model, best_path, validation_loader, criterion, device)
    mature_holdout = load_metrics(model, mature_path, holdout_loader, criterion, device)
    mature_validation = load_metrics(model, mature_path, validation_loader, criterion, device)

    final = {
        "best_holdout_epoch": best_epoch,
        "best_holdout_css": best_holdout,
        "mature_epoch": selected["epoch"],
        "mature_train_css": selected["train_css"],
        "mature_holdout_css_at_selection": selected["holdout_css"],
        "mature_holdout_cutoff": cutoff,
        **{f"strict_best_holdout/{key}": value for key, value in best_holdout_metrics.items()},
        **{f"strict_best_validation/{key}": value for key, value in best_validation.items()},
        **{f"mature_holdout/{key}": value for key, value in mature_holdout.items()},
        **{f"mature_validation/{key}": value for key, value in mature_validation.items()},
    }
    run.log(final)
    write_rows(
        run_dir / "final_metrics.csv",
        [{"metric": key, "value": value} for key, value in final.items()],
    )
    artifact = wandb.Artifact(f"{run_name}-checkpoints", type="model")
    for path in (best_path, mature_path, last_path):
        artifact.add_file(str(path))
    run.log_artifact(artifact)
    run.finish()

    print(f"Strict best: epoch {best_epoch}, holdout CSS={best_holdout:.4f}")
    print(
        f"Selected mature: epoch {selected['epoch']}, "
        f"train CSS={selected['train_css']:.4f}, "
        f"holdout CSS={selected['holdout_css']:.4f} (cutoff={cutoff:.4f})"
    )
    print(
        "Strict-best validation | "
        f"loss={best_validation['loss']:.4f} | CSS={best_validation['css']:.4f} | "
        f"TSS={best_validation['tss']:.4f} | HSS={best_validation['hss']:.4f} | "
        f"recall={best_validation['recall']:.4f} | "
        f"precision={best_validation['precision']:.4f} | "
        f"F1={best_validation['f1']:.4f} | "
        f"FPR={best_validation['false_positive_rate']:.4f}"
    )
    print(
        "Mature validation      | "
        f"loss={mature_validation['loss']:.4f} | CSS={mature_validation['css']:.4f} | "
        f"TSS={mature_validation['tss']:.4f} | HSS={mature_validation['hss']:.4f} | "
        f"recall={mature_validation['recall']:.4f} | "
        f"precision={mature_validation['precision']:.4f} | "
        f"F1={mature_validation['f1']:.4f} | "
        f"FPR={mature_validation['false_positive_rate']:.4f}"
    )
    print(f"Strict checkpoint: {best_path}")
    print(f"Mature checkpoint: {mature_path}")
    print(f"Last checkpoint: {last_path}")


if __name__ == "__main__":
    main()
