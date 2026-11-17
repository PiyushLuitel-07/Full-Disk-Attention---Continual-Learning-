"""Train Stage 1 for 100 epochs and select a mature near-best checkpoint."""

import argparse
import csv
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
    )
    if config["epochs"] != 100:
        print(f"Warning: requested {config['epochs']} epochs; the planned experiment uses 100.")
    if not 0 <= config["mature_holdout_tolerance"] <= 1:
        raise ValueError("mature_holdout_tolerance must be between 0 and 1")

    set_seed(config["random_seed"])
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = args.run_name or f"noaa_stage1_long_training_{timestamp}"
    run_dir = RESULTS_DIR / f"mobilenet_stage1_long_{timestamp}"
    candidate_dir = run_dir / "pareto_candidates"
    candidate_dir.mkdir(parents=True, exist_ok=False)

    batch_size, workers = config["batch_size"], config["num_workers"]
    train_loader = build_loader(TRAIN_CSV, IMAGE_DIR, batch_size, workers, shuffle=True)
    train_eval_loader = build_loader(TRAIN_CSV, IMAGE_DIR, batch_size, workers)
    holdout_loader = build_loader(HOLDOUT_CSV, IMAGE_DIR, batch_size, workers)
    validation_loader = build_loader(VALIDATION_CSV, IMAGE_DIR, batch_size, workers)

    counts = class_counts(train_loader)
    nf_count, fl_count = int(counts[0]), int(counts[1])
    config["nf_class_weight"] = 1.0
    config["fl_class_weight"] = nf_count / fl_count
    config["classifier_learning_rate"] = (
        config["backbone_learning_rate"] * config["classifier_lr_multiplier"]
    )
    config["checkpoint_selection"] = "highest_train_css_within_best_holdout_minus_tolerance"
    config["early_stopping"] = False

    criterion = nn.CrossEntropyLoss(
        weight=torch.tensor([1.0, config["fl_class_weight"]], device=device)
    )
    model = build_model(
        pretrained=config["pretrained"],
        hidden_dim=config["classifier_hidden_dim"],
        dropout=config["dropout"],
    ).to(device)
    optimizer, classifier_lr = build_optimizer(model, config)
    scheduler = ReduceLROnPlateau(
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
    print(f"Training: fixed {config['epochs']} epochs | early stopping disabled")
    print(f"Mature-selection holdout tolerance: {config['mature_holdout_tolerance']:.4f}")

    for epoch in range(1, config["epochs"] + 1):
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
            "selection/pareto_candidate_count": len(pareto),
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
            f"  Train(eval) | CSS={train_eval['css']:.4f} | "
            f"TSS={train_eval['tss']:.4f} | HSS={train_eval['hss']:.4f}"
        )
        print(
            f"  Holdout     | CSS={holdout['css']:.4f} | "
            f"TSS={holdout['tss']:.4f} | HSS={holdout['hss']:.4f}"
        )
        print(
            f"  Progress    | best={best_holdout:.4f} at epoch {best_epoch} | "
            f"Pareto={len(pareto)} | time={elapsed:.1f}s"
        )

    cutoff = best_holdout - config["mature_holdout_tolerance"]
    eligible = [item for item in pareto if item["holdout_css"] >= cutoff]
    selected = max(eligible, key=lambda item: (item["train_css"], item["holdout_css"]))
    shutil.copy2(selected["path"], mature_path)

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
    print(f"Strict checkpoint: {best_path}")
    print(f"Mature checkpoint: {mature_path}")
    print(f"Last checkpoint: {last_path}")


if __name__ == "__main__":
    main()
