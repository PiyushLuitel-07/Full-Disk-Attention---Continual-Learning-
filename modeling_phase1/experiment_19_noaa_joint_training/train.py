"""Run one three-stage NOAA joint-training Bayesian-sweep trial."""

import argparse
import csv
import json
import random
import time
from datetime import datetime

import torch
import wandb
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR

from config import (
    CONFIG,
    HOLDOUT_CSVS,
    IMAGE_DIR,
    JOINT_TRAIN_CSV,
    RESULTS_DIR,
    VALIDATION_CSV,
    WANDB_ENTITY,
    WANDB_PROJECT,
    WANDB_TAGS,
)
from dataloader import build_loader, class_counts, validate_csv_splits
from evaluation import classification_metrics
from model import build_model


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", default=None)
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def run_epoch(model, loader, criterion, device, optimizer=None, scheduler=None):
    training = optimizer is not None
    model.train(training)
    predictions, targets, scores, loss_sum, count = [], [], [], 0.0, 0
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
                scheduler.step()
            count += labels.numel()
            loss_sum += loss.item() * labels.numel()
            predictions.extend(logits.argmax(1).detach().cpu().tolist())
            scores.extend(logits.softmax(1)[:, 1].detach().cpu().tolist())
            targets.extend(labels.cpu().tolist())
    metrics = classification_metrics(predictions, targets, scores)
    metrics["loss"] = loss_sum / count
    return metrics


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


def macro_metrics(stage_metrics):
    keys = (
        "accuracy", "precision", "recall", "f1", "false_positive_rate",
        "false_negative_rate", "tss", "hss", "css", "roc_auc",
        "average_precision", "loss",
    )
    return {
        key: sum(metrics[key] for metrics in stage_metrics.values()) / len(stage_metrics)
        for key in keys
    }


def checkpoint(model, epoch, train_metrics, holdouts, macro, config):
    return {
        "architecture": "mobilenet_v3_large",
        "experiment": "noaa_joint_training_experiment_19",
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "train_metrics": train_metrics,
        "stage_holdout_metrics": holdouts,
        "macro_holdout_metrics": macro,
        "config": config,
    }


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def load_checkpoint_metrics(model, path, loaders, criterion, device):
    saved = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(saved["model_state_dict"])
    return {
        name: run_epoch(model, loader, criterion, device)
        for name, loader in loaders.items()
    }


def log_curves(run, rows):
    epochs = [row["epoch"] for row in rows]
    series = {
        "css": ["train_eval_css", "stage1_css", "stage2_css", "stage3_css", "macro_css"],
        "loss": ["train_eval_loss", "stage1_loss", "stage2_loss", "stage3_loss", "macro_loss"],
        "recall": ["train_eval_recall", "stage1_recall", "stage2_recall", "stage3_recall", "macro_recall"],
        "false_positive_rate": [
            "train_eval_false_positive_rate", "stage1_false_positive_rate",
            "stage2_false_positive_rate", "stage3_false_positive_rate",
            "macro_false_positive_rate",
        ],
        "roc_auc": [
            "train_eval_roc_auc", "stage1_roc_auc", "stage2_roc_auc",
            "stage3_roc_auc", "macro_roc_auc",
        ],
    }
    for metric, columns in series.items():
        run.log({
            f"learning_curves/{metric}": wandb.plot.line_series(
                xs=epochs,
                ys=[[row[column] for row in rows] for column in columns],
                keys=[column.replace("_", " ") for column in columns],
                title=f"Joint training and stage holdout {metric} across epochs",
                xname="Epoch",
            )
        })


def validate_config(config):
    if config["epochs"] <= 0 or config["batch_size"] <= 0:
        raise ValueError("epochs and batch_size must be positive")
    if config["backbone_learning_rate"] <= 0 or config["classifier_lr_multiplier"] <= 0:
        raise ValueError("learning rates and multiplier must be positive")
    if config["weight_decay"] < 0 or config["fl_class_weight"] <= 0:
        raise ValueError("weight decay must be nonnegative and class weight positive")
    if config["augmentation"] != "none":
        raise ValueError("Experiment 19 baseline does not use augmentation")
    if config["normalization"] != "shared_grayscale":
        raise ValueError("Experiment 19 uses shared grayscale normalization")


def main():
    args = parse_args()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    run_dir = RESULTS_DIR / f"mobilenet_joint_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=False)

    run = wandb.init(
        entity=WANDB_ENTITY,
        project=WANDB_PROJECT,
        name=args.run_name,
        tags=WANDB_TAGS,
        config=CONFIG,
        dir=str(run_dir),
        job_type="joint-training-sweep-trial",
    )
    config = dict(run.config)
    config["classifier_learning_rate"] = (
        config["backbone_learning_rate"] * config["classifier_lr_multiplier"]
    )
    validate_config(config)
    set_seed(int(config["random_seed"]))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    frames = validate_csv_splits(JOINT_TRAIN_CSV, HOLDOUT_CSVS, VALIDATION_CSV)
    norm = {"mean": config["grayscale_mean"], "std": config["grayscale_std"]}
    train_loader = build_loader(
        JOINT_TRAIN_CSV, IMAGE_DIR, config["batch_size"], config["num_workers"],
        shuffle=True, **norm,
    )
    train_eval_loader = build_loader(
        JOINT_TRAIN_CSV, IMAGE_DIR, config["batch_size"], config["num_workers"], **norm,
    )
    holdout_loaders = {
        name: build_loader(
            path, IMAGE_DIR, config["batch_size"], config["num_workers"], **norm,
        )
        for name, path in HOLDOUT_CSVS.items()
    }
    validation_loader = build_loader(
        VALIDATION_CSV, IMAGE_DIR, config["batch_size"], config["num_workers"], **norm,
    )

    counts = class_counts(train_loader)
    nf_count, fl_count = int(counts[0]), int(counts[1])
    config.update({
        "joint_nf_samples": nf_count,
        "joint_fl_samples": fl_count,
        "joint_samples": nf_count + fl_count,
        "exact_inverse_fl_weight": nf_count / fl_count,
        "sqrt_inverse_fl_weight": (nf_count / fl_count) ** 0.5,
        "run_name": run.name,
        "validation_role": "sweep_trial_ranking_only_after_macro_holdout_selection",
        "test_used": False,
    })
    run.config.update({
        key: config[key]
        for key in (
            "classifier_learning_rate", "joint_nf_samples", "joint_fl_samples",
            "joint_samples", "exact_inverse_fl_weight", "sqrt_inverse_fl_weight",
            "validation_role", "test_used",
        )
    }, allow_val_change=True)
    with (run_dir / "resolved_config.json").open("w", encoding="utf-8") as file:
        json.dump(config, file, indent=2, sort_keys=True)
        file.write("\n")

    model = build_model(
        pretrained=config["pretrained"],
        hidden_dim=config["classifier_hidden_dim"],
        dropout=config["dropout"],
    ).to(device)
    criterion = nn.CrossEntropyLoss(
        weight=torch.tensor(
            [config["nf_class_weight"], config["fl_class_weight"]], device=device
        )
    )
    optimizer, classifier_lr = build_optimizer(model, config)
    scheduler = OneCycleLR(
        optimizer,
        max_lr=[
            config["backbone_learning_rate"], config["backbone_learning_rate"],
            classifier_lr, classifier_lr,
        ],
        epochs=config["epochs"],
        steps_per_epoch=len(train_loader),
        pct_start=config["onecycle_pct_start"],
        anneal_strategy="cos",
        div_factor=config["onecycle_div_factor"],
        final_div_factor=config["onecycle_final_div_factor"],
    )

    run.define_metric("epoch")
    for group in (
        "train_online", "train_eval", "holdout_stage1", "holdout_stage2",
        "holdout_stage3", "holdout_macro", "diagnostics", "selection",
        "learning_rate", "timing",
    ):
        run.define_metric(f"{group}/*", step_metric="epoch")

    best_path = run_dir / "joint_best_macro_holdout.pt"
    last_path = run_dir / "joint_last.pt"
    rows, best_macro, best_epoch = [], float("-inf"), 0

    print(f"Device: {device}")
    print(f"Results: {run_dir}")
    print(f"Joint train: NF={nf_count:,} | FL={fl_count:,} | total={nf_count + fl_count:,}")
    print(f"Stage holdouts: " + " | ".join(f"{k}={len(frames[k]):,}" for k in HOLDOUT_CSVS))
    print(f"Loss weights: NF={config['nf_class_weight']:.4f} | FL={config['fl_class_weight']:.6f}")
    print(f"No augmentation | shared grayscale normalization")
    print(
        f"OneCycleLR: {config['epochs']} epochs | warmup={config['onecycle_pct_start']:.0%} | "
        f"backbone peak={config['backbone_learning_rate']:.2e} | classifier peak={classifier_lr:.2e}"
    )
    print("Checkpoint selection: maximum equally weighted macro CSS across Stage 1/2/3 holdouts")
    print("Validation is evaluated only after the macro-selected checkpoint; test is not used.")

    for epoch in range(1, config["epochs"] + 1):
        started = time.perf_counter()
        train_online = run_epoch(model, train_loader, criterion, device, optimizer, scheduler)
        train_eval = run_epoch(model, train_eval_loader, criterion, device)
        holdouts = {
            name: run_epoch(model, loader, criterion, device)
            for name, loader in holdout_loaders.items()
        }
        macro = macro_metrics(holdouts)
        improved = macro["css"] > best_macro
        if improved:
            best_macro, best_epoch = macro["css"], epoch
            torch.save(checkpoint(model, epoch, train_eval, holdouts, macro, config), best_path)
        if epoch == config["epochs"]:
            torch.save(checkpoint(model, epoch, train_eval, holdouts, macro, config), last_path)

        elapsed = time.perf_counter() - started
        logged = {
            "epoch": epoch,
            "learning_rate/backbone": optimizer.param_groups[0]["lr"],
            "learning_rate/classifier": optimizer.param_groups[2]["lr"],
            "selection/best_macro_css": best_macro,
            "selection/best_epoch": best_epoch,
            "selection/is_new_best": int(improved),
            "timing/epoch_seconds": elapsed,
            "diagnostics/train_macro_css_gap": train_eval["css"] - macro["css"],
            **{
                f"diagnostics/train_{name}_css_gap": train_eval["css"] - metrics["css"]
                for name, metrics in holdouts.items()
            },
            **{f"train_online/{k}": v for k, v in train_online.items()},
            **{f"train_eval/{k}": v for k, v in train_eval.items()},
            **{
                f"holdout_{name}/{k}": v
                for name, metrics in holdouts.items() for k, v in metrics.items()
            },
            **{f"holdout_macro/{k}": v for k, v in macro.items()},
        }
        run.log(logged)
        row = {key.replace("/", "_"): value for key, value in logged.items()}
        for name, metrics in holdouts.items():
            row.update({f"{name}_{key}": value for key, value in metrics.items()})
        row.update({f"macro_{key}": value for key, value in macro.items()})
        rows.append(row)
        write_csv(run_dir / "epoch_metrics.csv", rows)

        print(f"Epoch {epoch:03d}/{config['epochs']}")
        print(
            f"  Train | loss={train_eval['loss']:.4f} | CSS={train_eval['css']:.4f} | "
            f"TSS={train_eval['tss']:.4f} | HSS={train_eval['hss']:.4f} | "
            f"recall={train_eval['recall']:.4f} | FPR={train_eval['false_positive_rate']:.4f} | "
            f"AUC={train_eval['roc_auc']:.4f} | AP={train_eval['average_precision']:.4f}"
        )
        for name, metrics in holdouts.items():
            print(
                f"  {name.title()} holdout | loss={metrics['loss']:.4f} | "
                f"CSS={metrics['css']:.4f} | TSS={metrics['tss']:.4f} | "
                f"HSS={metrics['hss']:.4f} | recall={metrics['recall']:.4f} | "
                f"FPR={metrics['false_positive_rate']:.4f} | AUC={metrics['roc_auc']:.4f} | "
                f"AP={metrics['average_precision']:.4f}"
            )
        print(
            f"  Macro | CSS={macro['css']:.4f} | TSS={macro['tss']:.4f} | "
            f"HSS={macro['hss']:.4f} | gap={train_eval['css'] - macro['css']:+.4f} | "
            f"best={best_macro:.4f} at epoch {best_epoch} | time={elapsed:.1f}s"
        )

    log_curves(run, rows)
    evaluation_loaders = {**holdout_loaders, "validation": validation_loader}
    selected_metrics = load_checkpoint_metrics(
        model, best_path, evaluation_loaders, criterion, device
    )
    selected_macro = macro_metrics({k: selected_metrics[k] for k in HOLDOUT_CSVS})
    saved = torch.load(best_path, map_location="cpu", weights_only=False)
    final = {
        "selected_epoch": saved["epoch"],
        "selected_macro_holdout_css": selected_macro["css"],
        **{f"selected_macro_holdout/{k}": v for k, v in selected_macro.items()},
        **{
            f"selected_{name}/{k}": v
            for name, metrics in selected_metrics.items() for k, v in metrics.items()
        },
        **{f"validation/{k}": v for k, v in selected_metrics["validation"].items()},
    }
    run.log(final)
    run.summary.update(final)
    write_csv(
        run_dir / "final_metrics.csv",
        [{"metric": key, "value": value} for key, value in final.items()],
    )
    artifact = wandb.Artifact(f"{run.name}-joint-checkpoints", type="model")
    artifact.add_file(str(best_path))
    artifact.add_file(str(last_path))
    run.log_artifact(artifact)
    run.finish()

    validation = selected_metrics["validation"]
    print(f"Selected epoch: {saved['epoch']} | macro holdout CSS={selected_macro['css']:.4f}")
    print(
        f"Validation | loss={validation['loss']:.4f} | CSS={validation['css']:.4f} | "
        f"TSS={validation['tss']:.4f} | HSS={validation['hss']:.4f} | "
        f"recall={validation['recall']:.4f} | FPR={validation['false_positive_rate']:.4f} | "
        f"AUC={validation['roc_auc']:.4f} | AP={validation['average_precision']:.4f}"
    )
    print(f"Best checkpoint: {best_path}")


if __name__ == "__main__":
    main()
