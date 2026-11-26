"""Run one self-contained Stage 2 EWC-only Bayesian-sweep trial."""

import argparse
import csv
import math
import os
import random
import time
from datetime import datetime
from pathlib import Path

import torch
import wandb
from torch import nn
from torch.nn.modules.batchnorm import _BatchNorm
from torch.optim.lr_scheduler import ReduceLROnPlateau

from config import (
    ANCHOR,
    CONFIG,
    IMAGE_DIR,
    RESULTS_DIR,
    STAGE1_HOLDOUT_CSV,
    STAGE2_HOLDOUT_CSV,
    STAGE2_TRAIN_CSV,
    VALIDATION_CSV,
    WANDB_ENTITY,
    WANDB_PROJECT,
    WANDB_TAGS,
)
from dataloader import build_loaders, class_counts
from ewc import ewc_penalty, load_ewc_anchor
from metrics import classification_metrics, evaluate
from model import build_model, build_optimizer


def parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--anchor", type=Path, default=ANCHOR)
    parser.add_argument("--run-name")
    return parser.parse_args()


def prefix(name, metrics):
    return {f"{name}/{key}": value for key, value in metrics.items()}


def joint_css(first, second):
    return math.sqrt(max(0.0, first["css"]) * max(0.0, second["css"]))


def save_csv(path, data, multiple_rows=False):
    with path.open("w", newline="", encoding="utf-8") as file:
        if multiple_rows:
            writer = csv.DictWriter(file, fieldnames=data[0])
            writer.writeheader()
            writer.writerows(data)
        else:
            writer = csv.writer(file)
            writer.writerow(["metric", "value"])
            writer.writerows(data.items())


def resolve_config(run):
    """Combine sweep values with settings derived from them."""
    config = dict(run.config)
    config["classifier_learning_rate"] = (
        float(config["backbone_learning_rate"])
        * float(config["classifier_lr_multiplier"])
    )
    config["batchnorm_running_stats_mode"] = (
        "adapt_on_stage2"
        if config["adapt_batchnorm_running_stats"]
        else "frozen_at_stage1"
    )
    config["freeze_batchnorm_running_stats"] = not bool(
        config["adapt_batchnorm_running_stats"]
    )

    if float(config["ewc_lambda"]) <= 0:
        raise ValueError("Experiment 15 requires ewc_lambda > 0")

    run.config.update(config, allow_val_change=True)
    return config


def train_epoch(model, loader, criterion, optimizer, device, anchors, config):
    """Train one epoch with classification loss and EWC regularization."""
    model.train()

    if not config["adapt_batchnorm_running_stats"]:
        for module in model.modules():
            if isinstance(module, _BatchNorm):
                module.eval()

    totals = {"loss": 0.0, "classification_loss": 0.0, "ewc_loss": 0.0}
    predictions, targets = [], []
    sample_count = 0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        classification_loss = criterion(logits, labels)
        ewc_loss = 0.5 * config["ewc_lambda"] * ewc_penalty(model, anchors)
        total_loss = classification_loss + ewc_loss

        if not torch.isfinite(total_loss):
            raise FloatingPointError("Non-finite Stage 2 loss")

        total_loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), config["max_gradient_norm"]
        )
        if not torch.isfinite(gradient_norm):
            raise FloatingPointError("Non-finite gradient norm")
        optimizer.step()

        batch_size = len(images)
        sample_count += batch_size
        totals["loss"] += total_loss.detach().item() * batch_size
        totals["classification_loss"] += (
            classification_loss.detach().item() * batch_size
        )
        totals["ewc_loss"] += ewc_loss.detach().item() * batch_size
        predictions.extend(logits.argmax(dim=1).detach().cpu().tolist())
        targets.extend(labels.cpu().tolist())

    metrics = classification_metrics(predictions, targets)
    metrics.update(
        {name: value / sample_count for name, value in totals.items()}
    )
    return metrics


def main():
    args = parse_arguments()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_label = f"{timestamp}_{os.getpid()}"
    output = RESULTS_DIR / f"mobilenet_stage2_ewc_tuning_{run_label}"
    output.mkdir(parents=True, exist_ok=False)

    run = wandb.init(
        entity=WANDB_ENTITY,
        project=WANDB_PROJECT,
        config=CONFIG,
        dir=str(output),
        name=args.run_name or f"noaa_stage2_experiment_15_ewc_{run_label}",
        tags=WANDB_TAGS,
        job_type="stage2-ewc-bayesian-sweep-trial",
    )
    config = resolve_config(run)

    if not args.anchor.is_file():
        raise FileNotFoundError(f"Missing EWC anchor: {args.anchor}")

    random.seed(config["random_seed"])
    torch.manual_seed(config["random_seed"])
    torch.cuda.manual_seed_all(config["random_seed"])
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(0)

    # Load the selected Stage 1 model and its empirical Fisher information.
    saved_anchor = torch.load(args.anchor, map_location="cpu", weights_only=False)
    if len(saved_anchor.get("ewc_history", [])) != 1:
        raise ValueError("Expected one Stage 1 EWC anchor")

    model = build_model(
        hidden_dim=config["classifier_hidden_dim"],
        dropout=config["dropout"],
    ).to(device)
    model.load_state_dict(saved_anchor["model_state_dict"], strict=True)

    anchors = [load_ewc_anchor(saved_anchor["ewc_history"][0], device)]
    if anchors[0]["stage"] != 1:
        raise ValueError("Experiment 15 requires a Stage 1 EWC anchor")

    with torch.no_grad():
        initial_penalty = ewc_penalty(model, anchors).item()
    if not math.isfinite(initial_penalty) or abs(initial_penalty) > 1e-10:
        raise ValueError(f"Anchor/model mismatch: EWC penalty={initial_penalty}")

    paths = {
        "images": IMAGE_DIR,
        "stage1_holdout": STAGE1_HOLDOUT_CSV,
        "stage2_train": STAGE2_TRAIN_CSV,
        "stage2_holdout": STAGE2_HOLDOUT_CSV,
        "validation": VALIDATION_CSV,
    }
    loaders, sampling = build_loaders(config, paths)
    stage2_counts = class_counts(loaders["stage2_train"].dataset)

    class_weights = torch.tensor(
        [config["nf_class_weight"], config["fl_class_weight"]],
        dtype=torch.float32,
        device=device,
    )
    criterion = nn.CrossEntropyLoss(
        weight=class_weights,
        label_smoothing=float(config["label_smoothing"]),
    )
    optimizer = build_optimizer(model, config)
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=config["scheduler_factor"],
        patience=config["scheduler_patience"],
        threshold=config["early_stopping_min_delta"],
        threshold_mode="abs",
        min_lr=(
            [config["min_backbone_learning_rate"]] * 2
            + [config["min_classifier_learning_rate"]] * 2
        ),
    )

    run.config.update(
        {
            "source_ewc_anchor": str(args.anchor.resolve()),
            "stage2_train_nf": stage2_counts[0],
            "stage2_train_fl": stage2_counts[1],
            "sampling_metadata": sampling,
            "device": str(device),
            "initial_ewc_penalty": initial_penalty,
        },
        allow_val_change=True,
    )

    # Epoch 0 is the unchanged Stage 1 model before Stage 2 training.
    stage1_baseline = evaluate(
        model, loaders["stage1_holdout"], criterion, device
    )
    stage2_baseline = evaluate(
        model, loaders["stage2_holdout"], criterion, device
    )
    initial_metrics = {
        "epoch": 0,
        **prefix("stage1_holdout", stage1_baseline),
        **prefix("stage2_holdout", stage2_baseline),
    }

    run.define_metric("epoch")
    for group in ("stage2_train", "stage1_holdout", "stage2_holdout"):
        run.define_metric(f"{group}/*", step_metric="epoch")
    run.log(initial_metrics)
    save_csv(
        output / "initial_metrics.csv",
        {key: value for key, value in initial_metrics.items() if key != "epoch"},
    )

    print(
        f"Results: {output}\nDevice: {device} | batch={config['batch_size']} | "
        f"Stage 2: NF={stage2_counts[0]:,}, FL={stage2_counts[1]:,}"
    )
    print(
        f"lambda={config['ewc_lambda']:.3g} | "
        f"backbone LR={config['backbone_learning_rate']:.2e} | "
        f"classifier LR={config['classifier_learning_rate']:.2e}"
    )
    print(
        f"decay={config['weight_decay']:.4f} | "
        f"smoothing={config['label_smoothing']:.3f} | "
        f"BN={config['batchnorm_running_stats_mode']}"
    )

    # Train and select the epoch with the highest Stage 2 holdout CSS.
    best_css = float("-inf")
    best_epoch = None
    epochs_without_improvement = 0
    epoch_history = []
    checkpoint = output / "stage2_best.pt"

    for epoch in range(1, int(config["epochs"]) + 1):
        epoch_start = time.perf_counter()
        training = train_epoch(
            model,
            loaders["stage2_train"],
            criterion,
            optimizer,
            device,
            anchors,
            config,
        )
        stage1 = evaluate(model, loaders["stage1_holdout"], criterion, device)
        stage2 = evaluate(model, loaders["stage2_holdout"], criterion, device)

        stage2_css = stage2["css"]
        previous_best = best_css
        scheduler.step(stage2_css)

        if stage2_css > best_css:
            best_css = stage2_css
            best_epoch = epoch
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": {
                        name: value.detach().cpu()
                        for name, value in model.state_dict().items()
                    },
                    "config": config,
                    "stage2_train": training,
                    "stage2_holdout": stage2,
                    "stage1_retention": stage1,
                    "ewc_history": saved_anchor["ewc_history"],
                    "stage1_fisher_metadata": saved_anchor["fisher_metadata"],
                },
                checkpoint,
            )

        if stage2_css > previous_best + config["early_stopping_min_delta"]:
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        epoch_metrics = {
            "epoch": epoch,
            "selection/stage2_css": stage2_css,
            "diagnostics/train_holdout_css_gap": training["css"] - stage2_css,
            "diagnostics/joint_css": joint_css(stage1, stage2),
            "diagnostics/stage1_retention_ratio": (
                stage1["css"] / stage1_baseline["css"]
            ),
            "best/epoch": best_epoch,
            "best/stage2_css": best_css,
            "epochs_without_improvement": epochs_without_improvement,
            "timing/epoch_seconds": time.perf_counter() - epoch_start,
            **prefix("stage2_train", training),
            **prefix("stage1_holdout", stage1),
            **prefix("stage2_holdout", stage2),
        }
        run.log(epoch_metrics)

        epoch_history.append(
            {key.replace("/", "_"): value for key, value in epoch_metrics.items()}
        )
        save_csv(output / "epoch_metrics.csv", epoch_history, multiple_rows=True)

        print(
            f"Epoch {epoch:02d} | train CSS={training['css']:.4f} | "
            f"holdout CSS={stage2_css:.4f} | recall={stage2['recall']:.4f} | "
            f"FPR={stage2['false_positive_rate']:.4f} | best={best_css:.4f}"
        )

        if epochs_without_improvement >= config["early_stopping_patience"]:
            break

    # Restore the selected checkpoint and evaluate validation once.
    selected = torch.load(checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(selected["model_state_dict"])

    stage1 = evaluate(model, loaders["stage1_holdout"], criterion, device)
    stage2 = evaluate(model, loaders["stage2_holdout"], criterion, device)
    validation = evaluate(model, loaders["validation"], criterion, device)

    final_metrics = {
        "selected_epoch": selected["epoch"],
        "selected_stage2_train_css": selected["stage2_train"]["css"],
        "selected_stage2_train_holdout_gap": (
            selected["stage2_train"]["css"] - stage2["css"]
        ),
        "stage2_css_delta_from_epoch0": stage2["css"] - stage2_baseline["css"],
        "stage1_css_retention_ratio": stage1["css"] / stage1_baseline["css"],
        **prefix("stage1_retention", stage1),
        **prefix("stage2_holdout", stage2),
        **prefix("validation", validation),
    }
    run.log(final_metrics)
    run.summary.update(final_metrics)
    save_csv(output / "final_metrics.csv", final_metrics)

    artifact = wandb.Artifact(
        f"noaa-stage2-ewc-tuning-{run.id}",
        type="model",
        metadata={"selected_epoch": selected["epoch"]},
    )
    for path in (
        checkpoint,
        output / "initial_metrics.csv",
        output / "epoch_metrics.csv",
        output / "final_metrics.csv",
    ):
        artifact.add_file(str(path))

    run.log_artifact(artifact)
    run.finish()
    print(
        f"Selected epoch={selected['epoch']} | Stage 2 CSS={stage2['css']:.4f} | "
        f"Stage 1 CSS={stage1['css']:.4f} | validation CSS={validation['css']:.4f}"
    )


if __name__ == "__main__":
    main()
