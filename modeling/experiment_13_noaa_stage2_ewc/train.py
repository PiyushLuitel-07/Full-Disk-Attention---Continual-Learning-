"""Train NOAA Stage 2 from the corrected Stage 1 EWC anchor.

Example (single physical GPU 1):

    CUDA_VISIBLE_DEVICES=1 python modeling/experiment_13_noaa_stage2_ewc/train.py \
        --ewc-lambda 1.0

Use ``--ewc-lambda 0`` as the required no-EWC fine-tuning baseline.
"""

import argparse
import csv
import math
import random
import sys
import time
from datetime import datetime
from pathlib import Path

import torch
import wandb
from torch import nn
from torch.nn.modules.batchnorm import _BatchNorm
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau


SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.append(str(SCRIPT_DIR.parent))

from config import (  # noqa: E402
    CONFIG,
    FUTURE_VALIDATION_CSV,
    IMAGE_DIR,
    RESULTS_DIR,
    STAGE1_HOLDOUT_CSV,
    STAGE1_TRAIN_CSV,
    STAGE2_HOLDOUT_CSV,
    STAGE2_TRAIN_CSV,
    WANDB_ENTITY,
    WANDB_PROJECT,
    WANDB_TAGS,
)
from dataloader import build_stage2_loaders, class_counts  # noqa: E402
from ewc import ewc_penalty, move_ewc_anchor_to_device  # noqa: E402
from evaluation import calculate_classification_metrics  # noqa: E402
from model import build_model  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--anchor",
        type=Path,
        default=RESULTS_DIR / "anchors" / "stage1_ewc_ready.pt",
        help="Stage 1 EWC anchor made by prepare_stage1_fisher.py.",
    )
    parser.add_argument(
        "--ewc-lambda",
        type=float,
        required=True,
        help="Non-negative EWC strength; use 0 for the no-EWC baseline.",
    )
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument(
        "--nf-class-weight",
        type=float,
        default=None,
        help="Override the NF cross-entropy weight from config.py.",
    )
    parser.add_argument(
        "--fl-class-weight",
        type=float,
        default=None,
        help="Override the FL cross-entropy weight from config.py.",
    )
    parser.add_argument(
        "--backbone-learning-rate",
        type=float,
        default=None,
        help="Override the backbone learning rate from config.py.",
    )
    parser.add_argument(
        "--classifier-lr-multiplier",
        type=float,
        default=None,
        help="Override the classifier/backbone learning-rate multiplier.",
    )
    parser.add_argument(
        "--recency-half-life-months",
        type=float,
        default=None,
        help=(
            "If positive, sample recent Stage 2 training images more often "
            "using this exponential half-life. Zero keeps uniform sampling."
        ),
    )
    parser.add_argument("--run-name", default=None)
    return parser.parse_args()


def add_rates(metrics):
    """Add operating-point rates needed to interpret class imbalance."""
    actual_nf = metrics["tn"] + metrics["fp"]
    actual_fl = metrics["tp"] + metrics["fn"]
    total = actual_nf + actual_fl
    metrics["false_positive_rate"] = metrics["fp"] / actual_nf if actual_nf else 0.0
    metrics["false_negative_rate"] = metrics["fn"] / actual_fl if actual_fl else 0.0
    metrics["predicted_fl_rate"] = (metrics["tp"] + metrics["fp"]) / total if total else 0.0
    return metrics


def set_batchnorm_evaluation_mode(model):
    """Stop unprotected BatchNorm running-statistics drift during Stage 2."""
    for module in model.modules():
        if isinstance(module, _BatchNorm):
            module.eval()


def build_optimizer(model, backbone_lr, classifier_lr, weight_decay):
    """Use selected Stage 1 differential learning rates and selective decay."""
    groups = {
        "backbone_decay": [],
        "backbone_no_decay": [],
        "classifier_decay": [],
        "classifier_no_decay": [],
    }
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        section = "classifier" if name.startswith("classifier.") else "backbone"
        decay_type = "no_decay" if name.endswith(".bias") or parameter.ndim == 1 else "decay"
        groups[f"{section}_{decay_type}"].append(parameter)

    return AdamW(
        [
            {"params": groups["backbone_decay"], "lr": backbone_lr, "weight_decay": weight_decay},
            {"params": groups["backbone_no_decay"], "lr": backbone_lr, "weight_decay": 0.0},
            {"params": groups["classifier_decay"], "lr": classifier_lr, "weight_decay": weight_decay},
            {"params": groups["classifier_no_decay"], "lr": classifier_lr, "weight_decay": 0.0},
        ]
    )


def evaluate(model, loader, criterion, device):
    """Evaluate a split without updating parameters or BatchNorm buffers."""
    model.eval()
    total_loss = 0.0
    number_of_images = 0
    predictions, targets = [], []

    with torch.no_grad():
        for images, batch_targets in loader:
            images = images.to(device, non_blocking=True)
            batch_targets = batch_targets.to(device, non_blocking=True)
            logits = model(images)
            loss = criterion(logits, batch_targets)
            total_loss += loss.item() * images.shape[0]
            number_of_images += images.shape[0]
            predictions.extend(logits.argmax(dim=1).cpu().tolist())
            targets.extend(batch_targets.cpu().tolist())

    metrics = calculate_classification_metrics(predictions, targets)
    metrics["loss"] = total_loss / number_of_images
    return add_rates(metrics)


def train_epoch(
    model,
    loader,
    criterion,
    optimizer,
    device,
    anchors,
    ewc_lambda,
    max_gradient_norm,
    freeze_batchnorm_running_stats,
):
    """Train Stage 2 while logging classification and EWC losses separately."""
    model.train()
    if freeze_batchnorm_running_stats:
        set_batchnorm_evaluation_mode(model)

    totals = {"total": 0.0, "classification": 0.0, "ewc": 0.0}
    number_of_images = 0
    predictions, targets = [], []

    for images, batch_targets in loader:
        images = images.to(device, non_blocking=True)
        batch_targets = batch_targets.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        logits = model(images)
        classification_loss = criterion(logits, batch_targets)
        # The lambda=0 baseline must be ordinary fine-tuning, including no
        # unnecessary EWC-penalty computation in its training loop.
        unscaled_penalty = (
            ewc_penalty(model, anchors)
            if anchors and ewc_lambda > 0
            else classification_loss.new_zeros(())
        )
        regularization_loss = 0.5 * ewc_lambda * unscaled_penalty
        total_loss = classification_loss + regularization_loss

        if not torch.isfinite(total_loss):
            raise FloatingPointError(
                "Non-finite loss before backward: "
                f"classification={classification_loss.item()}, "
                f"unscaled_ewc={unscaled_penalty.item()}, "
                f"lambda={ewc_lambda}, ewc={regularization_loss.item()}"
            )

        total_loss.backward()
        if max_gradient_norm is not None:
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=max_gradient_norm
            )
            if not torch.isfinite(gradient_norm):
                raise FloatingPointError("Non-finite gradient norm during Stage 2 training.")
        optimizer.step()

        batch_size = images.shape[0]
        totals["total"] += total_loss.detach().item() * batch_size
        totals["classification"] += classification_loss.detach().item() * batch_size
        totals["ewc"] += regularization_loss.detach().item() * batch_size
        number_of_images += batch_size
        predictions.extend(logits.argmax(dim=1).detach().cpu().tolist())
        targets.extend(batch_targets.detach().cpu().tolist())

    metrics = calculate_classification_metrics(predictions, targets)
    metrics.update(
        {
            "loss": totals["total"] / number_of_images,
            "classification_loss": totals["classification"] / number_of_images,
            "ewc_loss": totals["ewc"] / number_of_images,
        }
    )
    return add_rates(metrics)


def joint_css(stage1_metrics, stage2_metrics):
    """Return the cross-stage score for reporting, not checkpoint selection."""
    return math.sqrt(max(0.0, stage1_metrics["css"]) * max(0.0, stage2_metrics["css"]))


def cpu_state_dict(model):
    """Store a portable checkpoint without retaining GPU memory references."""
    return {name: value.detach().cpu() for name, value in model.state_dict().items()}


def save_checkpoint(path, model, optimizer, scheduler, epoch, config, metrics, anchor_cpu):
    """Write a selected Stage 2 checkpoint plus immutable Stage 1 EWC history."""
    torch.save(
        {
            # Only epochs that completed Stage 2 training are eligible.
            "completed_stage": 2,
            "stage2_adapted": True,
            "architecture": "mobilenet_v3_large",
            "experiment": "corrected_stage2_ewc",
            "epoch": epoch,
            "model_state_dict": cpu_state_dict(model),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "selection_metrics": metrics,
            "config": config,
            "ewc_history": anchor_cpu["ewc_history"],
            "stage1_fisher_metadata": anchor_cpu["fisher_metadata"],
        },
        path,
    )


def write_rows(path, rows):
    """Write progressively accumulated epoch metrics for inspection/recovery."""
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def log_curves(run, history):
    """Create comparison panels without using future validation in selection."""
    for metric in ("loss", "css", "tss", "hss"):
        run.log(
            {
                f"learning_curves/{metric}": wandb.plot.line_series(
                    xs=history["epoch"],
                    ys=[
                        history[f"stage2_train/{metric}"],
                        history[f"stage1_holdout/{metric}"],
                        history[f"stage2_holdout/{metric}"],
                    ],
                    keys=["Stage 2 training", "Stage 1 retention", "Stage 2 holdout"],
                    title=f"Stage 2 EWC: {metric.upper()} across epochs",
                    xname="Epoch",
                )
            }
        )


def main():
    args = parse_args()
    if args.ewc_lambda < 0:
        raise ValueError("--ewc-lambda must be non-negative.")
    if not args.anchor.is_file():
        raise FileNotFoundError(
            f"EWC anchor not found: {args.anchor}. Run prepare_stage1_fisher.py first."
        )

    config = dict(CONFIG)
    config["ewc_lambda"] = float(args.ewc_lambda)
    if args.epochs is not None:
        if args.epochs <= 0:
            raise ValueError("--epochs must be positive.")
        config["epochs"] = int(args.epochs)
    if args.recency_half_life_months is not None:
        if args.recency_half_life_months <= 0:
            raise ValueError("--recency-half-life-months must be positive.")
        config["recency_half_life_months"] = float(args.recency_half_life_months)
        config["sampling"] = "recency_weighted_full_training_data"
    for argument_name, config_name in (
        ("nf_class_weight", "nf_class_weight"),
        ("fl_class_weight", "fl_class_weight"),
        ("backbone_learning_rate", "backbone_learning_rate"),
        ("classifier_lr_multiplier", "classifier_lr_multiplier"),
    ):
        value = getattr(args, argument_name)
        if value is not None:
            if value <= 0:
                raise ValueError(f"--{argument_name.replace('_', '-')} must be positive.")
            config[config_name] = float(value)
    config["classifier_learning_rate"] = (
        float(config["backbone_learning_rate"])
        * float(config["classifier_lr_multiplier"])
    )

    seed = int(config["random_seed"])
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(0)

    anchor_cpu = torch.load(args.anchor, map_location="cpu", weights_only=False)
    required_anchor_keys = {"model_state_dict", "ewc_history", "fisher_metadata"}
    missing_anchor_keys = required_anchor_keys.difference(anchor_cpu)
    if missing_anchor_keys:
        raise ValueError(f"Invalid EWC anchor; missing {sorted(missing_anchor_keys)}")
    if len(anchor_cpu["ewc_history"]) != 1 or anchor_cpu["ewc_history"][0].get("stage") != 1:
        raise ValueError("Stage 2 requires exactly one Stage 1 EWC anchor.")

    model = build_model(
        pretrained=False,
        hidden_dim=int(config["classifier_hidden_dim"]),
        dropout=float(config["dropout"]),
    ).to(device)
    model.load_state_dict(anchor_cpu["model_state_dict"], strict=True)
    anchors = [move_ewc_anchor_to_device(anchor_cpu["ewc_history"][0], device)]

    paths = {
        "image_directory": IMAGE_DIR,
        "stage1_train_csv": STAGE1_TRAIN_CSV,
        "stage1_holdout_csv": STAGE1_HOLDOUT_CSV,
        "stage2_train_csv": STAGE2_TRAIN_CSV,
        "stage2_holdout_csv": STAGE2_HOLDOUT_CSV,
        "future_validation_csv": FUTURE_VALIDATION_CSV,
    }
    loaders, stage2_sampling_metadata = build_stage2_loaders(config, paths)
    stage1_counts = class_counts(loaders["stage1_fisher"].dataset)
    stage2_counts = class_counts(loaders["stage2_train"].dataset)

    class_weights = torch.tensor(
        [float(config["nf_class_weight"]), float(config["fl_class_weight"])],
        dtype=torch.float32,
        device=device,
    )
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = build_optimizer(
        model,
        float(config["backbone_learning_rate"]),
        float(config["classifier_learning_rate"]),
        float(config["weight_decay"]),
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

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = RESULTS_DIR / f"mobilenet_stage2_ewc_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=False)
    run = wandb.init(
        entity=WANDB_ENTITY,
        project=WANDB_PROJECT,
        name=args.run_name or f"noaa_stage2_ewc_lambda_{args.ewc_lambda:g}_{timestamp}",
        tags=WANDB_TAGS + (["ewc-baseline"] if args.ewc_lambda == 0 else ["ewc"]),
        job_type="stage2-ewc-training",
        dir=str(run_dir),
        config={
            **config,
            "source_ewc_anchor": str(args.anchor.resolve()),
            "source_stage1_selected_epoch": anchor_cpu["fisher_metadata"]["source_selected_epoch"],
            "stage1_fisher_statistics": anchor_cpu["fisher_metadata"]["statistics"],
            "stage1_train_nf": stage1_counts[0],
            "stage1_train_fl": stage1_counts[1],
            "stage2_train_nf": stage2_counts[0],
            "stage2_train_fl": stage2_counts[1],
            "stage2_sampling_metadata": stage2_sampling_metadata,
            "device": str(device),
            "gpu_count_used": 1 if device.type == "cuda" else 0,
            "multi_gpu_method": "none",
        },
    )
    config = dict(run.config)

    anchor_stage1 = evaluate(model, loaders["stage1_holdout"], criterion, device)
    anchor_stage2 = evaluate(model, loaders["stage2_holdout"], criterion, device)
    anchor_joint_css = joint_css(anchor_stage1, anchor_stage2)
    print(f"Device: {device} | GPUs=1")
    print(f"Results: {run_dir}")
    print(f"EWC anchor: {args.anchor}")
    print(f"EWC lambda: {config['ewc_lambda']:.6g}")
    print(f"Stage 1 anchor holdout CSS: {anchor_stage1['css']:.4f}")
    print(f"Before Stage 2 training | Stage 2 holdout CSS: {anchor_stage2['css']:.4f}")
    print(f"Epoch 0 joint CSS: {anchor_joint_css:.4f}")
    print("Epoch 0 is a reference baseline only; checkpoint selection starts at epoch 1.")
    print(
        "Stage 2 data: "
        f"NF={stage2_counts[0]:,}, FL={stage2_counts[1]:,}, total={sum(stage2_counts):,}"
    )
    print(f"Stage 2 sampling: {stage2_sampling_metadata}")

    run.define_metric("epoch")
    for group in ("stage2_train", "stage1_holdout", "stage2_holdout"):
        run.define_metric(f"{group}/*", step_metric="epoch")
    run.define_metric("best/*")
    run.define_metric("future_validation/*")
    initial_metrics = {
        "epoch": 0,
        "selection/stage2_css": anchor_stage2["css"],
        "selection/joint_css": anchor_joint_css,
        "selection/stage1_css_retention_ratio": 1.0,
        **{f"stage1_holdout/{key}": value for key, value in anchor_stage1.items()},
        **{f"stage2_holdout/{key}": value for key, value in anchor_stage2.items()},
        **{f"stage1_anchor/{key}": value for key, value in anchor_stage1.items()},
        **{f"stage2_before_training/{key}": value for key, value in anchor_stage2.items()},
    }
    run.log(initial_metrics)

    initial_path = run_dir / "initial_metrics.csv"
    with initial_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["metric", "value"])
        writer.writerows(
            (key, value)
            for key, value in initial_metrics.items()
            if key not in {"epoch"}
        )

    best_stage2_css = float("-inf")
    best_stage2_recall = float("-inf")
    best_joint_css = float("nan")
    best_epoch = None
    without_improvement = 0
    checkpoint_path = run_dir / "stage2_best.pt"
    rows = []
    history = {"epoch": []}
    for group in ("stage2_train", "stage1_holdout", "stage2_holdout"):
        for metric in ("loss", "css", "tss", "hss"):
            history[f"{group}/{metric}"] = []

    for epoch in range(1, int(config["epochs"]) + 1):
        started = time.perf_counter()
        stage2_train = train_epoch(
            model,
            loaders["stage2_train"],
            criterion,
            optimizer,
            device,
            anchors,
            float(config["ewc_lambda"]),
            float(config["max_gradient_norm"]),
            bool(config["freeze_batchnorm_running_stats"]),
        )
        stage1_holdout = evaluate(model, loaders["stage1_holdout"], criterion, device)
        stage2_holdout = evaluate(model, loaders["stage2_holdout"], criterion, device)
        stage2_selection_css = stage2_holdout["css"]
        stage2_css_gap = stage2_train["css"] - stage2_holdout["css"]
        diagnostic_joint_css = joint_css(stage1_holdout, stage2_holdout)
        scheduler.step(stage2_holdout["css"])

        previous_best_stage2_css = best_stage2_css
        checkpoint_improved = stage2_selection_css > best_stage2_css
        meaningful_improvement = stage2_selection_css > (
            previous_best_stage2_css + float(config["early_stopping_min_delta"])
        )
        if checkpoint_improved:
            best_stage2_css = stage2_selection_css
            best_stage2_recall = stage2_holdout["recall"]
            best_joint_css = diagnostic_joint_css
            best_epoch = epoch
            save_checkpoint(
                checkpoint_path,
                model,
                optimizer,
                scheduler,
                epoch,
                config,
                {
                    "selection_metric": "stage2_holdout_css",
                    "stage2_holdout_css": stage2_selection_css,
                    "stage2_holdout_recall": stage2_holdout["recall"],
                    "stage2_train_css": stage2_train["css"],
                    "stage2_train_holdout_css_gap": stage2_css_gap,
                    "joint_css": diagnostic_joint_css,
                    "stage1_css_retention_ratio": (
                        stage1_holdout["css"] / anchor_stage1["css"]
                        if anchor_stage1["css"] > 0
                        else 0.0
                    ),
                    "stage1_holdout": stage1_holdout,
                    "stage2_holdout": stage2_holdout,
                },
                anchor_cpu,
            )

        if meaningful_improvement:
            without_improvement = 0
        else:
            without_improvement += 1

        elapsed = time.perf_counter() - started
        logged = {
            "epoch": epoch,
            "selection/stage2_css": stage2_selection_css,
            "selection/joint_css": diagnostic_joint_css,
            "diagnostics/stage2_train_holdout_css_gap": stage2_css_gap,
            "selection/stage1_css_retention_ratio": (
                stage1_holdout["css"] / anchor_stage1["css"]
                if anchor_stage1["css"] > 0
                else 0.0
            ),
            "learning_rate/backbone": optimizer.param_groups[0]["lr"],
            "learning_rate/classifier": optimizer.param_groups[2]["lr"],
            "best/stage2_css": best_stage2_css,
            "best/stage2_recall": best_stage2_recall,
            "best/joint_css_at_selected_epoch": best_joint_css,
            "best/epoch": best_epoch,
            "epochs_without_improvement": without_improvement,
            "timing/epoch_seconds": elapsed,
            **{f"stage2_train/{key}": value for key, value in stage2_train.items()},
            **{f"stage1_holdout/{key}": value for key, value in stage1_holdout.items()},
            **{f"stage2_holdout/{key}": value for key, value in stage2_holdout.items()},
        }
        run.log(logged)
        rows.append({key.replace("/", "_"): value for key, value in logged.items()})
        write_rows(run_dir / "epoch_metrics.csv", rows)

        history["epoch"].append(epoch)
        for group, metrics in (
            ("stage2_train", stage2_train),
            ("stage1_holdout", stage1_holdout),
            ("stage2_holdout", stage2_holdout),
        ):
            for metric in ("loss", "css", "tss", "hss"):
                history[f"{group}/{metric}"].append(metrics[metric])

        print(f"Epoch {epoch:02d}/{int(config['epochs'])}")
        print(
            "  Stage 2 train   | "
            f"loss={stage2_train['loss']:.4f} | class={stage2_train['classification_loss']:.4f} | "
            f"EWC={stage2_train['ewc_loss']:.4f} | CSS={stage2_train['css']:.4f}"
        )
        print(
            "  Stage 1 retention | "
            f"CSS={stage1_holdout['css']:.4f} | recall={stage1_holdout['recall']:.4f} | "
            f"FPR={stage1_holdout['false_positive_rate']:.4f}"
        )
        print(
            "  Stage 2 holdout | "
            f"CSS={stage2_holdout['css']:.4f} | recall={stage2_holdout['recall']:.4f} | "
            f"FPR={stage2_holdout['false_positive_rate']:.4f}"
        )
        print(
            "  Progress | "
            f"Stage 2 CSS={stage2_selection_css:.4f} | joint CSS={diagnostic_joint_css:.4f} | "
            f"train-holdout gap={stage2_css_gap:.4f} | "
            f"best epoch={best_epoch} | best Stage 2 CSS={best_stage2_css:.4f} | "
            f"time={elapsed:.1f}s"
        )

        if without_improvement >= int(config["early_stopping_patience"]):
            print(f"Early stopping at epoch {epoch}; best epoch was {best_epoch}.")
            break

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    selected_stage1 = evaluate(model, loaders["stage1_holdout"], criterion, device)
    selected_stage2 = evaluate(model, loaders["stage2_holdout"], criterion, device)
    # This is the only future-validation evaluation for this run. It was never
    # used by training, the scheduler, early stopping, or checkpoint choice.
    future_validation = evaluate(model, loaders["future_validation"], criterion, device)

    final_metrics = {
        "selected_epoch": checkpoint["epoch"],
        "selected_stage2_css": checkpoint["selection_metrics"]["stage2_holdout_css"],
        "selected_stage2_recall": checkpoint["selection_metrics"]["stage2_holdout_recall"],
        "selected_stage2_train_css": checkpoint["selection_metrics"]["stage2_train_css"],
        "selected_stage2_train_holdout_css_gap": checkpoint["selection_metrics"][
            "stage2_train_holdout_css_gap"
        ],
        "selected_joint_css": checkpoint["selection_metrics"]["joint_css"],
        "stage2_css_delta_from_epoch0": selected_stage2["css"] - anchor_stage2["css"],
        "stage1_css_retention_ratio": (
            selected_stage1["css"] / anchor_stage1["css"]
            if anchor_stage1["css"] > 0
            else 0.0
        ),
        **{f"stage1_retention/{key}": value for key, value in selected_stage1.items()},
        **{f"stage2_holdout/{key}": value for key, value in selected_stage2.items()},
        **{f"future_validation/{key}": value for key, value in future_validation.items()},
    }
    run.log(final_metrics)
    for key, value in final_metrics.items():
        run.summary[key] = value
    log_curves(run, history)

    final_path = run_dir / "final_metrics.csv"
    with final_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["metric", "value"])
        writer.writerows(final_metrics.items())

    artifact = wandb.Artifact(
        f"noaa-stage2-ewc-{run.id}",
        type="model",
        metadata={
            "ewc_lambda": config["ewc_lambda"],
            "selected_epoch": checkpoint["epoch"],
            "selection": config["checkpoint_selection"],
        },
    )
    artifact.add_file(str(checkpoint_path))
    artifact.add_file(str(initial_path))
    artifact.add_file(str(run_dir / "epoch_metrics.csv"))
    artifact.add_file(str(final_path))
    run.log_artifact(artifact)

    print(f"Best epoch: {checkpoint['epoch']}")
    print(
        "Selection rule: maximum Stage 2 holdout CSS among adapted epochs; "
        "epoch 0 was reference-only."
    )
    print(
        "Selected Stage 1 retention | "
        f"CSS={selected_stage1['css']:.4f} | recall={selected_stage1['recall']:.4f} | "
        f"FPR={selected_stage1['false_positive_rate']:.4f}"
    )
    print(
        "Selected Stage 2 holdout | "
        f"CSS={selected_stage2['css']:.4f} | recall={selected_stage2['recall']:.4f} | "
        f"FPR={selected_stage2['false_positive_rate']:.4f}"
    )
    print(
        "Future validation (not used for selection) | "
        f"CSS={future_validation['css']:.4f} | recall={future_validation['recall']:.4f} | "
        f"FPR={future_validation['false_positive_rate']:.4f}"
    )
    print(f"Checkpoint: {checkpoint_path}")
    run.finish()


if __name__ == "__main__":
    main()
