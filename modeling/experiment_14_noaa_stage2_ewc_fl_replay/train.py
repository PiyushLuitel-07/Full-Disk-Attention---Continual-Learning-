"""Train NOAA Stage 2 with corrected EWC and Stage 1 FL-only replay.

Default run on physical GPU 1::

    CUDA_VISIBLE_DEVICES=1 python \
      modeling/experiment_14_noaa_stage2_ewc_fl_replay/train.py

Important settings can be overridden from the command line so controlled
ablation runs do not require source-code edits.
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
from torch.nn import functional as F
from torch.nn.modules.batchnorm import _BatchNorm
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau


SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.append(str(SCRIPT_DIR.parent))

from config import (  # noqa: E402
    CONFIG,
    DEFAULT_EWC_ANCHOR,
    FUTURE_VALIDATION_CSV,
    IMAGE_DIR,
    RESULTS_DIR,
    STAGE1_FL_REPLAY_CSV,
    STAGE1_HOLDOUT_CSV,
    STAGE1_TRAIN_CSV,
    STAGE2_HOLDOUT_CSV,
    STAGE2_TRAIN_CSV,
    WANDB_ENTITY,
    WANDB_PROJECT,
    WANDB_TAGS,
)
from dataloader import build_experiment_loaders, cycle_batches  # noqa: E402
from ewc import ewc_penalty, move_ewc_anchor_to_device  # noqa: E402
from evaluation import calculate_classification_metrics  # noqa: E402
from model import build_model  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--anchor",
        type=Path,
        default=DEFAULT_EWC_ANCHOR,
        help="Prepared Stage 1 model/Fisher anchor.",
    )
    parser.add_argument(
        "--replay-csv",
        type=Path,
        default=STAGE1_FL_REPLAY_CSV,
        help="Validated Stage 1 FL-only replay manifest.",
    )
    parser.add_argument(
        "--ewc-lambda",
        type=float,
        default=None,
        help="EWC strength. Default: value in config.py.",
    )
    parser.add_argument(
        "--total-batch-size",
        type=int,
        default=None,
        help="Maximum mixed batch size. Default: 64.",
    )
    parser.add_argument(
        "--replay-batch-size",
        type=int,
        default=None,
        help=(
            "Number of Stage 1 replay images in each mixed batch. The rest "
            "are Stage 2 images. Use 0 for a matched no-replay control."
        ),
    )
    parser.add_argument(
        "--replay-loss-weight",
        type=float,
        default=None,
        help="Per-replay-example loss multiplier. Default: 1.0.",
    )
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--random-seed", type=int, default=None)
    parser.add_argument("--early-stopping-patience", type=int, default=None)
    parser.add_argument(
        "--nf-class-weight",
        type=float,
        default=None,
        help="NF cross-entropy weight. Default: 1.0.",
    )
    parser.add_argument(
        "--fl-class-weight",
        type=float,
        default=None,
        help="FL cross-entropy weight. Default: 1.0.",
    )
    parser.add_argument(
        "--backbone-learning-rate",
        type=float,
        default=None,
        help="Backbone learning rate.",
    )
    parser.add_argument(
        "--classifier-lr-multiplier",
        type=float,
        default=None,
        help="Classifier/backbone learning-rate multiplier.",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=None,
        help="AdamW weight decay for non-bias, non-normalization weights.",
    )

    batchnorm_group = parser.add_mutually_exclusive_group()
    batchnorm_group.add_argument(
        "--adapt-batchnorm-running-stats",
        dest="freeze_batchnorm_running_stats",
        action="store_false",
        help="Update BatchNorm running statistics on mixed training batches.",
    )
    batchnorm_group.add_argument(
        "--freeze-batchnorm-running-stats",
        dest="freeze_batchnorm_running_stats",
        action="store_true",
        help="Keep BatchNorm running statistics fixed at the Stage 1 values.",
    )
    parser.set_defaults(freeze_batchnorm_running_stats=None)
    parser.add_argument("--run-name", default=None)
    return parser.parse_args()


def build_run_config(args):
    """Apply validated command-line overrides to the experiment defaults."""
    config = dict(CONFIG)

    scalar_overrides = (
        ("ewc_lambda", "ewc_lambda"),
        ("replay_loss_weight", "replay_loss_weight"),
        ("nf_class_weight", "nf_class_weight"),
        ("fl_class_weight", "fl_class_weight"),
        ("backbone_learning_rate", "backbone_learning_rate"),
        ("classifier_lr_multiplier", "classifier_lr_multiplier"),
        ("weight_decay", "weight_decay"),
        ("epochs", "epochs"),
        ("num_workers", "num_workers"),
        ("random_seed", "random_seed"),
        ("early_stopping_patience", "early_stopping_patience"),
        ("total_batch_size", "total_batch_size"),
        ("replay_batch_size", "replay_batch_size"),
    )
    for argument_name, config_name in scalar_overrides:
        value = getattr(args, argument_name)
        if value is not None:
            config[config_name] = value

    if float(config["ewc_lambda"]) < 0:
        raise ValueError("--ewc-lambda must be non-negative.")
    if int(config["total_batch_size"]) <= 0:
        raise ValueError("--total-batch-size must be positive.")
    if int(config["replay_batch_size"]) < 0:
        raise ValueError("--replay-batch-size cannot be negative.")
    if int(config["replay_batch_size"]) >= int(config["total_batch_size"]):
        raise ValueError(
            "--replay-batch-size must be smaller than --total-batch-size."
        )
    if float(config["replay_loss_weight"]) <= 0:
        raise ValueError("--replay-loss-weight must be positive.")
    if int(config["epochs"]) <= 0:
        raise ValueError("--epochs must be positive.")
    if int(config["num_workers"]) < 0:
        raise ValueError("--num-workers cannot be negative.")
    if int(config["early_stopping_patience"]) <= 0:
        raise ValueError("--early-stopping-patience must be positive.")
    for key in (
        "nf_class_weight",
        "fl_class_weight",
        "backbone_learning_rate",
        "classifier_lr_multiplier",
    ):
        if float(config[key]) <= 0:
            raise ValueError(f"{key} must be positive.")
    if float(config["weight_decay"]) < 0:
        raise ValueError("--weight-decay must be non-negative.")

    config["stage2_batch_size"] = (
        int(config["total_batch_size"]) - int(config["replay_batch_size"])
    )
    config["classifier_learning_rate"] = (
        float(config["backbone_learning_rate"])
        * float(config["classifier_lr_multiplier"])
    )

    if args.freeze_batchnorm_running_stats is not None:
        config["freeze_batchnorm_running_stats"] = bool(
            args.freeze_batchnorm_running_stats
        )
    config["batchnorm_running_stats_mode"] = (
        "frozen_at_stage1_anchor"
        if config["freeze_batchnorm_running_stats"]
        else "adapt_on_mixed_stage2_replay_batches"
    )

    config["replay_enabled"] = int(config["replay_batch_size"]) > 0
    return config


def add_rates(metrics):
    """Add FPR, FNR, and predicted-FL rate to classification metrics."""
    actual_nf = metrics["tn"] + metrics["fp"]
    actual_fl = metrics["tp"] + metrics["fn"]
    total = actual_nf + actual_fl
    metrics["false_positive_rate"] = (
        metrics["fp"] / actual_nf if actual_nf else 0.0
    )
    metrics["false_negative_rate"] = (
        metrics["fn"] / actual_fl if actual_fl else 0.0
    )
    metrics["predicted_fl_rate"] = (
        (metrics["tp"] + metrics["fp"]) / total if total else 0.0
    )
    return metrics


def set_batchnorm_evaluation_mode(model):
    """Freeze BatchNorm buffers while leaving affine parameters trainable."""
    for module in model.modules():
        if isinstance(module, _BatchNorm):
            module.eval()


def build_optimizer(model, backbone_lr, classifier_lr, weight_decay):
    """Build AdamW with differential LRs and selective weight decay."""
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
        decay_type = (
            "no_decay" if name.endswith(".bias") or parameter.ndim == 1 else "decay"
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


def evaluate(model, loader, criterion, device):
    """Evaluate without changing parameters or BatchNorm buffers."""
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

    if number_of_images == 0:
        raise ValueError("Cannot evaluate an empty loader.")
    metrics = calculate_classification_metrics(predictions, targets)
    metrics["loss"] = total_loss / number_of_images
    return add_rates(metrics)


def cross_entropy_parts(logits, targets, class_weights):
    """Return weighted-CE mean, numerator, and normalization weight.

    Keeping the numerator and denominator explicit lets replay loss weighting
    remain correct even when CLI class weights differ from 1.0.
    """
    per_example_loss = F.cross_entropy(
        logits,
        targets,
        weight=class_weights,
        reduction="none",
    )
    numerator = per_example_loss.sum()
    normalizer = class_weights[targets].sum()
    return numerator / normalizer, numerator, normalizer


def train_epoch(
    model,
    stage2_loader,
    replay_loader,
    criterion,
    optimizer,
    device,
    anchors,
    ewc_lambda,
    replay_loss_weight,
    max_gradient_norm,
    freeze_batchnorm_running_stats,
):
    """Train one epoch on mixed batches while reporting Stage 2 metrics only."""
    model.train()
    if freeze_batchnorm_running_stats:
        set_batchnorm_evaluation_mode(model)

    replay_iterator = cycle_batches(replay_loader) if replay_loader is not None else None
    totals = {
        "total": 0.0,
        "mixed_classification_numerator": 0.0,
        "mixed_classification_normalizer": 0.0,
        "stage2_classification_numerator": 0.0,
        "stage2_classification_normalizer": 0.0,
        "replay_classification_numerator": 0.0,
        "replay_classification_normalizer": 0.0,
        "ewc": 0.0,
    }
    stage2_image_count = 0
    replay_image_count = 0
    predictions, targets = [], []

    for stage2_images, stage2_targets in stage2_loader:
        stage2_images = stage2_images.to(device, non_blocking=True)
        stage2_targets = stage2_targets.to(device, non_blocking=True)
        current_count = stage2_images.shape[0]

        replay_count = 0
        if replay_iterator is not None:
            replay_images, replay_targets = next(replay_iterator)
            replay_images = replay_images.to(device, non_blocking=True)
            replay_targets = replay_targets.to(device, non_blocking=True)
            replay_count = replay_images.shape[0]
            mixed_images = torch.cat((stage2_images, replay_images), dim=0)
        else:
            mixed_images = stage2_images

        optimizer.zero_grad(set_to_none=True)
        mixed_logits = model(mixed_images)
        stage2_logits = mixed_logits[:current_count]
        stage2_loss, stage2_numerator, stage2_normalizer = cross_entropy_parts(
            stage2_logits,
            stage2_targets,
            criterion.weight,
        )

        if replay_count > 0:
            replay_logits = mixed_logits[current_count:]
            replay_loss, replay_numerator, replay_normalizer = cross_entropy_parts(
                replay_logits,
                replay_targets,
                criterion.weight,
            )
            mixed_numerator = stage2_numerator + replay_loss_weight * replay_numerator
            mixed_normalizer = (
                stage2_normalizer + replay_loss_weight * replay_normalizer
            )
            mixed_classification_loss = mixed_numerator / mixed_normalizer
        else:
            replay_loss = stage2_loss.new_zeros(())
            replay_numerator = stage2_loss.new_zeros(())
            replay_normalizer = stage2_loss.new_zeros(())
            mixed_numerator = stage2_numerator
            mixed_normalizer = stage2_normalizer
            mixed_classification_loss = stage2_loss

        unscaled_penalty = (
            ewc_penalty(model, anchors)
            if anchors and ewc_lambda > 0
            else mixed_classification_loss.new_zeros(())
        )
        regularization_loss = 0.5 * ewc_lambda * unscaled_penalty
        total_loss = mixed_classification_loss + regularization_loss

        if not torch.isfinite(total_loss):
            raise FloatingPointError(
                "Non-finite loss before backward: "
                f"stage2={stage2_loss.item()}, replay={replay_loss.item()}, "
                f"mixed={mixed_classification_loss.item()}, "
                f"unscaled_ewc={unscaled_penalty.item()}, "
                f"lambda={ewc_lambda}, ewc={regularization_loss.item()}"
            )

        total_loss.backward()
        if max_gradient_norm is not None:
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=max_gradient_norm
            )
            if not torch.isfinite(gradient_norm):
                raise FloatingPointError("Non-finite gradient norm during training.")
        optimizer.step()

        # Weight epoch-level optimization losses by the number of current
        # Stage 2 images. This remains comparable when the last current batch
        # is smaller. Replay loss itself is averaged over replay images.
        totals["total"] += total_loss.detach().item() * current_count
        totals["mixed_classification_numerator"] += mixed_numerator.detach().item()
        totals["mixed_classification_normalizer"] += mixed_normalizer.detach().item()
        totals["stage2_classification_numerator"] += stage2_numerator.detach().item()
        totals["stage2_classification_normalizer"] += stage2_normalizer.detach().item()
        totals["ewc"] += regularization_loss.detach().item() * current_count
        if replay_count > 0:
            totals["replay_classification_numerator"] += (
                replay_numerator.detach().item()
            )
            totals["replay_classification_normalizer"] += (
                replay_normalizer.detach().item()
            )

        stage2_image_count += current_count
        replay_image_count += replay_count
        predictions.extend(stage2_logits.argmax(dim=1).detach().cpu().tolist())
        targets.extend(stage2_targets.detach().cpu().tolist())

    if stage2_image_count != len(stage2_loader.dataset):
        raise RuntimeError(
            "Stage 2 sample count changed during the epoch: "
            f"{stage2_image_count} != {len(stage2_loader.dataset)}"
        )

    metrics = calculate_classification_metrics(predictions, targets)
    metrics.update(
        {
            "loss": totals["total"] / stage2_image_count,
            "classification_loss": (
                totals["mixed_classification_numerator"]
                / totals["mixed_classification_normalizer"]
            ),
            "stage2_classification_loss": (
                totals["stage2_classification_numerator"]
                / totals["stage2_classification_normalizer"]
            ),
            "replay_classification_loss": (
                totals["replay_classification_numerator"]
                / totals["replay_classification_normalizer"]
                if replay_image_count
                else 0.0
            ),
            "ewc_loss": totals["ewc"] / stage2_image_count,
            "stage2_images_seen": stage2_image_count,
            "replay_images_seen": replay_image_count,
        }
    )
    return add_rates(metrics)


def joint_css(stage1_metrics, stage2_metrics):
    """Return cross-stage CSS for diagnostics, not checkpoint selection."""
    return math.sqrt(
        max(0.0, stage1_metrics["css"]) * max(0.0, stage2_metrics["css"])
    )


def cpu_state_dict(model):
    """Create a portable CPU model state dictionary."""
    return {
        name: value.detach().cpu()
        for name, value in model.state_dict().items()
    }


def save_checkpoint(
    path,
    model,
    optimizer,
    scheduler,
    epoch,
    config,
    metrics,
    anchor_cpu,
    sampling_metadata,
):
    """Save the selected Stage 2 replay checkpoint and EWC history."""
    torch.save(
        {
            "completed_stage": 2,
            "stage2_adapted": True,
            "architecture": "mobilenet_v3_large",
            "experiment": "stage2_ewc_stage1_fl_replay",
            "epoch": epoch,
            "model_state_dict": cpu_state_dict(model),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "selection_metrics": metrics,
            "config": config,
            "sampling_metadata": sampling_metadata,
            "ewc_history": anchor_cpu["ewc_history"],
            "stage1_fisher_metadata": anchor_cpu["fisher_metadata"],
        },
        path,
    )


def write_rows(path, rows):
    """Write progressive epoch metrics for recovery and inspection."""
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def log_curves(run, history):
    """Create W&B comparison panels without future-validation leakage."""
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
                    keys=[
                        "Stage 2 training",
                        "Stage 1 retention",
                        "Stage 2 holdout",
                    ],
                    title=f"Stage 2 EWC + FL replay: {metric.upper()}",
                    xname="Epoch",
                )
            }
        )


def main():
    args = parse_args()
    config = build_run_config(args)

    if not args.anchor.is_file():
        raise FileNotFoundError(
            f"EWC anchor not found: {args.anchor}. "
            "Run Experiment 13 prepare_stage1_fisher.py first."
        )
    if config["replay_enabled"] and not args.replay_csv.is_file():
        raise FileNotFoundError(f"Replay CSV not found: {args.replay_csv}")

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
    if (
        len(anchor_cpu["ewc_history"]) != 1
        or anchor_cpu["ewc_history"][0].get("stage") != 1
    ):
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
        "replay_csv": args.replay_csv,
    }
    loaders, sampling_metadata = build_experiment_loaders(config, paths)

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
    run_dir = RESULTS_DIR / f"mobilenet_stage2_ewc_fl_replay_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=False)
    replay_tags = (
        ["replay", "stage1-fl-only-replay"]
        if config["replay_enabled"]
        else ["no-replay-control"]
    )
    run = wandb.init(
        entity=WANDB_ENTITY,
        project=WANDB_PROJECT,
        name=(
            args.run_name
            or f"noaa_stage2_ewc_fl_replay_{timestamp}"
        ),
        tags=(
            WANDB_TAGS
            + replay_tags
            + (
                ["batchnorm-stage1-frozen"]
                if config["freeze_batchnorm_running_stats"]
                else ["batchnorm-mixed-batch-adaptation"]
            )
        ),
        job_type="stage2-ewc-replay-training",
        dir=str(run_dir),
        config={
            **config,
            "source_ewc_anchor": str(args.anchor.resolve()),
            "source_replay_csv": (
                str(args.replay_csv.resolve()) if config["replay_enabled"] else None
            ),
            "source_stage1_selected_epoch": anchor_cpu["fisher_metadata"][
                "source_selected_epoch"
            ],
            "stage1_fisher_statistics": anchor_cpu["fisher_metadata"]["statistics"],
            "sampling_metadata": sampling_metadata,
            "device": str(device),
            "gpu_count_used": 1 if device.type == "cuda" else 0,
            "multi_gpu_method": "none",
        },
    )
    config = dict(run.config)

    anchor_stage1 = evaluate(model, loaders["stage1_holdout"], criterion, device)
    anchor_stage2 = evaluate(model, loaders["stage2_holdout"], criterion, device)
    anchor_joint_css = joint_css(anchor_stage1, anchor_stage2)

    print(f"Device: {device} | GPUs={1 if device.type == 'cuda' else 0}")
    print(f"Results: {run_dir}")
    print(f"EWC anchor: {args.anchor}")
    print(f"EWC lambda: {float(config['ewc_lambda']):.6g}")
    print(f"Replay enabled: {config['replay_enabled']}")
    if config["replay_enabled"]:
        print(f"Replay CSV: {args.replay_csv}")
    print(
        "Mixed batch: "
        f"Stage 2={config['stage2_batch_size']} | "
        f"replay={config['replay_batch_size']} | "
        f"maximum total={config['total_batch_size']}"
    )
    print(f"Replay loss weight: {float(config['replay_loss_weight']):.4f}")
    print(f"BatchNorm running statistics: {config['batchnorm_running_stats_mode']}")
    print(f"Sampling metadata: {sampling_metadata}")
    print(f"Stage 1 anchor holdout CSS: {anchor_stage1['css']:.4f}")
    print(f"Before Stage 2 training | Stage 2 holdout CSS: {anchor_stage2['css']:.4f}")
    print(f"Epoch 0 joint CSS: {anchor_joint_css:.4f}")
    print("Epoch 0 is reference-only; checkpoint selection starts at epoch 1.")

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
            if key != "epoch"
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
            model=model,
            stage2_loader=loaders["stage2_train"],
            replay_loader=loaders["stage1_replay"],
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            anchors=anchors,
            ewc_lambda=float(config["ewc_lambda"]),
            replay_loss_weight=float(config["replay_loss_weight"]),
            max_gradient_norm=float(config["max_gradient_norm"]),
            freeze_batchnorm_running_stats=bool(
                config["freeze_batchnorm_running_stats"]
            ),
        )
        stage1_holdout = evaluate(
            model, loaders["stage1_holdout"], criterion, device
        )
        stage2_holdout = evaluate(
            model, loaders["stage2_holdout"], criterion, device
        )

        stage2_selection_css = stage2_holdout["css"]
        stage2_css_gap = stage2_train["css"] - stage2_holdout["css"]
        diagnostic_joint_css = joint_css(stage1_holdout, stage2_holdout)
        scheduler.step(stage2_selection_css)

        previous_best = best_stage2_css
        checkpoint_improved = stage2_selection_css > best_stage2_css
        meaningful_improvement = stage2_selection_css > (
            previous_best + float(config["early_stopping_min_delta"])
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
                sampling_metadata,
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
            "  Stage 2 train | "
            f"loss={stage2_train['loss']:.4f} | "
            f"mixed class={stage2_train['classification_loss']:.4f} | "
            f"Stage 2 class={stage2_train['stage2_classification_loss']:.4f} | "
            f"replay class={stage2_train['replay_classification_loss']:.4f} | "
            f"EWC={stage2_train['ewc_loss']:.4f} | "
            f"CSS={stage2_train['css']:.4f}"
        )
        print(
            "  Images seen | "
            f"Stage 2={int(stage2_train['stage2_images_seen']):,} | "
            f"replay={int(stage2_train['replay_images_seen']):,}"
        )
        print(
            "  Stage 1 retention | "
            f"CSS={stage1_holdout['css']:.4f} | "
            f"recall={stage1_holdout['recall']:.4f} | "
            f"FPR={stage1_holdout['false_positive_rate']:.4f}"
        )
        print(
            "  Stage 2 holdout | "
            f"CSS={stage2_holdout['css']:.4f} | "
            f"recall={stage2_holdout['recall']:.4f} | "
            f"FPR={stage2_holdout['false_positive_rate']:.4f}"
        )
        print(
            "  Progress | "
            f"Stage 2 CSS={stage2_selection_css:.4f} | "
            f"joint CSS={diagnostic_joint_css:.4f} | "
            f"train-holdout gap={stage2_css_gap:.4f} | "
            f"best epoch={best_epoch} | "
            f"best Stage 2 CSS={best_stage2_css:.4f} | "
            f"time={elapsed:.1f}s"
        )

        if without_improvement >= int(config["early_stopping_patience"]):
            print(f"Early stopping at epoch {epoch}; best epoch was {best_epoch}.")
            break

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    selected_stage1 = evaluate(model, loaders["stage1_holdout"], criterion, device)
    selected_stage2 = evaluate(model, loaders["stage2_holdout"], criterion, device)
    future_validation = evaluate(
        model, loaders["future_validation"], criterion, device
    )

    final_metrics = {
        "selected_epoch": checkpoint["epoch"],
        "selected_stage2_css": checkpoint["selection_metrics"][
            "stage2_holdout_css"
        ],
        "selected_stage2_recall": checkpoint["selection_metrics"][
            "stage2_holdout_recall"
        ],
        "selected_stage2_train_css": checkpoint["selection_metrics"][
            "stage2_train_css"
        ],
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
        "replay/enabled": config["replay_enabled"],
        "replay/batch_size": config["replay_batch_size"],
        "replay/loss_weight": config["replay_loss_weight"],
        **{f"stage1_retention/{key}": value for key, value in selected_stage1.items()},
        **{f"stage2_holdout/{key}": value for key, value in selected_stage2.items()},
        **{
            f"future_validation/{key}": value
            for key, value in future_validation.items()
        },
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
        f"noaa-stage2-ewc-fl-replay-{run.id}",
        type="model",
        metadata={
            "ewc_lambda": config["ewc_lambda"],
            "replay_enabled": config["replay_enabled"],
            "replay_batch_size": config["replay_batch_size"],
            "selected_epoch": checkpoint["epoch"],
            "selection": config["checkpoint_selection"],
        },
    )
    artifact.add_file(str(checkpoint_path))
    artifact.add_file(str(initial_path))
    artifact.add_file(str(run_dir / "epoch_metrics.csv"))
    artifact.add_file(str(final_path))
    if config["replay_enabled"]:
        artifact.add_file(str(args.replay_csv))
    run.log_artifact(artifact)

    print(f"Best epoch: {checkpoint['epoch']}")
    print("Selection: maximum Stage 2 holdout CSS among adapted epochs.")
    print(
        "Selected Stage 1 retention | "
        f"CSS={selected_stage1['css']:.4f} | "
        f"recall={selected_stage1['recall']:.4f} | "
        f"FPR={selected_stage1['false_positive_rate']:.4f}"
    )
    print(
        "Selected Stage 2 holdout | "
        f"CSS={selected_stage2['css']:.4f} | "
        f"recall={selected_stage2['recall']:.4f} | "
        f"FPR={selected_stage2['false_positive_rate']:.4f}"
    )
    print(
        "Future validation (not used for selection) | "
        f"CSS={future_validation['css']:.4f} | "
        f"recall={future_validation['recall']:.4f} | "
        f"FPR={future_validation['false_positive_rate']:.4f}"
    )
    print(f"Checkpoint: {checkpoint_path}")
    run.finish()


if __name__ == "__main__":
    main()
