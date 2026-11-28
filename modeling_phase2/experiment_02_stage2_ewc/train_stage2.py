"""Train Phase 2 Stage 2 MobileNetV3-Large with EWC from the Stage 1 anchor.

Pipeline rules:
- Checkpoint selection within a run: Best V2 (Stage2_holdout) CSS across epoch >= 1.
- Hyperparameter / lambda tuning across runs: V1 + V2 (retention + adaptation).
- Test set (test_2025_2026_8m.csv): Evaluated once at Epoch 0 (zero-shot) and on final
  selected checkpoints; never used for tuning or checkpoint selection.
"""

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

import torch
import wandb
from torch import nn
from torch.nn.modules.batchnorm import _BatchNorm
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
STAGE1_DIR = ROOT / "modeling_phase2" / "experiment_01_stage1_full_ft"
sys.path[:0] = [str(HERE), str(STAGE1_DIR)]
sys.path.append(str(ROOT / "modeling_phase1"))

from dataloader import IMAGENET_MEAN, IMAGENET_STD, build_evaluation_loader, build_loaders  # noqa: E402
from evaluation import calculate_classification_metrics  # noqa: E402
from ewc import ewc_penalty, move_ewc_anchor_to_device  # noqa: E402
from model import build_model  # noqa: E402
from stage2_config import (  # noqa: E402
    CONFIG,
    DEFAULT_ANCHOR,
    IMAGE_DIR,
    RESULTS_DIR,
    STAGE1_HOLDOUT_CSV,
    STAGE2_HOLDOUT_CSV,
    STAGE2_TRAIN_CSV,
    TEST_CSV,
    WANDB_ENTITY,
    WANDB_PROJECT,
    WANDB_TAGS,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--anchor", type=Path, default=DEFAULT_ANCHOR)
    parser.add_argument("--ewc-lambda", type=float, default=None)
    parser.add_argument("--sampling", choices=["full_training_data", "cyclic_balanced"], default=None)
    parser.add_argument("--augmentation", choices=["none", "both_flip_rot5", "fl_flip_rot5"], default=None)
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
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--scheduler-patience", type=int, default=None)
    parser.add_argument("--scheduler-factor", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--freeze-bn", action="store_true", help="Freeze BatchNorm running stats at Stage 1 values.")
    parser.add_argument("--skip-wandb", action="store_true")
    parser.add_argument("--run-name", type=str, default=None)
    return parser.parse_args()


def resolve_config(base_config, anchor, args):
    cfg = dict(base_config)
    source = anchor.get("source_stage1_config", {})
    for key in (
        "model",
        "classifier_hidden_dim",
        "dropout",
        "bn_momentum",
        "freeze_backbone",
        "trainable_backbone_blocks",
        "image_size",
        "normalization",
    ):
        if key in source:
            cfg[key] = source[key]
    cfg["mean"] = list(source.get("mean", IMAGENET_MEAN))
    cfg["std"] = list(source.get("std", IMAGENET_STD))

    overrides = {
        "ewc_lambda": args.ewc_lambda,
        "sampling": args.sampling,
        "augmentation": args.augmentation,
        "loss": args.loss,
        "nf_class_weight": args.nf_weight,
        "backbone_learning_rate": args.backbone_lr,
        "classifier_learning_rate": args.classifier_lr,
        "weight_decay": args.weight_decay,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "scheduler_patience": args.scheduler_patience,
        "scheduler_factor": args.scheduler_factor,
        "random_seed": args.seed,
        "num_workers": args.num_workers,
    }
    for key, val in overrides.items():
        if val is not None:
            cfg[key] = val

    if args.fl_weight is not None:
        cfg["fl_class_weight"] = args.fl_weight
        if args.loss is None:
            cfg["loss"] = "mild_weighted_cross_entropy"
    if args.freeze_bn:
        cfg["adapt_batchnorm_running_stats"] = False
    if cfg["ewc_lambda"] < 0:
        raise ValueError("ewc_lambda must be non-negative.")
    return cfg


def setup_logger(log_path):
    logger = logging.getLogger("phase2_stage2_ewc")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    for handler in (logging.FileHandler(log_path, encoding="utf-8"), logging.StreamHandler(sys.stdout)):
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


def add_rates(metrics):
    total = metrics["tp"] + metrics["tn"] + metrics["fp"] + metrics["fn"]
    neg = metrics["tn"] + metrics["fp"]
    pos = metrics["tp"] + metrics["fn"]
    metrics["predicted_fl_rate"] = (metrics["tp"] + metrics["fp"]) / total if total else 0.0
    metrics["false_positive_rate"] = metrics["fp"] / neg if neg else 0.0
    metrics["false_negative_rate"] = metrics["fn"] / pos if pos else 0.0
    return metrics


def format_metrics(label, m):
    return (
        f"  {label:<15} | Loss={m['loss']:.4f} | "
        f"CSS={m['css']:.4f} TSS={m['tss']:.4f} HSS={m['hss']:.4f} F1={m['f1']:.4f} | "
        f"Prec={m['precision']:.4f} Rec={m['recall']:.4f} FPR={m['false_positive_rate']:.4f}"
    )


def compute_class_weights(targets, cfg):
    fl_count = sum(1 for y in targets if y == 1)
    nf_count = len(targets) - fl_count
    if cfg["loss"] == "ordinary_cross_entropy":
        return 1.0, 1.0, nf_count, fl_count
    if cfg["loss"] == "mild_weighted_cross_entropy":
        fl_w = float(cfg["fl_class_weight"] if cfg["fl_class_weight"] is not None else 1.5)
        return float(cfg["nf_class_weight"]), fl_w, nf_count, fl_count
    fl_w = float(cfg["fl_class_weight"]) if cfg["fl_class_weight"] is not None else math.sqrt(nf_count / fl_count)
    return float(cfg["nf_class_weight"]), fl_w, nf_count, fl_count


def build_optimizer(model, cfg):
    groups = {(part, wd): [] for part in ("backbone", "classifier") for wd in (1, 0)}
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        part = "classifier" if name.startswith("classifier.") else "backbone"
        decay = 0 if (name.endswith(".bias") or param.ndim == 1) else 1
        groups[(part, decay)].append(param)
    lrs = {"backbone": cfg["backbone_learning_rate"], "classifier": cfg["classifier_learning_rate"]}
    return AdamW([
        {"params": groups[(part, wd)], "lr": lrs[part], "weight_decay": cfg["weight_decay"] if wd else 0.0}
        for part in ("backbone", "classifier")
        for wd in (1, 0)
        if groups[(part, wd)]
    ])


def evaluate(model, loader, criterion, device):
    model.eval()
    loss_sum, preds, truths = 0.0, [], []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            logits = model(images)
            loss_sum += criterion(logits, labels).item() * labels.size(0)
            preds.extend(logits.argmax(dim=1).cpu().tolist())
            truths.extend(labels.cpu().tolist())
    metrics = calculate_classification_metrics(preds, truths)
    metrics["loss"] = loss_sum / len(truths)
    return add_rates(metrics)


def train_one_epoch(model, loader, criterion, optimizer, device, anchors, cfg):
    model.train()
    if not cfg["adapt_batchnorm_running_stats"]:
        for module in model.modules():
            if isinstance(module, _BatchNorm):
                module.eval()

    total_sum = task_sum = raw_ewc_sum = scaled_ewc_sum = grad_norm_sum = 0.0
    max_grad_norm = 0.0
    count = batches = 0
    preds, truths = [], []

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        logits = model(images)
        task_loss = criterion(logits, labels)
        raw_ewc = ewc_penalty(model, anchors) if cfg["ewc_lambda"] > 0 else task_loss.new_zeros(())
        scaled_ewc = 0.5 * cfg["ewc_lambda"] * raw_ewc
        total_loss = task_loss + scaled_ewc

        if not torch.isfinite(total_loss):
            raise FloatingPointError(
                f"Non-finite loss: task={task_loss.item()}, raw_ewc={raw_ewc.item()}, lambda={cfg['ewc_lambda']}"
            )
        total_loss.backward()

        if cfg["max_gradient_norm"] is not None:
            gnorm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["max_gradient_norm"]))
        else:
            gnorms = [p.grad.detach().norm(2) for p in model.parameters() if p.grad is not None]
            gnorm = float(torch.stack(gnorms).norm(2)) if gnorms else 0.0
        if not math.isfinite(gnorm):
            raise FloatingPointError("Non-finite gradient norm")

        optimizer.step()

        bs = labels.size(0)
        count += bs
        batches += 1
        total_sum += total_loss.detach().item() * bs
        task_sum += task_loss.detach().item() * bs
        raw_ewc_sum += raw_ewc.detach().item() * bs
        scaled_ewc_sum += scaled_ewc.detach().item() * bs
        grad_norm_sum += gnorm
        max_grad_norm = max(max_grad_norm, gnorm)
        preds.extend(logits.argmax(dim=1).detach().cpu().tolist())
        truths.extend(labels.cpu().tolist())

    metrics = calculate_classification_metrics(preds, truths)
    metrics.update({
        "loss": total_sum / count,
        "task_loss": task_sum / count,
        "raw_ewc_penalty": raw_ewc_sum / count,
        "ewc_loss": scaled_ewc_sum / count,
        "mean_grad_norm": grad_norm_sum / batches,
        "max_grad_norm": max_grad_norm,
    })
    return add_rates(metrics)


def save_checkpoint(path, model, epoch, v1_m, v2_m, train_m, anchor, cfg):
    torch.save(
        {
            "completed_stage": 2,
            "architecture": cfg.get("model", "mobilenet_v3_large"),
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "v1_metrics": v1_m,
            "v2_metrics": v2_m,
            "train_metrics": train_m,
            "config": cfg,
            "ewc_history": anchor["ewc_history"],
        },
        path,
    )


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()
    if not args.anchor.is_file():
        raise FileNotFoundError(f"Stage 1 EWC anchor not found: {args.anchor}")

    anchor = torch.load(args.anchor, map_location="cpu", weights_only=False)
    cfg = resolve_config(CONFIG, anchor, args)

    random.seed(cfg["random_seed"])
    torch.manual_seed(cfg["random_seed"])
    torch.cuda.manual_seed_all(cfg["random_seed"])

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(0)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    lam_tag = f"lam{cfg['ewc_lambda']:g}"
    suffix = f"_{args.run_name}" if args.run_name else f"_{lam_tag}"
    run_name = f"phase2_stage2_ewc_{timestamp}{suffix}"
    run_dir = RESULTS_DIR / f"mobilenet_stage2_ewc_{timestamp}{suffix}"
    run_dir.mkdir(parents=True, exist_ok=False)

    logger = setup_logger(run_dir / "training.log")
    mean, std = tuple(cfg["mean"]), tuple(cfg["std"])

    loaders = build_loaders(
        STAGE2_TRAIN_CSV,
        STAGE2_HOLDOUT_CSV,
        IMAGE_DIR,
        batch_size=cfg["batch_size"],
        num_workers=cfg["num_workers"],
        seed=cfg["random_seed"],
        mean=mean,
        std=std,
        augmentation=cfg["augmentation"],
        sampling=cfg["sampling"],
    )
    v1_loader = build_evaluation_loader(STAGE1_HOLDOUT_CSV, IMAGE_DIR, cfg["batch_size"], cfg["num_workers"], mean=mean, std=std)
    test_loader = build_evaluation_loader(TEST_CSV, IMAGE_DIR, cfg["batch_size"], cfg["num_workers"], mean=mean, std=std)

    nf_w, fl_w, nf_cnt, fl_cnt = compute_class_weights(loaders["targets"], cfg)
    cfg["fl_class_weight"] = fl_w
    cfg["stage2_train_samples"] = len(loaders["targets"])
    cfg["stage2_fl_samples"] = fl_cnt
    cfg["stage2_nf_samples"] = nf_cnt

    with (run_dir / "config.json").open("w", encoding="utf-8") as file:
        json.dump(cfg, file, indent=2)

    model = build_model(
        pretrained=False,
        freeze_backbone=bool(cfg.get("freeze_backbone", False)),
        trainable_backbone_blocks=int(cfg.get("trainable_backbone_blocks", 2)),
        hidden_dim=int(cfg.get("classifier_hidden_dim", 1280)),
        dropout=float(cfg.get("dropout", 0.2)),
        bn_momentum=cfg.get("bn_momentum", 0.1),
    ).to(device)
    model.load_state_dict(anchor["model_state_dict"], strict=True)

    anchors = [move_ewc_anchor_to_device(item, device) for item in anchor["ewc_history"]]
    criterion = nn.CrossEntropyLoss(weight=torch.tensor([nf_w, fl_w], dtype=torch.float32, device=device))
    optimizer = build_optimizer(model, cfg)
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=float(cfg["scheduler_factor"]),
        patience=int(cfg["scheduler_patience"]),
        min_lr=[cfg["min_backbone_learning_rate"], cfg["min_backbone_learning_rate"],
                cfg["min_classifier_learning_rate"], cfg["min_classifier_learning_rate"]],
    )

    run = None
    if not args.skip_wandb:
        run = wandb.init(
            entity=WANDB_ENTITY,
            project=WANDB_PROJECT,
            name=run_name,
            dir=str(run_dir),
            tags=WANDB_TAGS + [lam_tag],
            config=cfg,
        )
        run.define_metric("epoch")
        for g in ("stage2_train", "v1_holdout", "v2_holdout", "joint"):
            run.define_metric(f"{g}/*", step_metric="epoch")

    logger.info("=" * 85)
    logger.info(f"Phase 2 Stage 2 EWC | Run Dir: {run_dir}")
    logger.info(f"Anchor: {args.anchor}")
    logger.info(f"Device: {device} | Seed: {cfg['random_seed']} | EWC lambda={cfg['ewc_lambda']:g} | Adapt BN={cfg['adapt_batchnorm_running_stats']}")
    logger.info(f"Optimizer: adamw | Backbone LR={cfg['backbone_learning_rate']} | Classifier LR={cfg['classifier_learning_rate']} | WD={cfg['weight_decay']}")
    logger.info(f"Data: Stage2_train={len(loaders['targets']):,} (FL={fl_cnt:,}, NF={nf_cnt:,}) | V1={len(v1_loader.dataset):,} | V2={len(loaders['holdout'].dataset):,} | Test={len(test_loader.dataset):,}")
    logger.info(f"Loss: {cfg['loss']} (NF={nf_w:.2f}, FL={fl_w:.4f}) | Epochs={cfg['epochs']}")
    logger.info("=" * 85)

    # Epoch 0: Zero-shot Stage 1 baseline on V1, V2, and Test
    v1_ep0 = evaluate(model, v1_loader, criterion, device)
    v2_ep0 = evaluate(model, loaders["holdout"], criterion, device)
    test_ep0 = evaluate(model, test_loader, criterion, device)
    logger.info("Epoch 00 (Stage 1 Zero-Shot Baseline):")
    logger.info(format_metrics("V1 Holdout", v1_ep0))
    logger.info(format_metrics("V2 Holdout", v2_ep0))
    logger.info(format_metrics("Test (8m)", test_ep0))

    best_v2_ckpt = run_dir / "stage2_best_v2.pt"
    best_joint_ckpt = run_dir / "stage2_best_joint.pt"
    last_ckpt = run_dir / "stage2_last.pt"
    best_v2_css = float("-inf")
    best_v2_ep = 0
    best_joint_css = float("-inf")
    best_joint_ep = 0
    epoch_rows = []

    for epoch in range(1, int(cfg["epochs"]) + 1):
        t0 = time.perf_counter()
        train_m = train_one_epoch(model, loaders["train"], criterion, optimizer, device, anchors, cfg)
        v1_m = evaluate(model, v1_loader, criterion, device)
        v2_m = evaluate(model, loaders["holdout"], criterion, device)
        elapsed = time.perf_counter() - t0

        blr = optimizer.param_groups[0]["lr"]
        clr = optimizer.param_groups[-1]["lr"]
        scheduler.step(v2_m["css"])
        new_blr = optimizer.param_groups[0]["lr"]
        new_clr = optimizer.param_groups[-1]["lr"]

        mean_v1_v2 = 0.5 * (v1_m["css"] + v2_m["css"])
        harm_v1_v2 = (2 * max(0.0, v1_m["css"]) * max(0.0, v2_m["css"])) / (max(0.0, v1_m["css"]) + max(0.0, v2_m["css"])) if (v1_m["css"] > 0 and v2_m["css"] > 0) else 0.0
        v1_ret = v1_m["css"] / v1_ep0["css"] if v1_ep0["css"] > 0 else 0.0

        saved = []
        if v2_m["css"] > best_v2_css:
            best_v2_css = v2_m["css"]
            best_v2_ep = epoch
            save_checkpoint(best_v2_ckpt, model, epoch, v1_m, v2_m, train_m, anchor, cfg)
            saved.append("best_v2")
        if harm_v1_v2 > best_joint_css:
            best_joint_css = harm_v1_v2
            best_joint_ep = epoch
            save_checkpoint(best_joint_ckpt, model, epoch, v1_m, v2_m, train_m, anchor, cfg)
            saved.append("best_joint")
        if epoch == int(cfg["epochs"]):
            save_checkpoint(last_ckpt, model, epoch, v1_m, v2_m, train_m, anchor, cfg)
            saved.append("last_epoch")

        logger.info(f"Epoch {epoch:02d}/{cfg['epochs']} | BLR={blr:.2e} CLR={clr:.2e} | Time={elapsed:.1f}s")
        logger.info(
            f"  Stage2 Train    | Total={train_m['loss']:.4f} Task={train_m['task_loss']:.4f} "
            f"EWC={train_m['ewc_loss']:.4f} (raw={train_m['raw_ewc_penalty']:.4f}) | "
            f"CSS={train_m['css']:.4f} Prec={train_m['precision']:.4f} Rec={train_m['recall']:.4f}"
        )
        logger.info(format_metrics("V1 Retention", v1_m))
        logger.info(format_metrics("V2 Holdout", v2_m))
        logger.info(f"  Joint V1+V2     | Mean={mean_v1_v2:.4f} Harmonic={harm_v1_v2:.4f} V1_RetRatio={v1_ret:.3f}")
        if saved:
            logger.info(f"  [SAVED] {', '.join(saved)} (Best V2={best_v2_css:.4f}@ep{best_v2_ep}, Best Joint={best_joint_css:.4f}@ep{best_joint_ep})")
        if new_blr < blr or new_clr < clr:
            logger.info(f"  [LR DECAY] Backbone {blr:.2e} -> {new_blr:.2e} | Classifier {clr:.2e} -> {new_clr:.2e}")

        row = {
            "epoch": epoch,
            "backbone_lr": blr,
            "classifier_lr": clr,
            "mean_v1_v2_css": mean_v1_v2,
            "harmonic_v1_v2_css": harm_v1_v2,
            "v1_retention_ratio": v1_ret,
            **{f"stage2_train_{k}": v for k, v in train_m.items()},
            **{f"v1_{k}": v for k, v in v1_m.items()},
            **{f"v2_{k}": v for k, v in v2_m.items()},
        }
        epoch_rows.append(row)
        write_csv(run_dir / "epoch_metrics.csv", epoch_rows)

        if run is not None:
            run.log({
                "epoch": epoch,
                "learning_rate/backbone": blr,
                "learning_rate/classifier": clr,
                "joint/mean_v1_v2_css": mean_v1_v2,
                "joint/harmonic_v1_v2_css": harm_v1_v2,
                "joint/v1_retention_ratio": v1_ret,
                **{f"stage2_train/{k}": v for k, v in train_m.items()},
                **{f"v1_holdout/{k}": v for k, v in v1_m.items()},
                **{f"v2_holdout/{k}": v for k, v in v2_m.items()},
            })

    logger.info("=" * 85)
    logger.info("FINAL CHECKPOINT EVALUATIONS (V1, V2, and Untouched 8-Month Test Set)")
    final_rows = [
        {"checkpoint": "epoch0_stage1", "epoch": 0, "split": "v1_holdout", **v1_ep0},
        {"checkpoint": "epoch0_stage1", "epoch": 0, "split": "v2_holdout", **v2_ep0},
        {"checkpoint": "epoch0_stage1", "epoch": 0, "split": "test", **test_ep0},
    ]
    for label, path in [("best_v2", best_v2_ckpt), ("best_joint", best_joint_ckpt), ("last_epoch", last_ckpt)]:
        if not path.is_file():
            continue
        ckpt = torch.load(path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"], strict=True)
        ep = ckpt["epoch"]
        v1_f = evaluate(model, v1_loader, criterion, device)
        v2_f = evaluate(model, loaders["holdout"], criterion, device)
        test_f = evaluate(model, test_loader, criterion, device)
        logger.info(f"Checkpoint [{label}] (Epoch {ep}):")
        logger.info(format_metrics("V1 Holdout", v1_f))
        logger.info(format_metrics("V2 Holdout", v2_f))
        logger.info(format_metrics("Test (8m)", test_f))
        final_rows.extend([
            {"checkpoint": label, "epoch": ep, "split": "v1_holdout", **v1_f},
            {"checkpoint": label, "epoch": ep, "split": "v2_holdout", **v2_f},
            {"checkpoint": label, "epoch": ep, "split": "test", **test_f},
        ])
        if run is not None:
            run.log({
                **{f"final_{label}_v1/{k}": v for k, v in v1_f.items()},
                **{f"final_{label}_v2/{k}": v for k, v in v2_f.items()},
                **{f"final_{label}_test/{k}": v for k, v in test_f.items()},
            })

    write_csv(run_dir / "final_metrics.csv", final_rows)
    if run is not None:
        run.finish()


if __name__ == "__main__":
    main()
