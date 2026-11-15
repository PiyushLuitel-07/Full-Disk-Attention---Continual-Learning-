"""Confirm three fixed Stage 2 EWC candidates across the Stage 1 seeds."""

import csv
import json
import math
import random
import sys
import time
from datetime import datetime

import torch
import wandb
from torch import nn
from torch.nn.modules.batchnorm import _BatchNorm
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau

from config import (
    ANCHOR, CANDIDATES, IMAGE_DIR, RESULTS_DIR, ROOT, STAGE1_HOLDOUT_CSV,
    STAGE1_TRAIN_CSV, STAGE2_HOLDOUT_CSV, STAGE2_TRAIN_CSV, TEST_CSV,
    VALIDATION_CSV, WANDB_ENTITY, WANDB_PROJECT, WANDB_TAGS,
)

sys.path.insert(0, str(ROOT))

from modeling.evaluation import calculate_classification_metrics
from modeling.experiment_13_noaa_stage2_ewc.dataloader import build_loader, build_stage2_loaders, class_counts
from modeling.experiment_13_noaa_stage2_ewc.ewc import ewc_penalty, move_ewc_anchor_to_device
from modeling.experiment_13_noaa_stage2_ewc.model import build_model


def rates(metrics):
    nf = metrics["tn"] + metrics["fp"]
    fl = metrics["tp"] + metrics["fn"]
    metrics.update(
        false_positive_rate=metrics["fp"] / nf if nf else 0.0,
        false_negative_rate=metrics["fn"] / fl if fl else 0.0,
        predicted_fl_rate=(metrics["tp"] + metrics["fp"]) / (nf + fl) if nf + fl else 0.0,
    )
    return metrics


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, count, predictions, targets = 0.0, 0, [], []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            logits = model(images)
            total_loss += criterion(logits, labels).item() * len(images)
            count += len(images)
            predictions.extend(logits.argmax(1).cpu().tolist())
            targets.extend(labels.cpu().tolist())
    metrics = calculate_classification_metrics(predictions, targets)
    metrics["loss"] = total_loss / count
    return rates(metrics)


def build_optimizer(model, config):
    groups = {(part, decay): [] for part in ("backbone", "classifier") for decay in (0, 1)}
    for name, parameter in model.named_parameters():
        part = "classifier" if name.startswith("classifier.") else "backbone"
        groups[part, int(not (name.endswith(".bias") or parameter.ndim == 1))].append(parameter)
    learning_rates = {
        "backbone": float(config["backbone_learning_rate"]),
        "classifier": float(config["classifier_learning_rate"]),
    }
    return AdamW([
        {
            "params": groups[part, decay],
            "lr": learning_rates[part],
            "weight_decay": float(config["weight_decay"]) if decay else 0.0,
        }
        for part in ("backbone", "classifier") for decay in (1, 0)
    ])


def train_epoch(model, loader, criterion, optimizer, device, anchors, config):
    model.train()
    if not config["adapt_batchnorm_running_stats"]:
        for module in model.modules():
            if isinstance(module, _BatchNorm):
                module.eval()
    sums = {"total": 0.0, "classification": 0.0, "ewc": 0.0}
    count, predictions, targets = 0, [], []
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        classification = criterion(logits, labels)
        regularization = 0.5 * float(config["ewc_lambda"]) * ewc_penalty(model, anchors)
        loss = classification + regularization
        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite Stage 2 loss")
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float(config["max_gradient_norm"]))
        if not torch.isfinite(norm):
            raise FloatingPointError("Non-finite gradient norm")
        optimizer.step()
        n = len(images)
        count += n
        for key, value in (("total", loss), ("classification", classification), ("ewc", regularization)):
            sums[key] += value.detach().item() * n
        predictions.extend(logits.argmax(1).detach().cpu().tolist())
        targets.extend(labels.cpu().tolist())
    metrics = calculate_classification_metrics(predictions, targets)
    metrics.update(
        loss=sums["total"] / count,
        classification_loss=sums["classification"] / count,
        ewc_loss=sums["ewc"] / count,
    )
    return rates(metrics)


def prefixed(name, metrics):
    return {f"{name}/{key}": value for key, value in metrics.items()}


def write_metrics(path, metrics):
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["metric", "value"])
        writer.writerows(metrics.items())


def write_epochs(path, rows):
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)


def resolved_configuration(run):
    selection = dict(run.config)
    candidate = str(selection["candidate_id"])
    seed = int(selection["random_seed"])
    if candidate not in CANDIDATES:
        raise ValueError(f"Unknown candidate: {candidate}")
    source_path = CANDIDATES[candidate]
    if not source_path.is_file():
        raise FileNotFoundError(f"Missing source checkpoint: {source_path}")
    source = torch.load(source_path, map_location="cpu", weights_only=False)
    config = dict(source["config"])
    config.update(
        experiment="noaa_stage2_ewc_seed_confirmation",
        candidate_id=candidate,
        source_candidate_checkpoint=str(source_path.resolve()),
        source_candidate_epoch=int(source["epoch"]),
        random_seed=seed,
    )
    config["classifier_learning_rate"] = (
        float(config["backbone_learning_rate"]) * float(config["classifier_lr_multiplier"])
    )
    run.config.update(config, allow_val_change=True)
    return candidate, seed, config


def main():
    run = wandb.init(
        entity=WANDB_ENTITY,
        project=WANDB_PROJECT,
        config={"candidate_id": "070421", "random_seed": 42},
        tags=WANDB_TAGS,
        job_type="stage2-ewc-seed-confirmation",
    )
    candidate, seed, config = resolved_configuration(run)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output = RESULTS_DIR / f"candidate_{candidate}_seed_{seed}_{stamp}"
    output.mkdir(parents=True, exist_ok=False)
    run.name = f"noaa_stage2_exp16_{candidate}_seed{seed}_{stamp}"
    run.config.update({"results_directory": str(output.resolve())}, allow_val_change=True)
    with (output / "resolved_config.json").open("w", encoding="utf-8") as file:
        json.dump(config, file, indent=2, default=str, sort_keys=True)

    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(0)

    if not ANCHOR.is_file():
        raise FileNotFoundError(f"Missing EWC anchor: {ANCHOR}")
    anchor = torch.load(ANCHOR, map_location="cpu", weights_only=False)
    if len(anchor["ewc_history"]) != 1:
        raise ValueError("Expected exactly one Stage 1 EWC anchor")
    model = build_model(pretrained=False, hidden_dim=int(config["classifier_hidden_dim"]), dropout=0.0).to(device)
    model.load_state_dict(anchor["model_state_dict"], strict=True)
    anchors = [move_ewc_anchor_to_device(anchor["ewc_history"][0], device)]
    with torch.no_grad():
        initial_penalty = ewc_penalty(model, anchors).item()
    if not math.isfinite(initial_penalty) or abs(initial_penalty) > 1e-10:
        raise ValueError(f"Anchor/model mismatch: initial EWC penalty={initial_penalty}")

    paths = {
        "image_directory": IMAGE_DIR,
        "stage1_train_csv": STAGE1_TRAIN_CSV,
        "stage1_holdout_csv": STAGE1_HOLDOUT_CSV,
        "stage2_train_csv": STAGE2_TRAIN_CSV,
        "stage2_holdout_csv": STAGE2_HOLDOUT_CSV,
        "future_validation_csv": VALIDATION_CSV,
    }
    loaders, sampling = build_stage2_loaders(config, paths)
    loaders["test"] = build_loader(
        TEST_CSV, IMAGE_DIR, int(config["batch_size"]), int(config["num_workers"])
    )
    counts = class_counts(loaders["stage2_train"].dataset)
    weights = torch.tensor(
        [float(config["nf_class_weight"]), float(config["fl_class_weight"])],
        dtype=torch.float32,
        device=device,
    )
    criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=float(config["label_smoothing"]))
    optimizer = build_optimizer(model, config)
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=float(config["scheduler_factor"]),
        patience=int(config["scheduler_patience"]),
        threshold=float(config["early_stopping_min_delta"]),
        threshold_mode="abs",
        min_lr=[float(config["min_backbone_learning_rate"])] * 2 + [float(config["min_classifier_learning_rate"])] * 2,
    )
    run.config.update(
        {"initial_ewc_penalty": initial_penalty, "sampling_metadata": sampling,
         "stage2_train_nf": counts[0], "stage2_train_fl": counts[1], "device": str(device)},
        allow_val_change=True,
    )

    base1 = evaluate(model, loaders["stage1_holdout"], criterion, device)
    base2 = evaluate(model, loaders["stage2_holdout"], criterion, device)
    initial = {**prefixed("stage1_holdout", base1), **prefixed("stage2_holdout", base2)}
    write_metrics(output / "initial_metrics.csv", initial)
    print(
        f"Candidate={candidate} | seed={seed} | device={device} | batch={config['batch_size']}\n"
        f"lambda={config['ewc_lambda']:.4g} | backbone LR={config['backbone_learning_rate']:.3g} | "
        f"classifier LR={config['classifier_learning_rate']:.3g}\n"
        f"Epoch 0 | Stage 1 CSS={base1['css']:.4f} | Stage 2 CSS={base2['css']:.4f}"
    )

    best, best_epoch, stale, rows = float("-inf"), None, 0, []
    checkpoint = output / "stage2_best.pt"
    for epoch in range(1, int(config["epochs"]) + 1):
        started = time.perf_counter()
        train = train_epoch(model, loaders["stage2_train"], criterion, optimizer, device, anchors, config)
        stage1 = evaluate(model, loaders["stage1_holdout"], criterion, device)
        stage2 = evaluate(model, loaders["stage2_holdout"], criterion, device)
        score, previous = stage2["css"], best
        scheduler.step(score)
        if score > best:
            best, best_epoch = score, epoch
            torch.save(
                {"epoch": epoch, "candidate_id": candidate, "random_seed": seed,
                 "model_state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
                 "config": config, "stage2_train": train, "stage1_retention": stage1,
                 "stage2_holdout": stage2, "ewc_history": anchor["ewc_history"],
                 "stage1_fisher_metadata": anchor["fisher_metadata"]},
                checkpoint,
            )
        stale = 0 if score > previous + float(config["early_stopping_min_delta"]) else stale + 1
        logged = {
            "epoch": epoch,
            "selection/stage2_css": score,
            "diagnostics/train_holdout_css_gap": train["css"] - score,
            "diagnostics/stage1_retention_ratio": stage1["css"] / base1["css"],
            "best/epoch": best_epoch,
            "best/stage2_css": best,
            "epochs_without_improvement": stale,
            "timing/epoch_seconds": time.perf_counter() - started,
            **prefixed("stage2_train", train),
            **prefixed("stage1_holdout", stage1),
            **prefixed("stage2_holdout", stage2),
        }
        run.log(logged)
        rows.append({key.replace("/", "_"): value for key, value in logged.items()})
        write_epochs(output / "epoch_metrics.csv", rows)
        print(
            f"Epoch {epoch:02d} | train CSS={train['css']:.4f} | Stage 2 CSS={score:.4f} | "
            f"recall={stage2['recall']:.4f} | FPR={stage2['false_positive_rate']:.4f} | best={best:.4f}"
        )
        if stale >= int(config["early_stopping_patience"]):
            break

    selected = torch.load(checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(selected["model_state_dict"])
    stage1 = evaluate(model, loaders["stage1_holdout"], criterion, device)
    stage2 = evaluate(model, loaders["stage2_holdout"], criterion, device)
    validation = evaluate(model, loaders["future_validation"], criterion, device)
    test = evaluate(model, loaders["test"], criterion, device)
    final = {
        "candidate_id": candidate,
        "random_seed": seed,
        "selected_epoch": selected["epoch"],
        "selected_stage2_train_css": selected["stage2_train"]["css"],
        "selected_stage2_train_holdout_gap": selected["stage2_train"]["css"] - stage2["css"],
        "stage2_css_delta_from_epoch0": stage2["css"] - base2["css"],
        "stage1_css_retention_ratio": stage1["css"] / base1["css"],
        **prefixed("stage1_retention", stage1),
        **prefixed("stage2_holdout", stage2),
        **prefixed("validation", validation),
        **prefixed("test", test),
    }
    write_metrics(output / "final_metrics.csv", final)
    run.log(final)
    for key, value in final.items():
        run.summary[key] = value
    artifact = wandb.Artifact(
        f"noaa-stage2-seed-confirmation-{candidate}-seed-{seed}-{run.id}",
        type="model",
        metadata={"candidate_id": candidate, "seed": seed, "selected_epoch": selected["epoch"]},
    )
    for path in (checkpoint, output / "resolved_config.json", output / "initial_metrics.csv",
                 output / "epoch_metrics.csv", output / "final_metrics.csv"):
        artifact.add_file(str(path))
    run.log_artifact(artifact)
    print(
        f"Selected epoch={selected['epoch']} | Stage 1 CSS={stage1['css']:.4f} | "
        f"Stage 2 CSS={stage2['css']:.4f} | validation CSS={validation['css']:.4f} | test CSS={test['css']:.4f}"
    )
    run.finish()


if __name__ == "__main__":
    main()

