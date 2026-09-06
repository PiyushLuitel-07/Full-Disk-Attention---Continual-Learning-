"""Train exactly one chronological stage.

Typical sequence from the project root::

    python -m modeling.train_continual --config configs/pilot.json --stage 1
    python -m modeling.train_continual --config configs/pilot.json --stage 2

Stage 2 refuses to start unless Stage 1's checkpoint and EWC state exist.
This makes the learning sequence visible and prevents accidental future-data
initialization.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import platform
import random
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from .attention_model import AttnNet
from .dataset import (
    MagnetogramDataset,
    Sample,
    make_ordered_loader,
    make_train_loader,
    read_manifest,
)
from .ewc import EWCState, estimate_diagonal_fisher
from .metrics import binary_metrics, predictions_from_probabilities
from .tracking import WandbTracker


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--stage", required=True, type=int, choices=(1, 2))
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, mps, or cuda:N")
    parser.add_argument(
        "--run-dir",
        type=Path,
        help="Override paths.run_dir in the JSON configuration",
    )
    return parser.parse_args()


def project_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def load_config(path: Path) -> dict[str, Any]:
    with project_path(path).open(encoding="utf-8") as handle:
        config = json.load(handle)
    required_sections = {
        "experiment",
        "stages",
        "model",
        "training",
        "ewc",
        "data",
        "paths",
        "tracking",
    }
    missing = required_sections - set(config)
    if missing:
        raise ValueError(f"Configuration is missing sections: {sorted(missing)}")
    if [stage["id"] for stage in config["stages"]] != [1, 2]:
        raise ValueError("This focused implementation expects exactly stages 1 and 2")
    if config["experiment"]["method"] not in {"ewc", "finetune"}:
        raise ValueError("experiment.method must be 'ewc' or 'finetune'")
    tracking = config["tracking"]
    if tracking.get("enabled", False):
        missing_tracking = {"entity", "project"} - set(tracking)
        if missing_tracking:
            raise ValueError(
                f"Enabled W&B tracking is missing settings: {sorted(missing_tracking)}"
            )
    return config


def select_device(requested: str) -> torch.device:
    if requested != "auto":
        device = torch.device(requested)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested but is not available")
    return device


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def validate_stage_years(samples: list[Sample], config: dict[str, Any], stage: int) -> None:
    stage_config = next(item for item in config["stages"] if item["id"] == stage)
    start_year, end_year = stage_config["start_year"], stage_config["end_year"]
    invalid = [
        sample.relative_path
        for sample in samples
        if not start_year <= int(sample.relative_path.split("/", 1)[0]) <= end_year
    ]
    if invalid:
        raise ValueError(
            f"Stage {stage} manifest contains rows outside {start_year}-{end_year}: "
            f"{invalid[:3]}"
        )


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(json_safe(payload), indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def atomic_torch_save(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    os.close(descriptor)
    try:
        torch.save(payload, temporary_name)
        os.replace(temporary_name, path)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)


def load_torch(path: Path, device: torch.device) -> dict[str, Any]:
    # These are project-created local artifacts, not untrusted downloads.
    return torch.load(path, map_location=device, weights_only=False)


def append_history(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        if not exists:
            writer.writeheader()
        writer.writerows(rows)


def train_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    ewc_state: EWCState,
    ewc_lambda: float,
    use_ewc: bool,
    device: torch.device,
    max_batches: int,
) -> dict[str, Any]:
    model.train()
    total_examples = 0
    ce_sum = raw_penalty_sum = weighted_penalty_sum = total_loss_sum = 0.0
    all_targets: list[int] = []
    all_predictions: list[int] = []

    for batch_number, (images, targets, _paths) in enumerate(loader, start=1):
        if max_batches and batch_number > max_batches:
            break
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        logits = model(images)[0]
        cross_entropy = criterion(logits, targets)
        raw_penalty = ewc_state.penalty(model) if use_ewc else logits.new_zeros(())
        weighted_penalty = 0.5 * ewc_lambda * raw_penalty
        loss = cross_entropy + weighted_penalty

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        batch_size = targets.shape[0]
        total_examples += batch_size
        ce_sum += cross_entropy.item() * batch_size
        raw_penalty_sum += raw_penalty.item() * batch_size
        weighted_penalty_sum += weighted_penalty.item() * batch_size
        total_loss_sum += loss.item() * batch_size
        all_targets.extend(targets.detach().cpu().tolist())
        all_predictions.extend(logits.argmax(dim=1).detach().cpu().tolist())

    if not total_examples:
        raise RuntimeError("No training batches were processed")
    result = binary_metrics(all_targets, all_predictions)
    result.update(
        {
            "ce_loss": ce_sum / total_examples,
            "raw_ewc_penalty": raw_penalty_sum / total_examples,
            "weighted_ewc_penalty": weighted_penalty_sum / total_examples,
            "total_loss": total_loss_sum / total_examples,
        }
    )
    return result


@torch.inference_mode()
def evaluate(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    threshold: float,
    device: torch.device,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    model.eval()
    targets: list[int] = []
    probabilities: list[float] = []
    paths: list[str] = []
    loss_sum = 0.0

    for images, batch_targets, batch_paths in loader:
        images = images.to(device, non_blocking=True)
        batch_targets = batch_targets.to(device, non_blocking=True)
        logits = model(images)[0]
        loss_sum += criterion(logits, batch_targets).item() * batch_targets.shape[0]
        flare_probabilities = torch.softmax(logits, dim=1)[:, 1]
        targets.extend(batch_targets.cpu().tolist())
        probabilities.extend(flare_probabilities.cpu().tolist())
        paths.extend(batch_paths)

    predictions = predictions_from_probabilities(probabilities, threshold)
    result = binary_metrics(targets, predictions)
    result["ce_loss"] = loss_sum / len(targets)
    rows = [
        {
            "image_path": path,
            "target": target,
            "probability_fl": probability,
            "predicted_class": prediction,
            "threshold": threshold,
        }
        for path, target, probability, prediction in zip(
            paths, targets, probabilities, predictions
        )
    ]
    return result, rows


def write_predictions(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def fisher_summary(fisher: dict[str, torch.Tensor], sample_count: int) -> dict[str, Any]:
    layers = {}
    for name, values in fisher.items():
        layers[name] = {
            "shape": list(values.shape),
            "mean": values.mean().item(),
            "max": values.max().item(),
            "sum": values.sum().item(),
            "nonzero": int(torch.count_nonzero(values).item()),
        }
    return {"sample_count": sample_count, "layers": layers}


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    stage = args.stage
    seed = int(config["experiment"]["seed"])
    method = config["experiment"]["method"]
    seed_everything(seed)
    device = select_device(args.device)

    manifest_dir = project_path(config["paths"]["manifest_dir"])
    image_root = project_path(config["paths"]["image_root"])
    run_dir = project_path(args.run_dir or config["paths"]["run_dir"])
    checkpoint_path = run_dir / "checkpoints" / f"stage{stage}.pt"
    ewc_path = run_dir / "ewc" / f"through_stage{stage}.pt"
    if checkpoint_path.exists() or ewc_path.exists():
        raise FileExistsError(
            f"Stage {stage} output already exists in {run_dir}. Use a new run directory "
            "to preserve provenance."
        )

    tracker = WandbTracker.start(config, stage, str(device))

    model = AttnNet(
        num_classes=int(config["model"]["num_classes"]),
        attention=bool(config["model"]["attention"]),
    ).to(device)
    ewc_state = EWCState()
    if stage > 1:
        previous_checkpoint = run_dir / "checkpoints" / f"stage{stage - 1}.pt"
        previous_ewc = run_dir / "ewc" / f"through_stage{stage - 1}.pt"
        if not previous_checkpoint.is_file() or not previous_ewc.is_file():
            raise FileNotFoundError(
                f"Train Stage {stage - 1} first. Expected {previous_checkpoint} and {previous_ewc}"
            )
        checkpoint = load_torch(previous_checkpoint, device)
        if checkpoint.get("config") != config:
            raise ValueError(
                "The Stage 2 configuration differs from Stage 1. Use the same "
                "configuration for one continual sequence, or start a new run directory."
            )
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        ewc_state = EWCState.from_state_dict(
            load_torch(previous_ewc, torch.device("cpu"))
        ).to(device)

    training = config["training"]
    batch_size = int(training["batch_size"])
    workers = int(training["num_workers"])
    pin_memory = device.type == "cuda"
    train_samples = read_manifest(manifest_dir / f"stage{stage}_train.csv")
    validate_stage_years(train_samples, config, stage)
    train_dataset = MagnetogramDataset(
        train_samples,
        image_root,
        int(config["model"]["image_size"]),
        augment_positive=bool(training["augment_positive"]),
    )
    train_loader = make_train_loader(
        train_dataset, batch_size, seed + stage, workers, pin_memory
    )

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=int(training["lr_halving_period_epochs"]),
        gamma=0.5,
    )
    criterion = nn.CrossEntropyLoss()
    use_ewc = stage > 1 and method == "ewc"
    start_time = time.time()
    history: list[dict[str, Any]] = []

    print(f"Training Stage {stage} on {device}; method={method}; EWC active={use_ewc}")
    for epoch in range(1, int(training["epochs_per_stage"]) + 1):
        epoch_result = train_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            ewc_state,
            float(config["ewc"]["lambda"]),
            use_ewc,
            device,
            int(training["max_batches_per_epoch"]),
        )
        epoch_result = {
            "stage": stage,
            "epoch": epoch,
            "learning_rate": optimizer.param_groups[0]["lr"],
            **epoch_result,
        }
        history.append(epoch_result)
        tracker.log_training_epoch(stage, epoch_result)
        scheduler.step()
        print(
            f"epoch={epoch} ce={epoch_result['ce_loss']:.4f} "
            f"weighted_ewc={epoch_result['weighted_ewc_penalty']:.6g} "
            f"tss={epoch_result['tss']:.4f}"
        )

    threshold = float(training["threshold"])
    evaluations: dict[str, Any] = {}
    # Evaluate every chronological period after every checkpoint. In particular,
    # after Stage 1 this records R_1,2: zero-shot performance on 2013--2014
    # before any Stage 2 optimization. Evaluation runs under inference_mode and
    # never contributes gradients, Fisher values, or checkpoint selection.
    evaluation_stage_ids = [item["id"] for item in config["stages"]]
    for evaluated_stage in evaluation_stage_ids:
        eval_samples = read_manifest(manifest_dir / f"stage{evaluated_stage}_eval.csv")
        validate_stage_years(eval_samples, config, evaluated_stage)
        eval_dataset = MagnetogramDataset(
            eval_samples,
            image_root,
            int(config["model"]["image_size"]),
            augment_positive=False,
        )
        eval_loader = make_ordered_loader(
            eval_dataset, batch_size, workers, pin_memory
        )
        result, prediction_rows = evaluate(
            model, eval_loader, criterion, threshold, device
        )
        evaluations[f"stage{evaluated_stage}"] = result
        tracker.log_evaluation(stage, evaluated_stage, result, prediction_rows)
        write_predictions(
            run_dir
            / "predictions"
            / f"after_stage{stage}_eval_stage{evaluated_stage}.csv",
            prediction_rows,
        )
        print(
            f"eval_stage={evaluated_stage} tss={result['tss']:.4f} "
            f"hss={result['hss']:.4f} tp={result['tp']} fp={result['fp']} "
            f"tn={result['tn']} fn={result['fn']}"
        )

    # Consolidation uses deterministic, unaugmented, class-balanced rows only.
    fisher_samples = read_manifest(manifest_dir / f"stage{stage}_fisher.csv")
    validate_stage_years(fisher_samples, config, stage)
    fisher_dataset = MagnetogramDataset(
        fisher_samples,
        image_root,
        int(config["model"]["image_size"]),
        augment_positive=False,
    )
    fisher_loader = make_ordered_loader(
        fisher_dataset,
        int(config["ewc"]["fisher_loader_batch_size"]),
        workers,
        pin_memory,
    )
    print(f"Estimating Stage {stage} empirical Fisher from {len(fisher_samples)} rows...")
    maximum_fisher = int(config["ewc"]["maximum_total_fisher_examples"])
    if maximum_fisher and maximum_fisher < len(fisher_samples):
        raise ValueError(
            "maximum_total_fisher_examples would truncate the time-sorted Fisher "
            "manifest and break its class balance. Regenerate a smaller balanced "
            "Fisher manifest instead."
        )
    fisher, fisher_count = estimate_diagonal_fisher(
        model,
        fisher_loader,
        device,
        max_samples=0,
    )
    ewc_state.add_stage(model, fisher)

    checkpoint = {
        "stage": stage,
        "method": method,
        "seed": seed,
        "model_state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "config": config,
        "evaluations": evaluations,
        "elapsed_seconds": time.time() - start_time,
    }
    atomic_torch_save(checkpoint, checkpoint_path)
    atomic_torch_save(ewc_state.state_dict(), ewc_path)
    append_history(run_dir / "metrics" / "history.csv", history)
    summary = {
        "trained_through_stage": stage,
        "method": method,
        "device": str(device),
        "train_examples_in_manifest": len(train_samples),
        "epochs": int(training["epochs_per_stage"]),
        "max_batches_per_epoch": int(training["max_batches_per_epoch"]),
        "ewc_was_active": use_ewc,
        "ewc_lambda": float(config["ewc"]["lambda"]),
        "evaluations": evaluations,
        "fisher_examples": fisher_count,
        "checkpoint": str(checkpoint_path),
        "ewc_state": str(ewc_path),
        "elapsed_seconds": checkpoint["elapsed_seconds"],
    }
    write_json(run_dir / "metrics" / f"stage{stage}_summary.json", summary)
    fisher_result = fisher_summary(fisher, fisher_count)
    write_json(run_dir / "ewc" / f"stage{stage}_fisher_summary.json", fisher_result)
    tracker.log_fisher(stage, fisher_result)
    tracker.log_stage_summary(summary)

    provenance_path = run_dir / f"provenance_stage{stage}.json"
    provenance = {
        "command": sys.argv,
        "config_path": str(project_path(args.config)),
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torchvision": __import__("torchvision").__version__,
        "device": str(device),
        "cuda_version": torch.version.cuda,
    }
    write_json(provenance_path, provenance)
    tracker.log_run_artifact(stage, run_dir, project_path(args.config))
    tracker.finish()
    print(f"Finished Stage {stage}. Summary: {run_dir / 'metrics' / f'stage{stage}_summary.json'}")


if __name__ == "__main__":
    main()
