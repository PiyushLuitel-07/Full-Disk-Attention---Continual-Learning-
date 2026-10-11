"""Diagnose Stage 3 EWC numerical stability without saving any model.

Stage 3 applies the selected EWC lambda to two accumulated Fisher entries.
Before changing the training procedure, this script reproduces a small number
of Stage 3 optimizer steps in memory and reports the classification loss,
per-stage EWC penalties, total loss, gradient norm, and parameter finiteness.

The selected checkpoint is read-only. This script does not initialize W&B,
write a checkpoint, or modify any dataset file. All simulated parameter
updates disappear when the process exits.

Example
-------
python modeling/stage3_00_check_stability.py --batches 10
"""

import argparse
import math
import random

import torch
from torch import nn
from torch.optim import SGD

from attention_model import Attn_Net
from dataloader import build_stage_loaders, discover_stages
from ewc import ewc_penalty, unwrap_model
from ewc_train import (
    IMAGE_DIRECTORY,
    NUM_WORKERS,
    RESULTS_DIRECTORY,
    STAGE_DIRECTORY,
)


STAGE2_EWC_CHECKPOINT = (
    RESULTS_DIRECTORY
    / "ewc_continual_20261010_130641"
    / "checkpoints"
    / "stage2_ewc_ready.pt"
)


def parse_arguments():
    """Read the number of in-memory optimizer steps to simulate."""
    parser = argparse.ArgumentParser(
        description="Inspect Stage 3 EWC losses and gradients for instability."
    )
    parser.add_argument(
        "--batches",
        type=int,
        default=10,
        help="Maximum number of Stage 3 training batches to simulate.",
    )
    return parser.parse_args()


def select_stage(stages, stage_number):
    """Return exactly one discovered stage or fail clearly."""
    matches = [stage for stage in stages if stage["number"] == stage_number]
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected exactly one Stage {stage_number} dataset, "
            f"found {len(matches)}."
        )
    return matches[0]


def load_checkpoint(device):
    """Load and validate the fixed EWC-ready Stage 2 checkpoint."""
    if not STAGE2_EWC_CHECKPOINT.is_file():
        raise FileNotFoundError(
            f"Checkpoint was not found: {STAGE2_EWC_CHECKPOINT}"
        )

    checkpoint = torch.load(
        STAGE2_EWC_CHECKPOINT,
        map_location=device,
        weights_only=False,
    )
    required_keys = {
        "completed_stage",
        "model_state_dict",
        "config",
        "ewc_history",
    }
    missing_keys = required_keys - set(checkpoint)
    if missing_keys:
        raise ValueError(
            f"Checkpoint is missing keys: {sorted(missing_keys)}"
        )
    if checkpoint["completed_stage"] != 2:
        raise ValueError("Expected a completed Stage 2 checkpoint.")

    history_stages = [
        entry.get("stage") for entry in checkpoint["ewc_history"]
    ]
    if history_stages != [1, 2]:
        raise ValueError(
            f"Expected EWC history stages [1, 2], found {history_stages}."
        )
    return checkpoint


def combined_fisher_maximum(ewc_history):
    """Find max_i(F1_i + F2_i) and the parameter containing that value."""
    fisher_names = set(ewc_history[0]["fisher"])
    for entry in ewc_history[1:]:
        fisher_names &= set(entry["fisher"])

    maximum = float("-inf")
    maximum_name = None
    all_finite = True
    all_nonnegative = True

    for name in sorted(fisher_names):
        combined = sum(entry["fisher"][name] for entry in ewc_history)
        all_finite = all_finite and bool(torch.isfinite(combined).all())
        all_nonnegative = all_nonnegative and bool((combined >= 0).all())
        candidate = combined.max().item()
        if candidate > maximum:
            maximum = candidate
            maximum_name = name

    if maximum_name is None:
        raise RuntimeError("The EWC histories have no common Fisher tensors.")
    return maximum, maximum_name, all_finite, all_nonnegative


def gradient_statistics(model):
    """Return total L2 norm, maximum absolute gradient, and finiteness."""
    squared_norm = 0.0
    maximum_absolute = 0.0
    all_finite = True

    for parameter in unwrap_model(model).parameters():
        if parameter.grad is None:
            continue
        gradient = parameter.grad.detach()
        all_finite = all_finite and bool(torch.isfinite(gradient).all())
        squared_norm += gradient.double().pow(2).sum().item()
        maximum_absolute = max(
            maximum_absolute,
            gradient.abs().max().item(),
        )

    return math.sqrt(squared_norm), maximum_absolute, all_finite


def model_parameters_are_finite(model):
    """Return whether every trainable model parameter is finite."""
    return all(
        bool(torch.isfinite(parameter).all())
        for parameter in unwrap_model(model).parameters()
    )


def main():
    args = parse_arguments()
    if args.batches <= 0:
        raise ValueError("--batches must be positive.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = load_checkpoint(device)
    config = dict(checkpoint["config"])

    image_size = int(config["image_size"])
    batch_size = int(config["batch_size"])
    learning_rate = float(config["learning_rate"])
    weight_decay = float(config["weight_decay"])
    ewc_lambda = float(config["ewc_lambda"])
    random_seed = int(config["random_seed"])

    random.seed(random_seed)
    torch.manual_seed(random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(random_seed)

    model = Attn_Net(
        im_size=image_size,
        num_classes=2,
        attention=True,
        init="kaimingUniform",
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])

    number_of_gpus = torch.cuda.device_count()
    if number_of_gpus > 1:
        model = nn.DataParallel(model)
        print(f"Using DataParallel on {number_of_gpus} GPUs")

    ewc_history = checkpoint["ewc_history"]
    combined_max, maximum_name, fisher_finite, fisher_nonnegative = (
        combined_fisher_maximum(ewc_history)
    )
    stability_quantity = learning_rate * ewc_lambda * combined_max
    theoretical_limit = (
        2.0 / (ewc_lambda * combined_max)
        if ewc_lambda > 0 and combined_max > 0
        else float("inf")
    )

    print(f"Checkpoint: {STAGE2_EWC_CHECKPOINT}")
    print(f"Device: {device}")
    print(f"Learning rate: {learning_rate:.10g}")
    print(f"EWC lambda: {ewc_lambda:.10g}")
    print(f"Combined Fisher maximum: {combined_max:.10g}")
    print(f"Combined Fisher maximum parameter: {maximum_name}")
    print(f"Combined Fisher finite: {fisher_finite}")
    print(f"Combined Fisher nonnegative: {fisher_nonnegative}")
    print(
        "Approximate stability quantity "
        f"lr * lambda * max(F1 + F2): {stability_quantity:.10g}"
    )
    print(
        "Approximate quadratic stability learning-rate limit: "
        f"{theoretical_limit:.10g}"
    )
    print(
        "Note: this scalar condition is diagnostic, not a complete stability "
        "guarantee for the neural network."
    )

    stage3 = select_stage(discover_stages(STAGE_DIRECTORY), 3)
    stage3_loaders = build_stage_loaders(
        train_csv=stage3["train_file"],
        holdout_csv=stage3["holdout_file"],
        image_directory=IMAGE_DIRECTORY,
        batch_size=batch_size,
        image_size=image_size,
        num_workers=NUM_WORKERS,
        pin_memory=device.type == "cuda",
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = SGD(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    model.train()

    print("\nSimulating Stage 3 optimizer steps in memory (nothing is saved):")
    for batch_index, (images, targets) in enumerate(
        stage3_loaders["train"],
        start=1,
    ):
        if batch_index > args.batches:
            break

        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        scores = model(images)[0]
        classification_loss = criterion(scores, targets)
        stage1_raw_penalty = ewc_penalty(model, [ewc_history[0]])
        stage2_raw_penalty = ewc_penalty(model, [ewc_history[1]])
        stage1_ewc_loss = (ewc_lambda / 2.0) * stage1_raw_penalty
        stage2_ewc_loss = (ewc_lambda / 2.0) * stage2_raw_penalty
        total_loss = (
            classification_loss + stage1_ewc_loss + stage2_ewc_loss
        )

        losses_finite = all(
            bool(torch.isfinite(value))
            for value in (
                classification_loss,
                stage1_ewc_loss,
                stage2_ewc_loss,
                total_loss,
            )
        )

        if losses_finite:
            total_loss.backward()
            gradient_norm, gradient_max, gradients_finite = (
                gradient_statistics(model)
            )
        else:
            gradient_norm = float("nan")
            gradient_max = float("nan")
            gradients_finite = False

        print(
            f"Batch {batch_index:02d} | "
            f"classification={classification_loss.item():.8g} | "
            f"stage1_EWC={stage1_ewc_loss.item():.8g} | "
            f"stage2_EWC={stage2_ewc_loss.item():.8g} | "
            f"total={total_loss.item():.8g} | "
            f"grad_norm={gradient_norm:.8g} | "
            f"grad_max={gradient_max:.8g} | "
            f"finite={losses_finite and gradients_finite}"
        )

        if not losses_finite or not gradients_finite:
            print("Stopped before optimizer.step(): non-finite value detected.")
            break

        optimizer.step()
        if not model_parameters_are_finite(model):
            print(
                "Stopped after optimizer.step(): model parameters became "
                "non-finite."
            )
            break
    else:
        batch_index = 0

    print("\nDiagnostic finished. No checkpoint or W&B run was written.")


if __name__ == "__main__":
    main()
