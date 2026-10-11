"""Prepare the selected Stage 2 checkpoint for Stage 3 EWC training.

The selected Stage 2 checkpoint already contains the Stage 1 Fisher
information used during Stage 2 training. Before Stage 3 can begin, it also
needs a Fisher-information estimate for the selected Stage 2 model and a
snapshot of the corresponding Stage 2 parameter values.

This script calculates those Stage 2 quantities once from the complete,
original, unaugmented Stage 2 training set. It preserves the existing Stage 1
EWC-history entry, appends exactly one Stage 2 entry, and writes a separate
``stage2_ewc_ready.pt`` checkpoint. The selected ``stage2.pt`` checkpoint is
never modified.

Example
-------
python modeling/stage2_02_calculate_fisher.py \
    --checkpoint results/<winning_run>/checkpoints/stage2.pt
"""

import argparse
import time
from datetime import datetime
from pathlib import Path

import torch
from torch import nn

from attention_model import Attn_Net
from dataloader import build_evaluation_loader, discover_stages
from ewc import calculate_fisher, save_parameters
from ewc_train import IMAGE_DIRECTORY, NUM_WORKERS, STAGE_DIRECTORY


def parse_arguments():
    """Read the selected checkpoint and optional Fisher-loader settings."""
    parser = argparse.ArgumentParser(
        description=(
            "Calculate Fisher information for the selected Stage 2 model "
            "and save a separate EWC-ready checkpoint for Stage 3."
        )
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Path to the winning Stage 2 stage2.pt checkpoint.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Output checkpoint path. By default, stage2_ewc_ready.pt is "
            "created beside the input checkpoint."
        ),
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help=(
            "Fisher batch size. By default, use the winning checkpoint's "
            "training batch size."
        ),
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=NUM_WORKERS,
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacement of an existing output checkpoint.",
    )
    return parser.parse_args()


def find_stage2_training_csv():
    """Return the Stage 2 training CSV discovered by the normal pipeline."""
    stage2_matches = [
        stage
        for stage in discover_stages(STAGE_DIRECTORY)
        if stage["number"] == 2
    ]

    if len(stage2_matches) != 1:
        raise RuntimeError(
            f"Expected exactly one Stage 2 dataset, found {len(stage2_matches)}."
        )

    return stage2_matches[0]["train_file"]


def validate_checkpoint(checkpoint, checkpoint_path):
    """Require the selected Stage 2 artifact and its Stage 1 EWC history."""
    required_keys = {
        "completed_stage",
        "model_state_dict",
        "config",
        "ewc_history",
    }
    missing_keys = required_keys - set(checkpoint)
    if missing_keys:
        raise ValueError(
            f"{checkpoint_path} is missing keys: {sorted(missing_keys)}"
        )

    if checkpoint["completed_stage"] != 2:
        raise ValueError(
            "Fisher preparation requires a completed Stage 2 checkpoint, "
            f"but completed_stage={checkpoint['completed_stage']!r}."
        )

    ewc_history = checkpoint["ewc_history"]
    if len(ewc_history) != 1:
        raise ValueError(
            "The selected Stage 2 checkpoint must contain exactly one "
            f"existing EWC entry, but found {len(ewc_history)}."
        )
    if ewc_history[0].get("stage") != 1:
        raise ValueError(
            "The existing EWC-history entry must represent Stage 1."
        )


def move_dictionary_to_cpu(values):
    """Detach a tensor dictionary so the saved checkpoint is portable."""
    return {
        name: value.detach().cpu()
        for name, value in values.items()
    }


def move_history_to_cpu(ewc_history):
    """Copy the existing EWC history to CPU without changing its contents."""
    return [
        {
            "stage": entry["stage"],
            "fisher": move_dictionary_to_cpu(entry["fisher"]),
            "parameters": move_dictionary_to_cpu(entry["parameters"]),
        }
        for entry in ewc_history
    ]


def main():
    args = parse_arguments()
    checkpoint_path = args.checkpoint.expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint was not found: {checkpoint_path}")

    output_path = (
        args.output.expanduser().resolve()
        if args.output is not None
        else checkpoint_path.with_name("stage2_ewc_ready.pt")
    )
    if output_path == checkpoint_path:
        raise ValueError("Output must not replace the selected Stage 2 checkpoint.")
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(
            f"Output already exists: {output_path}. "
            "Use --overwrite only if replacement is intentional."
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )
    validate_checkpoint(checkpoint, checkpoint_path)

    config = dict(checkpoint["config"])
    image_size = int(config.get("image_size", 256))
    fisher_batch_size = int(
        args.batch_size
        if args.batch_size is not None
        else config.get("batch_size", 128)
    )
    if fisher_batch_size <= 0:
        raise ValueError("Fisher batch size must be positive.")
    if args.num_workers < 0:
        raise ValueError("Number of workers cannot be negative.")

    # Recreate the exact network architecture and restore the selected weights.
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

    stage2_training_csv = find_stage2_training_csv()

    # This loader applies only resize plus 0-to-1 tensor scaling. Consequently,
    # every original Stage 2 sample is used exactly once, with no augmentation,
    # class balancing, repetition, or shuffle.
    fisher_loader = build_evaluation_loader(
        csv_file=stage2_training_csv,
        image_directory=IMAGE_DIRECTORY,
        batch_size=fisher_batch_size,
        image_size=image_size,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    print(f"Input checkpoint: {checkpoint_path}")
    print(f"Stage 2 training CSV: {stage2_training_csv}")
    print(f"Original Fisher samples: {len(fisher_loader.dataset):,}")
    print(f"Fisher batch size: {fisher_batch_size}")
    print(f"Device: {device}")
    print("Calculating Stage 2 Fisher information...")

    fisher_start = time.perf_counter()
    fisher = calculate_fisher(model, fisher_loader, device)
    fisher_seconds = time.perf_counter() - fisher_start
    protected_parameters = save_parameters(model)

    if set(fisher) != set(protected_parameters):
        raise RuntimeError("Fisher and protected-parameter names do not match.")
    if not all(torch.isfinite(value).all() for value in fisher.values()):
        raise RuntimeError("Fisher information contains a non-finite value.")
    if not all((value >= 0).all() for value in fisher.values()):
        raise RuntimeError("Fisher information contains a negative value.")

    fisher_cpu = move_dictionary_to_cpu(fisher)
    parameters_cpu = move_dictionary_to_cpu(protected_parameters)
    fisher_sum = sum(value.sum().item() for value in fisher_cpu.values())
    fisher_max = max(value.max().item() for value in fisher_cpu.values())

    # Preserve Stage 1's EWC entry and append the newly calculated Stage 2
    # entry. Stage 3 can therefore protect knowledge learned in both stages.
    ewc_history = move_history_to_cpu(checkpoint["ewc_history"])
    ewc_history.append(
        {
            "stage": 2,
            "fisher": fisher_cpu,
            "parameters": parameters_cpu,
        }
    )

    prepared_checkpoint = dict(checkpoint)
    prepared_checkpoint["ewc_history"] = ewc_history

    # Retain any earlier metadata and add a separate Stage 2 record rather
    # than overwriting information about the Stage 1 Fisher calculation.
    prepared_checkpoint["stage2_fisher_metadata"] = {
        "prepared_at": datetime.now().isoformat(timespec="seconds"),
        "source_checkpoint": str(checkpoint_path),
        "training_csv": str(stage2_training_csv),
        "samples": len(fisher_loader.dataset),
        "batch_size": fisher_batch_size,
        "num_workers": args.num_workers,
        "seconds": fisher_seconds,
        "augmentation": "none",
        "class_balancing": "none",
        "shuffle": False,
        "fisher_sum": fisher_sum,
        "fisher_max": fisher_max,
    }

    # Write to a temporary path first, then atomically replace the destination.
    temporary_path = output_path.with_name(f".{output_path.name}.tmp")
    torch.save(prepared_checkpoint, temporary_path)
    temporary_path.replace(output_path)

    print(f"Fisher calculation completed in {fisher_seconds:.1f} seconds")
    print(f"Fisher parameter tensors: {len(fisher_cpu):,}")
    print(f"Fisher sum: {fisher_sum:.6g}")
    print(f"Fisher maximum: {fisher_max:.6g}")
    print(f"EWC history stages: {[entry['stage'] for entry in ewc_history]}")
    print(f"EWC history entries: {len(ewc_history)}")
    print(f"EWC-ready checkpoint: {output_path}")
    print("The original Stage 2 checkpoint was not modified.")


if __name__ == "__main__":
    main()
