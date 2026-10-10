"""Prepare the selected Stage 1 checkpoint for Stage 2 EWC training.

Stage 1 itself does not use Elastic Weight Consolidation (EWC), because
there is no earlier task to preserve. Stage 2 does need two fixed items from
the selected Stage 1 model:

1. A snapshot of the Stage 1 parameter values.
2. A diagonal Fisher-information estimate describing the importance of each
   Stage 1 parameter.

This script calculates those items once from the complete, original,
unaugmented Stage 1 training set. It does not train Stage 1 or Stage 2, and it
does not modify the selected checkpoint. Instead, it writes a separate
``stage1_ewc_ready.pt`` checkpoint for all Stage 2 lambda trials to share.

Example
-------
python modeling/stage1_02_calculate_fisher.py \
    --checkpoint results/<winning_run>/checkpoints/stage1.pt
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
            "Calculate Fisher information for the selected Stage 1 model "
            "and save a separate EWC-ready checkpoint."
        )
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Path to the winning Stage 1 stage1.pt checkpoint.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Output checkpoint path. By default, stage1_ewc_ready.pt is "
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


def find_stage1_training_csv():
    """Return the Stage 1 training CSV discovered by the normal pipeline."""
    stage1_matches = [
        stage
        for stage in discover_stages(STAGE_DIRECTORY)
        if stage["number"] == 1
    ]

    if len(stage1_matches) != 1:
        raise RuntimeError(
            f"Expected exactly one Stage 1 dataset, found {len(stage1_matches)}."
        )

    return stage1_matches[0]["train_file"]


def validate_checkpoint(checkpoint, checkpoint_path):
    """Reject a checkpoint that is not the completed Stage 1 artifact."""
    required_keys = {"completed_stage", "model_state_dict", "config"}
    missing_keys = required_keys - set(checkpoint)
    if missing_keys:
        raise ValueError(
            f"{checkpoint_path} is missing keys: {sorted(missing_keys)}"
        )

    if checkpoint["completed_stage"] != 1:
        raise ValueError(
            "Fisher preparation requires a completed Stage 1 checkpoint, "
            f"but completed_stage={checkpoint['completed_stage']!r}."
        )


def move_dictionary_to_cpu(values):
    """Detach a tensor dictionary so the saved checkpoint is portable."""
    return {
        name: value.detach().cpu()
        for name, value in values.items()
    }


def main():
    args = parse_arguments()
    checkpoint_path = args.checkpoint.expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint was not found: {checkpoint_path}")

    output_path = (
        args.output.expanduser().resolve()
        if args.output is not None
        else checkpoint_path.with_name("stage1_ewc_ready.pt")
    )
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

    stage1_training_csv = find_stage1_training_csv()

    # build_evaluation_loader applies only resize plus 0-to-1 tensor scaling.
    # Therefore, every original Stage 1 sample is used exactly once, without
    # class balancing, repetition, shuffling, or augmentation.
    fisher_loader = build_evaluation_loader(
        csv_file=stage1_training_csv,
        image_directory=IMAGE_DIRECTORY,
        batch_size=fisher_batch_size,
        image_size=image_size,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    print(f"Input checkpoint: {checkpoint_path}")
    print(f"Stage 1 training CSV: {stage1_training_csv}")
    print(f"Original Fisher samples: {len(fisher_loader.dataset):,}")
    print(f"Fisher batch size: {fisher_batch_size}")
    print(f"Device: {device}")
    print("Calculating Stage 1 Fisher information...")

    fisher_start = time.perf_counter()
    fisher = calculate_fisher(model, fisher_loader, device)
    fisher_seconds = time.perf_counter() - fisher_start
    protected_parameters = save_parameters(model)

    if set(fisher) != set(protected_parameters):
        raise RuntimeError(
            "Fisher and protected-parameter names do not match."
        )
    if not all(torch.isfinite(value).all() for value in fisher.values()):
        raise RuntimeError("Fisher information contains a non-finite value.")
    if not all((value >= 0).all() for value in fisher.values()):
        raise RuntimeError("Fisher information contains a negative value.")

    fisher_cpu = move_dictionary_to_cpu(fisher)
    parameters_cpu = move_dictionary_to_cpu(protected_parameters)
    fisher_sum = sum(value.sum().item() for value in fisher_cpu.values())
    fisher_max = max(value.max().item() for value in fisher_cpu.values())

    # Preserve the selected checkpoint and add exactly one Stage 1 EWC entry.
    prepared_checkpoint = dict(checkpoint)
    prepared_checkpoint["ewc_history"] = [
        {
            "stage": 1,
            "fisher": fisher_cpu,
            "parameters": parameters_cpu,
        }
    ]
    prepared_checkpoint["fisher_metadata"] = {
        "prepared_at": datetime.now().isoformat(timespec="seconds"),
        "source_checkpoint": str(checkpoint_path),
        "training_csv": str(stage1_training_csv),
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
    print(f"EWC history entries: {len(prepared_checkpoint['ewc_history'])}")
    print(f"EWC-ready checkpoint: {output_path}")
    print("The original Stage 1 checkpoint was not modified.")


if __name__ == "__main__":
    main()
