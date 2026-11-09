"""Create a validated Stage 1 EWC anchor for corrected Stage 2 training.

Run this exactly once for the selected Stage 1 checkpoint. It saves a new
artifact containing the immutable Stage 1 model parameters plus a correctly
estimated, diagonal empirical Fisher. The original checkpoint is never edited.
"""

import argparse
import hashlib
import json
import random
import sys
import time
from datetime import datetime
from pathlib import Path

import torch
import wandb


SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from config import (  # noqa: E402
    CONFIG,
    IMAGE_DIR,
    RESULTS_DIR,
    STAGE1_CHECKPOINT,
    STAGE1_TRAIN_CSV,
    WANDB_ENTITY,
    WANDB_PROJECT,
    WANDB_TAGS,
)
from dataloader import build_loader, class_counts  # noqa: E402
from ewc import (  # noqa: E402
    calculate_per_example_empirical_fisher,
    fisher_statistics,
    save_parameter_snapshot,
    validate_fisher,
)
from model import build_model  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=STAGE1_CHECKPOINT)
    parser.add_argument(
        "--output",
        type=Path,
        default=RESULTS_DIR / "anchors" / "stage1_ewc_ready.pt",
        help="New EWC anchor path; the source checkpoint remains unchanged.",
    )
    parser.add_argument(
        "--fisher-batch-size",
        type=int,
        default=int(CONFIG["fisher_batch_size"]),
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=int(CONFIG["fisher_num_workers"]),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing output anchor only when explicitly intended.",
    )
    parser.add_argument("--skip-wandb", action="store_true")
    return parser.parse_args()


def sha256(path):
    """Return a reproducibility fingerprint without loading the checkpoint."""
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require_compatible_checkpoint(checkpoint, checkpoint_path):
    """Fail early if this is not a Stage 1 MobileNet checkpoint."""
    required = {"model_state_dict", "epoch", "config"}
    missing = required.difference(checkpoint)
    if missing:
        raise ValueError(
            f"{checkpoint_path} is missing required checkpoint keys: {sorted(missing)}"
        )
    if checkpoint.get("completed_stage") not in (None, 1):
        raise ValueError("The Fisher source must be a completed Stage 1 checkpoint.")
    if checkpoint.get("architecture") not in (None, "mobilenet_v3_large"):
        raise ValueError("The Fisher source is not a MobileNetV3-Large checkpoint.")


def main():
    args = parse_args()
    if args.fisher_batch_size <= 0 or args.num_workers < 0:
        raise ValueError("Fisher batch size must be positive and workers nonnegative.")
    if not args.checkpoint.is_file():
        raise FileNotFoundError(f"Stage 1 checkpoint not found: {args.checkpoint}")
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(
            f"Refusing to overwrite existing anchor: {args.output}. "
            "Use --overwrite only after confirming that replacement is intended."
        )

    config = dict(CONFIG)
    seed = int(config["random_seed"])
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(0)

    # Keep source weights on CPU until the model is constructed. This avoids
    # holding both a GPU checkpoint copy and the Fisher estimate unnecessarily.
    source_checkpoint = torch.load(
        args.checkpoint, map_location="cpu", weights_only=False
    )
    require_compatible_checkpoint(source_checkpoint, args.checkpoint)

    source_config = source_checkpoint["config"]
    hidden_dim = int(source_config.get("classifier_hidden_dim", 1280))
    dropout = float(source_config.get("dropout", 0.0))
    model = build_model(
        num_classes=2,
        pretrained=False,
        hidden_dim=hidden_dim,
        dropout=dropout,
    ).to(device)
    model.load_state_dict(source_checkpoint["model_state_dict"], strict=True)

    fisher_loader = build_loader(
        STAGE1_TRAIN_CSV,
        IMAGE_DIR,
        batch_size=args.fisher_batch_size,
        num_workers=args.num_workers,
        shuffle=False,
    )
    counts = class_counts(fisher_loader.dataset)

    run = None
    if not args.skip_wandb:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run = wandb.init(
            entity=WANDB_ENTITY,
            project=WANDB_PROJECT,
            name=f"noaa_stage1_fisher_preparation_{timestamp}",
            tags=WANDB_TAGS + ["stage1-fisher-preparation"],
            job_type="fisher-preparation",
            config={
                **config,
                "source_stage1_checkpoint": str(args.checkpoint),
                "source_stage1_checkpoint_sha256": sha256(args.checkpoint),
                "source_stage1_selected_epoch": source_checkpoint["epoch"],
                "fisher_batch_size": args.fisher_batch_size,
                "fisher_num_workers": args.num_workers,
                "fisher_samples": len(fisher_loader.dataset),
            },
        )

    print(f"Device: {device}")
    print(f"Source Stage 1 checkpoint: {args.checkpoint}")
    print(f"Source Stage 1 selected epoch: {source_checkpoint['epoch']}")
    print(f"Fisher data: NF={counts[0]:,}, FL={counts[1]:,}, total={sum(counts):,}")
    print(f"Fisher batch size: {args.fisher_batch_size}")
    print("Estimator: per-example empirical Fisher, model.eval(), no augmentation")

    started = time.perf_counter()
    fisher = calculate_per_example_empirical_fisher(model, fisher_loader, device)
    elapsed = time.perf_counter() - started
    parameters = save_parameter_snapshot(model)
    validate_fisher(fisher, expected_names=set(parameters))
    statistics = fisher_statistics(fisher)

    metadata = {
        "schema_version": 1,
        "stage": 1,
        "source_checkpoint": str(args.checkpoint.resolve()),
        "source_checkpoint_sha256": sha256(args.checkpoint),
        "source_selected_epoch": int(source_checkpoint["epoch"]),
        "estimator": config["fisher_estimator"],
        "formula": "mean_i(square(gradient_i(negative_log_likelihood)))",
        "uses_ground_truth_labels": bool(config["fisher_use_ground_truth_labels"]),
        "model_mode": config["fisher_model_mode"],
        "augmentation": "none",
        "data_csv": str(STAGE1_TRAIN_CSV.resolve()),
        "samples": len(fisher_loader.dataset),
        "class_counts": {"NF": counts[0], "FL": counts[1]},
        "batch_size": args.fisher_batch_size,
        "num_workers": args.num_workers,
        "seconds": elapsed,
        "statistics": statistics,
    }

    # The anchor contains only immutable Stage 1 information needed by EWC.
    # It intentionally excludes optimizer/scheduler state from the old run.
    anchor = {
        "completed_stage": 1,
        "architecture": "mobilenet_v3_large",
        "model_state_dict": {
            name: tensor.detach().cpu()
            for name, tensor in model.state_dict().items()
        },
        "source_stage1_config": source_config,
        "ewc_history": [
            {
                "stage": 1,
                "fisher": fisher,
                "parameters": parameters,
            }
        ],
        "fisher_metadata": metadata,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(anchor, args.output)
    metadata_path = args.output.with_suffix(".metadata.json")
    with metadata_path.open("w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2, sort_keys=True)

    print(f"Fisher calculation completed in {elapsed:.1f}s")
    print(
        "Fisher statistics | "
        f"sum={statistics['sum']:.6g} | mean={statistics['mean']:.6g} | "
        f"max={statistics['max']:.6g} | p99={statistics['p99']:.6g} | "
        f"zero fraction={statistics['zero_fraction']:.6g}"
    )
    print(f"EWC-ready Stage 1 anchor: {args.output}")
    print(f"Metadata: {metadata_path}")

    if run is not None:
        run.log(
            {
                "fisher/seconds": elapsed,
                **{f"fisher/{key}": value for key, value in statistics.items()},
            }
        )
        run.summary["ewc_anchor"] = str(args.output)
        run.summary["fisher_metadata"] = metadata
        artifact = wandb.Artifact(
            f"noaa-stage1-ewc-anchor-{run.id}",
            type="ewc-anchor",
            metadata=metadata,
        )
        artifact.add_file(str(args.output))
        artifact.add_file(str(metadata_path))
        run.log_artifact(artifact)
        run.finish()


if __name__ == "__main__":
    main()
