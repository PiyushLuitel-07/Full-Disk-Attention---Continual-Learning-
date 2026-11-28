"""Create a validated Phase 2 Stage 1 EWC anchor for Stage 2 training.

Run once for the selected Stage 1 checkpoint. Saves a new artifact containing
the immutable Stage 1 parameters plus a per-example diagonal empirical Fisher
estimated over the unaugmented Stage 1 training set.
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


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
STAGE1_DIR = ROOT / "modeling_phase2" / "experiment_01_stage1_full_ft"
sys.path[:0] = [str(HERE), str(STAGE1_DIR)]

from dataloader import IMAGENET_MEAN, IMAGENET_STD, MagnetogramDataset, _loader  # noqa: E402
from ewc import (  # noqa: E402
    calculate_per_example_empirical_fisher,
    fisher_statistics,
    save_parameter_snapshot,
    validate_fisher,
)
from model import build_model  # noqa: E402


IMAGE_DIR = ROOT / "downloaded_data" / "hmi_jpgs"
STAGE1_TRAIN_CSV = (
    ROOT
    / "data_labeling"
    / "data_labels"
    / "GOES_2010_2018_NOAA_2019_2026"
    / "CL_Labels_noaa_dates"
    / "Stage1_train.csv"
)
DEFAULT_STAGE1_CHECKPOINT = (
    ROOT
    / "results_phase2"
    / "stage1_full_ft"
    / "mobilenet_stage1_full_ft_20261127_141626_drop02_wd01_ep35"
    / "stage1_best_holdout.pt"
)
DEFAULT_OUTPUT = (
    ROOT / "results_phase2" / "stage2_ewc" / "anchors" / "stage1_ewc_ready.pt"
)

WANDB_ENTITY = "piyush-luitel-texas-christian-university"
WANDB_PROJECT = "Full disk attention solar flare prediction with continual learning"
WANDB_TAGS = [
    "MobileNetV3-Large",
    "Phase-2",
    "Hybrid-GOES-NOAA",
    "Stage-1",
    "stage1-fisher-preparation",
    "ewc-anchor",
]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_STAGE1_CHECKPOINT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--train-csv", type=Path, default=STAGE1_TRAIN_CSV)
    parser.add_argument("--fisher-batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
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
    if not args.train_csv.is_file():
        raise FileNotFoundError(f"Stage 1 training CSV not found: {args.train_csv}")
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(
            f"Refusing to overwrite existing anchor: {args.output}. "
            "Use --overwrite only after confirming replacement is intended."
        )

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(0)

    source_checkpoint = torch.load(
        args.checkpoint, map_location="cpu", weights_only=False
    )
    require_compatible_checkpoint(source_checkpoint, args.checkpoint)
    source_config = source_checkpoint["config"]

    model = build_model(
        pretrained=False,
        freeze_backbone=bool(source_config.get("freeze_backbone", False)),
        trainable_backbone_blocks=int(source_config.get("trainable_backbone_blocks", 2)),
        hidden_dim=int(source_config.get("classifier_hidden_dim", 1280)),
        dropout=float(source_config.get("dropout", 0.0)),
        bn_momentum=source_config.get("bn_momentum", 0.1),
    ).to(device)
    model.load_state_dict(source_checkpoint["model_state_dict"], strict=True)

    mean = tuple(source_config.get("mean", IMAGENET_MEAN))
    std = tuple(source_config.get("std", IMAGENET_STD))
    normalization = source_config.get("normalization", "imagenet")

    fisher_dataset = MagnetogramDataset(
        args.train_csv,
        IMAGE_DIR,
        mean=mean,
        std=std,
        augmentation="none",
    )
    fisher_loader = _loader(
        fisher_dataset,
        batch_size=args.fisher_batch_size,
        num_workers=args.num_workers,
        shuffle=False,
    )
    fl_count = sum(fisher_dataset.targets)
    nf_count = len(fisher_dataset.targets) - fl_count

    run = None
    if not args.skip_wandb:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run = wandb.init(
            entity=WANDB_ENTITY,
            project=WANDB_PROJECT,
            name=f"phase2_stage1_fisher_prep_{timestamp}",
            tags=WANDB_TAGS,
            job_type="fisher-preparation",
            config={
                "source_stage1_checkpoint": str(args.checkpoint),
                "source_stage1_checkpoint_sha256": sha256(args.checkpoint),
                "source_stage1_selected_epoch": int(source_checkpoint["epoch"]),
                "normalization": normalization,
                "mean": list(mean),
                "std": list(std),
                "fisher_batch_size": args.fisher_batch_size,
                "fisher_num_workers": args.num_workers,
                "fisher_samples": len(fisher_dataset),
                "fisher_estimator": "per_example_empirical_fisher_vmap",
            },
        )

    print(f"Device: {device}")
    print(f"Source Stage 1 checkpoint: {args.checkpoint}")
    print(f"Source Stage 1 selected epoch: {source_checkpoint['epoch']}")
    print(f"Normalization: {normalization} (mean={list(mean)}, std={list(std)})")
    print(f"Fisher data: NF={nf_count:,}, FL={fl_count:,}, total={len(fisher_dataset):,}")
    print(f"Fisher batch size: {args.fisher_batch_size}")
    print("Estimator: per-example empirical Fisher (vmap + grad), model.eval(), no augmentation")

    started = time.perf_counter()
    fisher = calculate_per_example_empirical_fisher(model, fisher_loader, device)
    elapsed = time.perf_counter() - started
    parameters = save_parameter_snapshot(model)
    validate_fisher(fisher, expected_names=set(parameters))
    statistics = fisher_statistics(fisher)

    metadata = {
        "schema_version": 1,
        "phase": 2,
        "stage": 1,
        "source_checkpoint": str(args.checkpoint.resolve()),
        "source_checkpoint_sha256": sha256(args.checkpoint),
        "source_selected_epoch": int(source_checkpoint["epoch"]),
        "normalization": normalization,
        "mean": list(mean),
        "std": list(std),
        "estimator": "per_example_empirical_fisher_vmap",
        "formula": "mean_i(square(gradient_i(negative_log_likelihood)))",
        "uses_ground_truth_labels": True,
        "model_mode": "eval",
        "augmentation": "none",
        "data_csv": str(args.train_csv.resolve()),
        "samples": len(fisher_dataset),
        "class_counts": {"NF": nf_count, "FL": fl_count},
        "batch_size": args.fisher_batch_size,
        "num_workers": args.num_workers,
        "seconds": elapsed,
        "statistics": statistics,
    }

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
            f"phase2-stage1-ewc-anchor-{run.id}",
            type="ewc-anchor",
            metadata=metadata,
        )
        artifact.add_file(str(args.output))
        artifact.add_file(str(metadata_path))
        run.log_artifact(artifact)
        run.finish()


if __name__ == "__main__":
    main()
