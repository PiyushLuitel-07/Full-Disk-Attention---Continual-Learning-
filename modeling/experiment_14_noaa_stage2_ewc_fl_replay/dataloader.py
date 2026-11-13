"""Data loading for Stage 2 EWC with Stage 1 FL-only replay."""

from pathlib import Path

import pandas as pd
import torch
import torchvision.transforms.functional as TF
from PIL import Image
from torch.utils.data import DataLoader, Dataset


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class MagnetogramDataset(Dataset):
    """Load original grayscale HMI magnetograms and binary NF/FL labels."""

    def __init__(self, csv_file, image_directory):
        self.csv_file = Path(csv_file)
        rows = pd.read_csv(self.csv_file)
        required_columns = {"label", "goes_class"}
        missing_columns = required_columns.difference(rows.columns)
        if missing_columns:
            raise ValueError(
                f"{self.csv_file} is missing required columns: "
                f"{sorted(missing_columns)}"
            )
        if rows.empty:
            raise ValueError(f"{self.csv_file} contains no samples.")

        self.relative_paths = rows["label"].astype(str).tolist()
        self.paths = [Path(image_directory) / value for value in self.relative_paths]
        self.targets = rows["goes_class"].astype(int).tolist()

        invalid_targets = set(self.targets).difference({0, 1})
        if invalid_targets:
            raise ValueError(
                f"Expected NF/FL labels 0 or 1 in {self.csv_file}, "
                f"found {sorted(invalid_targets)}"
            )

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        with Image.open(self.paths[index]) as image_file:
            # Match the selected Stage 1 model exactly: one HMI grayscale
            # channel replicated into the three ImageNet input channels.
            image = image_file.convert("L")
            tensor = TF.to_tensor(image).repeat(3, 1, 1)

        tensor = TF.normalize(tensor, IMAGENET_MEAN, IMAGENET_STD)
        target = torch.tensor(self.targets[index], dtype=torch.long)
        return tensor, target


def class_counts(dataset):
    """Return [NF, FL] counts for transparent split logging."""
    return torch.bincount(torch.tensor(dataset.targets), minlength=2).tolist()


def validate_fl_replay_manifest(replay_csv, stage1_train_csv, image_directory):
    """Reject leakage, duplicates, and non-FL rows in the replay manifest.

    Replay is allowed to contain only unique FL images from Stage 1 training.
    Stage 1 holdout, Stage 2, validation, and test images therefore cannot
    enter the replay buffer accidentally.
    """
    replay_csv = Path(replay_csv)
    stage1_train_csv = Path(stage1_train_csv)
    replay_rows = pd.read_csv(replay_csv)
    stage1_rows = pd.read_csv(stage1_train_csv)

    required_columns = {"label", "goes_class"}
    for csv_path, rows in (
        (replay_csv, replay_rows),
        (stage1_train_csv, stage1_rows),
    ):
        missing_columns = required_columns.difference(rows.columns)
        if missing_columns:
            raise ValueError(
                f"{csv_path} is missing required columns: {sorted(missing_columns)}"
            )

    if replay_rows.empty:
        raise ValueError(f"Replay manifest is empty: {replay_csv}")
    if replay_rows["label"].isna().any():
        raise ValueError("Replay manifest contains missing image paths.")
    if replay_rows["label"].duplicated().any():
        duplicate_count = int(replay_rows["label"].duplicated().sum())
        raise ValueError(
            f"Replay manifest contains {duplicate_count} duplicate image paths."
        )
    if not replay_rows["goes_class"].astype(int).eq(1).all():
        counts = replay_rows["goes_class"].value_counts(dropna=False).to_dict()
        raise ValueError(f"FL-only replay manifest contains non-FL rows: {counts}")

    stage1_fl_paths = set(
        stage1_rows.loc[
            stage1_rows["goes_class"].astype(int).eq(1), "label"
        ].astype(str)
    )
    replay_paths = set(replay_rows["label"].astype(str))
    outside_stage1_fl = sorted(replay_paths.difference(stage1_fl_paths))
    if outside_stage1_fl:
        preview = outside_stage1_fl[:5]
        raise ValueError(
            f"Replay contains {len(outside_stage1_fl)} images that are not "
            f"Stage 1 training FL samples. First paths: {preview}"
        )

    image_directory = Path(image_directory)
    missing_images = [
        relative_path
        for relative_path in sorted(replay_paths)
        if not (image_directory / relative_path).is_file()
    ]
    if missing_images:
        raise FileNotFoundError(
            f"Replay contains {len(missing_images)} missing image files under "
            f"{image_directory}. First paths: {missing_images[:5]}"
        )

    temporal_metadata = {}
    if "time_window" in replay_rows.columns:
        parsed_windows = pd.to_datetime(replay_rows["time_window"], errors="coerce")
        if parsed_windows.isna().any():
            raise ValueError("Replay manifest contains invalid time_window values.")
        if parsed_windows.duplicated().any():
            duplicate_count = int(parsed_windows.duplicated().sum())
            raise ValueError(
                f"Replay manifest contains {duplicate_count} duplicate time windows."
            )
        temporal_metadata["unique_time_windows"] = int(parsed_windows.nunique())

    return {
        "validated": True,
        "source": "stage1_training_fl_only",
        "samples": int(len(replay_rows)),
        "nf_samples": 0,
        "fl_samples": int(len(replay_rows)),
        "unique_image_paths": int(replay_rows["label"].nunique()),
        "all_image_files_exist": True,
        **temporal_metadata,
    }


def build_loader(
    dataset,
    batch_size,
    num_workers,
    *,
    shuffle=False,
    drop_last=False,
    seed=None,
):
    """Build one deterministic loader from an already validated dataset."""
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")

    settings = {
        "dataset": dataset,
        "batch_size": int(batch_size),
        "shuffle": bool(shuffle),
        "drop_last": bool(drop_last),
        "num_workers": int(num_workers),
        "pin_memory": torch.cuda.is_available(),
    }
    if seed is not None:
        settings["generator"] = torch.Generator().manual_seed(int(seed))
    if num_workers > 0:
        settings.update(persistent_workers=True, prefetch_factor=2)
    return DataLoader(**settings)


def cycle_batches(loader):
    """Yield replay batches indefinitely, reshuffling after each full pass."""
    if loader is None:
        raise ValueError("Cannot cycle a disabled replay loader.")
    while True:
        yield from loader


def build_experiment_loaders(config, paths):
    """Build current-domain, replay, retention, and evaluation loaders."""
    image_directory = paths["image_directory"]
    workers = int(config["num_workers"])
    stage2_batch_size = int(config["stage2_batch_size"])
    replay_batch_size = int(config["replay_batch_size"])
    seed = int(config["random_seed"])

    if stage2_batch_size <= 0:
        raise ValueError("stage2_batch_size must be positive.")
    if replay_batch_size < 0:
        raise ValueError("replay_batch_size cannot be negative.")
    if stage2_batch_size + replay_batch_size != int(config["total_batch_size"]):
        raise ValueError(
            "stage2_batch_size + replay_batch_size must equal total_batch_size."
        )

    stage1_holdout_dataset = MagnetogramDataset(
        paths["stage1_holdout_csv"], image_directory
    )
    stage2_train_dataset = MagnetogramDataset(
        paths["stage2_train_csv"], image_directory
    )
    stage2_holdout_dataset = MagnetogramDataset(
        paths["stage2_holdout_csv"], image_directory
    )
    future_validation_dataset = MagnetogramDataset(
        paths["future_validation_csv"], image_directory
    )

    stage2_train_loader = build_loader(
        stage2_train_dataset,
        batch_size=stage2_batch_size,
        num_workers=workers,
        shuffle=True,
        drop_last=False,
        seed=seed,
    )

    replay_loader = None
    if replay_batch_size > 0:
        replay_validation = validate_fl_replay_manifest(
            paths["replay_csv"], paths["stage1_train_csv"], image_directory
        )
        replay_dataset = MagnetogramDataset(paths["replay_csv"], image_directory)
        if replay_batch_size > len(replay_dataset):
            raise ValueError(
                f"replay_batch_size={replay_batch_size} exceeds the "
                f"{len(replay_dataset)}-image replay buffer."
            )
        replay_loader = build_loader(
            replay_dataset,
            batch_size=replay_batch_size,
            num_workers=workers,
            shuffle=True,
            # Keep every replay batch the requested size. The iterator cycles
            # and reshuffles, so omitted tail items remain eligible next pass.
            drop_last=True,
            seed=seed + 1,
        )
        replay_draws = len(stage2_train_loader) * replay_batch_size
        replay_metadata = {
            **replay_validation,
            "enabled": True,
            "batch_size": replay_batch_size,
            "batches_per_stage2_epoch": len(stage2_train_loader),
            "examples_drawn_per_stage2_epoch": replay_draws,
            "expected_reuses_per_stage2_epoch": replay_draws / len(replay_dataset),
            "drop_last_per_replay_pass": True,
        }
    else:
        replay_draws = 0
        replay_metadata = {
            "enabled": False,
            "samples": 0,
            "nf_samples": 0,
            "fl_samples": 0,
            "batch_size": 0,
            "batches_per_stage2_epoch": len(stage2_train_loader),
            "examples_drawn_per_stage2_epoch": 0,
            "expected_reuses_per_stage2_epoch": 0.0,
        }

    stage2_counts = class_counts(stage2_train_dataset)
    examples_per_epoch = len(stage2_train_dataset) + replay_draws
    positive_examples_per_epoch = stage2_counts[1] + replay_draws
    sampling_metadata = {
        "stage2": {
            "method": "uniform_shuffle_without_replacement",
            "samples": len(stage2_train_dataset),
            "nf_samples": stage2_counts[0],
            "fl_samples": stage2_counts[1],
            "batch_size": stage2_batch_size,
            "drop_last": False,
        },
        "replay": replay_metadata,
        "mixed_training": {
            "maximum_batch_size": int(config["total_batch_size"]),
            "optimization_steps_per_epoch": len(stage2_train_loader),
            "examples_per_epoch": examples_per_epoch,
            "expected_fl_examples_per_epoch": positive_examples_per_epoch,
            "expected_fl_fraction": (
                positive_examples_per_epoch / examples_per_epoch
                if examples_per_epoch
                else 0.0
            ),
        },
    }

    loaders = {
        "stage1_holdout": build_loader(
            stage1_holdout_dataset,
            batch_size=int(config["total_batch_size"]),
            num_workers=workers,
        ),
        "stage2_train": stage2_train_loader,
        "stage1_replay": replay_loader,
        "stage2_holdout": build_loader(
            stage2_holdout_dataset,
            batch_size=int(config["total_batch_size"]),
            num_workers=workers,
        ),
        "future_validation": build_loader(
            future_validation_dataset,
            batch_size=int(config["total_batch_size"]),
            num_workers=workers,
        ),
    }
    return loaders, sampling_metadata
