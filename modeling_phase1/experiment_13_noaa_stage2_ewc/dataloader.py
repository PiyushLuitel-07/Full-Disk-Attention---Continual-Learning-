"""Original-image NOAA magnetogram loading for corrected Stage 2 EWC."""

from pathlib import Path

import pandas as pd
import torch
import torchvision.transforms.functional as TF
from PIL import Image
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class MagnetogramDataset(Dataset):
    """Load original grayscale HMI magnetograms and binary NF/FL labels."""

    def __init__(self, csv_file, image_directory):
        rows = pd.read_csv(csv_file)
        required_columns = {"label", "goes_class"}
        missing_columns = required_columns.difference(rows.columns)
        if missing_columns:
            raise ValueError(
                f"{csv_file} is missing required columns: {sorted(missing_columns)}"
            )

        self.paths = [Path(image_directory) / value for value in rows["label"]]
        self.targets = rows["goes_class"].astype(int).tolist()
        # Image paths begin with YYYY/MM/DD. These timestamps are available in
        # the Stage 2 *training* CSV itself, so recency sampling never needs
        # to inspect the later holdout period.
        self.timestamps = pd.to_datetime(
            rows["label"].str.slice(0, 10), format="%Y/%m/%d", errors="raise"
        ).tolist()
        invalid_targets = set(self.targets).difference({0, 1})
        if invalid_targets:
            raise ValueError(f"Expected NF/FL labels 0 or 1, found {invalid_targets}")

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        with Image.open(self.paths[index]) as image_file:
            # Stage 1 was trained on grayscale HMI data replicated across the
            # three MobileNet input channels. Stage 2 must use the same input
            # representation to retain compatibility with its checkpoint.
            image = image_file.convert("L")
            tensor = TF.to_tensor(image).repeat(3, 1, 1)

        tensor = TF.normalize(tensor, IMAGENET_MEAN, IMAGENET_STD)
        target = torch.tensor(self.targets[index], dtype=torch.long)
        return tensor, target


def build_loader(
    csv_file,
    image_directory,
    batch_size,
    num_workers,
    *,
    shuffle=False,
    sampler=None,
):
    """Build one loader; a sampler and ``shuffle`` are mutually exclusive."""
    dataset = MagnetogramDataset(csv_file, image_directory)
    settings = {
        "dataset": dataset,
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
    }
    if sampler is None:
        settings["shuffle"] = shuffle
    else:
        settings["sampler"] = sampler
    if num_workers > 0:
        settings.update(persistent_workers=True, prefetch_factor=2)
    return DataLoader(**settings)


def class_counts(dataset):
    """Return [NF, FL] counts for transparent per-split logging."""
    return torch.bincount(torch.tensor(dataset.targets), minlength=2).tolist()


def build_recency_sampler(dataset, half_life_months, seed):
    """Sample recent Stage 2 training images more often, with replacement.

    Every original image remains eligible. A sample one half-life older than
    the most recent Stage 2 training example receives half its relative draw
    weight. ``num_samples=len(dataset)`` preserves the usual epoch length.
    """
    if half_life_months <= 0:
        raise ValueError("recency half-life must be positive.")

    dates = pd.DatetimeIndex(dataset.timestamps)
    reference_date = dates.max()
    ages_days = (reference_date - dates).days.to_numpy(dtype=float)
    half_life_days = float(half_life_months) * 30.4375
    weights = torch.as_tensor(0.5 ** (ages_days / half_life_days), dtype=torch.double)
    generator = torch.Generator().manual_seed(int(seed))
    sampler = WeightedRandomSampler(
        weights=weights,
        num_samples=len(dataset),
        replacement=True,
        generator=generator,
    )

    targets = torch.as_tensor(dataset.targets, dtype=torch.double)
    normalized_weights = weights / weights.sum()
    metadata = {
        "method": "exponential_recency_weighted_sampling",
        "half_life_months": float(half_life_months),
        "half_life_days": half_life_days,
        "reference_date": reference_date.strftime("%Y-%m-%d"),
        "oldest_training_date": dates.min().strftime("%Y-%m-%d"),
        "epoch_samples": len(dataset),
        "replacement": True,
        "minimum_relative_weight": float(weights.min().item()),
        "maximum_relative_weight": float(weights.max().item()),
        "expected_flare_fraction": float((normalized_weights * targets).sum().item()),
        "expected_mean_age_days": float(
            (normalized_weights * torch.as_tensor(ages_days)).sum().item()
        ),
    }
    return sampler, metadata


def build_stage2_loaders(config, paths):
    """Build all data views needed for Stage 2 EWC.

    ``stage1_fisher`` and all evaluation loaders are unshuffled. Only
    ``stage2_train`` is shuffled, because its order is irrelevant to the
    definition of either Fisher information or chronological evaluation.
    """
    image_directory = paths["image_directory"]
    batch_size = int(config["batch_size"])
    workers = int(config["num_workers"])
    fisher_workers = int(config["fisher_num_workers"])

    stage2_train_dataset = MagnetogramDataset(paths["stage2_train_csv"], image_directory)
    half_life_months = float(config.get("recency_half_life_months", 0.0))
    if half_life_months > 0:
        stage2_sampler, stage2_sampling_metadata = build_recency_sampler(
            stage2_train_dataset,
            half_life_months=half_life_months,
            seed=int(config["random_seed"]),
        )
    else:
        stage2_sampler = None
        stage2_sampling_metadata = {
            "method": "uniform_shuffle_without_replacement",
            "epoch_samples": len(stage2_train_dataset),
            "replacement": False,
        }

    loaders = {
        "stage1_fisher": build_loader(
            paths["stage1_train_csv"],
            image_directory,
            batch_size=int(config["fisher_batch_size"]),
            num_workers=fisher_workers,
            shuffle=False,
        ),
        "stage1_holdout": build_loader(
            paths["stage1_holdout_csv"], image_directory, batch_size, workers
        ),
        "stage2_train": DataLoader(
            stage2_train_dataset,
            batch_size=batch_size,
            shuffle=False if stage2_sampler is not None else True,
            sampler=stage2_sampler,
            num_workers=workers,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=workers > 0,
            prefetch_factor=2 if workers > 0 else None,
        ),
        "stage2_holdout": build_loader(
            paths["stage2_holdout_csv"], image_directory, batch_size, workers
        ),
        "future_validation": build_loader(
            paths["future_validation_csv"], image_directory, batch_size, workers
        ),
    }
    return loaders, stage2_sampling_metadata
