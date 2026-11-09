"""Original-image NOAA magnetogram loading for corrected Stage 2 EWC."""

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
        rows = pd.read_csv(csv_file)
        required_columns = {"label", "goes_class"}
        missing_columns = required_columns.difference(rows.columns)
        if missing_columns:
            raise ValueError(
                f"{csv_file} is missing required columns: {sorted(missing_columns)}"
            )

        self.paths = [Path(image_directory) / value for value in rows["label"]]
        self.targets = rows["goes_class"].astype(int).tolist()
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
):
    """Build one loader; only current-stage training is shuffled."""
    dataset = MagnetogramDataset(csv_file, image_directory)
    settings = {
        "dataset": dataset,
        "batch_size": batch_size,
        "shuffle": shuffle,
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
    }
    if num_workers > 0:
        settings.update(persistent_workers=True, prefetch_factor=2)
    return DataLoader(**settings)


def class_counts(dataset):
    """Return [NF, FL] counts for transparent per-split logging."""
    return torch.bincount(torch.tensor(dataset.targets), minlength=2).tolist()


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
        "stage2_train": build_loader(
            paths["stage2_train_csv"],
            image_directory,
            batch_size,
            workers,
            shuffle=True,
        ),
        "stage2_holdout": build_loader(
            paths["stage2_holdout_csv"], image_directory, batch_size, workers
        ),
        "future_validation": build_loader(
            paths["future_validation_csv"], image_directory, batch_size, workers
        ),
    }
    return loaders
