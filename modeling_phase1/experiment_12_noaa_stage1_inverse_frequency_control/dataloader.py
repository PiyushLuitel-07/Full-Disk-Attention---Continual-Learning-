"""Unaugmented NOAA magnetogram data loading for Experiment 12."""

from pathlib import Path

import pandas as pd
import torch
import torchvision.transforms.functional as TF
from PIL import Image
from torch.utils.data import DataLoader, Dataset


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class MagnetogramDataset(Dataset):
    """Load original 512x512 grayscale magnetograms and NF/FL targets."""

    def __init__(self, csv_file, image_directory):
        rows = pd.read_csv(csv_file)
        self.paths = [Path(image_directory) / value for value in rows["label"]]
        self.targets = rows["goes_class"].astype(int).tolist()

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        with Image.open(self.paths[index]) as image_file:
            image = image_file.convert("L")
            # Repeating grayscale data gives MobileNet three input channels
            # without inventing new visual information.
            tensor = TF.to_tensor(image).repeat(3, 1, 1)

        tensor = TF.normalize(tensor, IMAGENET_MEAN, IMAGENET_STD)
        return tensor, torch.tensor(self.targets[index], dtype=torch.long)


def _loader(dataset, batch_size, num_workers, shuffle=False):
    settings = {
        "dataset": dataset,
        "batch_size": batch_size,
        "shuffle": shuffle,
        "num_workers": num_workers,
        "pin_memory": True,
    }
    if num_workers > 0:
        settings.update(persistent_workers=True, prefetch_factor=2)
    return DataLoader(**settings)


def build_loaders(train_csv, holdout_csv, image_directory, batch_size, num_workers):
    """Build full-data training and chronological holdout loaders."""
    train_data = MagnetogramDataset(train_csv, image_directory)
    holdout_data = MagnetogramDataset(holdout_csv, image_directory)
    return {
        "train": _loader(train_data, batch_size, num_workers, shuffle=True),
        "holdout": _loader(holdout_data, batch_size, num_workers),
        "class_counts": torch.bincount(
            torch.tensor(train_data.targets), minlength=2
        ),
    }


def build_evaluation_loader(csv_file, image_directory, batch_size, num_workers):
    """Build an unshuffled, unaugmented evaluation loader."""
    return _loader(
        MagnetogramDataset(csv_file, image_directory),
        batch_size,
        num_workers,
    )
